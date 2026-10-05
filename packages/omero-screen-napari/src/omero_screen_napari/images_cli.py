"""Click command-line interface for headless image outputs: ``omero-screen-images``.

Renders the same images the napari widgets produce, without a viewer, so a
script or an agent working from an analysis notebook can regenerate them
reproducibly:

- ``gallery`` — per-well cell galleries, optionally restricted to one
  classifier class or to an explicit cell selection.

Every run writes a JSON manifest next to the images recording the settings,
seed, display limits and per-well outcome; ``--json`` also prints it.

The exported :data:`cli` group must stay importable without pulling in
napari, Qt or matplotlib's GUI backends. The rendering modules are imported
inside the command callbacks, after ``--env`` has been applied and the Agg
backend selected.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

    from omero_screen_napari.omero_data import OmeroData

CELLCYCLE_PHASES = ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]
CELLS_KEY_COLUMNS = ("image_id", "label")


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--env",
    default=None,
    help="Environment name (loads the configuration file .env.{name}).",
)
def cli(env: str | None) -> None:
    """Render cell galleries from OMERO-Screen plates without napari."""
    if env:
        os.environ["ENV"] = env
        from omero_screen.config import set_env_vars

        set_env_vars()
    import matplotlib

    matplotlib.use("Agg")


@cli.command()
@click.argument("plate_id", type=int)
@click.option(
    "--wells",
    default=None,
    help=(
        "Comma-separated wells, e.g. 'E2,G5', or 'All'. Defaults to the "
        "wells in --cells, otherwise required."
    ),
)
@click.option(
    "--channels",
    required=True,
    help=(
        "Comma-separated channel names, packed into R,G,B in order. One "
        "channel renders an inverted greyscale gallery."
    ),
)
@click.option(
    "--classifier-column",
    default="",
    help="Classifier column to filter on, e.g. classifier_nuclei4.",
)
@click.option(
    "--class",
    "class_value",
    default="",
    help="Class to keep in --classifier-column, e.g. micronuclei.",
)
@click.option(
    "--cellcycle",
    type=click.Choice(CELLCYCLE_PHASES),
    default="All",
    show_default=True,
    help="Cell-cycle phase to keep.",
)
@click.option(
    "--cells",
    "cells_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "CSV or Parquet of cells to draw from: columns image_id and label "
        "(the CellView nucleus label), optionally well and timepoint."
    ),
)
@click.option(
    "--grid",
    default="4x4",
    show_default=True,
    help="Gallery size as ROWSxCOLUMNS.",
)
@click.option(
    "--crop-size",
    type=click.IntRange(min=4),
    default=50,
    show_default=True,
    help="Crop edge length in pixels.",
)
@click.option(
    "--segmentation",
    type=click.Choice(["nucleus", "cell"]),
    default="nucleus",
    show_default=True,
    help="Mask that centres and outlines each crop.",
)
@click.option(
    "--timepoint",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="0-based timepoint for timelapse plates.",
)
@click.option(
    "--contour/--no-contour",
    default=True,
    show_default=True,
    help="Outline the mask in each crop.",
)
@click.option(
    "--keep-background",
    is_flag=True,
    help=(
        "Keep the pixels around the cell instead of blanking them. Needed "
        "to see objects outside the cell's own mask, e.g. micronuclei."
    ),
)
@click.option(
    "--limits",
    "limit_specs",
    multiple=True,
    metavar="CHANNEL=LO:HI",
    help=(
        "Fixed display limits for a channel, e.g. DAPI=200:12000; "
        "repeatable. Default: 0.1/99.9 percentiles pooled over the "
        "requested wells (zarr plates) or the plate's CellView intensity "
        "range (per-field plates). The limits used are in the manifest."
    ),
)
@click.option(
    "--title/--no-title",
    default=True,
    show_default=True,
    help="Print the well/settings header above the gallery.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for the cell draw; the same seed reproduces the gallery.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("galleries"),
    show_default=True,
    help="Output directory for the images and the manifest.",
)
@click.option(
    "--fmt",
    type=click.Choice(["pdf", "png", "svg", "tif"]),
    default="pdf",
    show_default=True,
    help="Image format.",
)
@click.option(
    "--dpi",
    type=click.IntRange(min=1),
    default=300,
    show_default=True,
    help="Raster resolution.",
)
@click.option(
    "--json",
    "print_json",
    is_flag=True,
    help="Print the manifest to stdout (logs go to stderr).",
)
def gallery(
    plate_id: int,
    wells: str | None,
    channels: str,
    classifier_column: str,
    class_value: str,
    cellcycle: str,
    cells_path: Path | None,
    grid: str,
    crop_size: int,
    segmentation: str,
    timepoint: int,
    contour: bool,
    keep_background: bool,
    limit_specs: tuple[str, ...],
    title: bool,
    seed: int,
    out_dir: Path,
    fmt: str,
    dpi: int,
    print_json: bool,
) -> None:
    """Render one cell gallery per well of PLATE_ID.

    Cells come from CellView; pixels from the plate's zarr cache when it
    has one, otherwise from the per-field images (downloaded from OMERO
    on a cache miss, one well at a time).

    \b
    Example:
      omero-screen-images gallery 5108 --wells E2,G5 \\
          --classifier-column classifier_nuclei4 --class micronuclei \\
          --channels DAPI --grid 5x5 --seed 1 --out qc/ --json
    """
    if print_json:
        # Library code (CellView's console, for one) prints to stdout; keep
        # stdout for the manifest alone so it can be piped into a parser.
        with contextlib.redirect_stdout(sys.stderr):
            manifest_path, written, n_wells = _render_gallery(**locals())
        click.echo(manifest_path.read_text())
    else:
        manifest_path, written, n_wells = _render_gallery(**locals())
        click.echo(
            f"Wrote {written}/{n_wells} galleries to {manifest_path.parent} "
            f"(manifest: {manifest_path.name})",
            err=True,
        )
    if not written:
        sys.exit(1)


def _render_gallery(
    plate_id: int,
    wells: str | None,
    channels: str,
    classifier_column: str,
    class_value: str,
    cellcycle: str,
    cells_path: Path | None,
    grid: str,
    crop_size: int,
    segmentation: str,
    timepoint: int,
    contour: bool,
    keep_background: bool,
    limit_specs: tuple[str, ...],
    title: bool,
    seed: int,
    out_dir: Path,
    fmt: str,
    dpi: int,
    print_json: bool,
) -> tuple[Path, int, int]:
    """Do the work of :func:`gallery`; return (manifest, written, wells)."""
    rows, columns = _parse_grid(grid)
    channel_list = _split_list(channels)
    if not channel_list:
        raise click.BadParameter(
            "give at least one channel", param_hint="--channels"
        )
    if len(channel_list) > 3:
        raise click.BadParameter(
            "at most three channels (R, G, B)", param_hint="--channels"
        )
    if class_value and not classifier_column:
        raise click.BadParameter(
            "--class needs --classifier-column", param_hint="--class"
        )

    from omero_screen_napari import __version__
    from omero_screen_napari.gallery_export import (
        MANIFEST_NAME,
        export_galleries,
    )
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroConnection, OmeroData
    from omero_screen_napari.plate_cache import _load_plate_data_from_cellview
    from omero_screen_napari.well_context import (
        WellContextError,
        load_well_context,
        well_source,
    )

    limits = _parse_limits(limit_specs)
    cells = _read_cells(cells_path) if cells_path is not None else None
    source = well_source(plate_id)
    plate_wells = _cellview_wells(
        _restrict_to_cells(_load_plate_data_from_cellview(plate_id), cells)
    )
    if not plate_wells:
        raise click.ClickException(
            f"Plate {plate_id} has no CellView rows"
            + (" for the cells in --cells." if cells is not None else ".")
            + " Import it into CellView first."
        )
    target_wells = _resolve_wells(wells, plate_wells, cells, plate_id)

    omero_data = OmeroData()
    connection = OmeroConnection()

    def load(well_list: list[str]) -> None:
        load_well_context(
            plate_id,
            well_list,
            omero_data=omero_data,
            timepoint=timepoint,
            connection=connection,
        )
        omero_data.plate_data = _restrict_to_cells(
            omero_data.plate_data, cells
        )
        _apply_limits(limits, omero_data)

    # A zarr plate loads its metadata once for all wells; a per-field
    # plate loads one well's fields at a time to bound memory.
    try:
        load(target_wells if source == "zarr" else target_wells[:1])
    except WellContextError as exc:
        raise click.ClickException(str(exc)) from exc
    _check_channels([*channel_list, *limits], omero_data)

    def prepare_well(well: str) -> None:
        if source == "fields" and omero_data.well_pos_list != [well]:
            load([well])

    user_data = UserData(
        segmentation=segmentation,
        reload=True,
        crop_size=crop_size,
        cellcycle=cellcycle,
        classifier_filter=class_value,
        classifier_column=classifier_column,
        timepoint=timepoint,
        columns=columns,
        rows=rows,
        contour=contour,
        no_background=not keep_background,
        show_title=title,
        channels=channel_list,
    )
    written = export_galleries(
        out_dir,
        target_wells,
        fmt=fmt,
        dpi=dpi,
        seed=seed,
        prepare_well=prepare_well,
        manifest_extra={
            "command": "gallery",
            "omero_screen_version": __version__,
            "source": source,
            "cells_file": str(cells_path) if cells_path is not None else None,
        },
        omero_data=omero_data,
        user_data=user_data,
    )

    return (
        out_dir.expanduser() / MANIFEST_NAME,
        len(written),
        len(target_wells),
    )


# ---------------------------------------------------------------------------
# Helpers (pure; unit-tested directly)
# ---------------------------------------------------------------------------


def _split_list(value: str) -> list[str]:
    """Split a comma-separated option value, dropping blanks."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_grid(grid: str) -> tuple[int, int]:
    """Parse ``ROWSxCOLUMNS`` (e.g. ``5x5``) into positive integers."""
    parts = grid.lower().replace("×", "x").split("x")
    try:
        rows, columns = (int(p) for p in parts)
    except ValueError:
        raise click.BadParameter(
            f"expected ROWSxCOLUMNS, e.g. 5x5, got {grid!r}",
            param_hint="--grid",
        ) from None
    if rows < 1 or columns < 1:
        raise click.BadParameter(
            "rows and columns must be at least 1", param_hint="--grid"
        )
    return rows, columns


def _read_cells(path: Path) -> pl.DataFrame:
    """Read a cell-selection table; require ``image_id`` and ``label``."""
    import polars as pl

    if path.suffix.lower() in {".parquet", ".pq"}:
        cells = pl.read_parquet(path)
    else:
        cells = pl.read_csv(path)
    missing = [c for c in CELLS_KEY_COLUMNS if c not in cells.columns]
    if missing:
        raise click.BadParameter(
            f"{path.name} is missing column(s) {', '.join(missing)}",
            param_hint="--cells",
        )
    if cells.height == 0:
        raise click.BadParameter(
            f"{path.name} has no rows", param_hint="--cells"
        )
    return cells


def _cells_keys(cells: pl.DataFrame, plate_columns: list[str]) -> list[str]:
    """Columns to match cells on: the key, plus timepoint when both have it."""
    keys = list(CELLS_KEY_COLUMNS)
    if "timepoint" in cells.columns and "timepoint" in plate_columns:
        keys.append("timepoint")
    return keys


def _restrict_to_cells(
    plate_data: pl.LazyFrame, cells: pl.DataFrame | None
) -> pl.LazyFrame:
    """Keep only the CellView rows named in ``cells`` (no-op without it).

    Keys are compared as strings: CellView may store multi-nucleate labels
    as stringified lists, and a CSV round trip can change integer widths.
    """
    import polars as pl

    if cells is None:
        return plate_data
    names = plate_data.collect_schema().names()
    if not names:
        return plate_data
    keys = _cells_keys(cells, names)
    selection = cells.select(
        [pl.col(k).cast(pl.Utf8).alias(f"__sel_{k}") for k in keys]
    ).unique()
    return (
        plate_data.with_columns(
            [pl.col(k).cast(pl.Utf8).alias(f"__sel_{k}") for k in keys]
        )
        .join(
            selection.lazy(),
            on=[f"__sel_{k}" for k in keys],
            how="semi",
        )
        .drop([f"__sel_{k}" for k in keys])
    )


def _cellview_wells(plate_data: pl.LazyFrame) -> list[str]:
    """Wells with CellView rows, sorted; empty for an empty frame."""
    names = plate_data.collect_schema().names()
    if "well" not in names:
        return []
    return sorted(
        str(w)
        for w in plate_data.select("well").unique().collect()["well"].to_list()
    )


def _resolve_wells(
    wells: str | None,
    plate_wells: list[str],
    cells: pl.DataFrame | None,
    plate_id: int,
) -> list[str]:
    """Turn ``--wells`` into the wells to render, validated against CellView."""
    if wells is None:
        if cells is None:
            raise click.UsageError("give --wells (or --cells to derive them)")
        return plate_wells
    if wells.strip().lower() == "all":
        return plate_wells
    requested = [w.upper() for w in _split_list(wells)]
    missing = [w for w in requested if w not in plate_wells]
    if missing:
        raise click.BadParameter(
            f"no CellView rows for {', '.join(missing)} in plate {plate_id}"
            + (" among the --cells selection" if cells is not None else "")
            + f". Wells available: {', '.join(plate_wells)}.",
            param_hint="--wells",
        )
    return requested


def _parse_limits(specs: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Parse ``CHANNEL=LO:HI`` specs into ``{channel: (lo, hi)}``."""
    limits: dict[str, tuple[int, int]] = {}
    for spec in specs:
        name, sep, window = spec.partition("=")
        lo_text, colon, hi_text = window.partition(":")
        try:
            if not (sep and colon and name.strip()):
                raise ValueError
            lo, hi = int(lo_text), int(hi_text)
        except ValueError:
            raise click.BadParameter(
                f"expected CHANNEL=LO:HI, e.g. DAPI=200:12000, got {spec!r}",
                param_hint="--limits",
            ) from None
        if hi <= lo:
            raise click.BadParameter(
                f"{spec!r}: HI must be greater than LO", param_hint="--limits"
            )
        limits[name.strip()] = (lo, hi)
    return limits


def _apply_limits(
    limits: dict[str, tuple[int, int]], omero_data: OmeroData
) -> None:
    """Override the loaded display limits for the named channels."""
    if not limits:
        return
    intensities = dict(omero_data.intensities or {})
    for name, window in limits.items():
        index = (omero_data.channel_data or {}).get(name)
        if index is not None:
            intensities[int(float(index))] = window
    omero_data.intensities = intensities


def _check_channels(channels: list[str], omero_data: OmeroData) -> None:
    """Fail early on channel names the plate does not have."""
    available = list((omero_data.channel_data or {}).keys())
    unknown = [c for c in channels if c not in available]
    if unknown:
        raise click.BadParameter(
            f"unknown channel(s) {', '.join(unknown)}; plate channels: "
            f"{', '.join(available) or 'none'}",
            param_hint="--channels",
        )


def main() -> None:
    """Console-script entry point."""
    cli()
