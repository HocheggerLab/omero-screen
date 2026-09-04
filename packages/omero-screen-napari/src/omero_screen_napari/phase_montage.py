"""Per-well cell-cycle montage figures from the zarr cache.

A single example cell is not representative of a well: cell-cycle phase drives
nuclear size, DNA content and morphology, so which cell you happen to pick
decides what the figure says. This builds a montage instead -- a few cells from
each phase, one row per cell, so a reader sees the range rather than one draw
from it.

Layout, per page (one page per well)::

    G1     [DAPI+Tub composite, mask outline] [ch3 grey] [ch4 grey] ...
           ...4 cells...
    S      ...
    G2/M   ...
    Poly   ...

Two choices in here are about honesty rather than looks:

* **Cells are drawn at random within each phase**, with the seed stamped on the
  page and every cell's well/label printed under its row. Picking the cell
  closest to each phase's median gives prettier panels, but it selects on the
  outcome and understates real variability. Random panels include the
  out-of-focus and awkwardly segmented cells, because those are in the data.
* **Display limits are computed once per channel for the whole well** and
  applied to every crop. Scaling each crop to its own range would make a G1 and
  a polyploid nucleus equally bright, which is the opposite of what the data
  says.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger

from omero_screen_napari.zarr_cache.crop import (
    fetch_crop,
    fetch_label_crop,
    resolve_to_zarr,
)
from omero_screen_napari.zarr_cache.palette import base_name
from omero_screen_napari.zarr_cache.reader import (
    cached_wells,
    plate_info,
    read_well,
)

#: Phases shown, in cell-cycle order. Sub-G1 is excluded by default: it is
#: mostly apoptotic bodies and debris, so it reads as a row of noise rather
#: than a population. Pass it explicitly to include it.
DEFAULT_PHASES: tuple[str, ...] = ("G1", "S", "G2/M", "Polyploid")

#: Channel roles rendered as a colour composite in the first column. Everything
#: else is shown greyscale, which is what a reader can actually compare.
DEFAULT_OVERLAY: tuple[str, ...] = ("dapi", "tub")

#: Composite colours, matching the conventions used elsewhere in the plugin.
_OVERLAY_RGB: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.3, 1.0),  # nuclear -> blue
    (0.0, 1.0, 0.2),  # tubulin -> green
)

_OUTLINE_RGB = (1.0, 1.0, 0.25)


class MontageError(Exception):
    """Raised when a montage cannot be built for a well."""


@dataclass(frozen=True)
class MontageConfig:
    """Settings for a well montage.

    Attributes:
        phases: Phase names to show, in row order.
        cells_per_phase: Rows per phase.
        seed: RNG seed for the per-phase draw. Stamped on the figure so a
            montage can be regenerated exactly.
        crop_um: Crop edge in microns. ``None`` sizes it from the largest cell
            selected, so polyploid cells are not clipped by a constant chosen
            for G1 ones.
        overlay: Channel base names composited in the first column.
        mask: Label layer outlined, falling back to ``"nuclei"`` when a plate
            has no cell masks.
        percentiles: Low/high percentiles for the per-channel display limits.
        size_percentile: Percentile of the plate's cell-diameter distribution
            used to size the crop. High by design -- see :func:`_crop_pixels`.
        crop_factor: Multiple of that diameter to use as the crop edge.
        exclude_edge: Drop cells whose crop would run off the canvas. This is
            about the crop being incomplete, not about the cell, so it does not
            bias which cells are shown.
        pages: Axis that splits pages -- ``"well"``, ``"phase"`` or
            ``"condition"``.
        rows: Axis that groups rows. Must differ from ``pages``.
        condition_col: Column holding the phenotype. ``None`` auto-detects the
            condition variable that actually varies on the plate.
    """

    pages: str = "well"
    rows: str = "phase"
    condition_col: str | None = None
    phases: tuple[str, ...] = DEFAULT_PHASES
    cells_per_phase: int = 4
    seed: int = 0
    crop_um: float | None = None
    overlay: tuple[str, ...] = DEFAULT_OVERLAY
    mask: str = "cells"
    percentiles: tuple[float, float] = (0.1, 99.9)
    exclude_edge: bool = True
    size_percentile: float = 99.0
    crop_factor: float = 1.4


@dataclass(frozen=True)
class CellRef:
    """One selected cell."""

    phase: str
    well: str
    label: int
    centroid: tuple[float, float]
    timepoint: int = 0
    diameter: float | None = None
    """Cell extent in pixels, used to size the crop. None if unavailable."""
    condition: str = ""
    """Phenotype label, e.g. the siRNA. Empty when the plate has no variable."""


@dataclass
class WellMontage:
    """One built page, ready to render.

    A page is a slice along ``config.pages`` -- one well, one phase or one
    condition -- and its rows are groups along ``config.rows``. The name is kept
    for continuity with the original well-per-page layout.
    """

    plate_id: int
    page_label: str
    channel_names: list[str]
    overlay_indices: list[int]
    grey_indices: list[int]
    cells: dict[str, list[CellRef]]
    limits: dict[int, tuple[float, float]]
    crop_px: int
    pixel_size_um: float | None
    config: MontageConfig
    missing: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def _phase_column(df: pl.DataFrame) -> str:
    for candidate in ("cell_cycle", "cell_cycle_detailed"):
        if candidate in df.columns:
            return candidate
    raise MontageError(
        "No cell-cycle column in the CellView data for this plate. Run the "
        "cell-cycle analysis, or import the plate into CellView first."
    )


def _first_column(df: pl.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


#: Centroid columns, in preference order. omero-screen writes regionprops names
#: per compartment (``centroid-0-nuc`` / ``-cell``); the underscored forms are
#: what a plain single-round export carries. The **nucleus** centroid is
#: preferred as the crop centre: it is the segmentation anchor the cell-cycle
#: call is made on, and it stays inside the cell for elongated or lopsided
#: shapes where the whole-cell centroid can drift off the nucleus entirely.
_Y_COLUMNS = ("centroid_y", "centroid-0-nuc", "centroid-0", "centroid-0-cell")
_X_COLUMNS = ("centroid_x", "centroid-1-nuc", "centroid-1", "centroid-1-cell")

#: Columns giving a cell's extent, in preference order, paired with whether the
#: value is already a diameter. Whole-cell extent is preferred over nuclear:
#: the crop has to hold the cell, and in polyploid cells the two diverge.
_EXTENT_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("equivalent_diameter_area_cell", True),
    ("area_cell", False),
    ("area_convex_cell", False),
    ("area_cyto", False),
    ("equivalent_diameter_area_nucleus", True),
    ("area_nucleus", False),
    ("area", False),
)


def _extent_column(df: pl.DataFrame) -> tuple[str, bool] | None:
    """The best available cell-extent column and whether it is a diameter."""
    for name, is_diameter in _EXTENT_COLUMNS:
        if name in df.columns:
            return name, is_diameter
    return None


def _row_diameter(
    row: dict[str, Any], extent: tuple[str, bool] | None
) -> float | None:
    """Cell diameter in pixels from whichever extent column the plate carries."""
    if extent is None:
        return None
    name, is_diameter = extent
    value = row.get(name)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    # A region of area A has an equivalent diameter of 2*sqrt(A/pi).
    return numeric if is_diameter else 2.0 * float(np.sqrt(numeric / np.pi))


#: Condition columns to consider as the phenotype axis, in preference order.
#: These are the CellView condition variables; the one actually used is the
#: first that takes more than one value on the plate, since a column with a
#: single value carries no comparison.
_CONDITION_COLUMNS = (
    "condition",
    "sirna",
    "drug",
    "antibody",
    "cell_line",
    "stimulus",
)

#: Valid values for ``MontageConfig.pages`` / ``.rows``.
AXES = ("well", "phase", "condition")


def condition_column(
    df: pl.DataFrame, config: MontageConfig | None = None
) -> str | None:
    """The column holding the phenotype, or None if nothing varies.

    Auto-detection takes the first candidate that has more than one value: a
    column with a single value (``cell_line: RPE-1`` on plate 4127) describes
    the plate rather than distinguishing anything on it.
    """
    if config is not None and config.condition_col:
        if config.condition_col not in df.columns:
            raise MontageError(
                f"Condition column {config.condition_col!r} is not in the "
                f"CellView data. Available: "
                f"{[c for c in _CONDITION_COLUMNS if c in df.columns]}"
            )
        return config.condition_col
    for name in _CONDITION_COLUMNS:
        if name in df.columns and df[name].drop_nulls().n_unique() > 1:
            return name
    return None


def axis_column(df: pl.DataFrame, axis: str, config: MontageConfig) -> str:
    """Map an axis name to the DataFrame column that carries it."""
    if axis == "well":
        return "well"
    if axis == "phase":
        return _phase_column(df)
    if axis == "condition":
        column = condition_column(df, config)
        if column is None:
            raise MontageError(
                "This plate has no condition variable that varies, so there is "
                "nothing to compare across. Use --rows phase / --pages well, "
                "or name a column with --group-by."
            )
        return column
    raise MontageError(f"Unknown axis {axis!r}; expected one of {list(AXES)}")


def axis_values(
    df: pl.DataFrame, axis: str, config: MontageConfig
) -> list[str]:
    """Ordered values along an axis.

    Phases keep cell-cycle order; wells and conditions sort alphabetically so a
    figure is reproducible rather than following the database's row order.
    """
    if axis == "phase":
        return list(config.phases)
    column = axis_column(df, axis, config)
    return sorted(str(v) for v in df[column].drop_nulls().unique().to_list())


def _draw_spread_across_wells(
    subset: pl.DataFrame, count: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Take ``count`` rows, rotating through the wells the subset spans.

    A condition usually has several replicate wells. Drawing all of its cells
    from one well makes the row a portrait of that well rather than of the
    condition, so a well-specific artefact would read as a phenotype. Taking one
    cell per well in rotation shows the phenotype recurring across replicates.

    Falls back to a plain draw when the subset is confined to one well, which is
    what happens when the pages axis is ``well``.
    """
    wells = sorted(str(w) for w in subset["well"].unique().to_list())
    if len(wells) <= 1:
        picks = rng.choice(subset.height, size=count, replace=False)
        return subset[sorted(int(i) for i in picks)]

    per_well: dict[str, list[int]] = {}
    for well in wells:
        rows = subset.with_row_index("_idx").filter(pl.col("well") == well)
        indices = [int(i) for i in rows["_idx"].to_list()]
        rng.shuffle(indices)
        per_well[well] = indices

    chosen: list[int] = []
    position = 0
    while len(chosen) < count and any(per_well.values()):
        well = wells[position % len(wells)]
        if per_well[well]:
            chosen.append(per_well[well].pop())
        position += 1
    return subset[sorted(chosen)]


def select_cells(
    df: pl.DataFrame,
    well: str,
    config: MontageConfig,
    canvas_hw: tuple[int, int] | None = None,
    crop_px: int | None = None,
) -> tuple[dict[str, list[CellRef]], list[str]]:
    """Draw ``cells_per_phase`` cells at random from each phase.

    The draw is uniform within a phase and seeded, so the same well always
    yields the same cells. Phases with fewer cells than requested contribute
    what they have and are reported.

    Args:
        df: Measurements for one plate, as loaded from CellView.
        well: Well position.
        config: Montage settings.
        canvas_hw: Well canvas size, used with ``crop_px`` to drop cells whose
            crop would run off the edge.
        crop_px: Crop edge in pixels, for the same check.

    Returns:
        ``({phase: [CellRef, ...]}, warnings)``.

    Raises:
        MontageError: if the well has no measurements or no phase column.
    """
    phase_col = _phase_column(df)
    y_col = _first_column(df, *_Y_COLUMNS)
    x_col = _first_column(df, *_X_COLUMNS)
    if y_col is None or x_col is None:
        found = sorted(c for c in df.columns if "centroid" in c.lower())
        raise MontageError(
            "No usable centroid columns in the CellView data for this plate. "
            f"Looked for {list(_Y_COLUMNS)} and {list(_X_COLUMNS)}; the plate "
            f"has {found or 'no centroid columns at all'}."
        )
    extent = _extent_column(df)

    well_df = df.filter(pl.col("well") == well)
    if well_df.is_empty():
        raise MontageError(f"Well {well} has no measurements in CellView")

    if config.exclude_edge and canvas_hw is not None and crop_px is not None:
        half = crop_px / 2
        h, w = canvas_hw
        well_df = well_df.filter(
            (pl.col(y_col) >= half)
            & (pl.col(y_col) <= h - half)
            & (pl.col(x_col) >= half)
            & (pl.col(x_col) <= w - half)
        )

    rng = np.random.default_rng(config.seed)
    selected: dict[str, list[CellRef]] = {}
    warnings: list[str] = []
    cond_col = condition_column(df, config)
    for phase in config.phases:
        phase_df = well_df.filter(pl.col(phase_col) == phase)
        group, note = _draw_group(
            phase_df,
            phase,
            well,
            config,
            rng,
            cond_col,
            phase_col,
            y_col,
            x_col,
            extent,
        )
        selected[phase] = group
        warnings.extend(note)
    return selected, warnings


def _draw_group(
    subset: pl.DataFrame,
    label: str,
    context: str,
    config: MontageConfig,
    rng: np.random.Generator,
    cond_col: str | None,
    phase_col: str,
    y_col: str,
    x_col: str,
    extent: tuple[str, bool] | None,
) -> tuple[list[CellRef], list[str]]:
    """Draw one group's cells and turn them into :class:`CellRef`s."""
    warnings: list[str] = []
    n_available = subset.height
    if n_available == 0:
        return [], [f"{context}: no {label} cells"]
    take = min(config.cells_per_phase, n_available)
    if take < config.cells_per_phase:
        warnings.append(
            f"{context}: only {n_available} {label} cell(s), "
            f"wanted {config.cells_per_phase}"
        )
    # Sorted first so the draw does not depend on row order from the DB.
    rows = _draw_spread_across_wells(subset.sort(["well", "label"]), take, rng)
    cells = [
        CellRef(
            phase=str(row[phase_col]),
            well=str(row["well"]),
            label=int(row["label"]),
            centroid=(float(row[y_col]), float(row[x_col])),
            timepoint=int(row.get("timepoint", 0) or 0),
            diameter=_row_diameter(row, extent),
            condition=str(row[cond_col]) if cond_col else "",
        )
        for row in rows.iter_rows(named=True)
    ]
    return cells, warnings


def select_grid(
    df: pl.DataFrame,
    config: MontageConfig,
    canvas_lookup: Callable[[str], tuple[int, int] | None] | None = None,
    crop_px: int | None = None,
) -> tuple[dict[str, dict[str, list[CellRef]]], list[str]]:
    """Select cells for the whole ``pages`` x ``rows`` grid.

    Returns:
        ``({page_value: {row_value: [CellRef, ...]}}, warnings)``.

    Raises:
        MontageError: if the two axes are the same, or an axis is unknown.
    """
    if config.pages == config.rows:
        raise MontageError(
            f"pages and rows are both {config.pages!r}; they must differ, "
            f"otherwise every page has exactly one row"
        )
    phase_col = _phase_column(df)
    y_col = _first_column(df, *_Y_COLUMNS)
    x_col = _first_column(df, *_X_COLUMNS)
    if y_col is None or x_col is None:
        found = sorted(c for c in df.columns if "centroid" in c.lower())
        raise MontageError(
            "No usable centroid columns in the CellView data for this plate. "
            f"Looked for {list(_Y_COLUMNS)} and {list(_X_COLUMNS)}; the plate "
            f"has {found or 'no centroid columns at all'}."
        )
    extent = _extent_column(df)
    cond_col = condition_column(df, config)

    page_col = axis_column(df, config.pages, config)
    row_col = axis_column(df, config.rows, config)
    rng = np.random.default_rng(config.seed)

    grid: dict[str, dict[str, list[CellRef]]] = {}
    warnings: list[str] = []
    for page in axis_values(df, config.pages, config):
        page_df = df.filter(pl.col(page_col).cast(pl.Utf8) == page)
        if config.exclude_edge and canvas_lookup and crop_px:
            page_df = _drop_edge_cells(
                page_df, canvas_lookup, crop_px, y_col, x_col
            )
        rows_out: dict[str, list[CellRef]] = {}
        for row_value in axis_values(df, config.rows, config):
            subset = page_df.filter(pl.col(row_col).cast(pl.Utf8) == row_value)
            cells, note = _draw_group(
                subset,
                row_value,
                page,
                config,
                rng,
                cond_col,
                phase_col,
                y_col,
                x_col,
                extent,
            )
            rows_out[row_value] = cells
            warnings.extend(note)
        grid[page] = rows_out
    return grid, warnings


def _drop_edge_cells(
    df: pl.DataFrame,
    canvas_lookup: Callable[[str], tuple[int, int] | None],
    crop_px: int,
    y_col: str,
    x_col: str,
) -> pl.DataFrame:
    """Drop cells whose crop would run off their well's canvas.

    Canvas size is per well -- a well that lost a field to autofocus has a
    smaller one -- so the bound is applied per well rather than plate-wide.
    """
    half = crop_px / 2
    keep = []
    for well in sorted(str(w) for w in df["well"].unique().to_list()):
        canvas = canvas_lookup(well)
        part = df.filter(pl.col("well") == well)
        if canvas is not None:
            h, w = canvas
            part = part.filter(
                (pl.col(y_col) >= half)
                & (pl.col(y_col) <= h - half)
                & (pl.col(x_col) >= half)
                & (pl.col(x_col) <= w - half)
            )
        keep.append(part)
    return pl.concat(keep) if keep else df


# ----------------------------------------------------------------------
# Display limits
# ----------------------------------------------------------------------


#: Wells sampled when computing plate-wide display limits. Pooling a handful is
#: indistinguishable from pooling all of them for a percentile, at a fraction of
#: the reads.
_LIMIT_SAMPLE_WELLS = 4


def channel_limits(
    plate_id: int,
    wells: Sequence[str],
    percentiles: tuple[float, float] = (0.1, 99.9),
) -> dict[int, tuple[float, float]]:
    """Per-channel display limits pooled across ``wells``.

    Computed once and applied to every crop on every page. The scope has to
    match the comparison the figure invites: a page that spans conditions spans
    several wells, so per-well limits would make conditions differ in brightness
    for reasons that are pure scaling. Plate-wide limits mean a dim condition
    looks dim because it is.

    Pixels are pooled from the smallest pyramid level of a few evenly spaced
    wells rather than every well, which changes a percentile negligibly and
    keeps the read cost flat as the plate grows.
    """
    chosen = list(wells)
    if len(chosen) > _LIMIT_SAMPLE_WELLS:
        step = len(chosen) / _LIMIT_SAMPLE_WELLS
        chosen = [chosen[int(i * step)] for i in range(_LIMIT_SAMPLE_WELLS)]

    pooled: dict[int, list[npt.NDArray[Any]]] = {}
    for well in chosen:
        try:
            smallest = read_well(plate_id, well)["image"][-1]
        except (KeyError, FileNotFoundError):
            continue
        for c in range(int(smallest.shape[1])):
            pooled.setdefault(c, []).append(np.asarray(smallest[0, c]).ravel())
    if not pooled:
        raise MontageError(
            f"Could not read any of wells {chosen} from plate {plate_id}'s "
            f"zarr cache to compute display limits"
        )

    limits: dict[int, tuple[float, float]] = {}
    for channel, planes in pooled.items():
        values = np.concatenate(planes)
        lo, hi = np.percentile(values, list(percentiles))
        if hi <= lo:
            hi = lo + 1.0
        limits[channel] = (float(lo), float(hi))
    return limits


def _resolve_overlay(
    channel_names: list[str], overlay: tuple[str, ...]
) -> tuple[list[int], list[int]]:
    """Split channels into the composite ones and the greyscale rest.

    Matched on the base name so a cyclic-IF store's ``DAPI_R1`` resolves, and
    only the first match for each role is taken -- a repeated stain belongs in
    the greyscale row, not the composite.
    """
    overlay_indices: list[int] = []
    for wanted in overlay:
        for index, name in enumerate(channel_names):
            if index in overlay_indices:
                continue
            if base_name(name).startswith(wanted.lower()):
                overlay_indices.append(index)
                break
    grey = [i for i in range(len(channel_names)) if i not in overlay_indices]
    return overlay_indices, grey


def _crop_pixels(
    config: MontageConfig,
    df: pl.DataFrame,
    well: str,
    pixel_size_um: float | None,
) -> int:
    """Crop edge in pixels.

    Sized from a high percentile of the cell diameter **across the whole plate**,
    restricted to the phases being shown, rather than from the cells this draw
    happened to pick. Two reasons: a draw's maximum is noisy, so the crop would
    change with the seed; and a plate-level figure wants every well at the same
    scale, or panels cannot be compared side by side.

    The percentile has to be high. On plate 4127 the p95 cell diameter is 74 px
    but the polyploid p95 is 114 px, so a crop sized on the pooled p95 would clip
    exactly the phenotype the montage exists to show.
    """
    if config.crop_um is not None and pixel_size_um:
        return max(16, int(round(config.crop_um / pixel_size_um)))

    extent = _extent_column(df)
    phase_col = _phase_column(df)
    if extent is None:
        return 128
    name, is_diameter = extent
    shown = df.filter(pl.col(phase_col).is_in(list(config.phases)))
    values = shown[name].drop_nulls().to_numpy()
    values = values[values > 0]
    if values.size == 0:
        return 128
    if not is_diameter:
        values = 2.0 * np.sqrt(values / np.pi)
    diameter = float(np.percentile(values, config.size_percentile))
    return int(np.clip(round(diameter * config.crop_factor / 2) * 2, 32, 1024))


def _canvas_lookup(plate_id: int, built: set[str]) -> Any:
    """Return a memoised ``well -> (h, w)``; None for a well not in the cache."""
    cache: dict[str, tuple[int, int] | None] = {}

    def lookup(well: str) -> tuple[int, int] | None:
        if well not in cache:
            if well not in built:
                cache[well] = None
            else:
                shape = read_well(plate_id, well)["image"][0].shape[-2:]
                cache[well] = (int(shape[0]), int(shape[1]))
        return cache[well]

    return lookup


def build_pages(
    plate_id: int,
    df: pl.DataFrame,
    config: MontageConfig | None = None,
) -> list[WellMontage]:
    """Build every page of the ``pages`` x ``rows`` grid.

    Does no drawing, so it can be tested without matplotlib.

    Raises:
        MontageError: if the plate has no zarr store, or no page has any cells.
    """
    config = config or MontageConfig()
    if resolve_to_zarr(plate_id) is None:
        raise MontageError(
            f"Plate {plate_id} has no zarr cache. Build one first "
            f"(Cache Plate in the plate-info dialog, or zarr_cache_build.py)."
        )
    info = plate_info(plate_id)
    channel_names: list[str] = list(info["channel_names"])
    pixel_size_um = info.get("pixel_size_um")

    built = set(cached_wells(plate_id))
    if not built:
        raise MontageError(f"Plate {plate_id} has no wells in its zarr cache")

    # Only wells that are actually cached can be cropped from. Dropping them
    # is reported rather than silent: a well vanishing from a figure because
    # nobody built it looks identical to a well with no cells in that phase.
    measured = {str(w) for w in df["well"].unique().to_list()}
    uncached = sorted(measured - built)
    dropped_note = (
        [
            f"{len(uncached)} measured well(s) not in the zarr cache, "
            f"excluded: {uncached}"
        ]
        if uncached
        else []
    )
    df = df.filter(pl.col("well").cast(pl.Utf8).is_in(sorted(built)))
    if df.is_empty():
        raise MontageError(
            f"None of the plate's measured wells are in the zarr cache "
            f"({len(built)} built). Cache the wells first."
        )

    crop_px = _crop_pixels(config, df, "", pixel_size_um)
    grid, warnings = select_grid(
        df, config, _canvas_lookup(plate_id, built), crop_px
    )
    warnings = dropped_note + warnings
    overlay_indices, grey_indices = _resolve_overlay(
        channel_names, config.overlay
    )
    # Plate-scope limits: every panel on every page is directly comparable.
    limits = channel_limits(plate_id, sorted(built), config.percentiles)

    pages = [
        WellMontage(
            plate_id=plate_id,
            page_label=page,
            channel_names=channel_names,
            overlay_indices=overlay_indices,
            grey_indices=grey_indices,
            cells=rows,
            limits=limits,
            crop_px=crop_px,
            pixel_size_um=pixel_size_um,
            config=config,
            missing=warnings,
        )
        for page, rows in grid.items()
        if any(rows.values())
    ]
    if not pages:
        raise MontageError(
            f"Plate {plate_id}: no cells matched "
            f"pages={config.pages} rows={config.rows} "
            f"phases={list(config.phases)}"
        )
    return pages


def build_montage(
    plate_id: int,
    well: str,
    df: pl.DataFrame,
    config: MontageConfig | None = None,
) -> WellMontage:
    """Build the single page for one well.

    Thin wrapper over :func:`build_pages` for the well-per-page layout.

    Raises:
        MontageError: if the well is not cached or has no cells.
    """
    config = config or MontageConfig()
    if config.pages != "well":
        raise MontageError(
            f"build_montage is for the well-per-page layout; this config uses "
            f"pages={config.pages!r}. Use build_pages instead."
        )
    built = set(cached_wells(plate_id)) if resolve_to_zarr(plate_id) else set()
    if resolve_to_zarr(plate_id) is not None and well not in built:
        raise MontageError(
            f"Well {well} is not in plate {plate_id}'s zarr cache "
            f"({len(built)} well(s) built). Cache it first, or pick from: "
            f"{sorted(built)[:12]}"
        )
    if not df.filter(pl.col("well").cast(pl.Utf8) == well).height:
        raise MontageError(f"Well {well} has no measurements in CellView")
    pages = build_pages(
        plate_id,
        df.filter(pl.col("well").cast(pl.Utf8) == well),
        config,
    )
    return pages[0]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _normalise(
    plane: npt.NDArray[Any], limits: tuple[float, float]
) -> npt.NDArray[np.float32]:
    lo, hi = limits
    out = (plane.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    clipped: npt.NDArray[np.float32] = np.clip(out, 0.0, 1.0)
    return clipped


def _composite(
    crop: npt.NDArray[Any],
    indices: list[int],
    limits: dict[int, tuple[float, float]],
) -> npt.NDArray[np.float32]:
    """Additive RGB composite of the overlay channels."""
    h, w = crop.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for slot, channel in enumerate(indices):
        colour = _OVERLAY_RGB[slot % len(_OVERLAY_RGB)]
        norm = _normalise(crop[channel], limits[channel])
        for band in range(3):
            rgb[..., band] += norm * colour[band]
    return np.clip(rgb, 0.0, 1.0)


def resolve_mask_label(
    mask: npt.NDArray[Any], nucleus_label: int
) -> int | None:
    """Find which label in ``mask`` belongs to the target cell.

    **The centre pixel is authoritative, not the ID.** CellView's ``label`` is
    the nucleus label; the nuclei mask uses those IDs directly but the cell mask
    is labelled independently, so on the cell mask the ID means nothing. Since
    labels are dense small integers and a crop holds ten to twenty cells, some
    neighbour's *cell* label frequently equals the target's *nucleus* label by
    coincidence -- measured at 3.9% of cells on plate 4127, and in 12 of those
    13 cases matching on the ID outlined an unrelated neighbour.

    The nucleus centroid is the crop centre by construction, so the object under
    the centre pixel is the target in either mask.

    Args:
        mask: Label crop, centred on the target's nucleus centroid.
        nucleus_label: CellView's ``label`` for the target.

    Returns:
        The label to outline, or None if nothing identifiable sits at the centre.
    """
    centre_y, centre_x = mask.shape[0] // 2, mask.shape[1] // 2
    centre = int(mask[centre_y, centre_x])
    if centre:
        return centre

    # Centre on background. Only then is the ID worth trying, and only if the
    # matching region is actually near the centre -- otherwise it is the same
    # coincidence again, just without a centre label to contradict it.
    matched = mask == nucleus_label
    if not matched.any():
        return None
    radius = max(2, min(mask.shape) // 10)
    near = matched[
        max(0, centre_y - radius) : centre_y + radius + 1,
        max(0, centre_x - radius) : centre_x + radius + 1,
    ]
    return nucleus_label if near.any() else None


def _outline(
    mask: npt.NDArray[Any], label: int | None, width: int = 2
) -> npt.NDArray[np.bool_]:
    """Boundary of the target cell only, so neighbours are not outlined.

    Drawn ``width`` pixels thick: a single-pixel boundary is invisible once the
    page is scaled down to a montage panel.
    """
    from skimage.segmentation import find_boundaries

    if label is None:
        return np.zeros(mask.shape, dtype=bool)
    target = mask == label
    if not target.any():
        return np.zeros(mask.shape, dtype=bool)
    boundary = find_boundaries(target, mode="outer")
    for _ in range(max(0, width - 1)):
        boundary |= np.roll(boundary, 1, axis=0) | np.roll(boundary, 1, axis=1)
    result: npt.NDArray[np.bool_] = boundary & ~target
    return result


#: Figure typography. Arial at 7pt is the lab's figure convention; setting it
#: through an rc_context in :func:`render_montage` rather than mutating
#: ``rcParams`` keeps it from leaking into any other plot the session draws.
FONT_FAMILY = "Arial"
FONT_SIZE = 7

#: Scale-bar length in microns. Stated once in the figure subtitle rather than
#: labelled on all ~180 panels, which would be pure clutter at this density.
SCALE_BAR_UM = 20.0


def _add_scale_bar(ax: Any, crop_px: int, pixel_size_um: float | None) -> None:
    """Draw an unlabelled scale bar in the bottom-left of one panel.

    Every panel gets one: a montage is cropped per cell, so a reader looking at
    any single panel in isolation -- which is how a figure is actually read --
    has no other cue to size. The length is identical everywhere and is stated
    in the subtitle, so the bars themselves stay unlabelled.
    """
    from matplotlib.patches import Rectangle

    if not pixel_size_um:
        return
    bar_px = SCALE_BAR_UM / pixel_size_um
    if bar_px >= crop_px * 0.6:
        # Too long to read as a bar rather than as a line across the panel.
        return
    ax.add_patch(
        Rectangle(
            (crop_px * 0.05, crop_px * 0.93),
            bar_px,
            max(crop_px * 0.018, 1.0),
            color="white",
            zorder=5,
        )
    )


@contextmanager
def montage_style() -> Iterator[Any]:
    """Apply the figure typography, yielding ``pyplot``.

    Must wrap **saving** as well as drawing. Matplotlib bakes a font *size* into
    a Text artist when it is created but resolves the *family* at draw time, so
    a context that only covered figure construction still wrote DejaVu Sans into
    the PDF -- which is exactly what happened first time round.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            # TrueType rather than Type 3, so the text stays selectable and
            # editable in Illustrator instead of arriving as outlines.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        yield plt


def render_montage(montage: WellMontage) -> Any:
    """Draw the montage and return the matplotlib Figure."""
    with montage_style() as plt:
        return _render(montage, plt)


def _render(montage: WellMontage, plt: Any) -> Any:
    """Draw the figure. Called inside the typography rc_context."""
    cfg = montage.config
    rows = [
        (group, cell)
        for group, cells in montage.cells.items()
        for cell in cells
    ]
    if not rows:
        raise MontageError(
            f"{montage.page_label}: no cells in any {cfg.rows} group"
        )
    n_cols = 1 + len(montage.grey_indices)
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(1.35 * n_cols, 1.35 * len(rows)),
        squeeze=False,
    )

    mask_name = montage.config.mask
    unoutlined: list[str] = []
    for row_index, (group, cell) in enumerate(rows):
        crop = fetch_crop(
            montage.plate_id,
            cell.well,
            cell.label,
            centroid=cell.centroid,
            size=montage.crop_px,
            t=cell.timepoint,
        )
        try:
            mask = fetch_label_crop(
                montage.plate_id,
                cell.well,
                centroid=cell.centroid,
                size=montage.crop_px,
                t=cell.timepoint,
                mask_name=mask_name,
            )
        except KeyError:
            mask = fetch_label_crop(
                montage.plate_id,
                cell.well,
                centroid=cell.centroid,
                size=montage.crop_px,
                t=cell.timepoint,
                mask_name="nuclei",
            )
        # CellView's label is the nucleus label; the cell mask is labelled
        # independently, so the cell has to be found by what sits under the
        # centre pixel rather than by matching the ID.
        mask_label = resolve_mask_label(mask, cell.label)
        if mask_label is None:
            unoutlined.append(f"{cell.well} {cell.label}")
        outline = _outline(mask, mask_label)

        rgb = _composite(crop, montage.overlay_indices, montage.limits)
        rgb[outline] = _OUTLINE_RGB
        ax = axes[row_index][0]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_ylabel(
            f"{group}\n{cell.well} · {cell.label}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=22,
        )
        if row_index == 0:
            overlay_label = " + ".join(
                montage.channel_names[i] for i in montage.overlay_indices
            )
            ax.set_title(overlay_label)

        for col, channel in enumerate(montage.grey_indices, start=1):
            gax = axes[row_index][col]
            # Rendered as RGB rather than through a grey colormap so the
            # contour can be drawn in colour on top. The image itself is still
            # greyscale -- the three bands carry the same values.
            grey = _normalise(crop[channel], montage.limits[channel])
            panel = np.repeat(grey[..., np.newaxis], 3, axis=2)
            panel[outline] = _OUTLINE_RGB
            gax.imshow(panel, interpolation="nearest")
            if row_index == 0:
                gax.set_title(montage.channel_names[channel])

        for ax_ in axes[row_index]:
            ax_.set_xticks([])
            ax_.set_yticks([])
            for spine in ax_.spines.values():
                spine.set_visible(False)
            _add_scale_bar(ax_, montage.crop_px, montage.pixel_size_um)

    subtitle = (
        f"plate {montage.plate_id} · {cfg.pages} {montage.page_label} · "
        f"rows: {cfg.rows} · {cfg.cells_per_phase} random cell(s) per group · "
        f"seed {cfg.seed} · crop {montage.crop_px}px"
    )
    if montage.pixel_size_um:
        subtitle += (
            f" ({montage.crop_px * montage.pixel_size_um:.0f} µm)"
            f" · scale bar {SCALE_BAR_UM:.0f} µm"
        )
    if unoutlined:
        # Visible rather than silent: a panel with no outline is ambiguous
        # about which cell it is showing.
        logger.warning(
            f"{montage.page_label}: no mask found for {len(unoutlined)} cell(s), "
            f"drawn without an outline: {unoutlined}"
        )
    fig.suptitle(subtitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def _safe(value: str) -> str:
    """Filename-safe form of an axis value, e.g. ``"G2/M"`` -> ``"G2-M"``."""
    return "".join(
        c if c.isalnum() or c in "-_" else "-" for c in value
    ).strip("-")


def export_page_pdf(
    page: WellMontage, out_dir: Path, plt: Any | None = None
) -> Path:
    """Write one page as a vector PDF.

    The filename carries the *axis* as well as the value: plate 4127 has both a
    well called G1 and a phase called G1, so ``plate4127_G1.pdf`` would be
    ambiguous and silently overwritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = (
        out_dir / f"plate{page.plate_id}_{page.config.pages}-"
        f"{_safe(page.page_label)}_montage.pdf"
    )
    with montage_style():
        fig = render_montage(page)
        fig.savefig(path, format="pdf", bbox_inches="tight")
        _close(fig)
    logger.info(f"Wrote {path}")
    return path


def export_well_pdf(
    plate_id: int,
    well: str,
    df: pl.DataFrame,
    out_dir: Path,
    config: MontageConfig | None = None,
) -> Path:
    """Build and write one well's montage as a vector PDF.

    Returns:
        The path written.
    """
    montage = build_montage(plate_id, well, df, config)
    for warning in montage.missing:
        logger.warning(warning)
    return export_page_pdf(montage, out_dir)


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def plate_wells(df: pl.DataFrame) -> list[str]:
    """Every well the plate has measurements for, sorted."""
    return sorted(str(w) for w in df["well"].unique().to_list())


def export_plate_pdfs(
    plate_id: int,
    df: pl.DataFrame,
    out_dir: Path,
    config: MontageConfig | None = None,
    wells: list[str] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Path], list[str]]:
    """Export every page of the configured layout.

    For the default well-per-page layout this is one PDF per well, and ``wells``
    restricts it. For any other layout the pages are phases or conditions, so
    ``wells`` instead restricts which wells the cells are drawn *from*.

    One bad page does not abandon the rest -- a plate usually has a well or two
    with no cells in some phase, or not built into the zarr cache, and losing
    twenty good pages to one bad one helps nobody.

    Returns:
        ``(paths_written, failures)`` where each failure is a
        ``"<page>: <reason>"`` line.
    """
    config = config or MontageConfig()
    if wells is not None:
        df = df.filter(pl.col("well").cast(pl.Utf8).is_in(wells))

    try:
        pages = build_pages(plate_id, df, config)
    except MontageError as exc:
        logger.warning(f"Plate {plate_id}: {exc}")
        return [], [f"plate {plate_id}: {exc}"]

    for warning in dict.fromkeys(pages[0].missing):
        logger.warning(warning)

    written: list[Path] = []
    failures: list[str] = []
    for index, page in enumerate(pages):
        if on_progress is not None:
            on_progress(page.page_label, index, len(pages))
        try:
            written.append(export_page_pdf(page, out_dir))
        except MontageError as exc:
            logger.warning(f"Skipping {page.page_label}: {exc}")
            failures.append(f"{page.page_label}: {exc}")
    return written, failures


def load_plate_measurements(plate_id: int) -> pl.DataFrame:
    """Load a plate's CellView measurements as an eager DataFrame.

    Raises:
        MontageError: if the plate is not in CellView.
    """
    from omero_screen_napari.plate_cache import _load_plate_data_from_cellview

    lf = _load_plate_data_from_cellview(plate_id)
    df = lf.collect()
    if df.is_empty():
        raise MontageError(
            f"Plate {plate_id} has no measurements in CellView. Import it "
            f"with 'cellview import plate {plate_id}' first."
        )
    return df
