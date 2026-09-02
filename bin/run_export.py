"""CLI for exporting OMERO plates as re-importable Harmony measurements.

Exposed as the ``omero-screen-export`` console script. The exported bundle is
re-imported with the same command ``scripts/load_plates.sh`` already uses::

    omero import <out>/<plate>/Images/Index.idx.xml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from omero_screen.export import PlateSpec

TIMEPOINT_HELP = (
    "Timepoints to export, as a range (``0-5``) or list (``0,2,4``). "
    "Default: all."
)


def _parse_timepoints(spec: str | None) -> set[int] | None:
    """Parse a ``--timepoints`` spec into a set of indices.

    Args:
        spec: ``"0-5"``, ``"0,2,4"``, or ``None`` for all timepoints.

    Returns:
        The selected indices, or ``None`` meaning "keep everything".

    Raises:
        click.BadParameter: If the spec is malformed.
    """
    if not spec:
        return None
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = (int(v) for v in part.split("-", 1))
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
        except ValueError:
            raise click.BadParameter(
                f"Cannot parse timepoint spec {spec!r}",
                param_hint="--timepoints",
            ) from None
    if not selected:
        raise click.BadParameter(
            f"Timepoint spec {spec!r} selects nothing",
            param_hint="--timepoints",
        )
    return selected


def _export_options(func):  # type: ignore[no-untyped-def]
    """Apply the options shared by ``plate`` and ``screen``."""
    options = [
        click.option(
            "--out",
            "out_dir",
            required=True,
            type=click.Path(file_okay=False, path_type=Path),
            help="Directory the measurement folder is written into.",
        ),
        click.option(
            "--wells",
            default=None,
            metavar="A1,B2,...",
            help="Comma-separated well positions to export. Default: all.",
        ),
        click.option(
            "--max-fields",
            type=int,
            default=None,
            metavar="N",
            help="Export at most N fields per well. Default: all.",
        ),
        click.option(
            "--timepoints", default=None, metavar="SPEC", help=TIMEPOINT_HELP
        ),
        click.option(
            "--env",
            default=None,
            help="Environment name (requires configuration file .env.{name}).",
        ),
        click.option(
            "--dry-run",
            is_flag=True,
            default=False,
            help="Report what would be exported, including estimated size, "
            "without reading any pixels.",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Export OMERO plates as re-importable Harmony measurements."""


def _run_export(
    plate_ids: list[int],
    out_dir: Path,
    wells: str | None,
    max_fields: int | None,
    timepoints: str | None,
    dry_run: bool,
) -> None:
    """Export each plate in ``plate_ids`` into ``out_dir``."""
    # Imported here so ``--env`` has already been applied to os.environ.
    from omero_utils.omero_connect import omero_connect

    from omero_screen.export import (
        estimate_size_bytes,
        read_plate,
        write_measurement,
        write_metadata_excel,
    )

    keep_timepoints = _parse_timepoints(timepoints)
    well_filter = (
        tuple(w.strip() for w in wells.split(",") if w.strip())
        if wells
        else None
    )

    @omero_connect
    def run(conn=None):  # type: ignore[no-untyped-def]
        for plate_id in plate_ids:
            spec = read_plate(
                conn,
                plate_id,
                wells=well_filter,
                max_fields=max_fields,
            )
            if keep_timepoints is not None:
                spec.images = [
                    i for i in spec.images if i.timepoint in keep_timepoints
                ]
                if not spec.images:
                    raise click.ClickException(
                        f"Plate {plate_id}: --timepoints {timepoints} "
                        f"matched no planes"
                    )

            size_mb = estimate_size_bytes(spec) / (1024 * 1024)
            click.echo(
                f"Plate {plate_id} ({spec.name}): "
                f"{len(spec.well_positions)} well(s), "
                f"{len(spec.images)} plane(s), ~{size_mb:.1f} MB"
            )
            if dry_run:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            index_path = write_measurement(conn, spec, out_dir)
            write_metadata_excel(
                conn,
                plate_id,
                well_positions=spec.well_positions,
                fallback_channels=_channel_names(spec),
                path=out_dir / spec.name / "metadata.xlsx",
            )
            click.echo(f"  -> {index_path}")
            click.echo(f"  re-import with: omero import {index_path}")

    run()


def _channel_names(spec: PlateSpec) -> list[str]:
    """Distinct channel names in channel order."""
    names: dict[int, str] = {}
    for image in spec.images:
        names.setdefault(image.channel, image.channel_name)
    return [names[k] for k in sorted(names)]


@cli.command("plate")
@click.argument("plate_id", type=int)
@_export_options
def export_plate(
    plate_id: int,
    out_dir: Path,
    wells: str | None,
    max_fields: int | None,
    timepoints: str | None,
    env: str | None,
    dry_run: bool,
) -> None:
    """Export a single plate by ID."""
    if env:
        os.environ["ENV"] = env
    _run_export([plate_id], out_dir, wells, max_fields, timepoints, dry_run)


@cli.command("screen")
@click.argument("screen_id", type=int)
@_export_options
def export_screen(
    screen_id: int,
    out_dir: Path,
    wells: str | None,
    max_fields: int | None,
    timepoints: str | None,
    env: str | None,
    dry_run: bool,
) -> None:
    """Export every plate in a screen."""
    if env:
        os.environ["ENV"] = env

    from omero_utils.omero_connect import omero_connect

    @omero_connect
    def plate_ids(conn=None):  # type: ignore[no-untyped-def]
        screen = conn.getObject("Screen", screen_id)
        if screen is None:
            raise click.ClickException(f"Screen {screen_id} was not found")
        return [int(p.getId()) for p in screen.listChildren()]

    ids = plate_ids()
    if not ids:
        raise click.ClickException(f"Screen {screen_id} contains no plates")
    click.echo(f"Screen {screen_id}: {len(ids)} plate(s)")
    _run_export(ids, out_dir, wells, max_fields, timepoints, dry_run)


def main() -> None:
    """Entry point for the ``omero-screen-export`` console script."""
    try:
        cli.main(standalone_mode=False)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
