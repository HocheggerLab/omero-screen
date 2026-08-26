"""Batch export of per-well galleries for figure generation.

The interactive gallery widget builds one well at a time and leaves the
figure on screen. For a figure panel you usually want the *same* settings
applied across every well of the plate, written to disk as files you can
drop into Illustrator — plus a record of which settings produced them.

Intended use is the napari console, after loading a plate and dialling in
the gallery settings (channels, crop size, cell-cycle phase, grid) you
want::

    from omero_screen_napari.gallery_export import export_galleries
    export_galleries("~/figures/plate3868", seed=0)

Each well becomes ``<well>.<fmt>`` and the run is described by a single
``gallery_export.json`` manifest next to the figures.

Settings come from the live ``userdata`` singleton — whatever the gallery
widget last ran with — so what you see interactively is what you get for
every well. ``well`` is the one field overridden per export.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from loguru import logger

from omero_screen_napari.gallery_api import build_gallery_figure
from omero_screen_napari.gallery_userdata import UserData

if TYPE_CHECKING:
    from omero_screen_napari.omero_data import OmeroData

MANIFEST_NAME = "gallery_export.json"


def export_galleries(
    output_dir: str | Path,
    wells: list[str] | None = None,
    *,
    fmt: str = "pdf",
    dpi: int = 300,
    seed: int | None = None,
    show_title: bool | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    omero_data: OmeroData | None = None,
    user_data: UserData | None = None,
) -> list[Path]:
    """Export one gallery per well using the current gallery settings.

    Args:
        output_dir: Directory for the figures and the manifest. Created if
            missing; ``~`` is expanded.
        wells: Wells to export. Defaults to every well the plate can
            serve — the zarr cache's wells when the plate is cached, else
            the wells loaded in the viewer — intersected with CellView.
        fmt: Figure extension, e.g. ``"pdf"`` (vector container, best for
            figure assembly) or ``"png"``.
        dpi: Raster resolution passed to ``savefig``.
        seed: Seed the crop sampling per well, so a re-run reproduces the
            same gallery. ``None`` leaves sampling random.
        show_title: Override the gallery title. ``False`` gives a bare
            panel for figure placement, ``True`` keeps the diagnostic
            well/settings header, ``None`` uses the current setting.
        on_progress: Called as ``(well, index, total)`` before each well.
            The export is synchronous — a 21-well plate takes ~30 s — so
            the GUI uses this to drive a progress bar.
        omero_data: Override the ``omero_data`` singleton (tests).
        user_data: Override the ``userdata`` singleton (tests).

    Returns:
        Paths of the figures written, in well order.

    Raises:
        ValueError: No plate loaded, no wells resolvable, or the gallery
            settings are incomplete (no channels selected).
    """
    if omero_data is None:
        from omero_screen_napari.omero_data_singleton import (
            omero_data as _omero_data,
        )

        omero_data = _omero_data
    if user_data is None:
        from omero_screen_napari.gallery_userdata_singleton import (
            userdata as _userdata,
        )

        user_data = _userdata

    _validate_settings(omero_data, user_data)
    target_wells = wells or available_wells(omero_data)
    if not target_wells:
        raise ValueError(
            f"No wells to export for plate {omero_data.plate_id}. Load a "
            f"plate first, or pass wells=[...] explicitly."
        )

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # The gallery mutates the singleton's crop pool. Snapshot it so an
    # export from the console doesn't disturb the well the user is
    # working on interactively.
    snapshot = _snapshot_crops(omero_data)
    written: list[Path] = []
    entries: dict[str, Any] = {}
    try:
        for index, well in enumerate(target_wells):
            if on_progress is not None:
                on_progress(well, index, len(target_wells))
            path, entry = _export_one(
                omero_data,
                user_data,
                well,
                out,
                fmt,
                dpi,
                seed,
                show_title,
            )
            entries[well] = entry
            if path is not None:
                written.append(path)
    finally:
        _restore_crops(omero_data, snapshot)

    manifest = _write_manifest(
        out,
        omero_data,
        _effective_settings(user_data, show_title),
        fmt,
        dpi,
        seed,
        entries,
    )
    logger.info(
        f"Exported {len(written):d}/{len(target_wells):d} gallery/ies to "
        f"{out} (manifest: {manifest.name})"
    )
    return written


def available_wells(omero_data: OmeroData) -> list[str]:
    """Wells this plate can produce a gallery for, sorted.

    The zarr cache can serve any well it has built, not just the ones on
    screen (see ``gallery_api._filter_well_centroids``), so prefer its
    well list and fall back to the viewer's. Either way the result is
    intersected with the wells CellView has rows for — no rows means no
    centroids to crop around.
    """
    candidates: list[str] = []
    try:
        from omero_screen_napari.zarr_cache import cached_wells

        candidates = cached_wells(omero_data.plate_id)
    except Exception as exc:  # noqa: BLE001 — zarr stack is optional
        logger.debug(f"No zarr cache for plate {omero_data.plate_id}: {exc}")
    if not candidates:
        candidates = list(omero_data.well_pos_list or [])

    in_cellview = _cellview_wells(omero_data)
    if in_cellview is None:
        return sorted(candidates)
    missing = sorted(set(candidates) - in_cellview)
    if missing:
        logger.info(f"Skipping wells with no CellView rows: {missing}")
    return sorted(w for w in candidates if w in in_cellview)


def _cellview_wells(omero_data: OmeroData) -> set[str] | None:
    """Wells present in the plate's CellView data, or None if unknown."""
    plate_data = omero_data.plate_data
    if plate_data is None:
        return None
    try:
        if "well" not in plate_data.collect_schema().names():
            return None
        return set(
            plate_data.select("well").unique().collect()["well"].to_list()
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not list CellView wells: {exc}")
        return None


def _validate_settings(omero_data: OmeroData, user_data: UserData) -> None:
    """Fail early on the two states that produce an unusable export."""
    if not omero_data.plate_id:
        raise ValueError(
            "No plate loaded. Load a plate in the welldata widget before "
            "exporting galleries."
        )
    if not user_data.channels:
        raise ValueError(
            "No channels selected. Run the gallery widget once to set the "
            "channels, crop size and grid used for the export."
        )


def _effective_settings(
    user_data: UserData, show_title: bool | None
) -> UserData:
    """The settings actually used, with the title override applied."""
    if show_title is None:
        return user_data
    return replace(user_data, show_title=show_title)


def _export_one(
    omero_data: OmeroData,
    user_data: UserData,
    well: str,
    out: Path,
    fmt: str,
    dpi: int,
    seed: int | None,
    show_title: bool | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Build and save one well's gallery; never raises for one bad well.

    A well that yields no crops (all fields dropped, a cell-cycle phase
    absent, a classifier value that never occurs there) should not abort
    a 21-well export, so its failure is recorded in the manifest and the
    loop moves on.
    """
    well_settings = replace(
        _effective_settings(user_data, show_title), well=well
    )
    if seed is not None:
        # Per-well offset keeps wells independent while staying
        # reproducible across runs.
        random.seed(f"{seed}:{well}")
    try:
        fig = build_gallery_figure(
            omero_data, well_settings, show=False, force_reload=True
        )
    except Exception as exc:  # noqa: BLE001 — one well must not kill the run
        logger.warning(f"Well {well}: gallery failed ({exc})")
        return None, {"exported": False, "reason": str(exc)}

    if fig is None:
        logger.warning(f"Well {well}: no crops, skipped")
        return None, {"exported": False, "reason": "no crops"}

    path = out / f"{well}.{fmt}"
    fig.savefig(
        str(path),
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    logger.info(f"Well {well}: wrote {path.name}")
    return path, {
        "exported": True,
        "file": path.name,
        "n_in_gallery": len(omero_data.selected_images),
        "n_crops_remaining": len(omero_data.cropped_images),
        "well_metadata": _well_metadata(omero_data, well),
    }


def _well_metadata(omero_data: OmeroData, well: str) -> dict[str, Any]:
    """Per-well annotations (cell line, condition, ...) if we have them."""
    pos_list = list(omero_data.well_pos_list or [])
    meta_list = list(omero_data.well_metadata_list or [])
    if well in pos_list:
        index = pos_list.index(well)
        if index < len(meta_list):
            return dict(meta_list[index] or {})
    try:
        from omero_screen_napari.zarr_cache import plate_info

        return dict(
            (plate_info(omero_data.plate_id).get("well_metadata") or {}).get(
                well, {}
            )
        )
    except Exception:  # noqa: BLE001 — annotations are informational
        return {}


def _write_manifest(
    out: Path,
    omero_data: OmeroData,
    user_data: UserData,
    fmt: str,
    dpi: int,
    seed: int | None,
    entries: dict[str, Any],
) -> Path:
    """Write the run's settings + per-well outcome as JSON.

    Deliberately records the full settings dict rather than a summary:
    the point is to be able to say months later exactly which crop size,
    channels and phase produced a panel — and to replay it.
    """
    settings = asdict(user_data)
    settings.pop("well", None)  # per-well, not a run setting
    manifest = {
        "plate_id": omero_data.plate_id,
        "plate_name": omero_data.plate_name,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "format": fmt,
        "dpi": dpi,
        "seed": seed,
        "settings": settings,
        "channel_data": dict(omero_data.channel_data or {}),
        "intensities": {
            str(k): list(v) for k, v in (omero_data.intensities or {}).items()
        },
        "pixel_size_um": (
            omero_data.pixel_size[0] if omero_data.pixel_size else None
        ),
        "wells": entries,
    }
    path = out / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def _snapshot_crops(omero_data: OmeroData) -> dict[str, Any]:
    """Copy the crop-pool fields the gallery overwrites."""
    return {
        name: list(getattr(omero_data, name, []) or [])
        for name in (
            "cropped_images",
            "cropped_labels",
            "cropped_cell_meta",
            "selected_images",
            "selected_cell_meta",
        )
    }


def _restore_crops(omero_data: OmeroData, snapshot: dict[str, Any]) -> None:
    """Put the interactive crop pool back after a batch run."""
    for name, value in snapshot.items():
        if hasattr(omero_data, name):
            setattr(omero_data, name, value)
