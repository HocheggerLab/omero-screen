"""Export a tracked well's lineage for curation in Mastodon (Fiji).

The OME-Zarr cache now lives in a visible folder (``~/omero-cache`` by
default) and is written one-timepoint-per-chunk, so Mastodon can open a well's
image group *directly* — no image copy or symlink. This module therefore only
needs to write a Mastodon CSV-importer file for the tracks (beside the cached
image) and a README with the exact paths to paste into Fiji plus the column
mapping.

Protecting a plate from eviction during curation is a separate, explicit
choice: the napari Tracks widget has Pin / Unpin buttons. Export does not pin —
caching a plate doesn't mean you're curating it.

Track model translation (CellView is *track-level*, Mastodon CSV is
*spot-level*): each ``(track_id, timepoint)`` row becomes one spot with a
unique id; a spot links to the same track's most-recent earlier frame (which
bridges segmentation gaps); a daughter's first spot links to its parent
track's last spot (the division); founders' first spots use ``parent_id = -1``.
Coordinates are scaled to physical units (µm) to match the image.

Main Functions:
    - export_well_for_mastodon: write the tracks CSV + README for one well.
    - write_plate_tracks_csvs: best-effort per-well CSVs at cache-build time.
    - build_mastodon_csv: pure track-level -> spot-level CSV translation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
from loguru import logger

from omero_screen_napari.tracks_loader import (
    CENTROID_X_COL,
    CENTROID_Y_COL,
    PARENT_COL,
    TIME_COL,
    TRACK_ID_COL,
    has_tracks,
)
from omero_screen_napari.zarr_cache import plate_zarr_path

#: Default base directory for the per-well export README folders.
DEFAULT_EXPORT_BASE = Path.home() / "mastodon_exports"

_AREA_COL = "area_nucleus"


def _well_image_group(plate_id: int, well: str) -> Path:
    """Resolve ``<cache>/plate_<id>.zarr/<row>/<col>/0`` for a well.

    This is the multiscale image group Mastodon opens directly — it carries
    its own ``.zgroup`` / ``.zattrs`` and (for caches built with the current
    writer) one-timepoint-per-chunk data.
    """
    if not well or len(well) < 2:
        raise ValueError(f"Well {well!r} must look like 'B2'.")
    row, col = well[0].upper(), well[1:]
    group = plate_zarr_path(plate_id) / row / col / "0"
    if not group.exists():
        raise FileNotFoundError(
            f"No cached image for well {well} at {group}. Build the zarr "
            f"cache first (Welldata widget → Cache Plate)."
        )
    return group


def _pixel_size_um(well_group: Path) -> float:
    """Read y/x pixel size (µm) from the well's OME-Zarr multiscale metadata."""
    meta = json.loads((well_group / ".zattrs").read_text())
    scale = meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"]  # [t, c, y, x]
    return float(scale[-1])


def well_tracks_csv_path(plate_id: int, well: str) -> Path:
    """Where a well's Mastodon ``tracks.csv`` lives — beside its image group.

    Co-locating it with the cached image (``<row>/<col>/tracks.csv``, a
    sibling of the ``0`` multiscale group) makes a cached well self-sufficient
    for Mastodon: open the image group, import the sibling CSV.
    """
    return _well_image_group(plate_id, well).parent / "tracks.csv"


def write_well_tracks_csv(
    plate_id: int,
    well: str,
    plate_data: pl.LazyFrame,
    pixel_size: float | None = None,
) -> Path:
    """Write a well's Mastodon tracks CSV into the cache, return its path.

    Raises:
        KeyError: If the plate has no track columns.
        ValueError: If the well has no tracked rows.
        FileNotFoundError: If the well image is not cached.
    """
    well_group = _well_image_group(plate_id, well)
    px = pixel_size if pixel_size is not None else _pixel_size_um(well_group)
    csv = build_mastodon_csv(plate_data, well, px)
    out = well_group.parent / "tracks.csv"
    csv.write_csv(out)
    return out


def write_plate_tracks_csvs(plate_id: int) -> list[Path]:
    """Write a tracks CSV beside every cached, tracked well of a plate.

    Best-effort and side-effect-only: pulls the plate's measurements from
    CellView and writes one ``tracks.csv`` per cached well that has track
    data. Returns the CSV paths written (empty if CellView is unavailable,
    the plate isn't imported, or it carries no tracks). Never raises — it is
    called from the zarr-cache build, which must not fail on a missing or
    track-free CellView.
    """
    from omero_screen_napari.zarr_cache import cached_wells

    try:
        from cellview.db.db import CellViewDB
        from cellview.exporters.db_to_polars import export_polars_lf

        conn = CellViewDB().connect()
        plate_data, _ = export_polars_lf(plate_id, conn)
    except Exception as exc:  # noqa: BLE001 — best-effort; log and bail
        logger.info(
            f"No Mastodon tracks CSV for plate {plate_id} (CellView unavailable: {exc})"
        )
        return []

    if not has_tracks(plate_data):
        return []

    written: list[Path] = []
    for well in cached_wells(plate_id):
        try:
            written.append(write_well_tracks_csv(plate_id, well, plate_data))
        except (KeyError, ValueError, FileNotFoundError):
            continue  # well has no tracked rows / not cached — skip
    if written:
        logger.info(
            f"Wrote {len(written):d} Mastodon tracks CSV(s) for plate {plate_id}"
        )
    return written


def build_mastodon_csv(
    plate_data: pl.LazyFrame, well: str, pixel_size: float
) -> pl.DataFrame:
    """Translate a well's track-level rows into a Mastodon spot CSV.

    Args:
        plate_data: CellView measurements LazyFrame for the plate.
        well: Well position to export.
        pixel_size: µm per pixel, applied to x, y and the spot radius.

    Returns:
        DataFrame with columns ``id, parent_id, x, y, z, frame, radius, label``
        ready for Mastodon's CSV Importer.

    Raises:
        KeyError: If the plate has no track columns.
        ValueError: If the well has no tracked rows.
    """
    if not has_tracks(plate_data):
        raise KeyError(
            "plate_data has no track_id column — run the pipeline with --track."
        )
    cols = [
        TRACK_ID_COL,
        PARENT_COL,
        TIME_COL,
        CENTROID_Y_COL,
        CENTROID_X_COL,
        _AREA_COL,
    ]
    df = (
        plate_data.filter(pl.col("well") == well)
        .select(cols)
        .sort([TRACK_ID_COL, TIME_COL])
        .collect()
    )
    if df.height == 0:
        raise ValueError(f"No tracked rows for well {well!r}.")

    rows = df.to_dicts()
    # Unique spot id per (track_id, timepoint), and the inverse lookup.
    spot_id: dict[tuple[int, int], int] = {
        (int(r[TRACK_ID_COL]), int(r[TIME_COL])): i for i, r in enumerate(rows)
    }
    # Sorted frames per track, to resolve gap-bridged and division parents.
    track_frames: dict[int, list[int]] = {}
    for tid, t in spot_id:
        track_frames.setdefault(tid, []).append(t)
    for frames in track_frames.values():
        frames.sort()

    out: list[dict[str, float | int]] = []
    for r in rows:
        tid = int(r[TRACK_ID_COL])
        t = int(r[TIME_COL])
        earlier_same = [f for f in track_frames[tid] if f < t]
        if earlier_same:
            # Link to the same track's most-recent earlier frame (bridges
            # segmentation gaps so a gapped track stays one track).
            pid = spot_id[(tid, max(earlier_same))]
        elif int(r[PARENT_COL]) != 0:
            # Division: link the daughter's first spot to the parent track's
            # last spot before this frame.
            parent_tid = int(r[PARENT_COL])
            cand = [f for f in track_frames.get(parent_tid, []) if f < t]
            pid = spot_id[(parent_tid, max(cand))] if cand else -1
        else:
            pid = -1  # founder, first frame
        radius_px = math.sqrt(float(r[_AREA_COL]) / math.pi)
        out.append(
            {
                "id": spot_id[(tid, t)],
                "parent_id": pid,
                "x": float(r[CENTROID_X_COL]) * pixel_size,
                "y": float(r[CENTROID_Y_COL]) * pixel_size,
                "z": 0.0,
                "frame": t,
                "radius": radius_px * pixel_size,
                "label": tid,
            }
        )
    return pl.DataFrame(out)


_README = """\
Mastodon import — plate {plate_id} well {well}
================================================

The omero-cache is an LRU cache. If you will curate this well over time,
click "Pin plate" in the napari Tracks widget first so it is not evicted
mid-curation; "Unpin plate" when you are done.

1. Fiji → Plugins → Tracking → Mastodon → "new from OME-NGFF…"
   Paste this image path (it opens directly — no copy was made):
       {image_path}
   → Detect datasets → click the listed row → OK → save the BDV XML anywhere.

2. The image opens in BigDataViewer. Press P for the side panel to adjust
   per-channel contrast; 1 / 2 switch channels; F toggles a fused overlay.

3. Load the tracks: main Mastodon window → File → Import → CSV Importer →
   choose this CSV (it sits next to the image):
       {csv_path}
   and map the columns:
       X=x  Y=y  Z=z  Frame=frame  ID=id  Parent ID=parent_id
       Radius (column)=radius  Label=label
   Default Radius: 10 (only used if the radius column is blank).

4. Link the views: click the same group-lock number (e.g. 1) in BOTH the
   BigDataViewer and TrackScheme windows. Then double-click a spot to navigate.

{n_spots} spots, {n_tracks} tracks, {n_div} division(s). pixel size {px:.4f} µm.
"""


def export_well_for_mastodon(
    plate_id: int,
    well: str,
    plate_data: pl.LazyFrame,
    out_base: Path | None = None,
    pixel_size: float | None = None,
) -> dict[str, Path]:
    """Write the Mastodon tracks CSV (beside the cached image) and a README.

    No image data is copied: Mastodon opens the cached well image group in
    place (see :func:`_well_image_group`). Pinning the plate against eviction
    is a separate, explicit step (the Pin button) — export does not pin.

    Args:
        plate_id: Plate to export from.
        well: Well position (e.g. ``"B2"``).
        plate_data: CellView measurements LazyFrame (``omero_data.plate_data``).
        out_base: Base directory for the README folder; defaults to
            :data:`DEFAULT_EXPORT_BASE`.
        pixel_size: µm/pixel override; defaults to the cache's metadata.

    Returns:
        Mapping with ``dir``, ``image``, ``csv`` and ``readme`` paths. ``image``
        and ``csv`` both live in the cache well dir (not copied); ``readme``
        is the guided-import note under ``out_base``.
    """
    well_group = _well_image_group(plate_id, well)
    px = pixel_size if pixel_size is not None else _pixel_size_um(well_group)

    base = out_base or DEFAULT_EXPORT_BASE
    out_dir = base / f"plate_{plate_id}_{well}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the CSV next to the cached image (same location the build uses),
    # so there is a single canonical tracks.csv per well.
    csv = build_mastodon_csv(plate_data, well, px)
    csv_path = well_group.parent / "tracks.csv"
    csv.write_csv(csv_path)

    n_div = (
        csv.filter(pl.col("parent_id") != -1)
        .group_by("parent_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    readme_path = out_dir / "README.txt"
    readme_path.write_text(
        _README.format(
            plate_id=plate_id,
            well=well,
            image_path=well_group,
            csv_path=csv_path,
            n_spots=csv.height,
            n_tracks=csv["label"].n_unique(),
            n_div=n_div,
            px=px,
        )
    )
    return {
        "dir": out_dir,
        "image": well_group,
        "csv": csv_path,
        "readme": readme_path,
    }
