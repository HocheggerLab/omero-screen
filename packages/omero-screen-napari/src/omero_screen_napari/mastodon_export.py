"""Export a tracked well as a self-contained Mastodon bundle.

Curation of automated tracks happens in Mastodon (Fiji). This module writes
everything Mastodon needs into one accessible folder, sidestepping the three
traps we hit getting the OME-Zarr cache to open directly:

- the cache lives under ``~/.cache`` which Fiji's file browser can't reach;
- the cache packs the whole T axis into one chunk, so BigDataViewer renders
  only ``t=0`` (older caches built before the ``_T_CHUNK = 1`` fix);
- the HCS plate / row / column nesting confuses BDV's dataset discoverer.

The bundle is therefore a **flat, one-timepoint-per-chunk OME-Zarr copy** of
the single well's image, plus a Mastodon CSV-importer file for the tracks, plus
a README with the click-by-click import steps.

    ~/mastodon_exports/plate_<id>_<well>/
        image.zarr/     flat multiscale group, T=1 chunks, .zgroup marker
        tracks.csv      spots + links for Mastodon's CSV Importer
        README.txt      import instructions + column mapping

Track model translation (CellView is *track-level*, Mastodon CSV is
*spot-level*): each ``(track_id, timepoint)`` row becomes one spot with a
unique id; a spot links to the same track's most-recent earlier frame (which
bridges segmentation gaps); a daughter's first spot links to its parent
track's last spot (the division); founders' first spots use ``parent_id = -1``.
Coordinates are scaled to physical units (µm) to match the image.

Main Functions:
    - export_well_for_mastodon: write the full bundle for one well.
    - build_mastodon_csv: pure track-level -> spot-level CSV translation.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import polars as pl
import zarr

from omero_screen_napari.tracks_loader import (
    CENTROID_X_COL,
    CENTROID_Y_COL,
    PARENT_COL,
    TIME_COL,
    TRACK_ID_COL,
    has_tracks,
)
from omero_screen_napari.zarr_cache.paths import plate_zarr_path

#: Default base directory for exports — a plain local folder Fiji can reach
#: (NOT the hidden ~/.cache, and avoid the iCloud-synced ~/Desktop).
DEFAULT_EXPORT_BASE = Path.home() / "mastodon_exports"

_AREA_COL = "area_nucleus"


def _well_image_group(plate_id: int, well: str) -> Path:
    """Resolve ``<cache>/plate_<id>.zarr/<row>/<col>/0`` for a well."""
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


def rechunk_image_for_bdv(well_group: Path, out_zarr: Path) -> None:
    """Copy a well's image pyramid to a flat, BDV-friendly OME-Zarr.

    Rewrites every pyramid level with one timepoint per chunk and writes a
    ``.zgroup`` marker plus the original ``.zattrs`` at the root, so the result
    opens directly in BigDataViewer / Mastodon with all timepoints intact.

    Args:
        well_group: Source ``.../<row>/<col>/0`` group in the plate cache.
        out_zarr: Destination ``.zarr`` directory (overwritten if present).
    """
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.mkdir(parents=True)

    shutil.copy(well_group / ".zattrs", out_zarr / ".zattrs")
    (out_zarr / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    multiscale = json.loads((well_group / ".zattrs").read_text())[
        "multiscales"
    ][0]
    for entry in multiscale["datasets"]:
        level = entry["path"]
        src = zarr.open(str(well_group / level), mode="r")
        _, _, y, x = src.shape
        tile = min(256, y, x)
        dst = zarr.open(
            str(out_zarr / level),
            mode="w",
            shape=src.shape,
            dtype=src.dtype,
            chunks=(1, 1, tile, tile),
            compressor=src.compressor,
        )
        dst[:] = src[:]


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

1. Fiji → Plugins → Tracking → Mastodon → "new from OME-NGFF…"
   Paste this path (Browse can't reach hidden folders; pasting works):
       {image_path}
   → Detect datasets → click the listed row → OK → save the BDV XML anywhere.

2. The image opens in BigDataViewer. Press P for the side panel to adjust
   per-channel contrast; 1 / 2 switch channels; F toggles a fused overlay.

3. Load the tracks: main Mastodon window → File → Import → CSV Importer →
   choose tracks.csv and map the columns:
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
    """Write a self-contained Mastodon bundle for one well.

    Args:
        plate_id: Plate to export from.
        well: Well position (e.g. ``"B2"``).
        plate_data: CellView measurements LazyFrame (``omero_data.plate_data``).
        out_base: Base directory for exports; defaults to
            :data:`DEFAULT_EXPORT_BASE`.
        pixel_size: µm/pixel override; defaults to the cache's metadata.

    Returns:
        Mapping with ``dir``, ``image``, ``csv`` and ``readme`` paths.
    """
    well_group = _well_image_group(plate_id, well)
    px = pixel_size if pixel_size is not None else _pixel_size_um(well_group)

    base = out_base or DEFAULT_EXPORT_BASE
    out_dir = base / f"plate_{plate_id}_{well}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "image.zarr"
    rechunk_image_for_bdv(well_group, image_path)

    csv = build_mastodon_csv(plate_data, well, px)
    csv_path = out_dir / "tracks.csv"
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
            image_path=image_path,
            n_spots=csv.height,
            n_tracks=csv["label"].n_unique(),
            n_div=n_div,
            px=px,
        )
    )
    return {
        "dir": out_dir,
        "image": image_path,
        "csv": csv_path,
        "readme": readme_path,
    }
