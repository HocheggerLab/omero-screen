"""Export a tracked well as a Cell Tracking Challenge (CTC) bundle for Mastodon.

Mastodon's plain *CSV* importer creates spots but **no links**, so it cannot
preserve tracks — and its label-image importer cannot create *division* links.
Only Mastodon's **CTC importer** rebuilds full lineages (including divisions),
because it reads the parent relationships from a ``res_track.txt`` file. So to
curate our tracks in Mastodon we export a CTC bundle, not a CSV.

A bundle is one folder per well containing:

* ``mask000.tif`` … ``mask{T-1}.tif`` — one nucleus label image per frame,
  read straight from the cached OME-Zarr (no OMERO round-trip). Pixel value =
  the track's CTC label.
* ``res_track.txt`` — the CTC track table, four space-separated integers per
  track: ``L B E P`` (label, begin frame, end frame, parent label; ``0`` =
  founder). See :func:`build_ctc_export`.
* ``manifest.json`` — a sidecar mapping each CTC label back to the original
  CellView ``track_id`` plus per-frame centroids, so corrected tracks can later
  be reconciled into CellView (the curated ``track_id`` / ``parent_track_id``
  columns). The CTC→CellView return trip itself is a separate, future step.
* ``mastodon_image/`` — a metadata-only OME-NGFF view of the well's image at
  **unit pixel scale** (symlinks into the cache, no pixel copy). Open this in
  Mastodon, not the raw cache group: the cached image is µm-calibrated, but the
  CTC importer places spots at raw pixel coordinates, so opening the calibrated
  image makes the tracks overshoot the picture by ``1 / pixel_size``. The view
  presents the same pixels at scale 1 so spots land on the nuclei
  (:func:`write_unit_scale_view`).

Track labels are renumbered ``1..N`` in begin-frame order (CTC-canonical, and
the same scheme used for both the TIFFs and ``res_track.txt`` so they always
agree). Coordinates in the manifest are pixel centroids, matching CellView's
``centroid-0-nuc`` / ``centroid-1-nuc``.

Main Functions:
    - export_well_ctc: write a well's full CTC bundle (masks + txt + manifest).
    - build_ctc_export: pure track-level -> CTC table + relabel map + manifest.
    - relabel_mask: remap a label frame's pixel values via the relabel map.
    - write_unit_scale_view: metadata-only unit-scale OME-NGFF view for Mastodon.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tifffile
from loguru import logger
from numpy.typing import NDArray

from omero_screen_napari.tracks_loader import (
    CENTROID_X_COL,
    CENTROID_Y_COL,
    PARENT_COL,
    TIME_COL,
    TRACK_ID_COL,
    has_tracks,
)
from omero_screen_napari.zarr_cache import plate_zarr_path, read_well

#: Default base directory for the per-well CTC export folders.
DEFAULT_EXPORT_BASE = Path.home() / "mastodon_exports"

#: Sidecar format version, so a future reconciliation reader can branch on it.
MANIFEST_VERSION = 1


@dataclass
class CtcExport:
    """Pure, file-free description of a well's CTC export.

    Attributes:
        track_table: ``L B E P`` rows, one per track, sorted by ``L`` — the
            ``res_track.txt`` content.
        relabel: Maps the original CellView ``track_id`` to its CTC label
            (``1..N``); ``0`` maps to ``0`` (background / untracked).
        manifest: Round-trip sidecar (see :func:`build_ctc_export`).
    """

    track_table: pl.DataFrame
    relabel: dict[int, int]
    manifest: dict[str, Any]


def _well_image_group(plate_id: int, well: str) -> Path:
    """Resolve ``<cache>/plate_<id>.zarr/<row>/<col>/0`` for a well.

    This is the multiscale image group Mastodon opens directly for the picture;
    the CTC bundle (masks + txt) is written to a separate export folder.
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


def build_ctc_export(plate_data: pl.LazyFrame, well: str) -> CtcExport:
    """Translate a well's track rows into a CTC table + relabel map + manifest.

    The CTC ``res_track.txt`` layout is one row per track, four space-separated
    integers ``L B E P``:

        L   track label (``1..N``)
        B   begin frame — first (0-based) timepoint the track appears
        E   end frame   — last timepoint the track appears
        P   parent track label (``0`` for a founder)

    Labels are reassigned ``1..N`` in begin-frame order (ties broken by the
    original ``track_id``) and parents remapped to the new labels, so ``B`` is
    non-decreasing and every parent label is smaller than its daughters'. The
    same ``relabel`` map is applied to the mask TIFFs (:func:`relabel_mask`) so
    the images and the table always agree.

    The manifest records, per CTC label, the original ``track_id`` /
    ``parent_track_id`` and the per-frame pixel centroids — enough to map a
    later Mastodon correction back onto CellView rows by position.

    Args:
        plate_data: CellView measurements LazyFrame for the plate.
        well: Well position to export.

    Returns:
        A :class:`CtcExport`.

    Raises:
        KeyError: If the plate has no track columns.
        ValueError: If the well has no tracked rows.
    """
    if not has_tracks(plate_data):
        raise KeyError(
            "plate_data has no track_id column — run the pipeline with --track."
        )
    cols = [TRACK_ID_COL, PARENT_COL, TIME_COL, CENTROID_Y_COL, CENTROID_X_COL]
    df = (
        plate_data.filter(pl.col("well") == well)
        .select(cols)
        .sort([TRACK_ID_COL, TIME_COL])
        .collect()
    )
    if df.height == 0:
        raise ValueError(f"No tracked rows for well {well!r}.")

    per_track = (
        df.group_by(TRACK_ID_COL)
        .agg(
            B=pl.col(TIME_COL).min(),
            E=pl.col(TIME_COL).max(),
            # parent_track_id is constant within a track; max() ignores any
            # stray 0 so a daughter keeps its real (non-zero) parent.
            parent=pl.col(PARENT_COL).max(),
        )
        .sort(["B", TRACK_ID_COL])
    )

    orig_ids = [int(o) for o in per_track[TRACK_ID_COL].to_list()]
    relabel: dict[int, int] = {o: i + 1 for i, o in enumerate(orig_ids)}
    relabel[0] = 0

    begins = [int(b) for b in per_track["B"].to_list()]
    ends = [int(e) for e in per_track["E"].to_list()]
    parents = [int(p) for p in per_track["parent"].to_list()]

    track_table = pl.DataFrame(
        {
            "L": [relabel[o] for o in orig_ids],
            "B": begins,
            "E": ends,
            "P": [relabel.get(p, 0) for p in parents],
        },
        schema={"L": pl.Int64, "B": pl.Int64, "E": pl.Int64, "P": pl.Int64},
    )

    # Per-track per-frame centroids for the round-trip sidecar.
    centroids: dict[int, list[dict[str, float | int]]] = {}
    for row in df.iter_rows(named=True):
        centroids.setdefault(int(row[TRACK_ID_COL]), []).append(
            {
                "t": int(row[TIME_COL]),
                "y": float(row[CENTROID_Y_COL]),
                "x": float(row[CENTROID_X_COL]),
            }
        )
    manifest_tracks = {
        str(relabel[o]): {
            "track_id": o,
            "parent_track_id": parents[i],
            "begin": begins[i],
            "end": ends[i],
            "frames": centroids.get(o, []),
        }
        for i, o in enumerate(orig_ids)
    }
    manifest = {
        "version": MANIFEST_VERSION,
        "well": well,
        "label_scheme": "relabel_1_to_n_by_begin_frame",
        "centroid_units": "pixels",
        "centroid_axes": "yx",
        "n_tracks": len(orig_ids),
        "tracks": manifest_tracks,
    }
    return CtcExport(
        track_table=track_table, relabel=relabel, manifest=manifest
    )


def _mask_dtype(max_label: int) -> type[np.unsignedinteger[Any]]:
    """Smallest unsigned int that holds ``max_label`` (CTC masks are integer)."""
    return np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32


def relabel_mask(
    mask: NDArray[Any],
    relabel: dict[int, int],
    dtype: type[np.unsignedinteger[Any]] = np.uint16,
) -> NDArray[Any]:
    """Remap a label image's pixel values via ``relabel`` (a lookup table).

    Background (``0``) stays ``0``, and any label **not** in ``relabel`` maps to
    ``0`` — so nuclei that carry no track (e.g. border-clipped cells dropped
    from the measurements) do not leak into the CTC masks as spurious spots.

    Args:
        mask: 2-D label frame (``track_id`` per pixel, as cached in the zarr).
        relabel: ``track_id -> CTC label`` map (from :func:`build_ctc_export`).
        dtype: Output integer dtype.

    Returns:
        The remapped frame, ``dtype``.
    """
    arr = np.asarray(mask)
    max_in = int(arr.max()) if arr.size else 0
    lut = np.zeros(max_in + 1, dtype=dtype)
    for old, new in relabel.items():
        if 0 <= old <= max_in:
            lut[old] = new
    remapped: NDArray[Any] = lut[arr]
    return remapped


def write_ctc_masks(
    nuclei_tyx: Any, relabel: dict[int, int], out_dir: Path
) -> list[Path]:
    """Write per-frame ``maskTTT.tif`` label images from a ``(T,Y,X)`` array.

    Reads one timepoint at a time (the zarr is chunked per timepoint), remaps
    its labels to the CTC scheme, and writes a compressed TIFF. File index ==
    frame index, so it lines up with the ``B``/``E`` frames in
    ``res_track.txt``.

    Returns the mask paths written, in frame order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = _mask_dtype(max(relabel.values(), default=0))
    paths: list[Path] = []
    n_t = int(nuclei_tyx.shape[0])
    for t in range(n_t):
        frame = relabel_mask(np.asarray(nuclei_tyx[t]), relabel, dtype)
        path = out_dir / f"mask{t:03d}.tif"
        tifffile.imwrite(path, frame, compression="zlib")
        paths.append(path)
    return paths


def write_unit_scale_view(src_group: Path, dst_group: Path) -> Path:
    """Write a metadata-only OME-NGFF view of an image group at unit scale.

    Mastodon's CTC importer places spots at **raw pixel** coordinates, but our
    cached OME-NGFF carries the microscope's physical pixel size in its
    ``coordinateTransformations`` — so Mastodon renders the image in microns
    and the imported pixel-space spots overshoot the picture by
    ``1 / pixel_size`` (e.g. a 0.59 µm/px well makes tracks ~1.69× too large).

    This writes a sibling group whose ``.zattrs`` spatial scales are divided by
    the level-0 pixel size (level 0 → 1, the pyramid kept as 1/2/4…) and
    **symlinks** the heavy per-level chunk directories — a few KB, no pixel
    copy. Open this in Mastodon instead of the raw cache group so the picture
    shares the masks' pixel grid. The napari cache keeps its real µm
    calibration, untouched.

    Falls back to returning ``src_group`` unchanged (with a warning) if the
    group has no readable multiscale metadata, so an export never fails here.

    Args:
        src_group: The cached image multiscale group (``…/<row>/<col>/0``).
        dst_group: Destination directory for the unit-scale view.

    Returns:
        The path to open in Mastodon — ``dst_group`` on success, else
        ``src_group``.
    """
    try:
        attrs = json.loads((src_group / ".zattrs").read_text())
        multiscale = attrs["multiscales"][0]
        axes = [a["name"] for a in multiscale["axes"]]
        xi, yi = axes.index("x"), axes.index("y")
        datasets = multiscale["datasets"]
        pixel_size = datasets[0]["coordinateTransformations"][0]["scale"][xi]
    except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
        logger.warning(
            f"No multiscale metadata at {src_group} ({exc!r}); pointing "
            f"Mastodon at the raw cache group — tracks may be mis-scaled."
        )
        return src_group

    if dst_group.exists() or dst_group.is_symlink():
        shutil.rmtree(dst_group, ignore_errors=True)
    dst_group.mkdir(parents=True)
    shutil.copy(src_group / ".zgroup", dst_group / ".zgroup")

    for dataset in datasets:
        scale = dataset["coordinateTransformations"][0]["scale"]
        if pixel_size:
            scale[xi] = round(scale[xi] / pixel_size, 6)
            scale[yi] = round(scale[yi] / pixel_size, 6)
        # Symlink the per-level chunk dir (absolute target) — no pixel copy.
        (dst_group / dataset["path"]).symlink_to(
            (src_group / dataset["path"]).resolve(), target_is_directory=True
        )
    (dst_group / ".zattrs").write_text(json.dumps(attrs, indent=2))
    return dst_group


_README = """\
Mastodon CTC import — plate {plate_id} well {well}
==================================================

This folder is a Cell Tracking Challenge (CTC) bundle:
    mask000.tif … mask{last:03d}.tif   per-frame nucleus label images
    res_track.txt                      lineage table (L B E P)
    manifest.json                      CellView round-trip metadata — keep it,
                                       do not edit
    mastodon_image/                    OME-NGFF image to open in Mastodon — a
                                       metadata-only view (symlinks, no pixel
                                       copy) of the cached well at UNIT pixel
                                       scale, so it lines up with the masks

Why CTC and not a CSV: Mastodon's CSV importer creates spots but NOT links, so
it loses every track and lineage. The CTC importer below rebuilds full
lineages, divisions included, from res_track.txt.

The omero-cache is an LRU cache. If you will curate this well over time, click
"Pin plate" in the napari Tracks widget first so it is not evicted
mid-curation; "Unpin plate" when you are done. (mastodon_image/ symlinks into
that cache, so the cache must stay present.)

1. Open the image. Fiji → Plugins → Tracking → Mastodon → "new from OME-NGFF…"
   Paste this path (opens directly; no pixel copy):
       {image_path}
   → Detect datasets → click the listed row → OK → save the BDV XML anywhere.
   It should report {n_frames} timepoints. Press P for the side panel to
   adjust per-channel contrast.

   IMPORTANT: open mastodon_image/, NOT the raw omero-cache zarr. The cache
   image carries the microscope's physical pixel size, but the CTC importer
   places spots at raw PIXEL coordinates — opening the calibrated image makes
   the tracks overshoot the picture by 1/pixel_size. mastodon_image/ is the
   same pixels at unit scale, so spots land on the nuclei.

2. Import the tracks. Main Mastodon window → Plugins → Cell Tracking Challenge
   → "Import from CTC format". In the "From where to import CTC tracking"
   dropdown choose "CTC: result data" — NOT "View: channel …" (that reads a
   fluorescence channel as a label image and scatters spurious spots/links
   everywhere). Set "Import till this time point" to {last}, click OK, then
   choose this folder:
       {out_dir}
   Mastodon reads res_track.txt for the parent / division links.

   Do NOT use "Plugins → Imports → Import from instance segmentation": it links
   by label/overlap and ignores res_track.txt, so divisions are lost.

3. Link the views: click the same group-lock number (e.g. 1) in BOTH the
   BigDataViewer and TrackScheme windows; double-click a spot to navigate.

4. When done, export back to CTC: Plugins → Cell Tracking Challenge → "Export
   to CTC format". Keep manifest.json with it — that is what a later step uses
   to reconcile the corrected tracks into CellView's curated track_id /
   parent_track_id columns.

{n_tracks} tracks, {n_div} division(s), {n_frames} frames. Labels are
renumbered 1..N by begin frame, so they do NOT match the raw CellView
track_id (manifest.json holds the mapping).
"""


def export_well_ctc(
    plate_id: int,
    well: str,
    plate_data: pl.LazyFrame,
    out_base: Path | None = None,
) -> dict[str, Path]:
    """Write a well's full CTC bundle and a guided-import README.

    Produces ``mask*.tif`` (from the cached zarr nuclei labels), ``res_track.txt``,
    ``manifest.json``, ``README.txt`` and a ``mastodon_image/`` unit-scale image
    view under ``<out_base>/plate_<id>_<well>_ctc/``. No OMERO round-trip — the
    masks come straight from the cache.

    Args:
        plate_id: Plate to export from.
        well: Well position (e.g. ``"B2"``).
        plate_data: CellView measurements LazyFrame (``omero_data.plate_data``).
        out_base: Base directory for the export folder; defaults to
            :data:`DEFAULT_EXPORT_BASE`.

    Returns:
        Mapping with ``dir``, ``res_track``, ``manifest``, ``readme`` and
        ``image`` (the unit-scale OME-NGFF view to open in Mastodon) paths.

    Raises:
        KeyError: If the plate has no track columns.
        ValueError: If the well has no tracked rows.
        FileNotFoundError: If the well's nuclei labels are not cached.
    """
    export = build_ctc_export(plate_data, well)

    well_group = _well_image_group(plate_id, well)
    nuclei = read_well(plate_id, well)["nuclei"]
    if not nuclei:
        raise FileNotFoundError(
            f"No cached nuclei labels for plate {plate_id} well {well}. Build "
            f"the zarr cache first (Welldata widget → Cache Plate)."
        )
    nuclei_tyx = nuclei[0]  # level-0 full-resolution (T, Y, X)

    base = out_base or DEFAULT_EXPORT_BASE
    out_dir = base / f"plate_{plate_id}_{well}_ctc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unit-scale image view to open in Mastodon (matches the pixel-space masks).
    image_view = write_unit_scale_view(well_group, out_dir / "mastodon_image")

    mask_paths = write_ctc_masks(nuclei_tyx, export.relabel, out_dir)

    res_track = out_dir / "res_track.txt"
    export.track_table.write_csv(
        res_track, separator=" ", include_header=False
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(export.manifest, indent=2))

    n_div = (
        export.track_table.filter(pl.col("P") != 0)
        .group_by("P")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    readme_path = out_dir / "README.txt"
    readme_path.write_text(
        _README.format(
            plate_id=plate_id,
            well=well,
            image_path=image_view,
            out_dir=out_dir,
            last=max(len(mask_paths) - 1, 0),
            n_tracks=export.manifest["n_tracks"],
            n_div=n_div,
            n_frames=len(mask_paths),
        )
    )
    logger.info(
        f"Wrote CTC bundle for plate {plate_id} well {well}: "
        f"{len(mask_paths)} masks, {export.manifest['n_tracks']} tracks -> {out_dir}"
    )
    return {
        "dir": out_dir,
        "res_track": res_track,
        "manifest": manifest_path,
        "readme": readme_path,
        "image": image_view,
    }
