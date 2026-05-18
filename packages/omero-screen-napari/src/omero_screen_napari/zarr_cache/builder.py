"""Build a plate.zarr cache from a stitched OMERO plate.

The builder is the heavy module that bridges OMERO data → stitched arrays →
``PlateZarrWriter``. It is the only entry point that talks to OMERO. The
napari widget invokes :func:`build_plate_zarr` as a thread worker; headless
callers (tests, scripts) call it directly.

Build flow per well:

1. Load N fields' raw pixel arrays (T, C, Y, X) and stage positions.
2. Apply flatfield per field, then stitch all channels in one call.
3. Fetch per-field stitched-mode segmentation masks via OMERO map
   annotations (``Stitched_Segmentation_Mask``) on the source images.
4. Recompose those per-field tiles into a single canvas via
   :func:`recompose_split_labels` (lossless: the masks were produced by a
   canvas-wide segmentation, then split for upload, so label IDs are
   globally unique).
5. Hand the result to :class:`PlateZarrWriter` for one well's worth of
   NGFF output.

Yields each completed well ID so a worker thread can stream progress to
the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from omero.gateway import BlitzGateway, WellWrapper
from omero_utils.images import fetch_stitched_field_masks
from omero_utils.stitching import (
    OPERETTA_STITCH_DEFAULTS,
    recompose_split_labels,
    stitch_from_positions,
)

from omero_screen_napari.omero_data import get_dataset_id
from omero_screen_napari.omero_image import get_image
from omero_screen_napari.plate_cache import (
    _detect_label_stitched_mode,
    _fetch_plate_metadata,
    _fetch_well_map,
)
from omero_screen_napari.zarr_cache.eviction import (
    enforce_size_cap,
    estimate_plate_size_bytes,
)
from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert
from omero_screen_napari.zarr_cache.writer import PlateZarrWriter

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dispatch helper
# ----------------------------------------------------------------------


def is_stitched_plate(conn: BlitzGateway, plate_id: int) -> bool:
    """Return True if the plate has any stitched-mode segmentation masks.

    Mirrors :func:`plate_cache._detect_label_stitched_mode` — kept here as
    the public entry point used by the widget to choose between the zarr
    builder and the existing diskcache builder.
    """
    return _detect_label_stitched_mode(conn, plate_id)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_flatfield_dict(
    conn: BlitzGateway,
    ff_mask_id: int,
    channel_data: dict[str, str],
) -> dict[str, npt.NDArray[Any]]:
    """Download the flatfield correction mask and split into per-channel arrays.

    The flatfield image has shape ``(T=1, Z=1, Y, X, C)`` with one C per
    plate channel. Returns ``{channel_name: (Y, X) float array}``.
    """
    ff_array = get_image(conn, ff_mask_id)
    # (T, Z, Y, X, C) → squeeze T and Z
    if ff_array.shape[0] != 1 or ff_array.shape[1] != 1:
        raise ValueError(
            f"Flatfield image {ff_mask_id} has unexpected shape {ff_array.shape}"
        )
    ff_yxc = ff_array[0, 0]
    out: dict[str, npt.NDArray[Any]] = {}
    for ch_name, idx_str in channel_data.items():
        ch_idx = int(idx_str)
        out[ch_name] = ff_yxc[..., ch_idx].astype(np.float32, copy=False)
    return out


def _load_well_fields(
    conn: BlitzGateway,
    well: WellWrapper,
    channel_data: dict[str, str],
    flatfield_dict: dict[str, npt.NDArray[Any]],
) -> tuple[
    npt.NDArray[Any],
    list[tuple[float, float]],
    int,
    int,
]:
    """Load one well's fields, flatfield-correct, and return as a stack.

    Returns:
        images: ``(N_fields, T, Y, X, C)`` float32 array.
        positions: list of stage (px, py) per field.
        tile_h, tile_w: per-field Y, X dimensions.
    """
    n_fields = len(list(well.listChildren()))
    channels = list(channel_data.keys())
    per_channel_fields: dict[str, list[npt.NDArray[Any]]] = {
        ch: [] for ch in channels
    }
    positions: list[tuple[float, float]] = []
    tile_h = tile_w = 0

    for n in range(n_fields):
        ws = well.getWellSample(n)
        image_obj = ws.getImage()
        px = ws.getPosX()
        py = ws.getPosY()
        positions.append(
            (
                px.getValue() if px is not None else 0.0,
                py.getValue() if py is not None else 0.0,
            )
        )

        # napari's cached get_image returns (T, Z, Y, X, C).
        array = get_image(conn, image_obj.getId(), tag=image_obj.getId())
        if array.shape[1] != 1:
            raise ValueError(
                f"Field image {image_obj.getId()} has Z={array.shape[1]}; "
                f"expected Z=1"
            )
        tile_h = array.shape[2]
        tile_w = array.shape[3]
        for ch_name, idx_str in channel_data.items():
            ch_idx = int(idx_str)
            # Flatfield correction needs float arithmetic; immediately cast
            # back to uint16 (the source pixel type). Flatfield masks are
            # normalised to median=1, so values typically stay within the
            # uint16 range; clip handles dark-corner over-correction.
            raw = array[:, 0, :, :, ch_idx].astype(np.float32, copy=False)
            corrected = raw / flatfield_dict[ch_name]
            field = np.clip(corrected, 0, np.iinfo(np.uint16).max).astype(
                np.uint16, copy=False
            )
            per_channel_fields[ch_name].append(field)  # (T, Y, X)

    # Stack into (N, T, Y, X, C). Channel order follows channel_data.
    per_channel_stacks = [
        np.stack(per_channel_fields[ch], axis=0) for ch in channels
    ]  # each (N, T, Y, X)
    stacked = np.stack(per_channel_stacks, axis=-1)  # (N, T, Y, X, C)
    return stacked, positions, tile_h, tile_w


def _stitch_image(
    images_ntyxc: npt.NDArray[Any],
    positions: list[tuple[float, float]],
) -> npt.NDArray[Any]:
    """Stitch (N, T, Y, X, C) → (T, C, Y, X)."""
    stitched_tyxc = stitch_from_positions(
        images_ntyxc, positions, **OPERETTA_STITCH_DEFAULTS
    )  # (T, Y, X, C)
    # Reorder to writer's expected layout (T, C, Y, X).
    return np.transpose(stitched_tyxc, (0, 3, 1, 2))


_LABEL_PLACEMENT_KEYS = (
    "overlap_x",
    "overlap_y",
    "translate_x",
    "translate_y",
)


def _recompose_labels(
    per_field_masks: list[npt.NDArray[Any]],
    positions: list[tuple[float, float]],
    tile_h: int,
    tile_w: int,
) -> npt.NDArray[Any]:
    """Recompose per-field label tiles (list of (T, Y, X)) → (T, Y, X).

    Filters ``OPERETTA_STITCH_DEFAULTS`` to the placement keys only:
    ``edge`` is image-blending and not accepted by the label recomposer.
    Mirrors the filter used by ``loops.py`` when splitting masks.
    """
    placement = {k: OPERETTA_STITCH_DEFAULTS[k] for k in _LABEL_PLACEMENT_KEYS}
    return recompose_split_labels(
        per_field_masks,
        positions,
        tile_h,
        tile_w,
        **placement,
    )


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def build_plate_zarr(
    plate_id: int,
    conn: BlitzGateway,
    *,
    wells: Iterable[str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Build (or extend) the plate.zarr cache for ``plate_id``.

    Generator yielding each well ID after it has been written. Callers
    (the widget worker thread or a script) can use that to drive a
    progress indicator.

    Args:
        plate_id: OMERO plate ID.
        conn: Live OMERO connection.
        wells: Subset of well labels to build (e.g. ``["A1", "B7"]``).
            If ``None``, every well advertised by the plate is built.
        progress_cb: Optional callback invoked with the well ID after
            each successful write — same value the generator yields.
            Lets non-iterator callers (UI thread) react.
    """
    metadata = _fetch_plate_metadata(conn, plate_id)
    channel_data: dict[str, str] = metadata["channel_data"]
    pixel_size = metadata["pixel_size"]
    plate_name = metadata["plate_name"]
    ff_mask_id = metadata["ff_mask_id"]

    well_map = _fetch_well_map(conn, plate_id)
    all_wells = sorted(well_map.keys())
    if not all_wells:
        raise ValueError(f"Plate {plate_id} has no wells")
    target_wells = sorted(wells) if wells is not None else all_wells

    # n_timepoints from the first image of any well. Plates are
    # homogeneous in T.
    first_well = well_map[all_wells[0]]
    first_image = first_well["images"][0]
    n_timepoints = int(first_image["dims"][0])  # dims = (T, C, Z, Y, X)

    pixel_size_um = pixel_size[0] if pixel_size else None

    # Pre-flight: estimate the build's footprint and evict LRU plates if
    # we'd otherwise blow the cap. Raises ZarrCacheTooSmall if a single
    # plate exceeds the cap on its own.
    field_dims = first_image["dims"]  # (T, C, Z, Y, X)
    field_h = int(field_dims[3])
    field_w = int(field_dims[4])
    n_fields_first = len(first_well["images"])
    # Stitched canvas is ~ sqrt(n_fields) × tile. Use a generous upper
    # bound: ceil(sqrt(N)) × tile to account for overlap subtraction not
    # quite reaching the theoretical max.
    import math

    grid_side = math.ceil(math.sqrt(max(n_fields_first, 1)))
    stitched_h = grid_side * field_h
    stitched_w = grid_side * field_w
    estimated = estimate_plate_size_bytes(
        n_wells=len(target_wells),
        n_timepoints=n_timepoints,
        n_channels=len(channel_data),
        stitched_h=stitched_h,
        stitched_w=stitched_w,
    )
    evicted = enforce_size_cap(extra_bytes=estimated)
    if evicted:
        logger.info(
            "Evicted %d plates to make room: %s", len(evicted), evicted
        )

    flatfield_dict = _load_flatfield_dict(conn, ff_mask_id, channel_data)

    # Need the live Plate object to iterate fields. _fetch_well_map only
    # returns ids + positions; per-field pixel download still wants the
    # OMERO wrapper.
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate {plate_id} not found in OMERO")
    well_objs: dict[str, WellWrapper] = {
        w.getWellPos(): w for w in plate.listChildren()
    }

    dataset_id = get_dataset_id(conn, plate_id)
    if not dataset_id:
        raise ValueError(
            f"Plate {plate_id} has no segmentation dataset — cannot find "
            f"stitched masks. Was the stitched analysis run?"
        )

    writer = PlateZarrWriter(
        plate_id=plate_id,
        plate_name=plate_name,
        channel_names=list(channel_data.keys()),
        pixel_size_um=pixel_size_um,
        n_timepoints=n_timepoints,
    )

    # Stash per-well metadata (cell_line, condition, timepoint, ...) so
    # the napari load path can render an overlay without re-hitting OMERO.
    well_meta_map = {
        well_pos: dict(well_map[well_pos].get("metadata", {}))
        for well_pos in all_wells
    }

    with writer:
        writer.ensure_plate(all_wells=all_wells, well_metadata=well_meta_map)

        for well_pos in target_wells:
            if well_pos not in well_objs:
                logger.warning(
                    "Well %s requested but not present on plate %s; skipping",
                    well_pos,
                    plate_id,
                )
                continue
            well_obj = well_objs[well_pos]
            logger.info(
                "Building zarr for well %s of plate %s", well_pos, plate_id
            )

            # --- Image branch -------------------------------------------------
            images_ntyxc, positions, tile_h, tile_w = _load_well_fields(
                conn, well_obj, channel_data, flatfield_dict
            )
            image_tcyx = _stitch_image(images_ntyxc, positions)

            # --- Label branch -------------------------------------------------
            try:
                nuc_per_field, cell_per_field, _ = fetch_stitched_field_masks(
                    conn, well_obj
                )
            except KeyError as e:
                # Well wasn't processed in stitched mode (no
                # Stitched_Segmentation_Mask annotation). Skip rather than
                # abort so partially-stitched plates still build for the
                # wells that *do* qualify.
                logger.warning(
                    "Skipping well %s: not processed in stitched mode (%s)",
                    well_pos,
                    e,
                )
                continue
            nuc_stitched = _recompose_labels(
                nuc_per_field, positions, tile_h, tile_w
            )
            cell_stitched: npt.NDArray[Any] | None
            if any(c is not None for c in cell_per_field):
                # All-or-nothing: if any field has a cell mask, expect them
                # all (the omero-screen pipeline writes both channels per
                # field when cell segmentation ran).
                if not all(c is not None for c in cell_per_field):
                    raise ValueError(
                        f"Well {well_pos} has cell masks for some fields but "
                        f"not all — refusing to recompose mixed coverage."
                    )
                cell_stitched = _recompose_labels(
                    [c for c in cell_per_field if c is not None],
                    positions,
                    tile_h,
                    tile_w,
                )
            else:
                cell_stitched = None

            # --- Write --------------------------------------------------------
            writer.write_well(
                well_pos, image_tcyx, nuc_stitched, cell_stitched
            )

            if progress_cb is not None:
                progress_cb(well_pos)
            yield well_pos

    # Register the plate (overwrites any prior entry, refreshing size and
    # well count). Doing this once at the end avoids hammering the JSON
    # with per-well writes.
    upsert(
        ZarrPlateEntry(
            plate_id=plate_id,
            plate_name=plate_name,
            size_bytes=_dir_size(plate_zarr_path(plate_id)),
            n_wells_written=len(target_wells),
        )
    )
    logger.info(
        "Finished zarr build for plate %s: %d wells",
        plate_id,
        len(target_wells),
    )
