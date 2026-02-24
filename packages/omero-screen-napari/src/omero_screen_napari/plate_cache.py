"""Cache orchestration for complete plate data.

Caches all data needed for display (metadata, flatfield-corrected images,
labels, stage positions) so that well navigation is fully offline once a
plate is cached. Supports concurrent downloads with progress reporting.

Cache key structure (stored in the shared diskcache from omero_image.py):
    plate:{plate_id}:meta   -> dict with channel_data, pixel_size, intensities, plate_name
    plate:{plate_id}:wells  -> dict mapping well_pos -> {well_id, metadata, images: [...]}
    plate:{plate_id}:labels -> dict mapping well_pos -> [label_image_id, ...]
    {image_id}:{timepoint}  -> flatfield-corrected float32 ZYXC numpy array
"""

import os
import queue
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
import omero
import polars as pl
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.rtypes import unwrap
from omero_screen.config import get_logger, getenv_as_int

from omero_screen_napari.omero_image import (
    _cache,
    _download_lock,
    _get_omero_image_timepoint,
    _get_omero_image_wrapper,
    get_image,
)

logger = get_logger(__name__)


# --------------- Public API ---------------


def get_all_cached_plates() -> list[tuple[int, str]]:
    """Return all cached plates as (plate_id, plate_name) pairs.

    Scans cache keys for ``plate:*:meta`` entries and extracts plate info.
    Results are sorted by plate_id descending (most recent first).

    Returns:
        List of (plate_id, plate_name) tuples.
    """
    plates: list[tuple[int, str]] = []
    try:
        for key in _cache.iterkeys():
            if not isinstance(key, str):
                continue
            if not key.startswith("plate:") or not key.endswith(":meta"):
                continue
            try:
                plate_id = int(key.split(":")[1])
            except (ValueError, IndexError):
                continue
            meta = _cache.get(key)
            if not isinstance(meta, dict):
                continue
            plate_name = meta.get("plate_name", str(plate_id))
            plates.append((plate_id, plate_name))
    except Exception:
        logger.debug("Error scanning cache keys", exc_info=True)
    plates.sort(key=lambda x: x[0], reverse=True)
    return plates


def get_plate_history() -> dict[int, dict[str, str]]:
    """Return persistent plate history from the cache.

    History survives cache eviction — evicted plates appear with status
    ``"removed"`` so the user can re-cache them later.

    On the first call after an upgrade, existing ``plate:*:meta`` keys are
    migrated into the history with status ``"cached"``.

    Returns:
        Dict mapping plate_id -> {"plate_name", "status", "last_cached"}.
    """
    history: dict[int, dict[str, str]] = _cache.get("plate_history") or {}

    # Migration: populate history from existing plate:*:meta keys
    migrated = False
    try:
        for key in _cache.iterkeys():
            if not isinstance(key, str):
                continue
            if not key.startswith("plate:") or not key.endswith(":meta"):
                continue
            try:
                plate_id = int(key.split(":")[1])
            except (ValueError, IndexError):
                continue
            if plate_id in history:
                continue
            meta = _cache.get(key)
            if not isinstance(meta, dict):
                continue
            history[plate_id] = {
                "plate_name": meta.get("plate_name", str(plate_id)),
                "status": "cached",
                "last_cached": str(date.today()),
            }
            migrated = True
    except Exception:
        logger.debug("Error during plate history migration", exc_info=True)

    if migrated:
        _cache["plate_history"] = history

    return history


def _update_plate_history(plate_id: int, plate_name: str, status: str) -> None:
    """Create or update a plate's history entry.

    Args:
        plate_id: OMERO plate ID.
        plate_name: Human-readable plate name.
        status: ``"cached"`` or ``"removed"``.
    """
    history: dict[int, dict[str, str]] = _cache.get("plate_history") or {}
    existing = history.get(plate_id, {})

    entry: dict[str, str] = {
        "plate_name": plate_name,
        "status": status,
    }

    if status == "cached":
        entry["last_cached"] = str(date.today())
    else:
        # Preserve the previous last_cached date
        entry["last_cached"] = existing.get("last_cached", str(date.today()))

    history[plate_id] = entry
    _cache["plate_history"] = history


def remove_plate_from_history(plate_id: int) -> None:
    """Forget a plate entirely — remove from history and delete cached data.

    Args:
        plate_id: OMERO plate ID.
    """
    # Delete cached data if present
    if is_plate_cached(plate_id):
        delete_plate_from_cache(plate_id)

    history: dict[int, dict[str, str]] = _cache.get("plate_history") or {}
    if plate_id in history:
        del history[plate_id]
        _cache["plate_history"] = history
        logger.info("Removed plate %d from history", plate_id)


def get_well_cache_status(plate_id: int) -> dict[str, bool]:
    """Check per-well cache completeness for a plate.

    For each well, checks whether **all** its images x timepoints exist
    in cache using the ``in`` operator (fast SQLite index lookup, no
    data deserialization).

    Args:
        plate_id: OMERO plate ID.

    Returns:
        Dict mapping well_pos -> True if fully cached, False otherwise.
        Empty dict if plate is not in cache.
    """
    wells = _cache.get(f"plate:{plate_id}:wells")
    if not isinstance(wells, dict) or not wells:
        return {}

    status: dict[str, bool] = {}
    for well_pos, well_info in wells.items():
        all_cached = True
        for img_info in well_info.get("images", []):
            image_id = img_info["image_id"]
            size_t = img_info.get("size_t", 1)
            for t in range(size_t):
                if f"{image_id}:{t}" not in _cache:
                    all_cached = False
                    break
            if not all_cached:
                break
        status[well_pos] = all_cached
    return status


def is_plate_cached(plate_id: int) -> bool:
    """Check if plate metadata and well data exist in cache."""
    return (
        _cache.get(f"plate:{plate_id}:meta") is not None
        and _cache.get(f"plate:{plate_id}:wells") is not None
    )


def is_plate_fully_cached(plate_id: int) -> bool:
    """Check if plate metadata AND all well images are cached.

    Unlike ``is_plate_cached`` (which only checks metadata), this verifies
    that every image x timepoint key exists in the cache.

    Args:
        plate_id: OMERO plate ID.

    Returns:
        True only when every well's images are fully cached.
    """
    if not is_plate_cached(plate_id):
        return False
    status = get_well_cache_status(plate_id)
    return bool(status) and all(status.values())


def get_cached_plate_metadata(plate_id: int) -> dict[str, Any] | None:
    """Return cached plate metadata or None."""
    return _cache.get(f"plate:{plate_id}:meta")  # type: ignore[no-any-return]


def get_cached_well_data(plate_id: int) -> dict[str, Any] | None:
    """Return cached wells dict or None."""
    return _cache.get(f"plate:{plate_id}:wells")  # type: ignore[no-any-return]


def get_cached_label_map(plate_id: int) -> dict[str, list[int]] | None:
    """Return cached label map or None."""
    return _cache.get(f"plate:{plate_id}:labels")  # type: ignore[no-any-return]


def delete_plate_from_cache(plate_id: int) -> int:
    """Delete all cached data for a plate (metadata, images, labels).

    Removes the three metadata keys plus every image and label key
    referenced by the plate's well map and label map.  The plate is
    preserved in the persistent history with status ``"removed"``.

    Args:
        plate_id: OMERO plate ID.

    Returns:
        Number of keys deleted.
    """
    deleted = 0
    keys_to_delete: list[str] = []

    # Read plate name before deleting metadata
    meta = _cache.get(f"plate:{plate_id}:meta")
    plate_name = (
        meta.get("plate_name", str(plate_id))
        if isinstance(meta, dict)
        else str(plate_id)
    )

    # Enumerate image keys from wells
    wells = _cache.get(f"plate:{plate_id}:wells")
    if isinstance(wells, dict):
        for well_info in wells.values():
            for img_info in well_info.get("images", []):
                image_id = img_info["image_id"]
                size_t = img_info.get("size_t", 1)
                for t in range(size_t):
                    keys_to_delete.append(f"{image_id}:{t}")

    # Enumerate label keys from label map
    label_map = _cache.get(f"plate:{plate_id}:labels")
    if isinstance(label_map, dict):
        for label_ids in label_map.values():
            for label_id in label_ids:
                keys_to_delete.append(f"{label_id}:0")

    # Metadata keys
    meta_keys = [
        f"plate:{plate_id}:meta",
        f"plate:{plate_id}:wells",
        f"plate:{plate_id}:labels",
    ]
    keys_to_delete.extend(meta_keys)

    with _download_lock:
        for key in keys_to_delete:
            if _cache.pop(key, None) is not None:  # type: ignore[arg-type]
                deleted += 1

    # Preserve the plate in history as "removed"
    _update_plate_history(plate_id, plate_name, "removed")

    logger.info("Deleted %d keys for plate %d", deleted, plate_id)
    return deleted


def _estimate_plate_bytes(wells: dict[str, dict[str, Any]]) -> int:
    """Estimate total bytes needed to cache a plate's images.

    Counts the total number of image x timepoint slots and multiplies
    by a per-image byte estimate (configurable via env var).

    Args:
        wells: Well map dict from ``_fetch_well_map()``.

    Returns:
        Estimated bytes.
    """
    per_image = getenv_as_int("OMERO_SCREEN_IMAGE_SIZE_ESTIMATE", 20 * 2**20)
    total_slots = 0
    for well_info in wells.values():
        for img_info in well_info.get("images", []):
            total_slots += img_info.get("size_t", 1)
    return total_slots * per_image


def ensure_cache_space(
    needed_bytes: int, exclude_plate_ids: set[int] | None = None
) -> list[int]:
    """Evict whole plates (oldest first) until enough space is available.

    Args:
        needed_bytes: Bytes to free up.
        exclude_plate_ids: Plate IDs to skip (e.g. the plate being cached).

    Returns:
        List of plate IDs that were evicted.
    """
    if _cache.size_limit <= 0:
        return []

    exclude = exclude_plate_ids or set()
    evicted: list[int] = []

    if _cache.volume() + needed_bytes <= _cache.size_limit:
        return []

    # Get plates sorted ascending by plate_id (oldest/smallest first)
    candidates = get_all_cached_plates()
    candidates.reverse()  # was desc, now asc

    for plate_id, _name in candidates:
        if plate_id in exclude:
            continue
        delete_plate_from_cache(plate_id)
        evicted.append(plate_id)
        if _cache.volume() + needed_bytes <= _cache.size_limit:
            break

    if _cache.volume() + needed_bytes > _cache.size_limit:
        logger.warning(
            "Cache still needs %d bytes after evicting %d plate(s). "
            "volume=%d, limit=%d",
            needed_bytes,
            len(evicted),
            _cache.volume(),
            _cache.size_limit,
        )

    return evicted


def _plate_image_completeness(plate_id: int) -> float:
    """Compute fraction of expected images actually present in cache.

    Args:
        plate_id: OMERO plate ID.

    Returns:
        Float between 0.0 and 1.0, or 0.0 if wells data is missing.
    """
    wells = _cache.get(f"plate:{plate_id}:wells")
    if not isinstance(wells, dict) or not wells:
        return 0.0

    total = 0
    present = 0
    for well_info in wells.values():
        for img_info in well_info.get("images", []):
            image_id = img_info["image_id"]
            size_t = img_info.get("size_t", 1)
            for t in range(size_t):
                total += 1
                if f"{image_id}:{t}" in _cache:
                    present += 1

    return present / total if total > 0 else 0.0


def clean_orphaned_plates(
    exclude_plate_ids: set[int] | None = None,
) -> list[int]:
    """Remove plates with less than 50% image completeness.

    Useful for cleaning up partially cached plates that were
    interrupted or corrupted by eviction.

    Args:
        exclude_plate_ids: Plate IDs to skip (e.g. plates being downloaded).

    Returns:
        List of cleaned plate IDs.
    """
    exclude = exclude_plate_ids or set()
    cleaned: list[int] = []

    for plate_id, _name in get_all_cached_plates():
        if plate_id in exclude:
            continue
        completeness = _plate_image_completeness(plate_id)
        if completeness < 0.5:
            logger.info(
                "Cleaning orphaned plate %d (%.0f%% complete)",
                plate_id,
                completeness * 100,
            )
            delete_plate_from_cache(plate_id)
            cleaned.append(plate_id)

    return cleaned


def cache_plate(
    plate_id: int, conn: BlitzGateway, max_workers: int = 3
) -> Generator[tuple[int, int], None, None]:
    """Cache entire plate: metadata + flatfield-corrected images + labels.

    Opens one OMERO connection for metadata, then spawns workers for images.
    Yields (images_done, images_total) for progress reporting.

    Args:
        plate_id: OMERO plate ID.
        conn: Active OMERO connection (caller manages lifecycle).
        max_workers: Number of concurrent download threads.

    Yields:
        Tuple of (images_done, images_total).
    """
    # Step 1: Fetch and cache plate metadata
    logger.info("Caching plate %d: fetching metadata", plate_id)
    meta = _fetch_plate_metadata(conn, plate_id)
    _cache[f"plate:{plate_id}:meta"] = meta

    # Record in persistent history
    _update_plate_history(plate_id, meta["plate_name"], "cached")

    # Step 2: Fetch well map (all wells with images and stage positions)
    logger.info("Caching plate %d: fetching well data", plate_id)
    wells = _fetch_well_map(conn, plate_id)
    _cache[f"plate:{plate_id}:wells"] = wells

    # Proactively evict old plates to make room for this one
    estimated_bytes = _estimate_plate_bytes(wells)
    evicted = ensure_cache_space(estimated_bytes, exclude_plate_ids={plate_id})
    if evicted:
        logger.info(
            "Evicted plates %s to make room for plate %d", evicted, plate_id
        )

    # Step 3: Fetch flatfield mask into memory (NOT cached)
    logger.info("Caching plate %d: fetching flatfield mask", plate_id)
    project_id = int(os.getenv("PROJECT_ID", "0"))
    flatfield_masks = _fetch_flatfield_mask(conn, plate_id, project_id)

    # Step 4: Fetch label map
    logger.info("Caching plate %d: fetching label map", plate_id)
    label_map = _fetch_label_map(conn, plate_id, project_id)
    _cache[f"plate:{plate_id}:labels"] = label_map

    # Squeeze flatfield mask from TZYXC to ZYXC so that division with
    # per-timepoint ZYXC images keeps the array 4-D (not 5-D).
    if flatfield_masks.ndim == 5:
        flatfield_masks = flatfield_masks[0]  # drop T dimension
        logger.debug(
            "Flatfield mask squeezed to shape %s", flatfield_masks.shape
        )

    # Step 5: Build downloads grouped by well, sorted by well position.
    # Well-grouped partitioning ensures wells complete sequentially so
    # users can start loading cached wells before the entire plate is done.
    sorted_well_keys = sorted(wells.keys(), key=_well_sort_key)
    well_groups: list[list[dict[str, Any]]] = []

    for well_pos in sorted_well_keys:
        group: list[dict[str, Any]] = []
        # Well images (need flatfield correction, skip if already cached)
        for img_info in wells[well_pos]["images"]:
            for t in range(img_info["size_t"]):
                if f"{img_info['image_id']}:{t}" not in _cache:
                    group.append(
                        {
                            "image_id": img_info["image_id"],
                            "timepoint": t,
                            "apply_flatfield": True,
                        }
                    )
        # Well labels (no flatfield, skip if already cached)
        if well_pos in label_map:
            for label_id in label_map[well_pos]:
                if f"{label_id}:0" not in _cache:
                    group.append(
                        {
                            "image_id": label_id,
                            "timepoint": 0,
                            "apply_flatfield": False,
                        }
                    )
        if group:
            well_groups.append(group)

    # Distribute whole well groups to workers round-robin
    batches: list[list[dict[str, Any]]] = [[] for _ in range(max_workers)]
    for i, group in enumerate(well_groups):
        batches[i % max_workers].extend(group)
    batches = [b for b in batches if b]

    total = sum(len(b) for b in batches)
    done = 0

    if total == 0:
        logger.info("Plate %d: all images already cached", plate_id)
        yield (0, 0)
        return

    logger.info(
        "Caching plate %d: downloading %d items (%d wells) with %d workers",
        plate_id,
        total,
        len(well_groups),
        max_workers,
    )

    # Step 6 & 7: Download images + labels with per-image progress.
    # Workers signal each completed image via a shared queue so the
    # generator can yield smooth progress to the napari progress bar.
    progress_q: queue.Queue[int] = queue.Queue()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _download_batch,
                batch,
                flatfield_masks,
                progress_q,
            )
            for batch in batches
        ]

        # Drain the queue, yielding after every image
        while done < total:
            try:
                progress_q.get(timeout=1.0)
                done += 1
                yield (done, total)
            except queue.Empty:
                # Check if all workers have finished (error or success)
                if all(f.done() for f in futures):
                    break

        # Re-raise the first worker exception, if any
        for f in futures:
            exc = f.exception()
            if exc is not None:
                logger.exception(
                    "Error in download worker for plate %d", plate_id
                )
                raise exc

    logger.info("Plate %d: caching complete (%d items)", plate_id, total)


# --------------- Metadata Fetching ---------------


def _fetch_plate_metadata(conn: BlitzGateway, plate_id: int) -> dict[str, Any]:
    """Fetch channel_data, pixel_size, intensities, plate_name from OMERO.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID.

    Returns:
        Dict with keys: channel_data, pixel_size, plate_name.
        Intensities are added later if CellView data is available.
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate {plate_id} not found in OMERO")

    plate_name = plate.getName()

    # Channel data from plate map annotations
    channel_data = _parse_channel_data(plate)

    # Pixel size from first well's first image
    pixel_size = _parse_pixel_size(plate)

    # Intensities from CellView if available, otherwise default
    intensities = _parse_intensities_from_cellview(plate_id, channel_data)

    return {
        "channel_data": channel_data,
        "pixel_size": pixel_size,
        "intensities": intensities,
        "plate_name": plate_name,
    }


def _parse_channel_data(plate: Any) -> dict[str, str]:
    """Extract channel_data dict from plate map annotations.

    Returns:
        Dict mapping channel_name -> channel_index (as string).
    """
    annotations = plate.listAnnotations()
    map_annotations = [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]

    if not map_annotations:
        raise ValueError(f"No MapAnnotations found for plate {plate.getId()}")

    for map_ann in map_annotations:
        ann_value = map_ann.getValue()
        for key, _value in ann_value:
            if key.lower() in ["dapi", "hoechst", "rfp"]:
                channel_data = dict(ann_value)
                sorted_data = dict(
                    sorted(channel_data.items(), key=lambda item: item[1])
                )
                result = {k.strip(): v for k, v in sorted_data.items()}
                if "Hoechst" in result:
                    result["DAPI"] = result.pop("Hoechst")
                if "RFP" in result:
                    result["DAPI"] = result.pop("RFP")
                return result

    raise ValueError(
        f"No DAPI or Hoechst channel information found for plate {plate.getId()}"
    )


def _parse_pixel_size(plate: Any) -> tuple[float, float]:
    """Extract pixel size from first well's first image.

    Returns:
        Tuple of (x_size, y_size) in microns.
    """
    wells = list(plate.listChildren())
    if not wells:
        raise ValueError("No wells found in the plate")

    image = wells[0].getImage(0)
    if image is None:
        raise ValueError("No image found in first well")

    x_size = image.getPixelSizeX()
    y_size = image.getPixelSizeY()

    if x_size is None or y_size is None:
        raise ValueError("No pixel size data found for the image")

    return (round(x_size, 1), round(y_size, 1))


def _parse_intensities_from_cellview(
    plate_id: int, channel_data: dict[str, str]
) -> dict[int, tuple[int, int]]:
    """Try to load intensity scaling from CellView database.

    Falls back to default (0, 65535) if CellView is unavailable.

    Args:
        plate_id: OMERO plate ID.
        channel_data: Channel name -> index mapping.

    Returns:
        Dict mapping channel_index -> (min_intensity, max_intensity).
    """
    try:
        from cellview.db.db import CellViewDB
        from cellview.exporters.db_to_polars import export_polars_lf

        db = CellViewDB()
        db_conn = db.connect()

        # Check if plate exists in DB
        result = db_conn.execute(
            "SELECT COUNT(*) FROM repeats WHERE plate_id = ?", [plate_id]
        ).fetchone()
        if not result or result[0] == 0:
            db_conn.close()
            return _default_intensities(channel_data)

        lf, _ = export_polars_lf(plate_id, db_conn)
        db_conn.close()

        intensity_dict: dict[int, tuple[int, int]] = {}
        columns = lf.collect_schema().names()

        for channel, channel_value in channel_data.items():
            cell_cols = (
                f"intensity_max_{channel}_cell",
                f"intensity_min_{channel}_cell",
            )
            nucleus_cols = (
                f"intensity_max_{channel}_nucleus",
                f"intensity_min_{channel}_nucleus",
            )

            target_cols = None
            if all(col in columns for col in cell_cols):
                max_val = lf.select(pl.col(cell_cols[0])).mean().collect()
                if max_val[0, 0] is not None:
                    target_cols = cell_cols

            if target_cols is None:
                if all(col in columns for col in nucleus_cols):
                    target_cols = nucleus_cols
                else:
                    intensity_dict[int(channel_value)] = (0, 65535)
                    continue

            max_value = lf.select(pl.col(target_cols[0])).mean().collect()
            min_value = lf.select(pl.col(target_cols[1])).min().collect()

            if max_value[0, 0] is not None and min_value[0, 0] is not None:
                intensity_dict[int(channel_value)] = (
                    int(min_value[0, 0]),
                    int(max_value[0, 0]),
                )
            else:
                intensity_dict[int(channel_value)] = (0, 65535)

        return intensity_dict

    except Exception:
        logger.debug(
            "CellView not available for plate %d, using default intensities",
            plate_id,
        )
        return _default_intensities(channel_data)


def _default_intensities(
    channel_data: dict[str, str],
) -> dict[int, tuple[int, int]]:
    """Return default intensity range for all channels."""
    return {int(v): (0, 65535) for v in channel_data.values()}


def _unwrap_length(value: Any) -> float | None:
    """Convert an OMERO Length value to a plain float.

    ``unwrap()`` on OMERO ``Length`` types returns a dict like
    ``{'value': 1234.5, 'unit': 'MICROMETER', 'symbol': 'µm'}``.
    This helper normalises that (or a plain numeric) to ``float``,
    returning ``None`` when the value cannot be converted.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        v = value.get("value")
        return float(v) if v is not None else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_well_map(
    conn: BlitzGateway, plate_id: int
) -> dict[str, dict[str, Any]]:
    """Fetch all wells with metadata, image lists, and stage positions.

    Uses a single HQL query for images + positions, plus well annotations.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID.

    Returns:
        Dict mapping well_pos -> {well_id, metadata, images: [{image_id, size_t, index, pos_x, pos_y}]}
    """
    query_service = conn.getQueryService()
    params = omero.sys.ParametersI()
    params.addLong("plate_id", plate_id)
    query = """
        select w.id, w.row, w.column, ws.posX, ws.posY, i.id, pi.sizeT
        from Plate as p
          left join p.wells as w
          left join w.wellSamples as ws
          left join ws.image as i
          left join i.pixels as pi
        where p.id = :plate_id
        order by w.row, w.column, ws.posX, ws.posY
    """
    results = query_service.projection(query, params, conn.SERVICE_OPTS)
    if not results:
        raise ValueError(f"No wells found for plate {plate_id}")

    # Group by well
    well_map: dict[str, dict[str, Any]] = {}
    for row_data in results:
        well_id = unwrap(row_data[0])
        well_row = unwrap(row_data[1])
        well_col = unwrap(row_data[2])
        pos_x = unwrap(row_data[3])
        pos_y = unwrap(row_data[4])
        image_id = unwrap(row_data[5])
        size_t = unwrap(row_data[6])

        well_pos = _row_col_to_well_pos(well_row, well_col)

        if well_pos not in well_map:
            well_map[well_pos] = {
                "well_id": well_id,
                "metadata": {},
                "images": [],
            }

        well_map[well_pos]["images"].append(
            {
                "image_id": image_id,
                "size_t": size_t,
                "index": len(well_map[well_pos]["images"]),
                "pos_x": _unwrap_length(pos_x),
                "pos_y": _unwrap_length(pos_y),
            }
        )

    # Fetch well metadata from annotations
    plate = conn.getObject("Plate", plate_id)
    if plate is not None:
        for well in plate.listChildren():
            well_pos = well.getWellPos()
            if well_pos in well_map:
                for ann in well.listAnnotations():
                    if ann and ann.getValue():
                        map_ann = dict(ann.getValue())
                        if "cell_line" in map_ann:
                            well_map[well_pos]["metadata"] = map_ann
                            break

    return well_map


def _row_col_to_well_pos(row: int, col: int) -> str:
    """Convert 0-based row/column to well position string (e.g. 'A1')."""
    return f"{chr(65 + row)}{col + 1}"


def _fetch_flatfield_mask(
    conn: BlitzGateway,
    plate_id: int,
    project_id: int,
) -> npt.NDArray[Any]:
    """Download flatfield mask from OMERO dataset.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID (used to find dataset and mask name).
        project_id: OMERO project ID containing the screen dataset.

    Returns:
        Flatfield mask array.
    """
    # Find the screen dataset
    project = conn.getObject("Project", project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    dataset = conn.getObject(
        "Dataset",
        attributes={"name": str(plate_id)},
        opts={"project": project.getId()},
    )
    if dataset is None:
        raise ValueError(f"Dataset for plate {plate_id} not found")

    flatfield_mask_name = f"{plate_id}_flatfield_masks"
    for image in dataset.listChildren():
        if image.getName() == flatfield_mask_name:
            return get_image(conn, image.getId())

    raise ValueError(
        f"No flatfield mask found in dataset for plate {plate_id}"
    )


def _fetch_label_map(
    conn: BlitzGateway,
    plate_id: int,
    project_id: int,
) -> dict[str, list[int]]:
    """Map well image_ids to their segmentation label image_ids.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID.
        project_id: OMERO project ID.

    Returns:
        Dict mapping well_pos -> [label_image_id, ...].
    """
    project = conn.getObject("Project", project_id)
    if project is None:
        return {}

    dataset = conn.getObject(
        "Dataset",
        attributes={"name": str(plate_id)},
        opts={"project": project.getId()},
    )
    if dataset is None:
        return {}

    # Build lookup: image_id -> label_image_id from dataset children
    label_lookup: dict[int, int] = {}
    for child in dataset.listChildren():
        name = child.getName()
        if "_segmentation" in name:
            try:
                source_id = int(name.split("_")[0])
                label_lookup[source_id] = child.getId()
            except (ValueError, IndexError):
                continue

    # Now get well map to associate labels with wells
    wells = get_cached_well_data(plate_id)
    if wells is None:
        # Well data should already be cached by cache_plate()
        return {}

    label_map: dict[str, list[int]] = {}
    for well_pos, well_info in wells.items():
        label_ids = []
        for img_info in well_info["images"]:
            image_id = img_info["image_id"]
            if image_id in label_lookup:
                label_ids.append(label_lookup[image_id])
        label_map[well_pos] = label_ids

    return label_map


def _well_sort_key(well_pos: str) -> tuple[str, int]:
    """Sort wells by letter then number (A1, A2, ..., B1, ...).

    Args:
        well_pos: Well position string like "A1", "B12".

    Returns:
        Tuple of (letter, number) for sorting.
    """
    letter = well_pos[0]
    try:
        number = int(well_pos[1:])
    except ValueError:
        number = 0
    return (letter, number)


# --------------- Download Workers ---------------


def _partition_round_robin(
    items: list[dict[str, Any]], n: int
) -> list[list[dict[str, Any]]]:
    """Partition items into n batches using round-robin."""
    batches: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        batches[i % n].append(item)
    return [b for b in batches if b]  # Remove empty batches


def _download_batch(
    batch: list[dict[str, Any]],
    flatfield_masks: npt.NDArray[Any] | None,
    progress_q: queue.Queue[int] | None = None,
) -> None:
    """Download and optionally flatfield-correct a batch of images.

    Each worker opens its own OMERO connection.  After every image is
    stored in the cache, a ``1`` is put on *progress_q* so the caller
    can track per-image progress.

    Args:
        batch: List of dicts with image_id, timepoint, and apply_flatfield.
        flatfield_masks: Flatfield correction array (ZYXC), or None.
        progress_q: Optional queue for per-image progress signalling.
    """
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")

    conn = BlitzGateway(username, password, host=host)
    conn.connect()
    if not conn.isConnected():
        raise RuntimeError(
            f"Download worker failed to connect to OMERO at {host}"
        )

    try:
        conn.c.enableKeepAlive(60)
        for item in batch:
            image_id = item["image_id"]
            timepoint = item["timepoint"]
            apply_flatfield = item.get("apply_flatfield", True)
            key = f"{image_id}:{timepoint}"

            image = _get_omero_image_wrapper(conn, image_id)
            arr = _get_omero_image_timepoint(image, timepoint)

            if apply_flatfield and flatfield_masks is not None:
                # flatfield_masks is ZYXC (T already squeezed by caller).
                # arr is also ZYXC, so division stays 4-D.
                arr = arr.astype(np.float32) / flatfield_masks.astype(
                    np.float32
                )
            else:
                arr = arr.astype(np.float32)

            with _download_lock:
                _cache[key] = arr

            logger.debug("Cached %s (shape %s)", key, arr.shape)

            if progress_q is not None:
                progress_q.put(1)
    finally:
        conn.close(hard=True)


# --------------- Cache-First Loading ---------------


def load_from_cache(
    omero_data: Any,
    plate_id: int,
    well_pos_input: str,
    image_input: str,
    time: str = "All",
) -> None:
    """Populate OmeroData entirely from diskcache. No OMERO connection needed.

    Args:
        omero_data: OmeroData instance to populate.
        plate_id: Plate ID.
        well_pos_input: Comma-separated well positions (e.g. "A1, A2").
        image_input: Image index input (e.g. "All", "0-3", "1, 3").
        time: Time input (e.g. "All", "1-3").
    """
    meta = get_cached_plate_metadata(plate_id)
    wells = get_cached_well_data(plate_id)
    label_map = get_cached_label_map(plate_id)

    if meta is None or wells is None:
        raise ValueError(f"Plate {plate_id} not fully cached")

    omero_data.plate_id = plate_id
    omero_data.channel_data = meta["channel_data"]
    omero_data.pixel_size = meta["pixel_size"]
    omero_data.intensities = meta["intensities"]
    omero_data.plate_name = meta["plate_name"]

    # Parse user well selection
    selected_wells = [w.strip() for w in well_pos_input.split(",")]
    omero_data.well_pos_list = selected_wells

    # Parse image index selection
    image_index = _parse_image_index(image_input, wells, selected_wells)
    omero_data.image_index = image_index
    omero_data.image_input = image_input

    # Parse time crop
    tstart, tend = _parse_time_range(time)

    # Reset well/image data
    omero_data.reset_well_and_image_data()

    image_arrays: list[npt.NDArray[Any]] = []
    label_arrays: list[npt.NDArray[Any]] = []
    image_ids: list[int] = []
    positions: list[tuple[float, float] | None] = []

    for well_pos in selected_wells:
        if well_pos not in wells:
            raise ValueError(f"Well {well_pos} not found in cached data")

        well_info = wells[well_pos]
        omero_data.well_id_list.append(well_info["well_id"])
        omero_data.well_metadata_list.append(well_info["metadata"])

        # Get images for selected indices
        well_images = well_info["images"]
        for idx in image_index:
            if idx >= len(well_images):
                logger.warning(
                    "Image index %d out of range for well %s", idx, well_pos
                )
                continue

            img_info = well_images[idx]
            image_id = img_info["image_id"]
            size_t = img_info["size_t"]
            image_ids.append(image_id)

            # Collect stage position (handle dict from old caches)
            px = _unwrap_length(img_info.get("pos_x"))
            py = _unwrap_length(img_info.get("pos_y"))
            if px is not None and py is not None:
                positions.append((px, py))
            else:
                positions.append(None)

            # Determine timepoint range
            t_start = tstart if tstart is not None else 0
            t_end = tend if tend is not None else size_t

            timepoint_arrays = []
            for t in range(t_start, t_end):
                arr = _cache.get(f"{image_id}:{t}")
                if arr is None:
                    raise ValueError(
                        f"Image {image_id}:{t} not found in cache"
                    )
                timepoint_arrays.append(arr)

            if len(timepoint_arrays) == 1:
                image_arrays.append(timepoint_arrays[0])
            else:
                image_arrays.append(np.stack(timepoint_arrays, axis=0))

        # Get labels for this well
        if label_map and well_pos in label_map:
            well_label_ids = label_map[well_pos]
            for _, idx in enumerate(image_index):
                if idx < len(well_label_ids):
                    label_id = well_label_ids[idx]
                    label_arr = _cache.get(f"{label_id}:0")
                    if label_arr is not None:
                        label_arrays.append(label_arr.squeeze())

    if image_arrays:
        # Log shapes for debugging mismatches
        shapes = {arr.shape for arr in image_arrays}
        if len(shapes) > 1:
            logger.warning("Image arrays have inconsistent shapes: %s", shapes)
            for i, arr in enumerate(image_arrays):
                logger.warning("  image_arrays[%d]: shape %s", i, arr.shape)

        # Stack images: result is (N, Z, Y, X, C) for single timepoints
        stacked = np.stack(image_arrays, axis=0)

        # Squeeze singleton dimensions (Z typically = 1) but keep N and C
        # Match the shape convention used by ImageParser._parse_images()
        shape = stacked.shape
        squeeze_axes = tuple(
            i
            for i in range(1, len(shape) - 2)  # skip N (0) and Y,X,C at end
            if shape[i] == 1
        )
        if squeeze_axes:
            stacked = np.squeeze(stacked, axis=squeeze_axes)
        omero_data.images = stacked
    else:
        omero_data.images = np.empty((0,))

    omero_data.image_ids = image_ids
    omero_data.image_positions = positions

    if label_arrays:
        # Log shapes for debugging mismatches
        label_shapes = {arr.shape for arr in label_arrays}
        if len(label_shapes) > 1:
            logger.warning(
                "Label arrays have inconsistent shapes: %s", label_shapes
            )
        omero_data.labels = np.stack(label_arrays, axis=0)
    else:
        omero_data.labels = np.empty((0,))

    logger.info(
        "Loaded plate %d from cache: %d images, %d labels",
        plate_id,
        len(image_arrays),
        len(label_arrays),
    )


def _parse_image_index(
    image_input: str,
    wells: dict[str, dict[str, Any]],
    selected_wells: list[str],
) -> list[int]:
    """Parse the image index input string.

    Args:
        image_input: User input string ("All", "0-3", "1, 3, 4").
        wells: Cached well data.
        selected_wells: Selected well positions.

    Returns:
        List of 0-based image indices.
    """
    if image_input.lower() == "all":
        # Use the first selected well to determine image count
        for wp in selected_wells:
            if wp in wells:
                return list(range(len(wells[wp]["images"])))
        return [0]
    elif "-" in image_input:
        start, end = map(int, image_input.split("-"))
        return list(range(start, end + 1))
    elif "," in image_input:
        return list(map(int, image_input.split(",")))
    else:
        return [int(image_input)]


def _parse_time_range(time: str) -> tuple[int | None, int | None]:
    """Parse the time input string.

    Args:
        time: User input ("All", "1-3", "2").

    Returns:
        Tuple of (start, end) as 0-based indices, or (None, None) for all.
    """
    if time.lower() == "all":
        return None, None
    elif "-" in time:
        start, end = map(int, time.split("-"))
        return start - 1, end  # Convert to 0-based, end is exclusive
    else:
        t = int(time)
        return t - 1, t
