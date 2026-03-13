"""Cache orchestration for complete plate data.

Caches all data needed for display (metadata, well layout,
stage positions) so that well image navigation is offline once a
plate and its images are cached. The image cache is managed in
``omero_image``.

Supports concurrent downloads with progress reporting.

Cache key structure:
    m{plate_id}  -> dict with channel_data, pixel_size, intensities, plate_name
    w{plate_id}  -> dict mapping well_pos -> {well_id, metadata, images: [{"image_id", "dims", "pos"}...]}
    l{plate_id}  -> dict mapping well_pos -> [{"label_id", "dims"}, ...]
    history      -> dict mapping plate_id -> {"plate_name", "status", "last_cached"}

Cache "dims" entries are a tuple of TCZYX dimensions.
Cache "pos" entries are a tuple of XY image stage positions.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
import omero
import polars as pl
from diskcache import Cache
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.rtypes import unwrap
from omero_screen.config import get_logger, getenv_as_int

from omero_screen_napari.omero_image import (
    add_cached_image,
    cache_size_limit,
    cache_volume,
    get_bytes_size,
    get_cache_path,
    get_cached_image,
    get_image,
    get_key,
    get_omero_image_timepoint,
    get_omero_image_wrapper,
    initialise_download,
    is_cached,
)

# Refactor to a method to free space to the image cache code
# using a sorted tag list.
from omero_screen_napari.omero_image import (
    evict as evict_images,
)

logger = get_logger(__name__)


_HISTORY_KEY = b"history"
# Bump this when the on-disk metadata format changes.
# Caching will delete and re-download plates cached with an older version.
_CACHE_VERSION = 1

# Configure cache path using environment.
# Uses the default size limit. The cache is not expected to grow very large
# as it stores plate metadata only. It is used for persistent access to read
# only data contained in OMERO.
#
# Example plate metadata:
# {
#     "channel_data": {"DAPI": "0", "Tub": "1"},
#     "pixel_size": (0.3, 0.3),
#     "intensities": {0: (100, 5000), 1: (50, 3000)},
#     "plate_name": "TestPlate",
#     "ff_mask_id": 999,
#     "cache_version": 1,
# }
#
# Example plate well metadata:
# Note that images are sorted by the well XY position, not the well index.
# {
#     "A1": {
#         "well_id": 10,
#         "metadata": {"cell_line": "RPE", "condition": "ctrl"},
#         "images": [
#             {
#                 "image_id": 100,
#                 "dims": (1, 2, 1, 100, 100),
#                 "pos": (0.0, 0.0),
#             },
#             {
#                 "image_id": 101,
#                 "dims": (1, 2, 1, 100, 100),
#                 "pos": (1.0, 0.0),
#             },
#         ],
#     },
#
# Example plate label metadata:
# Each well position has an array of length equal to the "images" key of the well metadata.
# Images with no corresponding label have an entry of None.
# {
#     "A1": [
#         {"label_id": 500, "dims": (1, 2, 1, 100, 100)},
#         {"label_id": 501, "dims": (1, 2, 1, 100, 100)},
#     ],
# }

_cache = Cache(
    get_cache_path("plates"),
    tag_index=True,
)
logger.info(
    "Plate cache: %s (size limit: %d)",
    _cache.directory,
    _cache.size_limit,
)


# Cache uses the shortest possible keys that will not clash.
# Use bytes encoded positive integers with a prefix.


def _get_bytes(plate_id: int) -> bytes:
    """Get the plate id encode as bytes

    Decode using int.from_bytes()."""
    # ceil(bit_length + 7) // 8
    size = (plate_id.bit_length() + 7) >> 3
    return plate_id.to_bytes(size)


def _get_meta_key(plate_id: int) -> bytes:
    """Get the key for the plate metadata."""
    return b"m" + _get_bytes(plate_id)


def _get_well_key(plate_id: int) -> bytes:
    """Get the key for the plate well data."""
    return b"w" + _get_bytes(plate_id)


def _get_label_key(plate_id: int) -> bytes:
    """Get the key for the plate label data."""
    return b"l" + _get_bytes(plate_id)


# --------------- Public API ---------------


def get_all_cached_plates() -> list[tuple[int, str]]:
    """Return all cached plates as (plate_id, plate_name) pairs.

    Scans cache keys for metadata entries and extracts plate info.
    Results are sorted by plate_id descending (most recent first).

    Returns:
        List of (plate_id, plate_name) tuples.
    """
    plates: list[tuple[int, str]] = []
    try:
        for key in _cache:
            if not isinstance(key, bytes):
                continue
            if key[0] != 109:  # ord("m"):
                continue
            # Decode the bytes representation
            plate_id = int.from_bytes(key[1:])
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

    Returns:
        Dict mapping plate_id -> {"plate_name", "status", "last_cached"}.
    """
    history: dict[int, dict[str, str]] = _cache.get(_HISTORY_KEY) or {}
    return history


def _update_plate_history(plate_id: int, plate_name: str, status: str) -> None:
    """Create or update a plate's history entry.

    Args:
        plate_id: OMERO plate ID.
        plate_name: Human-readable plate name.
        status: ``"cached"`` or ``"removed"``.
    """
    history: dict[int, dict[str, str]] = _cache.get(_HISTORY_KEY) or {}
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
    _cache.set(_HISTORY_KEY, history)


def remove_plate_from_history(plate_id: int) -> None:
    """Forget a plate entirely — remove from history and delete cached data.

    Args:
        plate_id: OMERO plate ID.
    """
    # Delete cached data if present
    _deleted: int = _cache.evict(plate_id) + evict_images(plate_id)

    history: dict[int, dict[str, str]] = _cache.get(_HISTORY_KEY) or {}
    if plate_id in history:
        del history[plate_id]
        _cache.set(_HISTORY_KEY, history)
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
    wells = get_cached_well_data(plate_id)
    if not isinstance(wells, dict) or not wells:
        return {}
    label_map = get_cached_label_map(plate_id)
    if not isinstance(label_map, dict) or not label_map:
        return {}

    # Require flat-field mask
    meta = get_cached_plate_metadata(plate_id)
    if meta is None:
        return {}
    ff_mask_id = meta.get("ff_mask_id", 0)
    if not is_cached(get_key(ff_mask_id, 0)):
        return {}

    status: dict[str, bool] = {}
    for well_pos, well_info in wells.items():
        all_cached = True
        # check images
        for img_info in well_info.get("images", []):
            image_id: int = img_info["image_id"]  # type: ignore[assignment]
            image_t: int = img_info["dims"][0]  # type: ignore[assignment]
            for t in range(image_t):
                if not is_cached(get_key(image_id, t)):
                    all_cached = False
                    break
            if not all_cached:
                break
        # check labels
        if all_cached:
            for label_entry in label_map[well_pos]:
                if label_entry is None:
                    # No label for corresponding image
                    continue
                label_id: int = label_entry["label_id"]  # type: ignore[assignment]
                label_t: int = label_entry["dims"][0]  # type: ignore[assignment, index]
                for t in range(label_t):
                    if not is_cached(get_key(label_id, t)):
                        all_cached = False
                        break
                if not all_cached:
                    break

        status[well_pos] = all_cached

    return status


def is_plate_cached(plate_id: int) -> bool:
    """Check if plate metadata and well data exist in cache."""
    return (
        get_cached_plate_metadata(plate_id) is not None
        and get_cached_well_data(plate_id) is not None
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
    meta = get_cached_plate_metadata(plate_id)
    if meta is None or get_cached_well_data(plate_id) is None:
        return False
    # Require flat-field mask
    ff_mask_id = meta.get("ff_mask_id", 0)
    if not is_cached(get_key(ff_mask_id, 0)):
        return False
    status = get_well_cache_status(plate_id)
    print(status)
    return bool(status) and all(status.values())


def get_cached_plate_metadata(plate_id: int) -> dict[str, Any] | None:
    """Return cached plate metadata or None."""
    return _cache.get(_get_meta_key(plate_id))  # type: ignore[no-any-return]


def get_cached_well_data(plate_id: int) -> dict[str, Any] | None:
    """Return cached wells dict or None."""
    return _cache.get(_get_well_key(plate_id))  # type: ignore[no-any-return]


def get_cached_label_map(
    plate_id: int,
) -> dict[str, list[dict[str, int | tuple[int, ...]] | None]] | None:
    """Return cached label map or None.

    Returns entries ``{"label_id": int, "dims": tuple[int, ...]}``. If labels are missing
    for a corresponding well image then the list entry is None.
    """
    return _cache.get(_get_label_key(plate_id))  # type: ignore[no-any-return]


def _get_project_id() -> int:
    """Get OMERO screen project."""
    return int(os.getenv("PROJECT_ID", "0"))


def get_plate_metadata(conn: BlitzGateway, plate_id: int) -> dict[str, Any]:
    """Return plate metadata from the cache, or from OMERO if not cached."""
    v = get_cached_plate_metadata(plate_id)
    if isinstance(v, dict):
        old_version = v.get("cache_version", 0)
        if old_version < _CACHE_VERSION:
            logger.info(
                "Plate %d: cache version %d < %d, deleting stale data",
                plate_id,
                old_version,
                _CACHE_VERSION,
            )
            # Evict stale metadata, keep images
            delete_plate_from_cache(plate_id, remove_images=False)
            v = None
    if v is None:
        logger.info("Caching plate %d: fetching metadata", plate_id)
        v = _fetch_plate_metadata(conn, plate_id, _get_project_id())
        _cache.set(_get_meta_key(plate_id), v, tag=plate_id)
    return v  # type: ignore[no-any-return]


def get_well_data(conn: BlitzGateway, plate_id: int) -> dict[str, Any]:
    """Return wells dict from the cache, or from OMERO if not cached."""
    v = get_cached_well_data(plate_id)
    if v is None:
        logger.info("Caching plate %d: fetching well data", plate_id)
        v = _fetch_well_map(conn, plate_id)
        _cache.set(_get_well_key(plate_id), v, tag=plate_id)
    return v  # type: ignore[no-any-return]


def get_label_map(
    conn: BlitzGateway,
    plate_id: int,
) -> dict[str, list[dict[str, int | tuple[int, ...]] | None]]:
    """Return label map from the cache, or from OMERO if not cached."""
    v = get_cached_label_map(plate_id)
    if v is None:
        logger.info("Caching plate %d: fetching label map", plate_id)
        v = _fetch_label_map(conn, plate_id, _get_project_id())
        _cache.set(_get_label_key(plate_id), v, tag=plate_id)
    return v  # type: ignore[no-any-return]


def delete_plate_from_cache(plate_id: int, remove_images: bool = True) -> int:
    """Delete all cached data for a plate (metadata, images, labels).

    Removes the three metadata keys plus every image and label key
    referenced by the plate's well map and label map.  The plate is
    preserved in the persistent history with status ``"removed"``.

    Args:
        plate_id: OMERO plate ID.
        remove_images: Whether to delete image data as well as metadata.

    Returns:
        Number of keys deleted.
    """
    # Read plate name before deleting metadata
    meta = get_cached_plate_metadata(plate_id)
    plate_name = (
        meta.get("plate_name", str(plate_id))
        if isinstance(meta, dict)
        else str(plate_id)
    )

    # All cache entries should be tagged with the plate ID
    deleted: int = _cache.evict(plate_id)
    if remove_images:
        deleted += evict_images(plate_id)

    # Preserve the plate in history as "removed"
    _update_plate_history(plate_id, plate_name, "removed")

    logger.info("Deleted %d keys for plate %d", deleted, plate_id)
    return deleted


def _estimate_plate_bytes(
    wells: dict[str, dict[str, Any]],
    label_map: dict[str, list[dict[str, int | tuple[int, ...]] | None]]
    | None = None,
) -> int:
    """Estimate total bytes needed to cache a plate's images and labels.

    Uses actual image dimensions (sizeZ, sizeY, sizeX, sizeC) from the
    well map and label map when available; falls back to a configurable
    per-image estimate otherwise.

    Args:
        wells: Well map dict from ``_fetch_well_map()``.
        label_map: Label map dict from ``_fetch_label_map()``.

    Returns:
        Estimated bytes.
    """
    fallback_per_image = getenv_as_int(
        # 4 channel 1080*1080 uint16 image: 9,331,200
        "OMERO_SCREEN_IMAGE_SIZE_ESTIMATE",
        4 * 1080**2 * 2,
    )
    total_bytes = 0

    # Assume all images and labels are the same
    for well_info in wells.values():
        image_entries = well_info.get("images", [])
        for img_info in image_entries:
            estimate = _estimate_entry_bytes(img_info, fallback_per_image)
            total_bytes += estimate * len(wells) * len(image_entries)
            # flat-field mask is 4 byte float64 so double uint16 estimate
            total_bytes += estimate * 2
            break
        break

    if label_map:
        for label_entries in label_map.values():
            for label_entry in label_entries:
                total_bytes += (
                    _estimate_entry_bytes(label_entry, fallback_per_image)
                    * len(label_map)
                    * len(label_entries)
                )
                break
            break

    return total_bytes


def _estimate_entry_bytes(entry: dict[str, Any] | None, fallback: int) -> int:
    """Estimate bytes for a single image/label entry.

    Args:
        entry: Dict with dims
        fallback: Bytes per timepoint when dimensions are missing.

    Returns:
        Estimated bytes.
    """
    if entry is None:
        return fallback
    dims: tuple[int, ...] | None = entry.get("dims")
    if dims is None:
        return fallback
    # Estimate 2 bytes per pixel (uint16 storage)
    return dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * 2


def ensure_cache_space(
    needed_bytes: int, exclude_plate_id: int, size_limit: int
) -> tuple[list[int], int, int]:
    """Evict whole plates (oldest first) until enough space is available.

    The evicted flag contain the number of plate evictions. If no evictions
    were required this is set to -1. A value of zero indicates that no
    evictions were possible.

    Args:
        needed_bytes: Bytes to free up.
        exclude_plate_id: Plate ID to skip (e.g. the plate being cached).
        size_limit: Cache size limit.

    Returns:
        List of plate IDs that were evicted; estimated cache volume; evicted flag
    """
    evicted: list[int] = []

    vol = cache_volume()
    if vol + needed_bytes <= size_limit:
        return [], vol, -1

    # Get plates sorted ascending by plate_id (oldest/smallest first)
    candidates = get_all_cached_plates()
    candidates.reverse()  # was desc, now asc

    for plate_id, _name in candidates:
        if plate_id == exclude_plate_id:
            continue
        delete_plate_from_cache(plate_id)
        evicted.append(plate_id)
        vol = cache_volume()
        if vol + needed_bytes <= size_limit:
            break

    if vol + needed_bytes > size_limit:
        logger.warning(
            "Cache still needs %d bytes after evicting %d plate(s). "
            "volume=%d, limit=%d",
            needed_bytes,
            len(evicted),
            cache_volume(),
            size_limit,
        )

    return evicted, vol, len(evicted)


def _plate_image_completeness(plate_id: int) -> float:
    """Compute fraction of expected images actually present in cache.

    This ignores counting labels.

    Args:
        plate_id: OMERO plate ID.

    Returns:
        Float between 0.0 and 1.0, or 0.0 if wells data is missing.
    """
    wells = get_cached_well_data(plate_id)
    if not isinstance(wells, dict) or not wells:
        return 0.0

    total = 0
    present = 0
    for well_info in wells.values():
        for img_info in well_info.get("images", []):
            image_id: int = img_info["image_id"]  # type: ignore[assignment]
            image_t: int = img_info["dims"][0]  # type: ignore[assignment]
            total += image_t
            for t in range(image_t):
                if is_cached(get_key(image_id, t)):
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
    plate_id: int,
    conn: BlitzGateway,
    stop_flag: threading.Event,
    max_workers: int = 3,
) -> Generator[tuple[int, int], None, None]:
    """Cache entire plate: metadata + images + labels.

    Opens one OMERO connection for metadata, then spawns workers for images.
    Yields (images_done, images_total) for progress reporting.

    If the ``stop_flag`` is set then downloading will stop and the method
    returns.

    Args:
        plate_id: OMERO plate ID.
        conn: Active OMERO connection (caller manages lifecycle).
        stop_flag: Flag to used to stop the download.
        max_workers: Number of concurrent download threads.

    Yields:
        Tuple of (images_done, images_total).
    """
    meta = get_plate_metadata(conn, plate_id)
    wells = get_well_data(conn, plate_id)
    label_map = get_label_map(conn, plate_id)

    # Quick check to determine if the plate will fit in the cache.
    # Done after fetching the label map so the estimate includes labels.
    # Note: Size estimate does not account for the cache compression or pixels type.
    estimated_bytes = _estimate_plate_bytes(wells, label_map)
    size_limit = cache_size_limit()
    if estimated_bytes >= size_limit:
        logger.warning(
            "Plate %d: estimated size %.1f GB exceeds cache size %.1f GB, skipping caching",
            plate_id,
            estimated_bytes / 2**30,
            size_limit / 2**30,
        )
        yield (0, 0)
        return

    logger.info(
        "Plate %d: estimated size %.1f GB (cache volume %.1f / %.1f GB)",
        plate_id,
        estimated_bytes / 2**30,
        cache_volume() / 2**30,
        size_limit / 2**30,
    )

    # Fetch flatfield mask image (image is cached)
    logger.info("Caching plate %d: fetching flatfield mask", plate_id)
    _ = get_image(conn, meta["ff_mask_id"], tag=plate_id)

    # Build downloads grouped by well, sorted by well position.
    # Well-sorted partitioning ensures wells complete sequentially so
    # users can start loading cached wells before the entire plate is done.
    sorted_well_keys = sorted(wells.keys(), key=_well_sort_key)
    keys: list[tuple[int, int]] = []

    # Re-estimate size using missing images. Assume all images are the same.
    n_wells = 0
    image_id: int = 0
    label_id: int = 0
    n_images = 0
    for well_pos in sorted_well_keys:
        last_len = len(keys)
        # Well images (skip if already cached)
        for img_info in wells[well_pos]["images"]:
            image_id: int = img_info["image_id"]  # type: ignore[assignment, no-redef]
            image_t: int = img_info["dims"][0]  # type: ignore[assignment, index]
            for t in range(image_t):
                key = get_key(image_id, t)
                if not is_cached(key):
                    n_images += 1
                    keys.append((image_id, t))
        # Well labels (skip if already cached)
        if well_pos in label_map:
            for label_entry in label_map[well_pos]:
                if label_entry is None:
                    # No label for corresponding image
                    continue
                label_id: int = label_entry["image_id"]  # type: ignore[assignment, no-redef]
                label_t: int = label_entry["dims"][0]  # type: ignore[assignment, index]
                for t in range(label_t):
                    key = get_key(label_id, t)
                    if not is_cached(key):
                        keys.append((label_t, t))
        if last_len < len(keys):
            n_wells += 1

    if n_wells == 0:
        logger.info("Plate %d: all images already cached", plate_id)
        # Record in persistent history
        _update_plate_history(plate_id, meta["plate_name"], "cached")
        yield (0, 0)
        return

    # Accurate download size (assuming all images the same)
    estimated_bytes = 0
    if image_id:
        estimated_bytes += get_bytes_size(conn, image_id) * n_images
    if label_id:
        estimated_bytes += get_bytes_size(conn, label_id) * (
            # number of labels
            len(keys) - n_images
        )

    # Repeat check to determine if the plate will fit in the cache.
    # This may can fail if the pixels type from previous estimate was wrong.
    if estimated_bytes >= size_limit:
        logger.warning(
            "Plate %d: estimated download size %.1f GB exceeds cache size %.1f GB, skipping caching",
            plate_id,
            estimated_bytes / 2**30,
            size_limit / 2**30,
        )
        yield (0, 0)
        return

    logger.info(
        "Plate %d: estimated download size %.1f GB",
        plate_id,
        estimated_bytes / 2**30,
    )

    if stop_flag.is_set():
        logger.warning("Plate %d: download cancelled", plate_id)
        yield (0, 0)
        return

    # Proactively evict old plates to make room for this one.
    # Note: The estimate does not account for image compression
    # so this may evict too many old plates.
    evicted, vol, evicted_flag = ensure_cache_space(
        estimated_bytes, plate_id, size_limit
    )
    if evicted:
        logger.info(
            "Evicted plates %s to make room for plate %d", evicted, plate_id
        )

    # Distribute keys to workers round-robin
    max_workers = max(1, max_workers)
    batches: list[list[tuple[int, int]]] = [[] for _ in range(max_workers)]
    for i, x in enumerate(keys):
        batches[i % max_workers].append(x)
    batches = [b for b in batches if b]

    # Adjust max workers if there are not enough images
    max_workers = len(batches)

    total = len(keys)
    logger.info(
        "Caching plate %d (%d wells): downloading %d items (%d wells) with %d workers",
        plate_id,
        len(sorted_well_keys),
        total,
        n_wells,
        max_workers,
    )

    # Download images + labels with per-image progress.
    # Workers signal each completed image via a shared queue so the
    # generator can yield smooth progress to the napari progress bar.
    progress_q: queue.Queue[int] = queue.Queue()

    # TODO: Test if removing reactive eviction is faster (no constant thread blocking).
    # The size estimate based on non-compressed data + evcition headroom will cause
    # eviction when it may not be necessary.

    # Pause event for reactive eviction: workers block on this before
    # each image.  The main loop clears it to pause workers, evicts old
    # plates, then sets it again to resume.
    pause_event = threading.Event()
    pause_event.set()  # not paused initially
    eviction_headroom = getenv_as_int(
        "OMERO_SCREEN_CACHE_EVICTION_HEADROOM", 500 * 2**20
    )

    done = 0
    try:
        downloaded = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _download_batch,
                    batch,
                    plate_id,
                    stop_flag,
                    progress_q,
                    pause_event,
                )
                for batch in batches
            ]

            # Drain the queue, yielding after every image
            while done < total:
                try:
                    cached = progress_q.get(timeout=5.0)
                    progress_q.task_done()
                    downloaded += cached
                    # Minimise cache volume calls using the downloaded size to estimate space
                    vol += cached
                    done += 1
                    yield (done, total)
                except queue.Empty:
                    # Check if all workers have finished (error or success)
                    if all(f.done() for f in futures):
                        break
                    continue

                # Check stop flag after any potential blocking wait
                if stop_flag.is_set():
                    logger.warning("Plate %d: download cancelled", plate_id)
                    break

                # Reactive eviction: when cache approaches its limit,
                # pause workers → evict old plates → resume workers.
                # Only check if previous eviction was possible.
                if evicted_flag and (vol + eviction_headroom >= size_limit):
                    pause_event.clear()
                    # Brief sleep so in-flight writes complete before we
                    # measure volume for eviction.
                    time.sleep(0.2)

                    needed = estimated_bytes - downloaded
                    evicted, vol, evicted_flag = ensure_cache_space(
                        max(needed, eviction_headroom),
                        plate_id,
                        size_limit,
                    )
                    pause_event.set()
                    if evicted:
                        logger.info(
                            "Reactive eviction for plate %d: "
                            "freed plates %s (volume now %.1f GB)",
                            plate_id,
                            evicted,
                            vol / 2**30,
                        )
                    elif evicted_flag == 0:
                        logger.warning(
                            "Plate %d: cache near limit but no plates "
                            "to evict (volume %.1f / %.1f GB). "
                            "Continuing anyway.",
                            plate_id,
                            vol / 2**30,
                            size_limit / 2**30,
                        )

            # Re-raise the first worker exception, if any
            for f in futures:
                exc = f.exception()
                if exc is not None:
                    logger.exception(
                        "Error in download worker for plate %d", plate_id
                    )
                    raise exc
    finally:
        # In the event of exceptions cancel the download and unblock any waiting workers
        stop_flag.set()
        pause_event.set()

    # Record in persistent history
    _update_plate_history(plate_id, meta["plate_name"], "cached")

    logger.info(
        "Plate %d: caching %s (%d/%d items)",
        plate_id,
        "cancelled" if stop_flag.is_set() else "complete",
        done,
        total,
    )


# --------------- Metadata Fetching ---------------


def _fetch_plate_metadata(
    conn: BlitzGateway, plate_id: int, project_id: int
) -> dict[str, Any]:
    """Fetch channel_data, pixel_size, intensities, plate_name, flat-field
    correction mask ID from OMERO.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID.
        project_id: OMERO project ID containing the screen dataset.

    Returns:
        Dict with keys: channel_data, pixel_size, intensities, plate_name, ff_mask_id.
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

    # Flat-field correction image
    ff_mask_id = _fetch_flatfield_mask_id(conn, plate_id, project_id)

    return {
        "channel_data": channel_data,
        "pixel_size": pixel_size,
        "intensities": intensities,
        "plate_name": plate_name,
        "ff_mask_id": ff_mask_id,
        "cache_version": _CACHE_VERSION,
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


def _fetch_flatfield_mask_id(
    conn: BlitzGateway,
    plate_id: int,
    project_id: int,
) -> int:
    """Find flatfield mask ID from OMERO dataset.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID (used to find dataset and mask name).
        project_id: OMERO project ID containing the screen dataset.

    Returns:
        Flatfield mask image ID.
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
            return int(image.getId())

    raise ValueError(
        f"No flatfield mask found in dataset for plate {plate_id}"
    )


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
        Dict mapping well_pos -> {well_id, metadata, images: [{image_id, dims, pos}]}
    """
    query_service = conn.getQueryService()
    params = omero.sys.ParametersI()
    params.addLong("plate_id", plate_id)
    query = """
        select w.id, w.row, w.column, ws.posX, ws.posY,
               i.id, pi.sizeT, pi.sizeC, pi.sizeZ, pi.sizeY, pi.sizeX
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
    for row in results:
        well_id = unwrap(row[0])
        well_row = unwrap(row[1])
        well_col = unwrap(row[2])
        pos = (int(unwrap(row[3])), int(unwrap(row[4])))
        image_id = unwrap(row[5])
        dims = (
            int(unwrap(row[6])),
            int(unwrap(row[7])),
            int(unwrap(row[8])),
            int(unwrap(row[9])),
            int(unwrap(row[10])),
        )

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
                "dims": dims,
                "pos": pos,
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


def _fetch_label_map(
    conn: BlitzGateway,
    plate_id: int,
    project_id: int,
) -> dict[str, list[dict[str, int | tuple[int, ...]] | None]]:
    """Map well image_ids to their segmentation label image_ids with dimensions.

    Args:
        conn: Active OMERO connection.
        plate_id: OMERO plate ID.
        project_id: OMERO project ID.

    Returns:
        Dict mapping well_pos -> [{"label_id", "dims"}, ...].
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

    if not label_lookup:
        return {}

    # Query dimensions for all label images via HQL
    label_dims: dict[int, tuple[int, ...]] = {}
    label_ids = list(label_lookup.values())
    query_service = conn.getQueryService()
    params = omero.sys.ParametersI()
    params.addIds(label_ids)
    query = (
        "select i.id, pi.sizeT, pi.sizeC, pi.sizeZ, pi.sizeY, pi.sizeX "
        "from Image i join i.pixels pi where i.id in (:ids)"
    )
    for row in query_service.projection(query, params, conn.SERVICE_OPTS):
        img_id = int(unwrap(row[0]))
        label_dims[img_id] = (
            int(unwrap(row[1])),
            int(unwrap(row[2])),
            int(unwrap(row[3])),
            int(unwrap(row[4])),
            int(unwrap(row[5])),
        )

    # Now get well map to associate labels with wells
    wells = get_cached_well_data(plate_id)
    if wells is None:
        # Well data should already be cached by cache_plate()
        return {}

    label_map: dict[str, list[dict[str, int | tuple[int, ...]] | None]] = {}
    for well_pos, well_info in wells.items():
        label_entries: list[dict[str, int | tuple[int, ...]] | None] = []
        for img_info in well_info["images"]:
            image_id = img_info["image_id"]
            label_id = label_lookup.get(image_id) or 0
            dims = label_dims.get(label_id)
            if dims is not None:
                label_entries.append(
                    {
                        "label_id": label_id,
                        "dims": dims,
                    }
                )
            else:
                # Place holder for unlabeled images
                label_entries.append(None)
        label_map[well_pos] = label_entries

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


def _download_batch(
    batch: list[tuple[int, int]],
    plate_id: int,
    stop_flag: threading.Event,
    progress_q: queue.Queue[int] | None = None,
    pause_event: threading.Event | None = None,
    conn: BlitzGateway | None = None,
) -> None:
    """Download a batch of images.

    Uses the provided OMERO connection (pre-created by the caller) or
    falls back to creating one.  The connection is always closed when
    the batch finishes.

    After every image is stored in the cache, a ``1`` is put on
    *progress_q* so the caller can track per-image progress.

    If *pause_event* is provided, workers block on it before each image.
    The caller can clear the event to pause workers while running
    cache eviction, then set it again to resume.

    Args:
        batch: List of cache keys (image_id:t).
        plate_id: Plate ID.
        stop_flag: Flag to used to stop the download.
        progress_q: Optional queue for per-image progress signalling.
        pause_event: Event that workers wait on before each image.
            Workers block when cleared, resume when set.
        conn: Pre-created OMERO connection. If ``None``, a new one is
            created (slower — connection overhead is paid inside the worker).
    """
    if conn is None:
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")

        conn = BlitzGateway(username, password, host=host)
        conn.connect()
        if not conn.isConnected():
            raise RuntimeError(
                f"Download worker failed to connect to OMERO at {host}"
            )

    # The finally block closes the connection and RawPixelsStore
    store = None
    try:
        conn.c.enableKeepAlive(60)

        last_image_id: int | None = None

        # Accumulators for per-phase timing (only when DEBUG logging)
        profiling = logger.isEnabledFor(logging.DEBUG)
        t_setup = 0.0
        t_download = 0.0
        t_cache_write = 0.0

        for image_id, timepoint in batch:
            if pause_event is not None:
                pause_event.wait()
            # Check stop flag after any potential blocking wait
            if stop_flag.is_set():
                break

            # Keep the RawPixelsStore open across timepoints of the
            # same image — setPixelsId is itself an RPC we only pay once.
            if image_id != last_image_id:
                t0 = time.perf_counter() if profiling else 0.0
                if store is not None:
                    with contextlib.suppress(Exception):
                        store.close()
                wrapper = get_omero_image_wrapper(conn, image_id)
                store, shape, dt_be = initialise_download(conn, wrapper)
                last_image_id = image_id
                if profiling:
                    t_setup += time.perf_counter() - t0

            assert store is not None

            t0 = time.perf_counter() if profiling else 0.0
            arr = get_omero_image_timepoint(store, timepoint, shape, dt_be)
            t1 = time.perf_counter() if profiling else 0.0
            t_download += t1 - t0
            add_cached_image(get_key(image_id, timepoint), arr, tag=plate_id)
            t0 = time.perf_counter() if profiling else 0.0
            t_cache_write += t0 - t1

            if progress_q is not None:
                progress_q.put(arr.nbytes)

        if profiling and batch:
            n_items = len(batch)
            logger.debug(
                "Batch timing (%d items): "
                "setup=%.2fs download=%.2fs write=%.2fs "
                "per-image: su=%.0fms dl=%.0fms wr=%.0fms",
                n_items,
                t_setup,
                t_download,
                t_cache_write,
                t_setup / n_items * 1000,
                t_download / n_items * 1000,
                t_cache_write / n_items * 1000,
            )
    finally:
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()
        conn.close(hard=True)


# --------------- Cache-First Loading ---------------


def load_from_cache(
    conn: BlitzGateway,
    omero_data: Any,
    plate_id: int,
    well_pos_input: str,
    image_input: str,
    time: str = "All",
) -> None:
    """Populate OmeroData using the cache, otherwise from OMERO.

    The connection will not be used if the data is fully cached.

    Args:
        conn: OMERO connection.
        omero_data: OmeroData instance to populate.
        plate_id: Plate ID.
        well_pos_input: Comma-separated well positions (e.g. "A1, A2").
        image_input: Image index input (e.g. "All", "0-3", "1, 3").
        time: Time input (e.g. "All", "1-3").
    """
    meta = get_plate_metadata(conn, plate_id)
    wells = get_well_data(conn, plate_id)
    label_map = get_label_map(conn, plate_id)

    # flatfield correction image: ZYXC
    ff_mask_id = meta.get("ff_mask_id") or 0
    flatfield_masks = get_cached_image(get_key(ff_mask_id, 0))
    if flatfield_masks is None:
        flatfield_masks = get_image(conn, ff_mask_id)
        if flatfield_masks is None:
            raise ValueError(f"Plate {plate_id} flat-field mask missing")
    flatfield_masks = flatfield_masks.astype(np.float32)

    # Parse user well selection
    well_pos_list = [w.strip() for w in well_pos_input.split(",")]

    # Parse image index selection
    image_index = _parse_image_index(image_input, wells, well_pos_list)

    # Parse time crop
    tstart, tend = _parse_time_range(time)

    # Reset well/image data
    well_id_list = []
    well_metadata_list = []

    image_arrays: list[npt.NDArray[Any]] = []
    label_arrays: list[npt.NDArray[Any]] = []
    image_ids: list[int] = []
    image_positions: list[tuple[float, float] | None] = []

    for i, well_pos in enumerate(well_pos_list):
        if well_pos not in wells:
            raise ValueError(f"Well {well_pos} not found in plate data")
        logger.info(
            "Loading well images %s (%d/%d)",
            well_pos,
            i + 1,
            len(well_pos_list),
        )

        well_info = wells[well_pos]
        well_id_list.append(well_info["well_id"])
        well_metadata_list.append(well_info["metadata"])

        # Get images for selected indices
        well_images = well_info["images"]
        for idx in image_index:
            if idx >= len(well_images):
                logger.warning(
                    "Image index %d out of range for well %s", idx, well_pos
                )
                continue

            img_info = well_images[idx]
            image_id = img_info["image_id"]  # type: ignore[assignment]
            image_t = img_info["dims"][0]  # type: ignore[assignment]
            image_ids.append(image_id)
            image_positions.append(img_info.get("pos"))

            # Determine timepoint range
            t_start = tstart if tstart is not None else 0
            t_end = tend if tend is not None else image_t

            timepoint_arrays: list[npt.NDArray[Any]] = []
            store = None
            for t in range(t_start, t_end):
                arr = get_cached_image(get_key(image_id, t))
                if arr is None:
                    store, shape, dt_be = initialise_download(
                        conn, get_omero_image_wrapper(conn, image_id)
                    )
                    arr = get_omero_image_timepoint(store, t, shape, dt_be)
                # Flatfield correction
                timepoint_arrays.append(
                    arr.astype(np.float32) / flatfield_masks
                )
            if store is not None:
                store.close()

            if len(timepoint_arrays) == 1:
                image_arrays.append(timepoint_arrays[0])
            else:
                image_arrays.append(np.stack(timepoint_arrays, axis=0))

        # Get labels for this well
        if well_pos in label_map:
            logger.info(
                "Loading well labels %s (%d/%d)",
                well_pos,
                i + 1,
                len(well_pos_list),
            )
            well_label_entries = label_map[well_pos]
            for idx in image_index:
                if idx < len(well_label_entries):
                    label_entry = well_label_entries[idx]
                    if label_entry is None:
                        # No label for corresponding image
                        continue

                    label_id: int = label_entry["label_id"]  # type: ignore[assignment]
                    label_t: int = label_entry["dims"][0]  # type: ignore[assignment, index]

                    # Determine timepoint range
                    t_start = tstart if tstart is not None else 0
                    t_end = tend if tend is not None else label_t

                    timepoint_label_arrays: list[npt.NDArray[Any]] = []
                    store = None
                    for t in range(t_start, t_end):
                        arr = get_cached_image(get_key(label_id, t))
                        if arr is None:
                            store, shape, dt_be = initialise_download(
                                conn, get_omero_image_wrapper(conn, label_id)
                            )
                            arr = get_omero_image_timepoint(
                                store, t, shape, dt_be
                            )
                        timepoint_label_arrays.append(arr)
                    if store is not None:
                        store.close()

                    if len(timepoint_label_arrays) == 1:
                        label_arrays.append(timepoint_label_arrays[0])
                    else:
                        label_arrays.append(
                            np.stack(timepoint_label_arrays, axis=0)
                        )

    # Re-populate OmeroData fields
    omero_data.reset_well_and_image_data()
    omero_data.well_id_list = well_id_list
    omero_data.well_metadata_list = well_metadata_list

    omero_data.plate_id = plate_id
    omero_data.channel_data = meta["channel_data"]
    omero_data.pixel_size = meta["pixel_size"]
    omero_data.intensities = meta["intensities"]
    omero_data.plate_name = meta["plate_name"]

    omero_data.well_pos_list = well_pos_list

    omero_data.image_index = image_index
    omero_data.image_input = image_input

    omero_data.image_ids = image_ids
    omero_data.image_positions = image_positions
    omero_data.images = _squeeze_stack(image_arrays, "images")
    omero_data.labels = _squeeze_stack(label_arrays, "labels")

    logger.info(
        "Loaded plate %d: %d images, %d labels",
        plate_id,
        len(image_arrays),
        len(label_arrays),
    )


def _squeeze_stack(
    arrays: list[npt.NDArray[Any]], name: str
) -> npt.NDArray[Any]:
    """Stack the arrays and remove unused dimensions from the images.

    Returns an empty stack if the list is empty.

    Args:
        arrays: List of images (TZYXC or ZYXC).

    Returns:
        Image stack.
    """
    if len(arrays) == 0:
        return np.empty((0,))

    # Log shapes for debugging mismatches
    shapes = {arr.shape for arr in arrays}
    if len(shapes) > 1:
        logger.warning("%s arrays have inconsistent shapes: %s", name, shapes)
        for i, arr in enumerate(arrays):
            logger.warning("  %s arrays[%d]: shape %s", name, i, arr.shape)
        # Not possible to stack the images
        raise ValueError(f"Cached {name} arrays have inconsistent shapes")

    # Stack images: result is (N,[ T,] Z, Y, X, C)
    stacked = np.stack(arrays, axis=0)

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
    return stacked


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
