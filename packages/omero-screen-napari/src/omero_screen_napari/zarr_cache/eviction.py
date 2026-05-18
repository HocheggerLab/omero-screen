"""LRU eviction for the zarr plate cache.

Stage 2 ships a simple, bounded cache. The whole point of "cache" is that
it cannot grow without limit. Policies:

* Size cap is set via ``OMERO_SCREEN_ZARR_MAX_GB`` (default 100; floor 10).
* On build, the caller can pre-check + evict to make room (see
  :func:`enforce_size_cap`).
* Eviction order is least-recently-accessed first (registry's
  ``last_accessed`` field). Reads ``touch`` the registry to push usage
  forward.
* Plates can be pinned in-process so a long-running viewer that has the
  store open is not evicted out from under it.
* If a single new plate's estimated size exceeds the cap, the build
  raises :class:`ZarrCacheTooSmall` rather than silently wiping the
  whole cache.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.registry import (
    list_plates,
    remove,
)

logger = logging.getLogger(__name__)


# Cap floor: refuse to operate below this. A 10 GB cap is the smallest
# size that can hold a typical fixed-cell screening plate.
_MIN_CAP_GB = 10
_DEFAULT_CAP_GB = 100


# In-process pin set. Readers register the plates they have open so the
# eviction logic skips them. Pins are cleared via ``unpin_plate`` or
# garbage collection of the reader.
_pinned_plates: set[int] = set()


class ZarrCacheTooSmall(RuntimeError):
    """Raised when a single plate cannot fit within the configured cap.

    The widget catches this and surfaces a dialog asking the user to raise
    ``OMERO_SCREEN_ZARR_MAX_GB``.
    """


def get_cap_bytes() -> int:
    """Read the configured size cap from ``OMERO_SCREEN_ZARR_MAX_GB``.

    Clamped to the ``_MIN_CAP_GB`` floor.
    """
    raw = os.environ.get("OMERO_SCREEN_ZARR_MAX_GB")
    try:
        cap_gb = int(raw) if raw else _DEFAULT_CAP_GB
    except ValueError:
        logger.warning(
            "OMERO_SCREEN_ZARR_MAX_GB=%r is not an integer; using default %d GB",
            raw,
            _DEFAULT_CAP_GB,
        )
        cap_gb = _DEFAULT_CAP_GB
    cap_gb = max(cap_gb, _MIN_CAP_GB)
    return cap_gb * (1024**3)


def pin_plate(plate_id: int) -> None:
    """Mark a plate as in-use; the evictor will skip it."""
    _pinned_plates.add(plate_id)


def unpin_plate(plate_id: int) -> None:
    """Release a pin acquired via :func:`pin_plate`."""
    _pinned_plates.discard(plate_id)


def _dir_size_bytes(path: Path) -> int:
    """Sum the on-disk size of a directory tree, bytes. Zero if missing."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            # File vanished mid-walk (concurrent eviction). Treat as 0.
            continue
    return total


def current_size_bytes() -> int:
    """Total bytes occupied by all registered plate stores."""
    return sum(
        _dir_size_bytes(plate_zarr_path(e.plate_id)) for e in list_plates()
    )


def evict_plate(plate_id: int) -> int:
    """Remove a plate's zarr directory and registry entry.

    Returns the number of bytes reclaimed. Pinned plates are skipped with
    a warning (returns 0).
    """
    if plate_id in _pinned_plates:
        logger.warning("Skipping eviction of pinned plate %d", plate_id)
        return 0
    path = plate_zarr_path(plate_id)
    size = _dir_size_bytes(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    remove(plate_id)
    logger.info("Evicted plate %d (%.1f MB)", plate_id, size / 1024 / 1024)
    return size


def enforce_size_cap(
    extra_bytes: int = 0,
    cap_bytes: int | None = None,
) -> list[int]:
    """Evict LRU plates until ``current_size + extra_bytes <= cap``.

    Args:
        extra_bytes: Bytes the caller is about to add (the new plate's
            estimated build size). Pre-flight check before a build.
        cap_bytes: Override the cap from the environment; mostly useful
            for tests.

    Returns:
        The plate IDs evicted, in eviction order.

    Raises:
        ZarrCacheTooSmall: If a single plate's ``extra_bytes`` alone
            exceeds the cap. Caller should surface this to the user
            rather than silently destroying the cache.
    """
    cap = cap_bytes if cap_bytes is not None else get_cap_bytes()
    if extra_bytes > cap:
        raise ZarrCacheTooSmall(
            f"New plate needs {extra_bytes / 1024**3:.1f} GB but cache "
            f"cap is {cap / 1024**3:.1f} GB. Raise OMERO_SCREEN_ZARR_MAX_GB."
        )

    evicted: list[int] = []
    while current_size_bytes() + extra_bytes > cap:
        # LRU first. list_plates() returns sorted by last_accessed ASC.
        candidates = [
            e for e in list_plates() if e.plate_id not in _pinned_plates
        ]
        if not candidates:
            logger.warning(
                "Cannot enforce cap: every remaining plate is pinned. "
                "current=%.1f GB, extra=%.1f GB, cap=%.1f GB",
                current_size_bytes() / 1024**3,
                extra_bytes / 1024**3,
                cap / 1024**3,
            )
            break
        victim = candidates[0]
        evict_plate(victim.plate_id)
        evicted.append(victim.plate_id)

    return evicted


def estimate_plate_size_bytes(
    n_wells: int,
    n_timepoints: int,
    n_channels: int,
    stitched_h: int,
    stitched_w: int,
    *,
    bytes_per_pixel: int = 2,
    label_overhead: float = 0.1,
    compression_ratio: float = 0.5,
) -> int:
    """Estimate the on-disk footprint of a plate build.

    Used to pre-flight :func:`enforce_size_cap`. Defaults assume uint16
    pixels, ~10 % label overhead (uint32 labels but heavily compressed by
    Blosc/zstd since values are mostly zero), and ~2× compression on
    image data.
    """
    raw_image_bytes = (
        n_wells
        * n_timepoints
        * n_channels
        * stitched_h
        * stitched_w
        * bytes_per_pixel
    )
    raw_label_bytes = int(
        n_wells * n_timepoints * stitched_h * stitched_w * 4 * label_overhead
    )
    return int((raw_image_bytes + raw_label_bytes) * compression_ratio)
