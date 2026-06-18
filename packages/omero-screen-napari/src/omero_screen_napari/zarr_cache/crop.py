"""Single-cell crop API for the zarr plate cache.

The Stage 3 surface that connects measurement rows (CellView) to pixels
(plate.zarr). Three usage modes:

* **Direct** — :func:`fetch_crop` / :func:`fetch_label_crop` take primitive
  coordinates. Callers that already have the centroid (gallery, training,
  inference) use these.
* **Row-based** — :func:`fetch_crop_from_row` extracts ``plate_id`` /
  ``well`` / ``timepoint`` / ``label`` / ``centroid_y`` / ``centroid_x``
  from a measurement row. Friendly for ``df.apply(...)`` and notebooks.
* **Batch pre-warm** — :func:`prepare` builds zarr stores for a set of
  plate IDs ahead of time. Headless consumers (classifier inference,
  plotting scripts) call this once at the top of a run so subsequent
  crop fetches never miss.

Centring policy: the caller supplies the centroid. The primitive does
not query CellView — it does not import ``cellview`` at all. Reason: it
keeps this module a pure pixel-fetch path, independent of the analytic
DB. Callers iterate measurements and already have the centroid columns
to hand.

Edge handling: crops near the canvas boundary are zero-padded so the
return is always exactly ``(size, size)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import numpy as np
import numpy.typing as npt
import zarr
from loguru import logger
from omero.gateway import BlitzGateway

from omero_screen_napari.zarr_cache.builder import build_plate_zarr
from omero_screen_napari.zarr_cache.paths import plate_zarr_path

# Default crop size in pixels (square). Matches the per-field gallery
# default so existing classifier inputs slot in unchanged.
DEFAULT_CROP_SIZE = 128


PrepareStatus = Literal["ready", "built", "failed"]


# ----------------------------------------------------------------------
# Plate resolution
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ZarrPlateHandle:
    """Lightweight pointer to a built zarr store.

    Kept as a frozen dataclass so callers can stash it without worrying
    about opened-file state — the zarr group is re-opened lazily on
    each access. (zarr-2's ``DirectoryStore`` is cheap to re-open.)
    """

    plate_id: int
    path: Path

    def open(self) -> zarr.hierarchy.Group:
        """Open the plate group (read-only). Does NOT refresh ``last_accessed``.

        ``touch()`` is intentionally not called here because crop-fetch
        loops can issue tens of thousands of opens per second; rewriting
        the registry JSON each time turned a 12k-cell gallery into a
        multi-minute hang. The high-level :func:`reader.open_plate`
        entry point (used by the napari viewer load) still touches once
        per plate-open, which is the right granularity for LRU.
        """
        return zarr.open_group(str(self.path), mode="r")


def resolve_to_zarr(plate_id: int) -> ZarrPlateHandle | None:
    """Return a handle if a zarr store exists for ``plate_id``, else None.

    Pure filesystem check — no OMERO, no CellView, no DB. The caller
    decides what to do on a miss (build via :func:`prepare`, fall back
    to the legacy per-field path, or surface an error).
    """
    path = plate_zarr_path(plate_id)
    if not path.exists():
        return None
    return ZarrPlateHandle(plate_id=plate_id, path=path)


# ----------------------------------------------------------------------
# Crop primitives
# ----------------------------------------------------------------------


def _crop_bounds(
    h: int, w: int, centre_y: float, centre_x: float, size: int
) -> tuple[int, int, int, int, int, int, int, int]:
    """Return ``(y0, y1, x0, x1, sy0, sy1, sx0, sx1)`` for a size×size crop.

    ``y0..y1`` / ``x0..x1`` are the world-coordinate bounds (may be
    negative or extend past canvas). ``sy0..sy1`` / ``sx0..sx1`` are
    those bounds clipped to ``[0, h)`` / ``[0, w)`` — the actual
    read region.
    """
    half = size // 2
    cy = int(centre_y)
    cx = int(centre_x)
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    sy0 = max(0, y0)
    sy1 = min(h, y1)
    sx0 = max(0, x0)
    sx1 = min(w, x1)
    return y0, y1, x0, x1, sy0, sy1, sx0, sx1


def _paste_into_padded(
    valid: npt.NDArray[Any],
    leading_shape: tuple[int, ...],
    dtype: np.dtype[Any],
    size: int,
    y0: int,
    sy0: int,
    sy1: int,
    x0: int,
    sx0: int,
    sx1: int,
) -> npt.NDArray[Any]:
    """Place a ``valid`` block (already read from zarr) into a zero-padded
    canvas of shape ``leading_shape + (size, size)``."""
    out = np.zeros(leading_shape + (size, size), dtype=dtype)
    py0 = sy0 - y0
    px0 = sx0 - x0
    out[..., py0 : py0 + (sy1 - sy0), px0 : px0 + (sx1 - sx0)] = valid
    return out


def _well_group_path(well: str) -> str:
    """``"B8"`` → ``"B/8/0"`` (row / col / field-of-view group)."""
    row = well[0]
    col = str(int(well[1:]))
    return f"{row}/{col}/0"


# Chunk-level cache size in bytes. A 3232×3232 uint16 4-channel canvas
# is ~80 MB at level 0; 256 MB lets a couple of wells live in memory at
# once and lets bursty crop-fetch workloads (gallery, training, inference)
# avoid re-decompressing the same chunks across cells.
_CHUNK_CACHE_BYTES = 256 * 2**20


@lru_cache(maxsize=16)
def _open_cached_root(plate_path: str) -> zarr.hierarchy.Group:
    """Open the plate root with an LRU chunk cache layered on top.

    Wrapping the ``DirectoryStore`` in :class:`zarr.LRUStoreCache` keeps
    decompressed chunks resident across :func:`fetch_crop` calls. Without
    this each ``arr[t, :, sy0:sy1, sx0:sx1]`` re-reads + re-decompresses
    every chunk it intersects, even when the previous call's crop
    overlapped the same chunks. Gallery loads on a 12 k-cell well went
    from ~30 s to a fraction of that after enabling the cache.
    """
    store = zarr.DirectoryStore(plate_path)
    cached = zarr.LRUStoreCache(store, max_size=_CHUNK_CACHE_BYTES)
    return zarr.open_group(store=cached, mode="r")


@lru_cache(maxsize=64)
def _open_well_image_array(plate_path: str, well: str) -> zarr.core.Array:
    """Cached open of the level-0 image array for one well.

    Both the array handle and (via the underlying group's
    ``LRUStoreCache``) its decoded chunks are kept resident between
    calls.
    """
    root = _open_cached_root(plate_path)
    return root[_well_group_path(well)]["0"]


@lru_cache(maxsize=64)
def _open_well_label_array(
    plate_path: str, well: str, mask_name: str
) -> zarr.core.Array:
    """Cached open of a level-0 label array for one well."""
    root = _open_cached_root(plate_path)
    img_grp = root[_well_group_path(well)]
    if (
        "labels" not in img_grp.group_keys()
        or mask_name not in img_grp["labels"].group_keys()
    ):
        raise KeyError(mask_name)
    return img_grp["labels"][mask_name]["0"]


# Materialised level-0 slabs per (plate_path, well, t). Galleries,
# training direct-load, and classifier inference all touch most chunks
# of a well across their many crop fetches; reading the slab once into
# RAM costs ~100 MB but cuts per-crop time from ~1 ms (zarr decode) to
# ~0.05 ms (numpy slice). LRU-bounded so multi-well batches don't blow
# memory: at ~80 MB per slab × 4 entries → 320 MB resident worst case.
_MATERIALISED_MAX_ENTRIES = 4
_materialised_slabs: dict[tuple[str, str, int], npt.NDArray[Any]] = {}
_materialised_lru: list[tuple[str, str, int]] = []


def _get_materialised_image_slab(
    plate_path: str, well: str, t: int
) -> npt.NDArray[Any]:
    """Return ``(C, Y, X)`` numpy slab for one well-timepoint.

    Reads the whole canvas into RAM the first time it's requested,
    then re-uses the array. LRU-evicts when the cache is full.
    """
    key = (plate_path, well, t)
    if key in _materialised_slabs:
        # Refresh LRU position.
        _materialised_lru.remove(key)
        _materialised_lru.append(key)
        return _materialised_slabs[key]

    arr = _open_well_image_array(plate_path, well)  # zarr (T, C, Y, X)
    slab = np.asarray(arr[t])  # (C, Y, X)
    _materialised_slabs[key] = slab
    _materialised_lru.append(key)
    while len(_materialised_lru) > _MATERIALISED_MAX_ENTRIES:
        evict_key = _materialised_lru.pop(0)
        _materialised_slabs.pop(evict_key, None)
    return slab


def _get_materialised_label_slab(
    plate_path: str, well: str, mask_name: str, t: int
) -> npt.NDArray[Any]:
    """Return ``(Y, X)`` numpy slab for one well's label group at ``t``."""
    key = (plate_path, f"{well}::{mask_name}", t)
    if key in _materialised_slabs:
        _materialised_lru.remove(key)
        _materialised_lru.append(key)
        return _materialised_slabs[key]

    arr = _open_well_label_array(plate_path, well, mask_name)
    slab = np.asarray(arr[t])
    _materialised_slabs[key] = slab
    _materialised_lru.append(key)
    while len(_materialised_lru) > _MATERIALISED_MAX_ENTRIES:
        evict_key = _materialised_lru.pop(0)
        _materialised_slabs.pop(evict_key, None)
    return slab


def _clear_zarr_cache() -> None:
    """Drop the per-well array + materialised slab caches. Call after
    eviction/rebuild."""
    _open_well_image_array.cache_clear()
    _open_well_label_array.cache_clear()
    _open_cached_root.cache_clear()
    _materialised_slabs.clear()
    _materialised_lru.clear()


def fetch_crop(
    plate_id: int,
    well: str,
    label: int,
    *,
    centroid: tuple[float, float],
    size: int = DEFAULT_CROP_SIZE,
    t: int = 0,
    channels: list[int] | None = None,
) -> npt.NDArray[Any]:
    """Return a ``(C, size, size)`` image crop centred on ``centroid``.

    Args:
        plate_id: OMERO plate ID. Must have a built zarr store; the caller
            verified this via :func:`resolve_to_zarr` or :func:`prepare`.
        well: Well label like ``"B8"``.
        label: The measurement's ``label`` value. Informational only —
            used for log messages and not to derive the centre.
        centroid: ``(y, x)`` pixel coordinates of the crop centre. From
            CellView's ``centroid_y`` / ``centroid_x`` columns.
        size: Square crop edge in pixels. Default 128.
        t: Timepoint index. Default 0 (fixed-cell). Live-cell consumers
            iterate over t per cell.
        channels: Optional list of channel indices to return. ``None``
            (default) returns every channel; the returned array's first
            axis length matches.

    Returns:
        ``(C, size, size)`` numpy array with the source dtype (uint16
        for omero-screen plates). Zero-padded at canvas edges.

    Raises:
        FileNotFoundError: If ``plate_id`` has no zarr cache. Callers
            should call :func:`resolve_to_zarr` first to dispatch.
    """
    path = plate_zarr_path(plate_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No zarr cache for plate {plate_id} at {path}"
        )
    # Materialise the whole (C, Y, X) slab on first access so subsequent
    # crops are pure numpy slices. Matches the diskcache path's
    # per-well-in-RAM model. ~80 MB per slab, LRU-bounded.
    slab = _get_materialised_image_slab(str(path), well, t)  # (C, Y, X)
    cy, cx = centroid
    n_channels, h, w = slab.shape
    y0, y1, x0, x1, sy0, sy1, sx0, sx1 = _crop_bounds(h, w, cy, cx, size)

    if sy1 <= sy0 or sx1 <= sx0:
        out = np.zeros((n_channels, size, size), dtype=slab.dtype)
        return out if channels is None else out[list(channels)]

    valid = slab[:, sy0:sy1, sx0:sx1]

    if (sy1 - sy0, sx1 - sx0) == (size, size):
        crop = np.ascontiguousarray(valid)
    else:
        crop = _paste_into_padded(
            valid, (n_channels,), slab.dtype, size, y0, sy0, sy1, x0, sx0, sx1
        )
    return crop if channels is None else crop[list(channels)]


def fetch_label_crop(
    plate_id: int,
    well: str,
    *,
    centroid: tuple[float, float],
    size: int = DEFAULT_CROP_SIZE,
    t: int = 0,
    mask_name: str = "nuclei",
) -> npt.NDArray[Any]:
    """Return a ``(size, size)`` label crop. Pads with zeros at edges.

    The returned mask is **not** isolated to the target label — consumers
    that want a single-cell mask apply their own isolation step (e.g.
    :func:`omero_screen_napari.gallery_api.erase_masks`). Reason: the
    isolation logic differs between gallery, training, and inference
    paths (multi-nucleate handling, tolerance behaviour) and we don't
    want this primitive making policy decisions.
    """
    path = plate_zarr_path(plate_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No zarr cache for plate {plate_id} at {path}"
        )
    try:
        slab = _get_materialised_label_slab(str(path), well, mask_name, t)
    except KeyError as e:
        raise KeyError(
            f"Plate {plate_id} well {well} has no '{mask_name}' label group"
        ) from e
    cy, cx = centroid
    h, w = slab.shape
    y0, y1, x0, x1, sy0, sy1, sx0, sx1 = _crop_bounds(h, w, cy, cx, size)
    if sy1 <= sy0 or sx1 <= sx0:
        return np.zeros((size, size), dtype=slab.dtype)
    valid = slab[sy0:sy1, sx0:sx1]
    if (sy1 - sy0, sx1 - sx0) == (size, size):
        return np.ascontiguousarray(valid)
    return _paste_into_padded(
        valid, (), slab.dtype, size, y0, sy0, sy1, x0, sx0, sx1
    )


# ----------------------------------------------------------------------
# Row convenience
# ----------------------------------------------------------------------


def _row_get(row: Mapping[str, Any], *keys: str) -> Any:
    """Return the first key that exists on ``row``.

    Accepts pandas Series, dicts, and Polars rows (via .get / item access).
    Raises KeyError if none match.
    """
    for k in keys:
        try:
            val = row[k]
        except (KeyError, IndexError):
            continue
        if val is not None:
            return val
    raise KeyError(f"Row missing all of {keys!r}")


def fetch_crop_from_row(
    row: Mapping[str, Any],
    *,
    size: int = DEFAULT_CROP_SIZE,
    channels: list[int] | None = None,
    mask_name: str | None = None,
) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Convenience wrapper that extracts coordinates from a measurement row.

    Reads ``plate_id``, ``well``, ``timepoint`` (or ``t``; defaults to 0),
    ``label``, ``centroid_y``, ``centroid_x``. If ``mask_name`` is set,
    also returns the matching label crop; otherwise returns just the
    image crop.

    Useful for ``df.apply`` or interactive notebook work. Performance-
    sensitive callers should iterate themselves and call :func:`fetch_crop`
    directly to avoid per-row column lookups.
    """
    plate_id = int(_row_get(row, "plate_id"))
    well = str(_row_get(row, "well"))
    label = int(_row_get(row, "label"))
    try:
        t = int(_row_get(row, "timepoint", "t"))
    except KeyError:
        t = 0
    cy = float(_row_get(row, "centroid_y", "centroid-0", "centroid-0-cell"))
    cx = float(_row_get(row, "centroid_x", "centroid-1", "centroid-1-cell"))

    image_crop = fetch_crop(
        plate_id,
        well,
        label,
        centroid=(cy, cx),
        size=size,
        t=t,
        channels=channels,
    )
    if mask_name is None:
        return image_crop
    label_crop = fetch_label_crop(
        plate_id,
        well,
        centroid=(cy, cx),
        size=size,
        t=t,
        mask_name=mask_name,
    )
    return image_crop, label_crop


# ----------------------------------------------------------------------
# Batch pre-warm
# ----------------------------------------------------------------------


def prepare(
    plate_ids: Iterable[int],
    *,
    conn: BlitzGateway,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict[int, PrepareStatus]:
    """Build zarr stores for plates that don't have them yet.

    Used by headless consumers (classifier inference, plotting scripts)
    that cannot tolerate a mid-run cache miss. Returns a status per
    plate:

    * ``"ready"`` — already built before this call
    * ``"built"`` — built successfully during this call
    * ``"failed"`` — build raised; the exception is logged

    Individual failures do not stop the loop — callers can decide
    whether a partial-success result is usable.

    Args:
        plate_ids: Iterable of OMERO plate IDs to ensure.
        conn: Live OMERO connection. The omero-screen pipeline already
            has one open; pass it through rather than re-authenticating.
        progress_cb: Optional ``(plate_id, message)`` callback. Useful
            for CLI / Qt progress bars.
    """
    results: dict[int, PrepareStatus] = {}
    for plate_id in plate_ids:
        if resolve_to_zarr(plate_id) is not None:
            results[plate_id] = "ready"
            if progress_cb is not None:
                progress_cb(plate_id, "ready")
            continue

        logger.info(f"prepare: building zarr for plate {plate_id}")
        try:
            for well_id in build_plate_zarr(plate_id, conn):
                if progress_cb is not None:
                    progress_cb(plate_id, f"built {well_id}")
            results[plate_id] = "built"
        except Exception as e:  # noqa: BLE001
            logger.error(f"prepare: build failed for plate {plate_id}: {e}")
            results[plate_id] = "failed"
            if progress_cb is not None:
                progress_cb(plate_id, f"failed: {e}")
    return results
