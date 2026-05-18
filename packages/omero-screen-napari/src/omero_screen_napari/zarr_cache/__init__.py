"""OME-Zarr cache layer for stitched plates.

This sub-package owns the on-disk OME-NGFF cache that lives alongside the
existing diskcache (``plate_cache.py``). Plates are written here on demand by
the napari Cache button when the plate has a stitched segmentation dataset;
otherwise the existing diskcache path is used.

Design notes:

* The cache is **single-user, single-process**. No cross-process locking.
* It is **bounded and self-managing** via LRU eviction (see ``eviction.py``).
* The link between a measurement row and its pixels is a filesystem
  derivation (``plate_zarr_path(plate_id).exists()``) — there is no
  ``zarr_plates`` table in CellView. Eviction never touches CellView.
"""

from omero_screen_napari.zarr_cache.builder import (
    build_plate_zarr,
    is_stitched_plate,
)
from omero_screen_napari.zarr_cache.crop import (
    DEFAULT_CROP_SIZE,
    ZarrPlateHandle,
    fetch_crop,
    fetch_crop_from_row,
    fetch_label_crop,
    prepare,
    resolve_to_zarr,
)
from omero_screen_napari.zarr_cache.display import load_plate_to_viewer
from omero_screen_napari.zarr_cache.eviction import (
    ZarrCacheTooSmall,
    current_size_bytes,
    enforce_size_cap,
    evict_plate,
    get_cap_bytes,
    pin_plate,
    unpin_plate,
)
from omero_screen_napari.zarr_cache.paths import (
    ZARR_ROOT,
    plate_zarr_path,
    registry_path,
)
from omero_screen_napari.zarr_cache.reader import (
    cached_wells,
    open_plate,
    plate_info,
    read_well,
)
from omero_screen_napari.zarr_cache.registry import (
    ZarrPlateEntry,
    list_plates,
    load_registry,
    remove,
    upsert,
)
from omero_screen_napari.zarr_cache.writer import PlateZarrWriter

__all__ = [
    "DEFAULT_CROP_SIZE",
    "ZARR_ROOT",
    "PlateZarrWriter",
    "ZarrCacheTooSmall",
    "ZarrPlateEntry",
    "ZarrPlateHandle",
    "build_plate_zarr",
    "cached_wells",
    "current_size_bytes",
    "enforce_size_cap",
    "evict_plate",
    "fetch_crop",
    "fetch_crop_from_row",
    "fetch_label_crop",
    "get_cap_bytes",
    "is_stitched_plate",
    "list_plates",
    "load_plate_to_viewer",
    "load_registry",
    "open_plate",
    "pin_plate",
    "plate_info",
    "plate_zarr_path",
    "prepare",
    "read_well",
    "registry_path",
    "remove",
    "resolve_to_zarr",
    "unpin_plate",
    "upsert",
]
