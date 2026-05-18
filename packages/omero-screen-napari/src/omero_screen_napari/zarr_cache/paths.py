"""Disk layout for the zarr cache.

Sibling of the existing diskcache, sharing the ``OMERO_SCREEN_CACHE_PATH``
root. Layout::

    ~/.cache/omero_screen/
    ├── images/                 # existing diskcache (per-field pixels)
    ├── plates/                 # existing diskcache (plate metadata)
    └── zarr/                   # this sub-package
        ├── registry.json
        ├── plate_1234.zarr/
        └── plate_1235.zarr/
"""

from __future__ import annotations

from pathlib import Path

from omero_screen_napari.omero_image import get_cache_path


def _zarr_root() -> Path:
    """Filesystem root for all plate.zarr stores."""
    return Path(get_cache_path("zarr"))


# Module-level handle for convenience. ``get_cache_path`` reads
# ``OMERO_SCREEN_CACHE_PATH`` at call time, so we evaluate lazily where
# possible — but expose this constant for code that just wants the default.
ZARR_ROOT: Path = _zarr_root()


def plate_zarr_path(plate_id: int) -> Path:
    """Filesystem path to the ``plate_<id>.zarr`` directory store."""
    return _zarr_root() / f"plate_{plate_id}.zarr"


def plate_zarr_tmp_path(plate_id: int) -> Path:
    """Staging directory used during writes for crash safety."""
    return _zarr_root() / f"plate_{plate_id}.zarr.tmp"


def registry_path() -> Path:
    """Filesystem path to the JSON registry of cached plates."""
    return _zarr_root() / "registry.json"
