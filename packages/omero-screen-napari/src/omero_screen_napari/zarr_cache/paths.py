"""Disk layout for the zarr cache.

Sibling of the existing diskcache, sharing the ``OMERO_SCREEN_CACHE_PATH``
root. Layout::

    ~/.cache/omero_screen/
    ├── images/                 # existing diskcache (per-field pixels)
    ├── plates/                 # existing diskcache (plate metadata)
    └── zarr/                   # this sub-package
        ├── registry.json
        ├── plate_1234.zarr/
        ├── plate_1235.zarr/
        └── aligned/            # cyclic-IF (4i) aligned artifacts — isolated
            ├── registry.json   #   own registry + LRU budget, keyed by
            └── plate_1234.zarr/ #  the master plate id (see aligned_builder)

The ``root`` parameter on the path helpers selects the namespace: ``None``
(default) is the plain plate cache; :func:`aligned_zarr_root` is the 4i
namespace. This lets the writer/registry/reader be reused for both without
the two colliding — a plain stitched cache of a master plate and its 4i
aligned assembly live in different directories with separate registries.
"""

from __future__ import annotations

from pathlib import Path

from omero_screen_napari.omero_image import get_cache_path


def _zarr_root() -> Path:
    """Filesystem root for all plate.zarr stores."""
    return Path(get_cache_path("zarr"))


def aligned_zarr_root() -> Path:
    """Filesystem root for the isolated cyclic-IF (4i) aligned namespace.

    A sub-directory of the plain zarr root with its own registry so aligned
    assemblies never collide with, or evict, plain plate caches.
    """
    return _zarr_root() / "aligned"


# Module-level handle for convenience. ``get_cache_path`` reads
# ``OMERO_SCREEN_CACHE_PATH`` at call time, so we evaluate lazily where
# possible — but expose this constant for code that just wants the default.
ZARR_ROOT: Path = _zarr_root()


def plate_zarr_path(plate_id: int, *, root: Path | None = None) -> Path:
    """Filesystem path to the ``plate_<id>.zarr`` directory store.

    ``root`` overrides the namespace (default: the plain zarr root); pass
    :func:`aligned_zarr_root` for the 4i aligned namespace.
    """
    return (root or _zarr_root()) / f"plate_{plate_id}.zarr"


def plate_zarr_tmp_path(plate_id: int, *, root: Path | None = None) -> Path:
    """Staging directory used during writes for crash safety."""
    return (root or _zarr_root()) / f"plate_{plate_id}.zarr.tmp"


def registry_path(*, root: Path | None = None) -> Path:
    """Filesystem path to the JSON registry of cached plates."""
    return (root or _zarr_root()) / "registry.json"
