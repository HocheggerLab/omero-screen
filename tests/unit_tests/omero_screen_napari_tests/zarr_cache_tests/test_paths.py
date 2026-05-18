"""Path helpers — env-controlled cache root, plate path conventions."""

from __future__ import annotations

import importlib

import pytest


def _fresh_paths_module():
    """Reload the paths module so ``ZARR_ROOT`` re-reads the env var."""
    import omero_screen_napari.zarr_cache.paths as paths_module

    importlib.reload(paths_module)
    return paths_module


def test_zarr_root_respects_cache_env_var(isolated_cache):
    paths = _fresh_paths_module()
    assert paths.ZARR_ROOT == isolated_cache / "zarr"


def test_plate_zarr_path_uses_id(isolated_cache):
    paths = _fresh_paths_module()
    assert paths.plate_zarr_path(1234) == isolated_cache / "zarr" / "plate_1234.zarr"


def test_plate_zarr_tmp_path_is_sibling_of_final(isolated_cache):
    paths = _fresh_paths_module()
    final = paths.plate_zarr_path(99)
    tmp = paths.plate_zarr_tmp_path(99)
    assert tmp.parent == final.parent
    assert tmp.name == final.name + ".tmp"


def test_registry_path_under_zarr_root(isolated_cache):
    paths = _fresh_paths_module()
    assert paths.registry_path() == isolated_cache / "zarr" / "registry.json"


@pytest.mark.parametrize("plate_id", [1, 42, 99999, 2**30])
def test_plate_paths_are_unique_per_id(plate_id, isolated_cache):
    paths = _fresh_paths_module()
    assert str(plate_id) in paths.plate_zarr_path(plate_id).name
