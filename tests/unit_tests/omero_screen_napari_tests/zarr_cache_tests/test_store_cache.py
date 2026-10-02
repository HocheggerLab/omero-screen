"""Tests for the zarr 3 LRU value cache that replaced zarr 2's LRUStoreCache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from omero_screen_napari.zarr_cache.store_cache import (
    LRUCacheStore,
    open_cached_group,
)
from zarr.storage import LocalStore


def _write_store(path: Path, n_chunks: int = 4, chunk: int = 64) -> np.ndarray:
    """Write a v2-format group holding one chunked uint16 array named "0".

    Random data so chunks stay near their raw size after compression, which
    keeps the size-bound arithmetic in these tests meaningful.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2**16, (n_chunks * chunk, chunk), dtype=np.uint16)
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", data=data, chunks=(chunk, chunk))
    return data


def _cached_root(
    path: Path, max_size: int
) -> tuple[zarr.Group, LRUCacheStore]:
    store = LRUCacheStore(
        LocalStore(str(path), read_only=True), max_size=max_size
    )
    return zarr.open_group(store=store, mode="r"), store


def test_reads_through_unchanged(tmp_path: Path) -> None:
    data = _write_store(tmp_path / "s.zarr")
    root = open_cached_group(str(tmp_path / "s.zarr"), max_size=2**20)
    np.testing.assert_array_equal(root["0"][:], data)


def test_second_read_is_served_from_cache(tmp_path: Path) -> None:
    data = _write_store(tmp_path / "s.zarr")
    root, store = _cached_root(tmp_path / "s.zarr", max_size=2**20)
    arr = root["0"]
    arr[:]
    misses_after_first = store.misses
    hits_before = store.hits

    np.testing.assert_array_equal(arr[:], data)

    assert store.misses == misses_after_first, "repeat read went to disk"
    assert store.hits > hits_before


def test_cache_respects_size_bound(tmp_path: Path) -> None:
    _write_store(tmp_path / "s.zarr", n_chunks=8)
    # Room for roughly two of the eight chunks' encoded bytes.
    root, store = _cached_root(tmp_path / "s.zarr", max_size=2 * 64 * 64 * 2)
    root["0"][:]
    assert store._current_size <= store.max_size
    chunk_keys = [
        k for k in store._values if k.startswith("0/") and k[2].isdigit()
    ]
    assert 0 < len(chunk_keys) < 8, chunk_keys  # some evicted, some kept


def test_value_larger_than_cache_is_returned_but_not_kept(
    tmp_path: Path,
) -> None:
    data = _write_store(tmp_path / "s.zarr", n_chunks=1, chunk=128)
    root, store = _cached_root(tmp_path / "s.zarr", max_size=64)
    np.testing.assert_array_equal(root["0"][:], data)
    assert store._current_size <= 64


def test_least_recently_used_entry_is_evicted_first(tmp_path: Path) -> None:
    _write_store(tmp_path / "s.zarr", n_chunks=3)
    root, store = _cached_root(tmp_path / "s.zarr", max_size=2**20)
    arr = root["0"]
    arr[0:64]  # chunk 0
    arr[64:128]  # chunk 1
    arr[0:64]  # touch chunk 0 again -> chunk 1 is now least recent
    keys = [k for k in store._values if k.startswith("0/") and k[2].isdigit()]
    assert keys[-1] == "0/0.0", keys

    # Shrink the budget to exactly the most recent entry and insert another.
    store.max_size = store._current_size
    arr[128:192]  # chunk 2 forces evictions from the LRU end
    assert "0/1.0" not in store._values
