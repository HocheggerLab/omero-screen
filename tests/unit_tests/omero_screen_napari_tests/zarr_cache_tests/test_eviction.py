"""Eviction: cap parsing, LRU order, pin protection, refusal."""

from __future__ import annotations

import shutil

import pytest

from omero_screen_napari.zarr_cache import (
    ZarrCacheTooSmall,
    ZarrPlateEntry,
    current_size_bytes,
    enforce_size_cap,
    evict_plate,
    get_cap_bytes,
    pin_plate,
    plate_zarr_path,
    unpin_plate,
    upsert,
)
from omero_screen_napari.zarr_cache.eviction import (
    _DEFAULT_CAP_GB,
    _MIN_CAP_GB,
    estimate_plate_size_bytes,
)


# ---------------------------------------------------------------------- #
# Cap config                                                             #
# ---------------------------------------------------------------------- #


def test_cap_default():
    assert get_cap_bytes() == _DEFAULT_CAP_GB * (1024**3)


def test_cap_custom(monkeypatch):
    monkeypatch.setenv("OMERO_SCREEN_ZARR_MAX_GB", "50")
    assert get_cap_bytes() == 50 * (1024**3)


def test_cap_floored(monkeypatch):
    monkeypatch.setenv("OMERO_SCREEN_ZARR_MAX_GB", "1")  # below floor
    assert get_cap_bytes() == _MIN_CAP_GB * (1024**3)


def test_cap_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OMERO_SCREEN_ZARR_MAX_GB", "not-a-number")
    assert get_cap_bytes() == _DEFAULT_CAP_GB * (1024**3)


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _fake_plate(plate_id: int, size_bytes: int, last_accessed: str) -> None:
    """Create a synthetic plate directory + registry entry of a given size."""
    path = plate_zarr_path(plate_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "blob.bin").write_bytes(b"x" * size_bytes)
    upsert(
        ZarrPlateEntry(
            plate_id=plate_id,
            size_bytes=size_bytes,
            last_accessed=last_accessed,
        )
    )


# ---------------------------------------------------------------------- #
# Eviction                                                               #
# ---------------------------------------------------------------------- #


def test_current_size_bytes_sums_disk():
    _fake_plate(1, 1000, "2026-01-01T00:00:00+00:00")
    _fake_plate(2, 2000, "2026-01-02T00:00:00+00:00")
    assert current_size_bytes() == 3000


def test_evict_plate_removes_dir_and_entry():
    _fake_plate(1, 1000, "2026-01-01T00:00:00+00:00")
    reclaimed = evict_plate(1)
    assert reclaimed == 1000
    assert not plate_zarr_path(1).exists()
    assert current_size_bytes() == 0


def test_evict_pinned_plate_is_skipped():
    _fake_plate(1, 1000, "2026-01-01T00:00:00+00:00")
    pin_plate(1)
    try:
        assert evict_plate(1) == 0
        assert plate_zarr_path(1).exists()
    finally:
        unpin_plate(1)


def test_persistent_pin_survives_inprocess_reset():
    """A pin must protect a plate across a napari restart.

    Mastodon curation pins a plate, then napari closes (clearing the
    in-process set) while the user curates for days. The persistent registry
    flag must still make the evictor skip the plate.
    """
    from omero_screen_napari.zarr_cache import eviction

    _fake_plate(1, 2000, "2026-01-01T00:00:00+00:00")  # pinned, oldest (LRU)
    _fake_plate(2, 2000, "2026-01-02T00:00:00+00:00")
    pin_plate(1)  # persists to the registry
    eviction._pinned_plates.clear()  # simulate a restart
    try:
        evicted = enforce_size_cap(extra_bytes=0, cap_bytes=1500)
        # Plate 1 is LRU so would normally go first; the persistent pin
        # protects it, so only plate 2 is evicted.
        assert 1 not in evicted
        assert plate_zarr_path(1).exists()
        assert 2 in evicted
    finally:
        unpin_plate(1)


def test_enforce_size_cap_evicts_lru_first():
    # Three plates totalling 6000 B. Cap at 4500 B with 0 extra → must
    # evict the LRU (plate 2, oldest last_accessed) to fit.
    _fake_plate(1, 2000, "2026-01-02T00:00:00+00:00")
    _fake_plate(2, 2000, "2026-01-01T00:00:00+00:00")  # oldest
    _fake_plate(3, 2000, "2026-01-03T00:00:00+00:00")
    evicted = enforce_size_cap(extra_bytes=0, cap_bytes=4500)
    assert evicted == [2]
    assert current_size_bytes() == 4000


def test_enforce_size_cap_evicts_until_fits():
    _fake_plate(1, 2000, "2026-01-03T00:00:00+00:00")
    _fake_plate(2, 2000, "2026-01-01T00:00:00+00:00")
    _fake_plate(3, 2000, "2026-01-02T00:00:00+00:00")
    # Cap 1500 with 0 extra → must evict all three.
    evicted = enforce_size_cap(extra_bytes=0, cap_bytes=1500)
    assert sorted(evicted) == [1, 2, 3]
    assert current_size_bytes() == 0


def test_enforce_size_cap_accounts_for_extra_bytes():
    _fake_plate(1, 2000, "2026-01-01T00:00:00+00:00")
    # Current 2000, planned 3000, cap 4000 → must free 1000 → evict #1.
    evicted = enforce_size_cap(extra_bytes=3000, cap_bytes=4000)
    assert evicted == [1]


def test_single_plate_too_big_raises():
    with pytest.raises(ZarrCacheTooSmall):
        enforce_size_cap(extra_bytes=5_000, cap_bytes=4_000)


def test_pinned_plate_protected_during_eviction():
    _fake_plate(1, 2000, "2026-01-01T00:00:00+00:00")  # pinned, LRU
    _fake_plate(2, 2000, "2026-01-02T00:00:00+00:00")
    pin_plate(1)
    try:
        evicted = enforce_size_cap(extra_bytes=0, cap_bytes=1500)
        assert evicted == [2]
        assert plate_zarr_path(1).exists()
    finally:
        unpin_plate(1)


# ---------------------------------------------------------------------- #
# Size estimation                                                        #
# ---------------------------------------------------------------------- #


def test_estimate_scales_with_inputs():
    small = estimate_plate_size_bytes(
        n_wells=1, n_timepoints=1, n_channels=1, stitched_h=100, stitched_w=100
    )
    big = estimate_plate_size_bytes(
        n_wells=10, n_timepoints=1, n_channels=4, stitched_h=1000, stitched_w=1000
    )
    assert big > small * 100  # 10 wells × 4 channels × 100× area → ≫ 100×
