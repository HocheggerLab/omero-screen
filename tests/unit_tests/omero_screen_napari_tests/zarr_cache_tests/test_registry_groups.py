"""Tests for 4i group awareness in the registry and evictor.

A cyclic-IF store is keyed by its master plate but holds several plates' pixels.
The dangerous interaction is ``evict_plate``: it removes a directory derived
from a plate ID, so handed a restain member ID it would delete the master's
store -- destroying every round while dropping one registry row. These tests pin
that down, along with size accounting and pin expansion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omero_screen_napari.zarr_cache import registry
from omero_screen_napari.zarr_cache.eviction import (
    _pinned_plates,
    current_size_bytes,
    evict_plate,
    is_pinned,
    pin_plate,
    pinned_plate_ids,
    transient_pin,
    unpin_plate,
)
from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.registry import (
    ZarrPlateEntry,
    find_group,
    upsert,
)

MASTER = 4127
ROUNDS = [4130, 4131]


@pytest.fixture(autouse=True)
def _clear_transient_pins():
    _pinned_plates.clear()
    yield
    _pinned_plates.clear()


def _make_store(plate_id: int, n_bytes: int = 1024) -> Path:
    path = plate_zarr_path(plate_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "data.bin").write_bytes(b"\0" * n_bytes)
    return path


def _register_group() -> None:
    _make_store(MASTER)
    upsert(
        ZarrPlateEntry(
            plate_id=MASTER,
            plate_name="4i master",
            member_plate_ids=list(ROUNDS),
        )
    )


class TestEntryRoundTrip:
    def test_members_survive_a_save_load_cycle(self) -> None:
        upsert(ZarrPlateEntry(plate_id=MASTER, member_plate_ids=[4130]))
        assert registry.load_registry()[MASTER].member_plate_ids == [4130]

    def test_legacy_registry_loads_without_members(self, tmp_path) -> None:
        """A registry written before 4i support must still load."""
        path = registry.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "plates": {
                        "1234": {
                            "plate_id": 1234,
                            "plate_name": "legacy",
                            "size_bytes": 10,
                            "n_wells_written": 2,
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "last_accessed": "2026-01-01T00:00:00+00:00",
                            "pinned": False,
                        }
                    }
                }
            )
        )
        entry = registry.load_registry()[1234]
        assert entry.member_plate_ids == []
        assert not entry.is_group

    def test_covered_plate_ids(self) -> None:
        entry = ZarrPlateEntry(plate_id=MASTER, member_plate_ids=list(ROUNDS))
        assert entry.covered_plate_ids == {MASTER, 4130, 4131}
        assert entry.is_group


class TestFindGroup:
    def test_finds_by_master(self) -> None:
        _register_group()
        assert find_group(MASTER).plate_id == MASTER

    def test_finds_by_member(self) -> None:
        _register_group()
        assert find_group(4130).plate_id == MASTER

    def test_unknown_plate_is_none(self) -> None:
        _register_group()
        assert find_group(9999) is None

    def test_own_store_wins_over_group_membership(self) -> None:
        """A restain plate may also have its own standalone store."""
        _register_group()
        _make_store(4130)
        upsert(ZarrPlateEntry(plate_id=4130, plate_name="standalone"))
        assert find_group(4130).plate_id == 4130


class TestSizeAccounting:
    def test_group_counted_once(self) -> None:
        _register_group()
        assert current_size_bytes() == 1024

    def test_standalone_member_store_counted_separately(self) -> None:
        """Two real directories, so two contributions -- not double counting."""
        _register_group()
        _make_store(4130, n_bytes=512)
        upsert(ZarrPlateEntry(plate_id=4130))
        assert current_size_bytes() == 1024 + 512


class TestEvictionSafety:
    def test_evicting_a_member_is_refused(self) -> None:
        _register_group()
        assert evict_plate(4130) == 0
        assert plate_zarr_path(MASTER).exists(), (
            "evicting a restain member must not delete the master's store"
        )
        assert find_group(MASTER) is not None

    def test_evicting_the_master_removes_the_store(self) -> None:
        _register_group()
        assert evict_plate(MASTER) == 1024
        assert not plate_zarr_path(MASTER).exists()
        assert find_group(MASTER) is None

    def test_member_with_own_store_evicts_only_its_own(self) -> None:
        _register_group()
        _make_store(4130, n_bytes=512)
        upsert(ZarrPlateEntry(plate_id=4130))
        assert evict_plate(4130) == 512
        assert not plate_zarr_path(4130).exists()
        assert plate_zarr_path(MASTER).exists()

    def test_ordinary_plate_still_evicts(self) -> None:
        _make_store(1234, n_bytes=256)
        upsert(ZarrPlateEntry(plate_id=1234))
        assert evict_plate(1234) == 256
        assert not plate_zarr_path(1234).exists()


class TestPinExpansion:
    def test_pinning_a_member_protects_the_store(self) -> None:
        _register_group()
        pin_plate(4130, persist=False)
        assert is_pinned(MASTER)
        assert evict_plate(MASTER) == 0
        assert plate_zarr_path(MASTER).exists()

    def test_pinning_the_master_protects_members(self) -> None:
        _register_group()
        pin_plate(MASTER, persist=False)
        assert is_pinned(4130)

    def test_pinned_ids_expand_across_the_group(self) -> None:
        _register_group()
        pin_plate(MASTER, persist=False)
        assert pinned_plate_ids() >= {MASTER, 4130, 4131}

    def test_durable_pin_on_a_member_protects_the_store(self) -> None:
        _register_group()
        upsert(ZarrPlateEntry(plate_id=4130, member_plate_ids=[]))
        registry.set_pinned(4130, True)
        assert is_pinned(4130)


class TestTransientPin:
    def test_pins_for_the_block_then_releases(self) -> None:
        _register_group()
        with transient_pin(MASTER):
            assert is_pinned(MASTER)
        assert not is_pinned(MASTER)

    def test_releases_on_exception(self) -> None:
        with pytest.raises(RuntimeError), transient_pin(MASTER):
            raise RuntimeError("build failed")
        assert not is_pinned(MASTER)

    def test_does_not_clear_a_pre_existing_pin(self) -> None:
        """A durable Mastodon pin must survive a build that pins the same plate."""
        _register_group()
        pin_plate(MASTER, persist=True)
        with transient_pin(MASTER):
            pass
        assert is_pinned(MASTER), (
            "the build's transient pin must not clear a user's durable pin"
        )
        unpin_plate(MASTER)
