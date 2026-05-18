"""Registry: load, upsert, remove, list_plates, touch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from omero_screen_napari.zarr_cache import registry as reg_mod
from omero_screen_napari.zarr_cache.registry import (
    ZarrPlateEntry,
    list_plates,
    load_registry,
    remove,
    touch,
    upsert,
)


def test_load_returns_empty_when_file_absent():
    assert load_registry() == {}


def test_upsert_roundtrip():
    upsert(ZarrPlateEntry(plate_id=42, plate_name="A"))
    entries = load_registry()
    assert set(entries) == {42}
    assert entries[42].plate_name == "A"


def test_upsert_overwrites_existing_entry():
    upsert(ZarrPlateEntry(plate_id=42, plate_name="A", size_bytes=100))
    upsert(ZarrPlateEntry(plate_id=42, plate_name="A2", size_bytes=200))
    entries = load_registry()
    assert len(entries) == 1
    assert entries[42].plate_name == "A2"
    assert entries[42].size_bytes == 200


def test_remove_is_idempotent():
    remove(999)  # no-op on empty
    upsert(ZarrPlateEntry(plate_id=999))
    remove(999)
    remove(999)
    assert load_registry() == {}


def test_list_plates_sorted_by_last_accessed_ascending():
    upsert(ZarrPlateEntry(plate_id=1, last_accessed="2026-01-03T00:00:00+00:00"))
    upsert(ZarrPlateEntry(plate_id=2, last_accessed="2026-01-01T00:00:00+00:00"))
    upsert(ZarrPlateEntry(plate_id=3, last_accessed="2026-01-02T00:00:00+00:00"))
    plates = list_plates()
    assert [p.plate_id for p in plates] == [2, 3, 1]


def test_touch_updates_last_accessed():
    upsert(ZarrPlateEntry(plate_id=7, last_accessed="2020-01-01T00:00:00+00:00"))
    touch(7)
    entries = load_registry()
    assert entries[7].last_accessed != "2020-01-01T00:00:00+00:00"


def test_touch_missing_plate_is_noop():
    touch(404)  # must not raise
    assert load_registry() == {}


def test_corrupt_registry_file_is_treated_as_empty(monkeypatch):
    path = reg_mod.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{{")
    assert load_registry() == {}


def test_atomic_write_uses_temp_file(tmp_path):
    upsert(ZarrPlateEntry(plate_id=1))
    upsert(ZarrPlateEntry(plate_id=2))
    # No leftover .tmp file after successful writes.
    leftover = list(reg_mod.registry_path().parent.glob("registry.json.*"))
    assert leftover == []


def test_from_dict_coerces_types():
    entry = ZarrPlateEntry.from_dict(
        {"plate_id": "55", "size_bytes": "100", "n_wells_written": "3"}
    )
    assert entry.plate_id == 55
    assert entry.size_bytes == 100
    assert entry.n_wells_written == 3
