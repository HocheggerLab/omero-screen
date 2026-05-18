"""JSON registry of cached plate.zarr stores.

The registry tracks every plate that has at least one well written to its
zarr store. It is the source of truth for LRU eviction (see
``eviction.py``) — the filesystem itself only tells you what is present,
not when it was last touched or how big a partially-written plate was
expected to be.

Schema (one line per plate)::

    {
      "plates": {
        "1234": {
          "plate_id": 1234,
          "plate_name": "PlateA",
          "size_bytes": 123456789,
          "n_wells_written": 4,
          "created_at": "2026-05-15T09:00:00+00:00",
          "last_accessed": "2026-05-15T11:23:45+00:00"
        },
        ...
      }
    }

Concurrency: single-user, single-process. All writes go through a
temp-file + rename to survive crashes mid-write.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from omero_screen_napari.zarr_cache.paths import registry_path

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ZarrPlateEntry:
    """One row in the registry."""

    plate_id: int
    plate_name: str = ""
    size_bytes: int = 0
    n_wells_written: int = 0
    created_at: str = field(default_factory=_now_iso)
    last_accessed: str = field(default_factory=_now_iso)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZarrPlateEntry:
        return cls(
            plate_id=int(data["plate_id"]),
            plate_name=str(data.get("plate_name", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            n_wells_written=int(data.get("n_wells_written", 0)),
            created_at=str(data.get("created_at", _now_iso())),
            last_accessed=str(data.get("last_accessed", _now_iso())),
        )


def load_registry() -> dict[int, ZarrPlateEntry]:
    """Read the registry from disk. Returns an empty dict if missing or corrupt."""
    path = registry_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Zarr registry %s could not be read; treating as empty", path
        )
        return {}
    plates = raw.get("plates", {})
    return {
        int(plate_id): ZarrPlateEntry.from_dict(entry)
        for plate_id, entry in plates.items()
    }


def _save_registry(entries: dict[int, ZarrPlateEntry]) -> None:
    """Atomic write of the registry via temp file + os.replace."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plates": {
            str(plate_id): asdict(entry) for plate_id, entry in entries.items()
        }
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def upsert(entry: ZarrPlateEntry) -> None:
    """Insert or update a single plate entry."""
    entries = load_registry()
    entries[entry.plate_id] = entry
    _save_registry(entries)


def remove(plate_id: int) -> None:
    """Drop a plate from the registry. No-op if absent."""
    entries = load_registry()
    if plate_id in entries:
        del entries[plate_id]
        _save_registry(entries)


def list_plates() -> list[ZarrPlateEntry]:
    """Return all registry entries, sorted by ``last_accessed`` ascending (LRU first)."""
    entries = load_registry()
    return sorted(entries.values(), key=lambda e: e.last_accessed)


def touch(plate_id: int) -> None:
    """Update ``last_accessed`` for a plate. No-op if absent.

    Called on read (open_plate, read_well) so eviction respects usage.
    """
    entries = load_registry()
    if plate_id not in entries:
        return
    entries[plate_id].last_accessed = _now_iso()
    _save_registry(entries)
