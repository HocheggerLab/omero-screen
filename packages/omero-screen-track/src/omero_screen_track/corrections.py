"""Read access to a plate's stored track correction.

``omero-screen-images correct`` stores a correction beside the plate's zarr
cache (``<cache>/zarr/plate_<id>.corrections/``; see
:mod:`omero_screen_napari.zarr_cache.correction`). After the corrected nuclei
are imported into CellView, ``track_id_raw`` *is* the final track id of that
correction, so the review needs nothing from the correction except its
repair events (for the ``fragment`` / ``refractory`` flags) and a hash that
identifies it: an edit log replayed on another correction names nuclei that
no longer exist.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def cache_root() -> Path:
    """The omero-screen cache root (``OMERO_SCREEN_CACHE_PATH`` or ``~/omero-cache``)."""
    return Path(
        os.environ.get("OMERO_SCREEN_CACHE_PATH")
        or Path.home() / "omero-cache"
    ).expanduser()


def correction_dir(plate_id: int) -> Path:
    """Where the plate's correction is stored, whether or not it exists."""
    return cache_root() / "zarr" / f"plate_{plate_id}.corrections"


@dataclass(frozen=True)
class StoredCorrection:
    """The parts of a stored correction the review uses."""

    well: str
    parents: dict[int, int]
    events: pd.DataFrame
    hash: str


def _files(plate_id: int, well: str) -> list[Path]:
    d = correction_dir(plate_id)
    return [d / f"{well}.table.parquet", d / f"{well}.lineage.parquet"]


def correction_hash(plate_id: int, wells: list[str]) -> str | None:
    """SHA-256 over the stored tables of ``wells``, or ``None`` if any is missing."""
    h = hashlib.sha256()
    for well in sorted(wells):
        for f in _files(plate_id, well):
            if not f.exists():
                return None
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def load_stored(plate_id: int, well: str) -> StoredCorrection | None:
    """The stored correction of one well, or ``None`` if there is none."""
    d = correction_dir(plate_id)
    table, lineage = _files(plate_id, well)
    if not (table.exists() and lineage.exists()):
        return None
    lin = pd.read_parquet(lineage)
    events_path = d / f"{well}.events.parquet"
    digest = correction_hash(plate_id, [well])
    assert digest is not None
    return StoredCorrection(
        well=well,
        parents=dict(
            zip(
                lin["track_id"].astype(int),
                lin["parent_track_id"].astype(int),
                strict=True,
            )
        ),
        events=pd.read_parquet(events_path)
        if events_path.exists()
        else pd.DataFrame(),
        hash=digest,
    )
