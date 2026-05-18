"""Read access to cached plate.zarr stores.

Thin convenience layer over zarr — no ome-zarr reader machinery, since the
napari side only needs raw multiscale arrays to hand to viewer layers.
"""

from __future__ import annotations

import logging
from typing import Any

import zarr

from omero_screen_napari.zarr_cache.paths import plate_zarr_path
from omero_screen_napari.zarr_cache.registry import touch

logger = logging.getLogger(__name__)


def open_plate(plate_id: int) -> zarr.hierarchy.Group:
    """Open the plate's zarr root group (read-only).

    Updates ``last_accessed`` in the registry as a side effect so eviction
    respects usage.
    """
    path = plate_zarr_path(plate_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No zarr cache for plate {plate_id} at {path}. Run the napari "
            f"Cache button or build_plate_zarr() to create it."
        )
    touch(plate_id)
    return zarr.open_group(str(path), mode="r")


def plate_info(plate_id: int) -> dict[str, Any]:
    """Return plate-level metadata for the load path.

    Returns a dict with::

        {
            "plate_name": str,
            "channel_names": list[str],
            "pixel_size_um": float | None,
            "n_timepoints": int,
            "rows": list[str],              # ["A", "B", ...]
            "columns": list[str],           # ["1", "2", ...]
            "wells": list[dict],            # NGFF well dicts (path/rowIndex/columnIndex)
            "well_metadata": dict[str, dict],  # {"A1": {"cell_line": ..., ...}}
        }
    """
    root = open_plate(plate_id)
    plate_attrs = root.attrs.get("plate", {})
    omero_attrs = root.attrs.get("omero_screen", {})
    return {
        "plate_name": plate_attrs.get("name", ""),
        "channel_names": list(omero_attrs.get("channel_names", [])),
        "pixel_size_um": omero_attrs.get("pixel_size_um"),
        "n_timepoints": int(omero_attrs.get("n_timepoints", 1)),
        "rows": [r["name"] for r in plate_attrs.get("rows", [])],
        "columns": [c["name"] for c in plate_attrs.get("columns", [])],
        "wells": list(plate_attrs.get("wells", [])),
        "well_metadata": dict(omero_attrs.get("well_metadata", {})),
    }


def cached_wells(plate_id: int) -> list[str]:
    """Return the list of wells that have data on disk for this plate.

    Format: OMERO well labels like ``A1``, ``B7`` etc., sorted.
    Returns an empty list if the plate has no zarr cache.
    """
    path = plate_zarr_path(plate_id)
    if not path.exists():
        return []
    root = zarr.open_group(str(path), mode="r")
    wells_meta = root.attrs.get("plate", {}).get("wells", [])
    found: list[str] = []
    for w in wells_meta:
        well_path = w["path"]  # e.g. "A/1"
        row, col = well_path.split("/")
        if (path / row / col).exists():
            found.append(f"{row}{col}")
    return sorted(found)


def read_well(plate_id: int, well: str) -> dict[str, Any]:
    """Return the multiscale arrays for one well.

    Returns a dict with keys:

    * ``image``: list of zarr arrays, one per pyramid level, shape
      ``(T, C, Y, X)``.
    * ``nuclei``: list of zarr arrays per level, shape ``(T, Y, X)``.
    * ``cells``: list of zarr arrays per level, or ``None`` if the well
      has no cell segmentation.
    * ``channel_names``: list of channel name strings from the plate
      ``omero_screen`` attrs.
    * ``pixel_size_um``: float or ``None``.
    """
    root = open_plate(plate_id)
    row = well[0]
    col = str(int(well[1:]))
    field_grp = root[f"{row}/{col}/0"]

    # Pyramid levels are stored as numeric subgroups "0", "1", "2", ...
    img_levels = sorted(
        [k for k in field_grp.array_keys() if k.isdigit()], key=int
    )
    images = [field_grp[k] for k in img_levels]

    labels_grp = field_grp.get("labels", None)
    nuclei = _read_label_pyramid(labels_grp, "nuclei") if labels_grp else []
    cells = _read_label_pyramid(labels_grp, "cells") if labels_grp else None

    omero_meta = root.attrs.get("omero_screen", {})
    return {
        "image": images,
        "nuclei": nuclei,
        "cells": cells,
        "channel_names": list(omero_meta.get("channel_names", [])),
        "pixel_size_um": omero_meta.get("pixel_size_um"),
    }


def _read_label_pyramid(
    labels_grp: zarr.hierarchy.Group, name: str
) -> list[zarr.core.Array] | None:
    """Return list of label arrays (one per pyramid level), or None if absent."""
    if name not in labels_grp:
        return None
    grp = labels_grp[name]
    levels = sorted([k for k in grp.array_keys() if k.isdigit()], key=int)
    return [grp[k] for k in levels]
