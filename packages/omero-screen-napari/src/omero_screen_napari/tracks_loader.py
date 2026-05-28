"""Build napari track data from CellView measurements.

Tracking is run upstream by the omero-screen pipeline (``--track``), which
writes a ``track_id`` and ``parent_track_id`` per cell per frame into the
CellView measurements table. No GEFF/zarr is needed to *view* tracks: the full
lineage graph is reconstructable from those two columns plus the nucleus
centroid and timepoint.

This module turns the per-well slice of the already-loaded CellView
``plate_data`` LazyFrame into the ``(N, 4)`` array and parent graph that
napari's built-in ``Tracks`` layer consumes.

Main Functions:
    - load_tracks_for_well: CellView LazyFrame slice -> :class:`TracksData`.
    - has_tracks: cheap check for whether a plate carries track data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

# CellView renames the raw regionprops ``centroid-0``/``centroid-1`` columns to
# these on import (see cellview.utils.state._rename_centroid_cols).
CENTROID_Y_COL = "centroid-0-nuc"
CENTROID_X_COL = "centroid-1-nuc"
TRACK_ID_COL = "track_id"
PARENT_COL = "parent_track_id"
TIME_COL = "timepoint"

# Per-track properties surfaced to the Tracks layer when present, so the user
# can colour tracks by them.
_OPTIONAL_PROPERTY_COLS = ("cell_cycle", "cell_cycle_detailed")


@dataclass
class TracksData:
    """napari-ready track data for one well.

    Attributes:
        data: ``(N, 4)`` array of ``[track_id, t, y, x]`` rows, sorted by track
            then time — the layout ``napari.Viewer.add_tracks`` expects.
        graph: Maps a child ``track_id`` to its parent ``track_id``\\ (s);
            founders are absent. Suitable for the ``graph`` kwarg of
            ``add_tracks``.
        properties: Per-row properties (same length/order as ``data``) for
            ``color_by`` — always includes ``track_id``.
    """

    data: NDArray[np.float64]
    graph: dict[int, list[int]]
    properties: dict[str, NDArray[Any]] = field(default_factory=dict)


def has_tracks(plate_data: pl.LazyFrame) -> bool:
    """Return True if the plate's measurements carry a ``track_id`` column."""
    return TRACK_ID_COL in plate_data.collect_schema().names()


def load_tracks_for_well(
    plate_data: pl.LazyFrame, well: str
) -> TracksData | None:
    """Build napari track data for a single well.

    Args:
        plate_data: The CellView measurements LazyFrame loaded for the plate
            (``OmeroData.plate_data``).
        well: Well position to slice (e.g. ``"C4"``), matched against the
            ``well`` column.

    Returns:
        A :class:`TracksData`, or ``None`` if the plate has no track data.

    Raises:
        KeyError: If track data is present but a required centroid/timepoint
            column is missing.
        ValueError: If the well has no tracked rows.
    """
    if not has_tracks(plate_data):
        return None

    columns = plate_data.collect_schema().names()
    required = [
        TRACK_ID_COL,
        PARENT_COL,
        TIME_COL,
        CENTROID_Y_COL,
        CENTROID_X_COL,
    ]
    missing = [c for c in required if c not in columns]
    if missing:
        raise KeyError(
            f"Track data present but required columns missing: {missing}. "
            f"Available: {columns}"
        )

    optional = [c for c in _OPTIONAL_PROPERTY_COLS if c in columns]
    df = (
        plate_data.filter(pl.col("well") == well)
        .select(required + optional)
        .sort([TRACK_ID_COL, TIME_COL])
        .collect()
    )
    if df.height == 0:
        raise ValueError(f"No tracked rows for well {well!r}.")

    track_ids = df[TRACK_ID_COL].to_numpy()
    # Swap centroid-0 / centroid-1 — try the alternate axis order to confirm
    # whether the alignment seen on plate 4155 was due to a y/x swap rather
    # than the OME-Zarr physical-unit scale on the image layer.
    data = np.column_stack(
        [
            track_ids,
            df[TIME_COL].to_numpy(),
            df[CENTROID_X_COL].to_numpy(),
            df[CENTROID_Y_COL].to_numpy(),
        ]
    ).astype(np.float64)

    graph = _build_graph(df)

    properties: dict[str, NDArray[Any]] = {TRACK_ID_COL: track_ids}
    for col in optional:
        properties[col] = df[col].to_numpy()

    return TracksData(data=data, graph=graph, properties=properties)


def _build_graph(df: pl.DataFrame) -> dict[int, list[int]]:
    """Map each child track id to its parent(s) from the parent column.

    A ``parent_track_id`` of 0 marks a founder (no parent) and is skipped.
    """
    pairs = (
        df.select([TRACK_ID_COL, PARENT_COL])
        .unique()
        .filter(pl.col(PARENT_COL) != 0)
    )
    graph: dict[int, list[int]] = {}
    for child, parent in zip(
        pairs[TRACK_ID_COL].to_list(), pairs[PARENT_COL].to_list(), strict=True
    ):
        graph.setdefault(int(child), []).append(int(parent))
    return graph
