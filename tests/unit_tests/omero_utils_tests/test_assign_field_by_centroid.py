"""Tests for ``omero_utils.stitching.assign_field_by_centroid``."""

from __future__ import annotations

import numpy as np
import pytest
from omero_utils.stitching import assign_field_by_centroid


def _grid_positions(n_cols: int, n_rows: int, step: float = 100.0):
    """Synthetic stage positions for an ``n_cols × n_rows`` grid.

    Returned in row-major order (col varies fastest within each row)
    so each ``(col, row)`` maps to ``positions[row * n_cols + col]``.
    """
    return [
        (col * step, row * step)
        for row in range(n_rows)
        for col in range(n_cols)
    ]


def test_tile_centre_returns_own_index():
    """Centroid at the centre of each tile maps to that tile's index."""
    tile_h = tile_w = 100
    positions = _grid_positions(3, 3)

    # Tile (col, row) has canvas origin (col*tile_w, row*tile_h) when
    # overlap=0; centroid at the centre belongs unambiguously.
    centroids = np.array(
        [
            [row * tile_h + tile_h / 2, col * tile_w + tile_w / 2]
            for row in range(3)
            for col in range(3)
        ]
    )
    out = assign_field_by_centroid(
        centroids, positions, tile_h, tile_w
    )
    assert out.tolist() == list(range(9))


def test_overlap_strip_picks_nearer_centre():
    """Centroid in an overlap strip is assigned to the nearer tile centre."""
    tile_h = tile_w = 100
    overlap = 20
    positions = _grid_positions(2, 1)  # two tiles side by side

    # With overlap_x=20, tile 0 spans x in [0, 100), tile 1 in [80, 180).
    # Overlap strip x ∈ [80, 100). A centroid at x=85 is nearer tile 0
    # (centre x=50) than tile 1 (centre x=130).
    centroid_left = np.array([[50.0, 85.0]])
    out_left = assign_field_by_centroid(
        centroid_left, positions, tile_h, tile_w, overlap_x=overlap
    )
    assert out_left.tolist() == [0]

    # A centroid at x=95 is nearer tile 1 (|95-130|=35) than tile 0
    # (|95-50|=45).
    centroid_right = np.array([[50.0, 95.0]])
    out_right = assign_field_by_centroid(
        centroid_right, positions, tile_h, tile_w, overlap_x=overlap
    )
    assert out_right.tolist() == [1]


def test_four_tile_corner_is_deterministic():
    """Centroid at a 4-tile shared corner returns one tile, deterministically."""
    tile_h = tile_w = 100
    overlap_x = overlap_y = 20
    positions = _grid_positions(2, 2)

    # Exact 4-tile shared corner pixel. Distances to the four tile
    # centres are all equal, so np.argmin returns the smallest index
    # deterministically. We assert determinism + that the choice is
    # one of the four candidates that contain it.
    corner = np.array([[90.0, 90.0]])  # inside all four after overlap
    out1 = assign_field_by_centroid(
        corner,
        positions,
        tile_h,
        tile_w,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
    )
    out2 = assign_field_by_centroid(
        corner,
        positions,
        tile_h,
        tile_w,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
    )
    assert out1.tolist() == out2.tolist()
    assert out1[0] in {0, 1, 2, 3}


def test_centroid_outside_all_rects_falls_back_to_nearest():
    """Centroid outside every tile rect falls back to globally nearest centre."""
    tile_h = tile_w = 100
    positions = _grid_positions(2, 1)

    # x=500 is outside both tiles ([0,100) and [100,200)); nearest
    # centre is tile 1 (centre x=150) vs tile 0 (centre x=50).
    centroid = np.array([[50.0, 500.0]])
    out = assign_field_by_centroid(
        centroid, positions, tile_h, tile_w
    )
    assert out.tolist() == [1]


def test_rejects_wrong_shape():
    """Non-(N, 2) centroid arrays raise ValueError."""
    positions = _grid_positions(2, 1)
    with pytest.raises(ValueError):
        assign_field_by_centroid(
            np.array([1.0, 2.0, 3.0]), positions, 100, 100
        )
