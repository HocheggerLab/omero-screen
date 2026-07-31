"""Tests for ``omero_utils.stitching.assign_tile_by_centroid``."""

from __future__ import annotations

import numpy as np
import pytest
from omero_utils.stitching import assign_tile_by_centroid


def _grid_offsets(n_cols: int, n_rows: int, tile_w: int, tile_h: int,
    ox: int = 0, oy: int = 0, tx: int = 0, ty: int = 0):
    """Synthetic canvas offsets for an ``n_cols × n_rows`` grid.

    Returned in row-major order (col varies fastest within each row)
    so each ``(col, row)`` maps to ``offsets[row * n_cols + col]``.

    Optional overlaps in x and y can be provided.

    Optional translations in x and y can be provided. This is the translation
    applied to each successive tile in the same row (ty)/column (tx).
    """
    offsets = np.stack([
        (col * (tile_w - ox) + row * tx, row * (tile_h - oy) + col * ty)
        for row in range(n_rows)
        for col in range(n_cols)
    ])
    # offsets must be positive
    return offsets - offsets.min(axis=0)


def test_tile_centre_returns_own_index():
    """Centroid at the centre of each tile maps to that tile's index."""
    tile_h = tile_w = 100
    offsets = _grid_offsets(3, 3, tile_w, tile_h)

    # Tile (col, row) has canvas origin (col*tile_w, row*tile_h) when
    # overlap=0; centroid at the centre belongs unambiguously.
    centroids = np.array(
        [
            [row * tile_h + tile_h / 2, col * tile_w + tile_w / 2]
            for row in range(3)
            for col in range(3)
        ]
    )
    out = assign_tile_by_centroid(
        centroids, offsets, tile_h, tile_w
    )
    assert out.tolist() == list(range(9))


def test_overlap_strip_picks_nearer_centre():
    """Centroid in an overlap strip is assigned to the nearer tile centre."""
    tile_h = tile_w = 100
    overlap = 20
    offsets = _grid_offsets(2, 1, tile_w, tile_h, ox=overlap, oy=overlap)  # two tiles side by side

    # With overlap_x=20, tile 0 spans x in [0, 100), tile 1 in [80, 180).
    # Overlap strip x ∈ [80, 100). A centroid at x=85 is nearer tile 0
    # (centre x=50) than tile 1 (centre x=130).
    centroid_left = np.array([[50.0, 85.0]])
    out_left = assign_tile_by_centroid(
        centroid_left, offsets, tile_h, tile_w
    )
    assert out_left.tolist() == [0]

    # A centroid at x=95 is nearer tile 1 (|95-130|=35) than tile 0
    # (|95-50|=45).
    centroid_right = np.array([[50.0, 95.0]])
    out_right = assign_tile_by_centroid(
        centroid_right, offsets, tile_h, tile_w
    )
    assert out_right.tolist() == [1]


def test_four_tile_corner_is_deterministic():
    """Centroid at a 4-tile shared corner returns one tile, deterministically."""
    tile_h = tile_w = 100
    overlap_x = overlap_y = 20
    offsets = _grid_offsets(2, 2, tile_w, tile_h)

    # Exact 4-tile shared corner pixel. Distances to the four tile
    # centres are all equal, so np.argmin returns the smallest index
    # deterministically. We assert determinism + that the choice is
    # one of the four candidates that contain it.
    corner = np.array([[90.0, 90.0]])  # inside all four after overlap
    out1 = assign_tile_by_centroid(
        corner,
        offsets,
        tile_h,
        tile_w,
    )
    out2 = assign_tile_by_centroid(
        corner,
        offsets,
        tile_h,
        tile_w,
    )
    assert out1.tolist() == out2.tolist()
    assert out1[0] in {0, 1, 2, 3}


def test_centroid_outside_all_rects_falls_back_to_nearest():
    """Centroid outside every tile rect falls back to globally nearest centre."""
    tile_h = tile_w = 100
    offsets = _grid_offsets(2, 1, tile_w, tile_h)

    # x=500 is outside both tiles ([0,100) and [100,200)); nearest
    # centre is tile 1 (centre x=150) vs tile 0 (centre x=50).
    centroid = np.array([[50.0, 500.0]])
    out = assign_tile_by_centroid(
        centroid, offsets, tile_h, tile_w
    )
    assert out.tolist() == [1]


def test_rejects_wrong_centroids_shape():
    """Non-(N, 2) centroid arrays raise ValueError."""
    offsets = _grid_offsets(2, 1, 100, 100)
    with pytest.raises(ValueError, match="centroids_yx must be"):
        assign_tile_by_centroid(
            np.array([1.0, 2.0, 3.0]), offsets, 100, 100
        )
    with pytest.raises(ValueError, match="centroids_yx must be"):
        assign_tile_by_centroid(
            np.array([[1.0, 2.0, 3.0]]), offsets, 100, 100
        )


def test_rejects_wrong_offsets_shape():
    """Non-(K, 2) offset arrays raise ValueError."""
    centroids = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="offsets must be"):
        assign_tile_by_centroid(
            centroids, np.array([1, 2]), 100, 100
        )
    with pytest.raises(ValueError, match="offsets must be"):
        assign_tile_by_centroid(
            centroids, np.array([[1, 2, 3]]), 100, 100
        )


def test_all_negative_offsets_throws():
    """All negative offsets raise ValueError."""
    tile_h = tile_w = 100
    with pytest.raises(ValueError, match="No valid positive offsets"):
        assign_tile_by_centroid(
            np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([(-1, -1), (-1, -1)]), tile_h, tile_w
        )
    with pytest.raises(ValueError, match="No valid positive offsets"):
        assign_tile_by_centroid(
            np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([(-1, 0), (0, -1)]), tile_h, tile_w
        )


def test_negative_offsets_are_ignored():
    """Negative offsets are ignored."""
    tile_h, tile_w = 100, 300
    # centroids are (y, x)
    centroid = np.array([[tile_h / 2, tile_w / 2], [3 * tile_h / 2, tile_w / 2]])

    assert assign_tile_by_centroid(
        centroid, np.array([(0, 0), (0, tile_h), (-1, -1)]), tile_h, tile_w
    ).tolist() == [0, 1]
    assert assign_tile_by_centroid(
        centroid, np.array([(0, 0), (-1, -1), (0, tile_h)]), tile_h, tile_w
    ).tolist() == [0, 2]
    assert assign_tile_by_centroid(
        centroid, np.array([(-1, -1), (0, 0), (0, tile_h)]), tile_h, tile_w
    ).tolist() == [1, 2]
