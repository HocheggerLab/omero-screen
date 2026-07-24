"""Tests for ``omero_utils.stitching``."""

from __future__ import annotations

import numpy as np
import pytest

from omero_utils.stitching import positions_to_offsets

@pytest.mark.parametrize("tile_w, tile_h", [
    [100, 100],
    [100, 50],
    [50, 100]
])
def test_positions_to_grid_2x2(tile_w, tile_h):
    # 2x2 grid:
    # 0 1
    # 2 3
    positions = [(300., 300.), (600., 300.), (300., 600.), (600., 600.)]
    offsets = positions_to_offsets(positions, tile_w, tile_h)

    assert offsets.shape == (4, 2)
    assert np.all(offsets.min(axis=0) == 0), "Min (x,y) must be zero"
    assert np.all(offsets == np.array([
       [0, 0],
       [tile_w, 0],
       [0, tile_h],
       [tile_w, tile_h],
    ]))


@pytest.mark.parametrize("overlap_x, overlap_y, translate_x, translate_y", [
    [0, 0, 0, 0],
    [5, 10, 0, 0],
    [0, 0, 2, 3],
    [5, 10, 2, 3],
    [5, 10, -2, -3],
    [-5, -10, -2, -3],
])
def test_positions_to_grid_2x2_with_alignment(overlap_x, overlap_y, translate_x, translate_y):
    # 2x2 grid:
    # 0 2
    # 1 3
    # This is intentionally a different order to the previous 2x2 test
    positions = [(300., 300.), (300., 600.), (600., 300.), (600., 600.)]
    tile_w = 40
    tile_h = 60
    offsets = positions_to_offsets(positions, tile_w, tile_h,
        overlap_x=overlap_x, overlap_y=overlap_y, translate_x=translate_x, translate_y=translate_y)

    assert offsets.shape == (4, 2)
    assert np.all(offsets.min(axis=0) == 0), "Min (x,y) must be zero"

    expected = np.array([
       [0, 0],
       [translate_x, tile_h - overlap_y],
       [tile_w - overlap_x, translate_y],
       [tile_w - overlap_x + translate_x, tile_h - overlap_y + translate_y],
    ])
    min_pos = expected.min(axis=0)
    expected -= min_pos

    assert np.all(offsets == expected)


@pytest.mark.parametrize("overlap_x, overlap_y, translate_x, translate_y", [
    [0, 0, 0, 0],
    [5, 10, 0, 0],
    [0, 0, 2, 3],
    [5, 10, 2, 3],
    [5, 10, -2, -3],
    [-5, -10, -2, -3],
])
def test_positions_to_grid_sparse_3x3_with_alignment(overlap_x, overlap_y, translate_x, translate_y):
    # Sparse 3x3 grid:
    # . 3 .
    # 1 0 2
    # . 4 .
    positions = [(300., 300.), (0., 300.), (600., 300.), (300., 0.), (300., 600.)]
    tile_w = 40
    tile_h = 60
    offsets = positions_to_offsets(positions, tile_w, tile_h,
        overlap_x=overlap_x, overlap_y=overlap_y, translate_x=translate_x, translate_y=translate_y)

    assert offsets.shape == (5, 2)
    assert np.all(offsets.min(axis=0) == 0), "Min (x,y) must be zero"

    # Note: translate_x * y, translate_y * x
    expected = np.array([
       # (1, 1)
       [tile_w - overlap_x + translate_x, tile_h - overlap_y + translate_y],
       # (0, 1)
       [translate_x, tile_h - overlap_y],
       # (2, 1)
       [2 * (tile_w - overlap_x) + translate_x, tile_h - overlap_y + 2 * translate_y],
       # (1, 0)
       [tile_w - overlap_x, translate_y],
       # (1, 2)
       [tile_w - overlap_x + 2 * translate_x, 2 * (tile_h - overlap_y) + translate_y],
    ])
    min_pos = expected.min(axis=0)
    expected -= min_pos

    assert np.all(offsets == expected)
