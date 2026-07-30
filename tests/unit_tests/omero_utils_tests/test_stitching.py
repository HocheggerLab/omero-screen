"""Tests for ``omero_utils.stitching``."""

from __future__ import annotations

import numpy as np
import pytest
import random

from omero_utils.stitching import (
    positions_to_offsets,
    get_overlap,
    positions_to_layout,
)

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


@pytest.mark.parametrize("offsets, tile_h, tile_w, overlap", [
    [np.array([[0, 0]]), 10, 10, 0],
    [np.array([
      [0, 0],
      [10, 0],
    ]), 10, 10, 0],
    [np.array([
      [0, 0],
      [9, 0],
    ]), 10, 10, 1],
    [np.array([
      [0, 0],
      [0, 8],
    ]), 10, 10, 2],
    # Largest overlap is r0c0 to r0c1 of 2.
    # This case is changed below with tile sizes.
    [np.array([
      [0, 0],
      [1, 8],
      [9, -2],
    ]), 10, 10, 2],
    # -> Increasing tile_w increases the overlap in x: 12 - 9 = 3
    [np.array([
      [0, 0],
      [1, 8],
      [9, -2],
    ]), 10, 12, 3],
    # -> Increasing tile_h increases the overlap in y: 12 - 8 = 4
    [np.array([
      [0, 0],
      [1, 8],
      [9, -2],
    ]), 12, 10, 4],
    # -> Increasing tile_w increases the overlap in x: 14 - 9 = 5
    [np.array([
      [0, 0],
      [1, 8],
      [9, -2],
    ]), 12, 14, 5],
])
def test_get_overlap(offsets, tile_h, tile_w, overlap):
    actual = get_overlap(offsets, tile_h, tile_w)
    assert actual == overlap


class TestPositionsToLayout:
    def test_not_enough_positions(self):
        positions = [(300.0, 300.0)]
        assert positions_to_layout(positions) is None

    def test_missing_positions(self):
        positions = [(300.0, 300.0), None]
        assert positions_to_layout(positions) is None

    def test_missing_x(self):
        positions = [(300.0, 300.0), (None, 300.0)]
        assert positions_to_layout(positions) is None

    def test_missing_y(self):
        positions = [(300.0, 300.0), (300.0, None)]
        assert positions_to_layout(positions) is None

    def test_duplicate_positions(self):
        positions = [(300.0, 300.0), (300.0, 300.0)]
        assert positions_to_layout(positions) is None

    def test_2x1(self):
        positions = [(300.0, 300.0), (600.0, 300.0)]
        layout = positions_to_layout(positions)
        assert layout == [(0, 0), (1, 0)]

    def test_1x2(self):
        positions = [(300.0, 300.0), (300.0, 600.0)]
        layout = positions_to_layout(positions)
        assert layout == [(0, 0), (0, 1)]

    def test_2x2(self):
        positions = [
            (300.0, 300.0),
            (600.0, 300.0),
            (300.0, 600.0),
            (600.0, 600.0),
        ]
        layout = positions_to_layout(positions)
        assert layout == [(0, 0), (1, 0), (0, 1), (1, 1)]

    def test_3x3_sparse(self):
        positions = [
            (300.0, 300.0),
            (0.0, 300.0),
            (600.0, 300.0),
            (300.0, 0.0),
            (300.0, 600.0),
        ]
        layout = positions_to_layout(positions)
        assert layout == [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]

    def test_5x5(self):
        # Operetta 5x5 position grid:
        positions = [
            (-45.4, 35.2),
            (-45.40127846185709, 35.19740664774529),
            (-45.39998681394756, 35.19741273194196),
            (-45.398695166038024, 35.19741881613863),
            (-45.39741011215991, 35.198718533854915),
            (-45.39870175906167, 35.19871244966299),
            (-45.399993406971205, 35.19870636546632),
            (-45.40128505488074, 35.19870028126965),
            (-45.40257670279027, 35.19869419707298),
            (-45.40258329581906, 35.19998783160666),
            (-45.40129164790953, 35.19999391580333),
            (-45.398708352090466, 35.20000608419667),
            (-45.397416705188704, 35.2000121683886),
            (-45.3974232982175, 35.20130580292228),
            (-45.39871494511926, 35.201299718730354),
            (-45.40000659302879, 35.201293634533684),
            (-45.401298240938324, 35.201287550337014),
            (-45.40258988884785, 35.201281466140344),
            (-45.40130483396711, 35.202581184870695),
            (-45.400013186057585, 35.202587269067365),
            (-45.39872153814805, 35.202593353264035),
        ]
        # Expected grid layout
        grid = [[-1, 1, 2, 3, -1], [8, 7, 6, 5, 4], [9, 10, 0, 11, 12], [17, 16, 15, 14, 13], [-1, 18, 19, 20, -1]]
        grid_map = np.array(grid)
        expected = [(-1, -1)] * sum(grid_map.ravel() >= 0)
        for y in range(5):
            for x in range(5):
                i = grid[y][x]
                if i >= 0:
                    expected[i] = (x, y)

        layout = positions_to_layout(positions)
        assert layout == expected

    def test_5x5_sparse(self):
        # Operetta 5x5 position grid:
        positions = [
            (-45.4, 35.2),
            (-45.40127846185709, 35.19740664774529),
            (-45.39998681394756, 35.19741273194196),
            (-45.398695166038024, 35.19741881613863),
            (-45.39741011215991, 35.198718533854915),
            (-45.39870175906167, 35.19871244966299),
            (-45.399993406971205, 35.19870636546632),
            (-45.40128505488074, 35.19870028126965),
            (-45.40257670279027, 35.19869419707298),
            (-45.40258329581906, 35.19998783160666),
            (-45.40129164790953, 35.19999391580333),
            (-45.398708352090466, 35.20000608419667),
            (-45.397416705188704, 35.2000121683886),
            (-45.3974232982175, 35.20130580292228),
            (-45.39871494511926, 35.201299718730354),
            (-45.40000659302879, 35.201293634533684),
            (-45.401298240938324, 35.201287550337014),
            (-45.40258988884785, 35.201281466140344),
            (-45.40130483396711, 35.202581184870695),
            (-45.400013186057585, 35.202587269067365),
            (-45.39872153814805, 35.202593353264035),
        ]
        # Expected grid layout
        grid = [[-1, 1, 2, 3, -1], [8, 7, 6, 5, 4], [9, 10, 0, 11, 12], [17, 16, 15, 14, 13], [-1, 18, 19, 20, -1]]
        grid_map = np.array(grid)
        expected = [(-1, -1)] * sum(grid_map.ravel() >= 0)
        for y in range(5):
            for x in range(5):
                i = grid[y][x]
                if i >= 0:
                    expected[i] = (x, y)

        # Omit each position in turn
        n = len(positions)
        for i in range(n):
            pos = positions.copy()
            exp = expected.copy()
            del pos[i]
            del exp[i]
            layout = positions_to_layout(pos)
            assert layout == exp

    def test_5x5_sparse_random(self):
        # Operetta 5x5 position grid:
        positions = [
            (-45.4, 35.2),
            (-45.40127846185709, 35.19740664774529),
            (-45.39998681394756, 35.19741273194196),
            (-45.398695166038024, 35.19741881613863),
            (-45.39741011215991, 35.198718533854915),
            (-45.39870175906167, 35.19871244966299),
            (-45.399993406971205, 35.19870636546632),
            (-45.40128505488074, 35.19870028126965),
            (-45.40257670279027, 35.19869419707298),
            (-45.40258329581906, 35.19998783160666),
            (-45.40129164790953, 35.19999391580333),
            (-45.398708352090466, 35.20000608419667),
            (-45.397416705188704, 35.2000121683886),
            (-45.3974232982175, 35.20130580292228),
            (-45.39871494511926, 35.201299718730354),
            (-45.40000659302879, 35.201293634533684),
            (-45.401298240938324, 35.201287550337014),
            (-45.40258988884785, 35.201281466140344),
            (-45.40130483396711, 35.202581184870695),
            (-45.400013186057585, 35.202587269067365),
            (-45.39872153814805, 35.202593353264035),
        ]
        # Expected grid layout
        grid = [[-1, 1, 2, 3, -1], [8, 7, 6, 5, 4], [9, 10, 0, 11, 12], [17, 16, 15, 14, 13], [-1, 18, 19, 20, -1]]
        grid_map = np.array(grid)
        expected = [(-1, -1)] * sum(grid_map.ravel() >= 0)
        for y in range(5):
            for x in range(5):
                i = grid[y][x]
                if i >= 0:
                    expected[i] = (x, y)

        # Omit 2 random positions
        n = len(positions)
        for i in range(10):
            # Use numpy arrays for slicing
            pos = np.array(positions)
            exp = np.array(expected)
            sample = random.sample(range(n), n - 2)
            pos = [tuple(x) for x in pos[sample]]
            exp = [tuple(x) for x in exp[sample]]
            layout = positions_to_layout(pos)
            assert layout == exp
