"""Tests for position-based stitching from stage coordinates."""

import numpy as np
import pytest
import re
from omero_utils.stitching import (
    _adaptive_tolerance,
    _cluster_values,
    has_valid_positions,
    positions_to_grid,
    recompose_split_labels,
    recompose_tiles,
    split_stitched_from_offsets,
    split_stitched_mask_to_fields,
    stitch_from_offsets,
    stitch_from_positions,
    stitch_labels_from_positions,
)

# --------------- has_valid_positions ---------------


class TestHasValidPositions:
    def test_all_valid_um(self):
        assert has_valid_positions([(0.0, 0.0), (100.0, 0.0)]) is True

    def test_single_position(self):
        assert has_valid_positions([(0.0, 0.0)]) is False

    def test_none_in_list(self):
        assert has_valid_positions([(0.0, 0.0), None]) is False

    def test_empty(self):
        assert has_valid_positions([]) is False

    def test_three_valid(self):
        assert (
            has_valid_positions([(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)])
            is True
        )

    def test_all_identical_positions(self):
        """Positions that all cluster to one point cannot form a grid."""
        assert (
            has_valid_positions([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])
            is False
        )

    def test_more_images_than_grid_cells(self):
        """5 images but only 2x2=4 grid cells → would lose an image."""
        positions = [
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (0.5, 0.5),  # clusters with (0, 0) due to small gap
        ]
        assert has_valid_positions(positions) is False

    def test_reference_frame_positions(self):
        """Real Operetta positions in reference-frame units (not µm)."""
        positions = [
            (-23.602595, 44.10129),
            (-23.602591, 44.09999),
            (-23.602587, 44.09870),
            (-23.601304, 44.10259),
            (-23.601300, 44.10129),
            (-23.601296, 44.09999),
            (-23.601292, 44.09870),
            (-23.600009, 44.10259),
            (-23.600005, 44.10129),
        ]
        assert has_valid_positions(positions) is True


# --------------- _adaptive_tolerance ---------------


class TestAdaptiveTolerance:
    def test_clean_grid(self):
        """Uniform spacing → tolerance < spacing."""
        values = [0.0, 100.0, 200.0, 300.0]
        tol = _adaptive_tolerance(values)
        assert 0 < tol < 100.0

    def test_noisy_operetta(self):
        """Within-column noise ~0.000004, between-column gaps ~0.0013."""
        values = [
            -23.602595, -23.602591, -23.602587,  # column 1
            -23.601304, -23.601300, -23.601296,  # column 2
            -23.600009, -23.600005, -23.600001,  # column 3
        ]
        tol = _adaptive_tolerance(values)
        # Must be > within-column noise but < between-column spacing
        assert tol > 0.000004
        assert tol < 0.001

    def test_single_value(self):
        assert _adaptive_tolerance([42.0]) == 0.0

    def test_two_values(self):
        tol = _adaptive_tolerance([0.0, 100.0])
        assert tol > 0

    def test_duplicates_ignored(self):
        """Duplicate values (zero gaps) shouldn't affect tolerance."""
        tol = _adaptive_tolerance([0.0, 0.0, 100.0, 100.0, 200.0])
        assert tol > 0


# --------------- _cluster_values ---------------


class TestClusterValues:
    def test_regular_spacing(self):
        values = [0.0, 100.0, 200.0]
        clusters = _cluster_values(values, tolerance=25.0)
        assert len(clusters) == 3
        assert clusters == pytest.approx([0.0, 100.0, 200.0])

    def test_tolerance_grouping(self):
        # Values within tolerance should be grouped
        values = [0.0, 0.5, 100.0, 100.3]
        clusters = _cluster_values(values, tolerance=1.0)
        assert len(clusters) == 2
        assert clusters[0] == pytest.approx(0.25)
        assert clusters[1] == pytest.approx(100.15)

    def test_single_value(self):
        clusters = _cluster_values([42.0], tolerance=1.0)
        assert clusters == [42.0]

    def test_empty(self):
        assert _cluster_values([], tolerance=1.0) == []

    def test_unsorted_input(self):
        values = [200.0, 0.0, 100.0]
        clusters = _cluster_values(values, tolerance=25.0)
        assert clusters == pytest.approx([0.0, 100.0, 200.0])


# --------------- positions_to_grid ---------------


class TestPositionsToGrid:
    def test_2x2_grid_um(self):
        # 2x2 grid with 100µm spacing
        positions = [
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
        ]
        grid_map = positions_to_grid(positions)
        assert len(grid_map) == 2  # 2 columns
        assert len(grid_map[0]) == 2  # 2 rows per column
        assert grid_map[0][0] == 0
        assert grid_map[1][0] == 1
        assert grid_map[0][1] == 2
        assert grid_map[1][1] == 3

    def test_3x3_grid_um(self):
        positions = [
            (0.0, 0.0),
            (100.0, 0.0),
            (200.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (200.0, 100.0),
            (0.0, 200.0),
            (100.0, 200.0),
            (200.0, 200.0),
        ]
        grid_map = positions_to_grid(positions)
        assert len(grid_map) == 3  # 3 columns
        for col in grid_map.values():
            assert len(col) == 3

    def test_single_row(self):
        positions = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
        grid_map = positions_to_grid(positions)
        assert len(grid_map) == 3
        for col in grid_map.values():
            assert list(col.keys()) == [0]

    def test_reference_frame_positions(self):
        positions = [
            (-23.602595, 44.10129),
            (-23.602591, 44.09999),
            (-23.601304, 44.10129),
            (-23.601300, 44.09999),
        ]
        grid_map = positions_to_grid(positions)
        assert len(grid_map) == 2  # 2 columns
        for col in grid_map.values():
            assert len(col) == 2  # 2 rows


# --------------- stitch_from_positions ---------------


class TestStitchFromPositions:
    def test_2x2_synthetic(self):
        # 4 tiles of 32x32 with 2 channels, spacing = perfect (no overlap)
        tile = np.ones((32, 32, 2), dtype=np.float32)
        images = np.stack([tile * (i + 1) for i in range(4)])  # (4, 32, 32, 2)
        # 50µm spacing, 1µm/px → spacing_px = 50, tile = 32 → no overlap (0)
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        result = stitch_from_positions(
            images, positions,
        )
        # Output should be (64, 64, 2) since no overlap
        assert result.ndim == 3
        assert result.shape[2] == 2
        assert result.shape[0] == 64
        assert result.shape[1] == 64

    def test_5d_time_series(self):
        # 4 tiles of (3, 16, 16, 1) — 3 timepoints
        tile = np.ones((3, 16, 16, 1), dtype=np.float32)
        images = np.stack([tile for _ in range(4)])  # (4, 3, 16, 16, 1)
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        result = stitch_from_positions(
            images, positions,
        )
        # (T, Y, X, C) = (3, 32, 32, 1)
        assert result.ndim == 4
        assert result.shape[0] == 3
        assert result.shape[3] == 1

    def test_with_um_overlap(self):
        # 4 tiles of 100x100, spacing 80µm, pixel_size 1µm/px → overlap 20
        tile = np.ones((100, 100, 1), dtype=np.float32)
        images = np.stack([tile for _ in range(4)])
        positions = [
            (0.0, 0.0),
            (80.0, 0.0),
            (0.0, 80.0),
            (80.0, 80.0),
        ]
        result = stitch_from_positions(
            images, positions, overlap_x=20, overlap_y=20
        )
        # Output: 2 tiles at 100px with 20px overlap → 180px per axis
        assert result.shape[0] == 180
        assert result.shape[1] == 180

    def test_reference_frame_stitch(self):
        """Stitching with reference-frame positions: layout correct, 0 overlap."""
        tile = np.ones((32, 32, 1), dtype=np.float32)
        images = np.stack([tile * (i + 1) for i in range(4)])
        positions = [
            (-23.6026, 44.1013),
            (-23.6013, 44.1013),
            (-23.6026, 44.1000),
            (-23.6013, 44.1000),
        ]
        result = stitch_from_positions(
            images, positions,
        )
        # 2x2 grid, no overlap → 64x64
        assert result.shape == (64, 64, 1)

    def test_2x2_labels_with_translation(self):
        tile = np.ones((32, 32, 2), dtype=np.float32)
        images = np.stack([tile * (i + 1) for i in range(4)])  # (4, 32, 32, 2)
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        for pos in [(0, 0), (0, 4), (2, 0), (0, -4), (-2, 0), (4, 3), (-2, 5)]:
            tx, ty = pos
            result = stitch_from_positions(
                images, positions,
                translate_x=tx,
                translate_y=ty,
            )
            assert result.ndim == 3
            assert result.shape[2] == 2
            assert result.shape[0] == 64 + abs(ty)
            assert result.shape[1] == 64 + abs(tx)
            # check corner values.
            top_left, top_right, bottom_left, bottom_right = 1.0, 2.0, 3.0, 4.0

            # Approximate centre of (32,32) tiles will not have moved
            assert result[16, 16, 0] == top_left
            assert result[16, -16, 0] == top_right
            assert result[-16, 16, 0] == bottom_left
            assert result[-16, -16, 0] == bottom_right

            # row shift
            if tx > 0:
                top_right, bottom_left = 0, 0
            if tx < 0:
                top_left, bottom_right = 0, 0
            # col shift
            if ty > 0:
                top_right, bottom_left = 0, 0
            if ty < 0:
                top_left, bottom_right = 0, 0

            assert result[0, 0, 0] == top_left
            assert result[0, -1, 0] == top_right
            assert result[-1, 0, 0] == bottom_left
            assert result[-1, -1, 0] == bottom_right


# --------------- stitch_labels_from_positions ---------------


class TestStitchLabelsFromPositions:
    def test_2x2_labels(self):
        label = np.ones((32, 32, 1), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        result = stitch_labels_from_positions(
            labels, positions,
        )
        assert result.ndim == 3
        assert result.shape[2] == 1
        assert result.shape[0] == 64
        assert result.shape[1] == 64

    def test_2x2_labels_with_translation(self):
        label = np.ones((32, 32, 1), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        # Note: translation can create overlap if both are used with same sign
        for pos in [(0, 0), (0, 4), (2, 0), (0, -4), (-2, 0), (-3, 4), (3, -2)]:
            tx, ty = pos
            result = stitch_labels_from_positions(
                labels, positions,
                translate_x=tx,
                translate_y=ty,
            )
            assert result.ndim == 3
            assert result.shape[2] == 1
            assert result.shape[0] == 64 + abs(ty)
            assert result.shape[1] == 64 + abs(tx)
            # check corner values.
            # Note: Stiching proceeds in x, y order.
            # Each additional label is added to the current max.
            # Stich order = 1, 3, 2, 4. Cumulative = 1, 4, 6, 10.
            top_left, top_right, bottom_left, bottom_right = 1, 6, 4, 10

            # Approximate centre of (32,32) tiles will not have moved
            assert result[16, 16, 0] == top_left
            assert result[16, -16, 0] == top_right
            assert result[-16, 16, 0] == bottom_left
            assert result[-16, -16, 0] == bottom_right

            # row shift
            if tx > 0:
                top_right, bottom_left = 0, 0
            if tx < 0:
                top_left, bottom_right = 0, 0
            # col shift
            if ty > 0:
                top_right, bottom_left = 0, 0
            if ty < 0:
                top_left, bottom_right = 0, 0

            assert result[0, 0, 0] == top_left
            assert result[0, -1, 0] == top_right
            assert result[-1, 0, 0] == bottom_left
            assert result[-1, -1, 0] == bottom_right

    def test_2x2_labels_with_translation_positive(self):
        label = np.ones((32, 32, 1), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        tx, ty = 4, 3
        result = stitch_labels_from_positions(
            labels, positions,
            translate_x=tx,
            translate_y=ty,
        )
        assert result.ndim == 3
        assert result.shape[2] == 1
        assert result.shape[0] == 64 + abs(ty)
        assert result.shape[1] == 64 + abs(tx)
        # check corner values.
        # Note: Stiching proceeds in x, y order.
        # Each additional label is added to the current max.
        # Unless it overlaps with a previous label, in which case it matches.
        # Here top-right (2) overlaps bottom-left (3), so bottom-left is preserved.
        # Stich order = 1, 3, 2, 4. Cumulative = 1, 4, 4 (overlap), 8.
        top_left, top_right, bottom_left, bottom_right = 1, 4, 4, 8

        # Approximate centre of (32,32) tiles will not have moved
        assert result[16, 16, 0] == top_left
        assert result[16, -16, 0] == top_right
        assert result[-16, 16, 0] == bottom_left
        assert result[-16, -16, 0] == bottom_right

        # row shift/col shift positive removes two corners
        top_right, bottom_left = 0, 0

        assert result[0, 0, 0] == top_left
        assert result[0, -1, 0] == top_right
        assert result[-1, 0, 0] == bottom_left
        assert result[-1, -1, 0] == bottom_right

    def test_2x2_labels_with_translation_negative(self):
        label = np.ones((32, 32, 1), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        positions = [
            (0.0, 0.0),
            (50.0, 0.0),
            (0.0, 50.0),
            (50.0, 50.0),
        ]
        tx, ty = -4, -3
        result = stitch_labels_from_positions(
            labels, positions,
            translate_x=tx,
            translate_y=ty,
        )
        assert result.ndim == 3
        assert result.shape[2] == 1
        assert result.shape[0] == 64 + abs(ty)
        assert result.shape[1] == 64 + abs(tx)
        # check corner values.
        # Note: Stiching proceeds in x, y order.
        # Each additional label is added to the current max.
        # Unless it overlaps with a previous label, in which case it matches.
        # Here bottom-right (4) overlaps top-left (1), so top-left is preserved.
        # Stich order = 1, 3, 2, 4. Cumulative = 1, 4, 6, 1 (overlap).
        top_left, top_right, bottom_left, bottom_right = 1, 6, 4, 1

        # Approximate centre of (32,32) tiles will not have moved
        assert result[16, 16, 0] == top_left
        assert result[16, -16, 0] == top_right
        assert result[-16, 16, 0] == bottom_left
        assert result[-16, -16, 0] == bottom_right

        # row shift/col shift negative removes two corners
        top_left, bottom_right = 0, 0

        assert result[0, 0, 0] == top_left
        assert result[0, -1, 0] == top_right
        assert result[-1, 0, 0] == bottom_left
        assert result[-1, -1, 0] == bottom_right


# --------------- stitch_from_offsets ---------------


class TestStitchFromOffsets:
    def test_negative_offsets_throws(self):
        # 4 tiles of 32x32 with 2 channels, spacing = perfect (no overlap)
        tile = np.ones((32, 32, 1), dtype=np.float32)
        images = np.stack([tile, tile])  # (2, 32, 32, 1)
        with pytest.raises(ValueError, match="Offsets must be positive"):
            stitch_from_offsets(images, np.array([(0, 0), (32, -1)]))
        with pytest.raises(ValueError, match="Offsets must be positive"):
            stitch_from_offsets(images, np.array([(0, 0), (-1, 32)]))

    def test_2x2_synthetic(self):
        # 4 tiles of 32x32 with 2 channels, spacing = perfect (no overlap)
        tile = np.ones((32, 32, 2), dtype=np.float32)
        images = np.stack([tile * (i + 1) for i in range(4)])  # (4, 32, 32, 2)
        # tile = 32 → no overlap (0)
        offsets = np.array([
            (0, 0),
            (32, 0),
            (0, 32),
            (32, 32),
        ])
        result = stitch_from_offsets(
            images, offsets,
        )
        # Output should be (64, 64, 2) since no overlap
        assert result.ndim == 3
        assert result.shape == (64, 64, 2)

    def test_5d_time_series(self):
        # 4 tiles of (3, 16, 16, 1) — 3 timepoints
        tile = np.ones((3, 16, 16, 1), dtype=np.float32)
        images = np.stack([tile for _ in range(4)])  # (4, 3, 16, 16, 1)
        offsets = np.array([
            (0, 0),
            (16, 0),
            (0, 16),
            (16, 16),
        ])
        result = stitch_from_offsets(
            images, offsets,
        )
        # (T, Y, X, C) = (3, 32, 32, 1)
        assert result.ndim == 4
        assert result.shape == (3, 32, 32, 1)

    def test_2x2_with_overlap(self):
        # 4 tiles of 100x100 → overlap 20
        tile = np.ones((100, 100, 1), dtype=np.float32)
        images = np.stack([tile for _ in range(4)])
        offsets = np.array([
            (0, 0),
            (80, 0),
            (0, 80),
            (80, 80),
        ])
        result = stitch_from_offsets(
            images, offsets,
        )
        # Output: 2 tiles at 100px with 20px overlap → 180px per axis
        assert result.ndim == 3
        assert result.shape == (180, 180, 1)

    def test_2x2_with_translation(self):
        tile = np.ones((32, 32, 2), dtype=np.float32)
        images = np.stack([tile * (i + 1) for i in range(4)])  # (4, 32, 32, 2)
        offsets = np.array([
            (0, 0),
            (32, 0),
            (0, 32),
            (32, 32),
        ])
        for pos in [(0, 0), (0, 4), (2, 0), (0, -4), (-2, 0), (4, 3), (-2, 5)]:
            tx, ty = pos
            off = offsets.copy()
            for a in off:
                if a[0]:
                    if a[1]:
                        a[0] += tx
                    a[1] += ty
                elif a[1]:
                    a[0] += tx
            # offsets must be positive
            off -= off.min(axis=0)
            result = stitch_from_offsets(
                images, off,
            )
            assert result.ndim == 3
            assert result.shape == (64 + abs(ty), 64 + abs(tx), 2)
            # check corner values.
            top_left, top_right, bottom_left, bottom_right = 1.0, 2.0, 3.0, 4.0

            # Approximate centre of (32,32) tiles will not have moved
            assert result[16, 16, 0] == top_left
            assert result[16, -16, 0] == top_right
            assert result[-16, 16, 0] == bottom_left
            assert result[-16, -16, 0] == bottom_right

            # row shift
            if tx > 0:
                top_right, bottom_left = 0, 0
            if tx < 0:
                top_left, bottom_right = 0, 0
            # col shift
            if ty > 0:
                top_right, bottom_left = 0, 0
            if ty < 0:
                top_left, bottom_right = 0, 0

            assert result[0, 0, 0] == top_left
            assert result[0, -1, 0] == top_right
            assert result[-1, 0, 0] == bottom_left
            assert result[-1, -1, 0] == bottom_right

    def test_2x2_with_auto_edge(self):
        # 4 tiles of 100x100 → overlap 20
        tile = np.arange(10000).reshape((100, 100, 1))
        images = np.stack([tile for _ in range(4)])
        offsets = np.array([
            (0, 0),
            (80, 0),
            (0, 80),
            (80, 80),
        ])
        expected = stitch_from_offsets(
            images, offsets, edge=20
        )
        actual = stitch_from_offsets(
            images, offsets, edge=-1
        )
        np.testing.assert_array_equal(actual, expected, strict=True)


# --------------- split_stitched_mask_to_fields / recompose_split_labels ---------------


def _make_2x2_canvas_and_positions(
    tile_h: int = 32,
    tile_w: int = 32,
    overlap_x: int = 0,
    overlap_y: int = 0,
    n_t: int = 1,
    dtype: np.dtype = np.uint16,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build a (T, Y, X) canvas matching a 2x2 grid with given tile size and overlap."""
    spacing_x = tile_w - overlap_x
    spacing_y = tile_h - overlap_y
    positions = [
        (0.0, 0.0),
        (float(spacing_x), 0.0),
        (0.0, float(spacing_y)),
        (float(spacing_x), float(spacing_y)),
    ]
    canvas_w = 2 * tile_w - overlap_x
    canvas_h = 2 * tile_h - overlap_y
    canvas = np.zeros((n_t, canvas_h, canvas_w), dtype=dtype)
    return canvas, positions


class TestSplitRecomposeLossless:
    """Round-trip: split(canvas) → recompose == canvas, pixel-for-pixel.

    Inputs come from a canvas-wide segmentation, so label IDs are globally
    unique. recompose_split_labels must preserve label IDs and reassemble
    boundary cells losslessly.
    """

    def test_label_fully_inside_one_tile(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        canvas[0, 5:10, 5:10] = 5

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_label_straddling_horizontal_boundary(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        # Boundary at canvas x = 28. Label spans 26..34 → crosses past overlap.
        canvas[0, 10:14, 26:34] = 7

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_label_straddling_vertical_boundary(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        canvas[0, 26:34, 10:14] = 11

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_label_crossing_four_corner_overlap(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        canvas[0, 26:34, 26:34] = 13

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_multiple_labels_mixed_positions(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        canvas[0, 2:6, 2:6] = 1
        canvas[0, 20:25, 30:34] = 2
        canvas[0, 30:34, 20:25] = 3
        canvas[0, 28:32, 28:32] = 4
        canvas[0, 50:54, 50:54] = 5

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_multiple_timepoints(self):
        """T>1 round-trip preserves per-frame labels."""
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=16, tile_w=16, overlap_x=2, overlap_y=2, n_t=3
        )
        canvas[0, 13:17, 13:17] = 1
        canvas[1, 5:9, 5:9] = 2
        canvas[2, 13:17, 5:9] = 3

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=16, tile_w=16,
            overlap_x=2, overlap_y=2,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=16, tile_w=16,
            overlap_x=2, overlap_y=2,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_zero_overlap(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=0, overlap_y=0
        )
        canvas[0, 10:14, 10:14] = 1
        canvas[0, 10:14, 40:44] = 2
        canvas[0, 40:44, 10:14] = 3
        canvas[0, 40:44, 40:44] = 4

        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=32, tile_w=32,
        )
        np.testing.assert_array_equal(recomposed, canvas)

    def test_empty_canvas(self):
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=16, tile_w=16, overlap_x=2, overlap_y=2
        )
        tiles = split_stitched_mask_to_fields(
            canvas, positions, tile_h=16, tile_w=16,
            overlap_x=2, overlap_y=2,
        )
        recomposed = recompose_split_labels(
            tiles, positions, tile_h=16, tile_w=16,
            overlap_x=2, overlap_y=2,
        )
        np.testing.assert_array_equal(recomposed, canvas)


class TestRecomposeSplitLabelsNapariShapes:
    """Recompose round-trip via the (N, [T,] H, W, C) array shape used by napari."""

    def test_4d_nhwc(self):
        """(N, H, W, C) input: fixed-cell labels with channel axis."""
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=32, tile_w=32, overlap_x=4, overlap_y=4
        )
        canvas[0, 20:25, 30:34] = 7  # crosses TL/TR boundary
        # Build a 2-channel (N, H, W, 2) input by sliding 2D split outputs
        # into channel 0 and zeros into channel 1.
        tiles_t = split_stitched_mask_to_fields(
            canvas, positions, tile_h=32, tile_w=32, overlap_x=4, overlap_y=4,
        )  # list of (T=1, H, W)
        n = len(tiles_t)
        nhwc = np.zeros((n, 32, 32, 2), dtype=canvas.dtype)
        for i, tile in enumerate(tiles_t):
            nhwc[i, :, :, 0] = tile[0]

        recomposed = recompose_split_labels(
            nhwc, positions, tile_h=32, tile_w=32,
            overlap_x=4, overlap_y=4,
        )
        # Output should be (Y, X, C). Channel 0 matches canvas[0]; channel 1 is zero.
        assert recomposed.shape == (canvas.shape[1], canvas.shape[2], 2)
        np.testing.assert_array_equal(recomposed[..., 0], canvas[0])
        np.testing.assert_array_equal(recomposed[..., 1], np.zeros_like(canvas[0]))

    def test_5d_ntyxc(self):
        """(N, T, H, W, C) input: live-cell labels with channel axis."""
        canvas, positions = _make_2x2_canvas_and_positions(
            tile_h=16, tile_w=16, overlap_x=2, overlap_y=2, n_t=2
        )
        canvas[0, 13:17, 5:9] = 1
        canvas[1, 5:9, 13:17] = 2

        tiles_t = split_stitched_mask_to_fields(
            canvas, positions, tile_h=16, tile_w=16, overlap_x=2, overlap_y=2,
        )  # list of (T=2, H, W)
        n = len(tiles_t)
        nthwc = np.zeros((n, 2, 16, 16, 1), dtype=canvas.dtype)
        for i, tile in enumerate(tiles_t):
            nthwc[i, :, :, :, 0] = tile

        recomposed = recompose_split_labels(
            nthwc, positions, tile_h=16, tile_w=16,
            overlap_x=2, overlap_y=2,
        )
        # Output should be (T, Y, X, C=1).
        assert recomposed.shape == (
            canvas.shape[0], canvas.shape[1], canvas.shape[2], 1
        )
        np.testing.assert_array_equal(recomposed[..., 0], canvas)


class TestRecomposeSplitLabelsValidation:
    def test_empty_tiles_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            recompose_split_labels([], [], tile_h=16, tile_w=16)

    def test_length_mismatch_raises(self):
        tile = np.zeros((1, 16, 16), dtype=np.uint16)
        with pytest.raises(ValueError, match="must match"):
            recompose_split_labels(
                [tile, tile], [(0.0, 0.0)], tile_h=16, tile_w=16
            )

    def test_wrong_ndim_raises(self):
        tile = np.zeros((16, 16), dtype=np.uint16)
        with pytest.raises(ValueError, match=r"\(T, tile_h, tile_w\)"):
            recompose_split_labels(
                [tile], [(0.0, 0.0)], tile_h=16, tile_w=16
            )


# --------------- split_stitched_from_offsets ---------------


def _test_image(shape: tuple[int, ...]) -> np.ndarray:
    """Return a test image of the provided shape."""
    return np.arange(np.prod(shape)).reshape(shape)


class TestSplitStitchedFromOffsets:
    def test_negative_offsets_throws(self):
        image = _test_image((1, 32, 64))
        with pytest.raises(ValueError, match="Offsets must be positive"):
            split_stitched_from_offsets(image, np.array([(0, 0), (32, -1)]), 32, 32)
        with pytest.raises(ValueError, match="Offsets must be positive"):
            split_stitched_from_offsets(image, np.array([(0, 0), (-1, 32)]), 32, 32)

    def test_dimensions_throws(self):
        image = _test_image((1, 32, 64))
        with pytest.raises(ValueError, match=re.escape("stitched must be (T, Y, X), got (32, 64)")):
            split_stitched_from_offsets(image[0], np.array([(0, 0), (32, 0)]), 32, 32)
        with pytest.raises(ValueError, match=re.escape("stitched must be (T, Y, X), got (1, 1, 32, 64)")):
            split_stitched_from_offsets(image[np.newaxis, ...], np.array([(0, 0), (32, 0)]), 32, 32)

    def test_1x2_no_overlap(self):
        image = _test_image((1, 32, 64))
        tiles = split_stitched_from_offsets(image, np.array([(0, 0), (32, 0)]), 32, 32)
        np.testing.assert_array_equal(tiles[0], image[:, 0:32, 0:32])
        np.testing.assert_array_equal(tiles[1], image[:, 0:32, 32:64])

    def test_2x2_with_overlap(self):
        # Ensure size is large enough for the internal crop
        image = _test_image((2, 62, 61))
        tiles = split_stitched_from_offsets(image, np.array([(0, 0), (2, 30), (29, 1), (29, 30)]), 32, 32)
        np.testing.assert_array_equal(tiles[0], image[:, 0:32, 0:32])
        np.testing.assert_array_equal(tiles[1], image[:, 30:62, 2:34])
        np.testing.assert_array_equal(tiles[2], image[:, 1:33, 29:61])
        np.testing.assert_array_equal(tiles[3], image[:, 30:62, 29:61])


# --------------- recompose_tiles ---------------


class TestRecomposeTiles:
    def test_negative_offsets_throws(self):
        tiles = [np.ones((1, 32, 64), dtype=np.uint16)] * 2
        with pytest.raises(ValueError, match="Offsets must be positive"):
            recompose_tiles(tiles, np.array([(0, 0), (32, -1)]))
        with pytest.raises(ValueError, match="Offsets must be positive"):
            recompose_tiles(tiles, np.array([(0, 0), (-1, 32)]))

    def test_1x2_no_overlap(self):
        image = _test_image((3, 32, 64))
        tiles = [image[:, 0:32, 0:32], image[:, 0:32, 32:64]]
        canvas = recompose_tiles(tiles, np.array([(0, 0), (32, 0)]))
        np.testing.assert_array_equal(canvas, image, strict=True)

    def test_2x2_TYX_with_overlap(self):
        image = _test_image((2, 62, 61))
        tiles = [image[:, 0:32, 0:32], image[:, 30:62, 2:34], image[:, 1:33, 29:61], image[:, 30:62, 29:61]]
        canvas = recompose_tiles(tiles, np.array([(0, 0), (2, 30), (29, 1), (29, 30)]))
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[:, 0:32, 0:32] = 1
        mask[:, 30:62, 2:34] = 1
        mask[:, 1:33, 29:61] = 1
        mask[:, 30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)

    def test_2x2_NYXC_with_overlap(self):
        image = _test_image((62, 61, 3))
        tiles = np.stack([image[0:32, 0:32, :], image[30:62, 2:34, :], image[1:33, 29:61, :], image[30:62, 29:61, :]])
        canvas = recompose_tiles(tiles, np.array([(0, 0), (2, 30), (29, 1), (29, 30)]))
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[0:32, 0:32] = 1
        mask[30:62, 2:34] = 1
        mask[1:33, 29:61] = 1
        mask[30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)

    def test_2x2_NTYXC_with_overlap(self):
        image = _test_image((4, 62, 61, 3))
        tiles = np.stack([
            image[:, 0:32, 0:32, :],
            image[:, 30:62, 2:34, :],
            image[:, 1:33, 29:61, :],
            image[:, 30:62, 29:61, :]
        ])
        canvas = recompose_tiles(tiles, np.array([(0, 0), (2, 30), (29, 1), (29, 30)]))
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[:, 0:32, 0:32] = 1
        mask[:, 30:62, 2:34] = 1
        mask[:, 1:33, 29:61] = 1
        mask[:, 30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)


class TestSplitRecomposeTiles:
    """Round-trip: split(canvas) → recompose == canvas, pixel-for-pixel."""
    def test_2x2_TYX_with_overlap(self):
        image = _test_image((2, 62, 61))
        offsets = np.array([(0, 0), (2, 30), (29, 1), (29, 30)])
        tiles = split_stitched_from_offsets(image, offsets, 32, 32)
        canvas = recompose_tiles(tiles, offsets)
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[:, 0:32, 0:32] = 1
        mask[:, 30:62, 2:34] = 1
        mask[:, 1:33, 29:61] = 1
        mask[:, 30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)

    def test_2x2_NYXC_with_overlap(self):
        image = _test_image((62, 61, 3))
        offsets = np.array([(0, 0), (2, 30), (29, 1), (29, 30)])
        # split each channel
        tiles = []
        for c in range(image.shape[-1]):
            # YX for channel -> TYX with T=1
            t = split_stitched_from_offsets(image[..., c][np.newaxis, ...], offsets, 32, 32)
            # stack NTYX and append NYX
            tiles.append(np.stack(t).squeeze(axis=1))
        # Build stack of NYXC
        stack = np.stack(tiles, axis=-1)
        canvas = recompose_tiles(stack, offsets)
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[0:32, 0:32] = 1
        mask[30:62, 2:34] = 1
        mask[1:33, 29:61] = 1
        mask[30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)

    def test_2x2_NTYXC_with_overlap(self):
        image = _test_image((4, 62, 61, 3))
        offsets = np.array([(0, 0), (2, 30), (29, 1), (29, 30)])
        # split each channel
        tiles = []
        for c in range(image.shape[-1]):
            # list of TYX for channel
            t = split_stitched_from_offsets(image[..., c], offsets, 32, 32)
            # append NTYX
            tiles.append(np.stack(t))
        # Build stack of NTYXC
        stack = np.stack(tiles, axis=-1)
        canvas = recompose_tiles(stack, offsets)
        # Remove parts of image not covered by tiles
        mask = np.zeros_like(image)
        mask[:, 0:32, 0:32] = 1
        mask[:, 30:62, 2:34] = 1
        mask[:, 1:33, 29:61] = 1
        mask[:, 30:62, 29:61] = 1
        image *= mask
        np.testing.assert_array_equal(canvas, image, strict=True)
