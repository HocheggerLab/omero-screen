"""Tests for position-based stitching from stage coordinates."""

import numpy as np
import pytest

from omero_utils.stitching import (
    _adaptive_tolerance,
    _cluster_values,
    _compute_overlap,
    has_valid_positions,
    positions_to_grid,
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


# --------------- _compute_overlap ---------------


class TestComputeOverlap:
    def test_um_spacing(self):
        """Spacing in µm close to tile size → valid overlap."""
        # 5 clusters, spacing = 200µm, pixel_size = 0.5µm/px → step = 400px
        # tile = 500px → overlap = 100
        clusters = [0.0, 200.0, 400.0, 600.0, 800.0]
        assert _compute_overlap(clusters, tile_size_px=500, pixel_size=0.5, axis_label="X") == 100

    def test_non_um_spacing_uses_fallback(self):
        """Spacing much smaller than tile → not µm, uses fallback."""
        clusters = [0.0, 0.0013, 0.0026]
        assert _compute_overlap(clusters, tile_size_px=1080, pixel_size=0.65, axis_label="X", fallback=7) == 7

    def test_non_um_spacing_default_fallback(self):
        """Spacing much smaller than tile → default fallback is 0."""
        clusters = [0.0, 0.0013, 0.0026]
        assert _compute_overlap(clusters, tile_size_px=1080, pixel_size=0.65, axis_label="X") == 0

    def test_single_cluster(self):
        assert _compute_overlap([0.0], tile_size_px=1080, pixel_size=0.65, axis_label="X") == 0


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
