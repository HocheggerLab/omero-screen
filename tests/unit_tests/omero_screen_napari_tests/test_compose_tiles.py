"""Tests for compose_tiles and compose_labels threaded stitching."""

import numpy as np
import pytest

from omero_screen_napari.welldata_api import compose_labels, compose_tiles


# --------------- compose_tiles ---------------


class TestComposeTiles:
    def _make_tiles(
        self,
        grid: tuple[int, int],
        tile_shape: tuple[int, int, int],
        dtype: np.dtype = np.dtype(np.float32),
        fill: float | None = None,
    ) -> dict[int, dict[int, np.ndarray]]:
        """Helper to build a tiles dict for an (nx, ny) grid."""
        tiles: dict[int, dict[int, np.ndarray]] = {}
        idx = 1
        for x in range(grid[0]):
            tiles[x] = {}
            for y in range(grid[1]):
                val = fill if fill is not None else float(idx)
                tiles[x][y] = np.full(tile_shape, val, dtype=dtype)
                idx += 1
        return tiles

    def test_single_tile_no_rotation(self) -> None:
        """1x1 grid, rotation=0 → output == input."""
        tile = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        tiles = {0: {0: tile}}
        result = compose_tiles(tiles, rotation=0)
        np.testing.assert_array_almost_equal(result, tile)

    def test_2x2_uniform_tiles(self) -> None:
        """2x2 grid, uniform values, no rotation → correct shape and values."""
        tiles = self._make_tiles((2, 2), (16, 16, 2), fill=1.0)
        result = compose_tiles(tiles, rotation=0)
        assert result.shape == (32, 32, 2)
        np.testing.assert_array_almost_equal(result, 1.0)

    def test_rotation_changes_shape(self) -> None:
        """Non-zero rotation → output larger than input tile."""
        tiles = self._make_tiles((1, 1), (32, 32, 1), fill=1.0)
        result = compose_tiles(tiles, rotation=15)
        # Rotated tile should be larger than original
        assert result.shape[0] > 32
        assert result.shape[1] > 32
        assert result.shape[2] == 1

    def test_overlap_blending(self) -> None:
        """Negative ox/oy with edge > 0 → overlap region is blended."""
        tiles = self._make_tiles((2, 1), (32, 32, 1), fill=1.0)
        result = compose_tiles(tiles, rotation=0, ox=-8, oy=0, edge=4)
        # Width: 2*32 - 8 = 56
        assert result.shape == (32, 56, 1)
        # All values should be 1.0 (uniform tiles blended with weights)
        np.testing.assert_array_almost_equal(result, 1.0)

    def test_preserves_integer_dtype(self) -> None:
        """uint16 input → uint16 output."""
        tiles = self._make_tiles(
            (1, 1), (16, 16, 1), dtype=np.dtype(np.uint16), fill=100.0
        )
        result = compose_tiles(tiles, rotation=0)
        assert result.dtype == np.uint16

    def test_preserves_float_dtype(self) -> None:
        """float32 input → float32 output."""
        tiles = self._make_tiles(
            (1, 1), (16, 16, 1), dtype=np.dtype(np.float32), fill=1.5
        )
        result = compose_tiles(tiles, rotation=0)
        assert result.dtype == np.float32

    def test_multichannel_correct_placement(self) -> None:
        """Each channel has different values; verify they land in correct output channels."""
        tile = np.zeros((8, 8, 3), dtype=np.float32)
        tile[..., 0] = 10
        tile[..., 1] = 20
        tile[..., 2] = 30
        tiles = {0: {0: tile}}
        result = compose_tiles(tiles, rotation=0)
        np.testing.assert_array_almost_equal(result[..., 0], 10.0)
        np.testing.assert_array_almost_equal(result[..., 1], 20.0)
        np.testing.assert_array_almost_equal(result[..., 2], 30.0)


# --------------- compose_labels ---------------


class TestComposeLabels:
    def _make_label_tiles(
        self,
        grid: tuple[int, int],
        tile_shape: tuple[int, int, int],
    ) -> dict[int, dict[int, np.ndarray]]:
        """Helper to build label tiles. Each tile gets a unique label value."""
        tiles: dict[int, dict[int, np.ndarray]] = {}
        label_id = 1
        for x in range(grid[0]):
            tiles[x] = {}
            for y in range(grid[1]):
                tiles[x][y] = np.full(tile_shape, label_id, dtype=np.int32)
                label_id += 1
        return tiles

    def test_single_tile_labels(self) -> None:
        """1x1 grid → labels preserved."""
        tile = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
        # Shape: (2, 2, 2) → YXC
        tiles = {0: {0: tile}}
        result = compose_labels(tiles, rotation=0)
        np.testing.assert_array_equal(result, tile)

    def test_2x2_non_overlapping(self) -> None:
        """2x2 with no overlap → all labels present in output."""
        tiles = self._make_label_tiles((2, 2), (16, 16, 1))
        result = compose_labels(tiles, rotation=0)
        assert result.shape == (32, 32, 1)
        unique_labels = np.unique(result)
        # Should have background (0 from rotation border if any) plus the 4 label IDs
        # With no rotation, tiles fill exactly, so we get remapped IDs
        assert len(unique_labels) >= 4  # At least 4 unique non-zero labels

    def test_label_ids_unique(self) -> None:
        """Overlapping tiles with distinct labels → merged IDs remain unique."""
        # Create two tiles with multiple distinct label regions each
        tile1 = np.zeros((16, 16, 1), dtype=np.int32)
        tile1[:8, :, 0] = 1  # top half = label 1
        tile1[8:, :, 0] = 2  # bottom half = label 2

        tile2 = np.zeros((16, 16, 1), dtype=np.int32)
        tile2[:8, :, 0] = 1  # top half = label 1
        tile2[8:, :, 0] = 2  # bottom half = label 2

        tiles = {0: {0: tile1}, 1: {0: tile2}}
        result = compose_labels(tiles, rotation=0, ox=-4)
        # Should have at least 2 distinct non-zero labels in the output
        unique = np.unique(result[result > 0])
        assert len(unique) >= 2

    def test_labels_no_rotation(self) -> None:
        """Verify compose_labels with rotation=0 produces correct shape."""
        tiles = self._make_label_tiles((3, 3), (8, 8, 1))
        result = compose_labels(tiles, rotation=0)
        assert result.shape == (24, 24, 1)

    def test_labels_with_rotation(self) -> None:
        """Rotation changes output shape for labels too."""
        tiles = self._make_label_tiles((1, 1), (32, 32, 1))
        result = compose_labels(tiles, rotation=15)
        assert result.shape[0] > 32
        assert result.shape[1] > 32
        assert result.shape[2] == 1
