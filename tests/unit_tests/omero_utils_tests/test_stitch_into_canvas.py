"""Tests for ``stitch_into_canvas``.

The helper exists so a cyclic-IF restain round can be placed into the master
round's frame: the alignment shift is subtracted from the master's canvas offsets
(which can go negative) and the result must land on a canvas of exactly the
master's size so the rounds stack channel-wise.

The size guarantee is the load-bearing property. ``build_plate_zarr`` declares a
dask block's shape from a probe of the first block, so a stitch that quietly
returned a different size would corrupt the written store rather than raise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from omero_utils.stitching import stitch_from_offsets, stitch_into_canvas


def _tiles(
    n: int, h: int = 8, w: int = 8, c: int = 1, seed: int = 0
) -> npt.NDArray[np.uint16]:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1000, size=(n, h, w, c), dtype=np.uint16)


def _grid_offsets() -> npt.NDArray[np.int_]:
    """2x2 grid of 8x8 tiles, no overlap -> a 16x16 canvas."""
    return np.array([[0, 0], [8, 0], [0, 8], [8, 8]])


class TestMatchesStitchFromOffsets:
    """With no shift the helper must be a pure pass-through."""

    def test_zero_shift_identical_to_stitch_from_offsets(self) -> None:
        tiles = _tiles(4)
        offsets = _grid_offsets()
        expected = stitch_from_offsets(tiles, offsets, edge=0)
        actual = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)

    def test_zero_shift_5d(self) -> None:
        tiles = _tiles(4).reshape(4, 1, 8, 8, 1).repeat(3, axis=1)
        offsets = _grid_offsets()
        expected = stitch_from_offsets(tiles, offsets, edge=0)
        actual = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert actual.shape == (3, 16, 16, 1)
        np.testing.assert_array_equal(actual, expected)


class TestCanvasSizeGuarantee:
    """The output is exactly canvas_hw whatever the offsets do."""

    @pytest.mark.parametrize(
        "dx, dy",
        [(0, 0), (3, 0), (0, 3), (-3, 0), (0, -3), (5, -5), (-5, 5)],
    )
    def test_shape_always_canvas_hw(self, dx: int, dy: int) -> None:
        tiles = _tiles(4)
        offsets = _grid_offsets() - (dx, dy)
        out = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert out.shape == (16, 16, 1)

    @pytest.mark.parametrize("hw", [(4, 4), (16, 16), (32, 32), (5, 27)])
    def test_shape_for_odd_canvases(self, hw: tuple[int, int]) -> None:
        tiles = _tiles(4)
        out = stitch_into_canvas(tiles, _grid_offsets(), hw, edge=0)
        assert out.shape == (*hw, 1)

    def test_canvas_larger_than_stitch_is_zero_filled(self) -> None:
        tiles = _tiles(4)
        out = stitch_into_canvas(tiles, _grid_offsets(), (32, 32), edge=0)
        assert out.shape == (32, 32, 1)
        # The stitched 16x16 lands top-left; everything beyond it is zero.
        assert out[16:, :].max() == 0
        assert out[:, 16:].max() == 0
        assert out[:16, :16].max() > 0

    def test_5d_shape_preserved(self) -> None:
        tiles = _tiles(4).reshape(4, 1, 8, 8, 1).repeat(2, axis=1)
        offsets = _grid_offsets() - (3, 4)
        out = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert out.shape == (2, 16, 16, 1)


class TestShiftPlacement:
    """A shift moves content by exactly that many pixels, in the right axis."""

    def test_positive_shift_moves_content_up_and_left(self) -> None:
        tiles = _tiles(4)
        base = stitch_into_canvas(tiles, _grid_offsets(), (16, 16), edge=0)
        # Subtracting (dx, dy) from the offsets moves tiles left/up, so the
        # canvas content appears shifted towards the origin.
        shifted = stitch_into_canvas(
            tiles, _grid_offsets() - (2, 3), (16, 16), edge=0
        )
        np.testing.assert_array_equal(shifted[: 16 - 3, : 16 - 2], base[3:, 2:])

    def test_x_shift_does_not_move_y(self) -> None:
        """Guards the axis order: offsets are (ox, oy), arrays are (y, x)."""
        tiles = _tiles(4)
        base = stitch_into_canvas(tiles, _grid_offsets(), (16, 16), edge=0)
        shifted = stitch_into_canvas(
            tiles, _grid_offsets() - (4, 0), (16, 16), edge=0
        )
        # Rows are unchanged; only columns moved.
        np.testing.assert_array_equal(shifted[:, : 16 - 4], base[:, 4:])

    def test_y_shift_does_not_move_x(self) -> None:
        tiles = _tiles(4)
        base = stitch_into_canvas(tiles, _grid_offsets(), (16, 16), edge=0)
        shifted = stitch_into_canvas(
            tiles, _grid_offsets() - (0, 4), (16, 16), edge=0
        )
        np.testing.assert_array_equal(shifted[: 16 - 4, :], base[4:, :])


class TestTilesOutsideCanvas:
    def test_tile_shifted_entirely_off_canvas_is_dropped(self) -> None:
        tiles = _tiles(4)
        # Push everything far past the canvas: nothing should survive.
        offsets = _grid_offsets() - (100, 100)
        out = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert out.shape == (16, 16, 1)
        assert out.max() == 0

    def test_partial_overlap_keeps_the_visible_part(self) -> None:
        tiles = _tiles(4)
        offsets = _grid_offsets() - (12, 0)
        out = stitch_into_canvas(tiles, offsets, (16, 16), edge=0)
        assert out.shape == (16, 16, 1)
        assert out.max() > 0
        # The rightmost columns are no longer covered by any tile.
        assert out[:, 12:].max() == 0

    def test_dtype_is_preserved(self) -> None:
        tiles = _tiles(4)
        out = stitch_into_canvas(
            tiles, _grid_offsets() - (3, 3), (16, 16), edge=0
        )
        assert out.dtype == tiles.dtype

    def test_multichannel(self) -> None:
        tiles = _tiles(4, c=3)
        out = stitch_into_canvas(
            tiles, _grid_offsets() - (2, 2), (16, 16), edge=0
        )
        assert out.shape == (16, 16, 3)
