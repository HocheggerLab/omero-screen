"""Tests for position-based stitching from stage coordinates."""

import numpy as np
import pytest
import re
from omero_utils.stitching import (
    recompose_tiles,
    split_stitched_from_offsets,
    stitch_from_offsets,
    stitch_labels_from_offsets,
)


# --------------- stitch_from_offsets ---------------


class TestStitchFromOffsets:
    def test_negative_offsets_throws(self):
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


# --------------- stitch_labels_from_offsets ---------------


class TestStitchLabelsFromOffsets:
    def test_negative_offsets_throws(self):
        tile = np.ones((32, 32, 1), dtype=np.float32)
        images = np.stack([tile, tile])  # (2, 32, 32, 1)
        with pytest.raises(ValueError, match="Offsets must be positive"):
            stitch_labels_from_offsets(images, np.array([(0, 0), (32, -1)]))
        with pytest.raises(ValueError, match="Offsets must be positive"):
            stitch_labels_from_offsets(images, np.array([(0, 0), (-1, 32)]))

    def test_2x2_labels(self):
        label = np.ones((32, 32, 1), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        # tile = 32 → no overlap (0)
        offsets = np.array([
            (0, 0),
            (32, 0),
            (0, 32),
            (32, 32),
        ])
        result = stitch_labels_from_offsets(
            labels, offsets,
        )
        # Output should be (64, 64, 1) since no overlap
        assert result.ndim == 3
        assert result.shape == (64, 64, 1)

    def test_2x2_labels_with_translation(self):
        label = np.ones((32, 32, 2), dtype=np.int32)
        labels = np.stack([label * (i + 1) for i in range(4)])
        offsets = np.array([
            (0, 0),
            (0, 32),
            (32, 0),
            (32, 32),
        ])
        # Note: translation can create overlap if both are used with same sign
        for pos in [(0, 0), (0, 4), (2, 0), (0, -4), (-2, 0), (-3, 4), (3, -2)]:
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
            result = stitch_labels_from_offsets(
                labels, off,
            )
            assert result.ndim == 3
            assert result.shape == (64 + abs(ty), 64 + abs(tx), 2)
            # check corner values.
            # Note: Stiching defined in x, y order:
            # 1 3
            # 2 4
            # Each additional label is added to the current max.
            # Stitch order = 1, 2, 3, 4. Cumulative = 1, 3, 6, 10.
            top_left, bottom_left, top_right, bottom_right = 1, 3, 6, 10

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
        offsets = np.array([
            (0, 0),
            (0, 32),
            (32, 0),
            (32, 32),
        ])
        tx, ty = 4, 3
        for a in offsets:
            if a[0]:
                if a[1]:
                    a[0] += tx
                a[1] += ty
            elif a[1]:
                a[0] += tx
        result = stitch_labels_from_offsets(
            labels, offsets,
        )
        assert result.ndim == 3
        assert result.shape == (64 + abs(ty), 64 + abs(tx), 1)
        # check corner values.
        # Note: Stiching defined in x, y order.
        # 1 3
        # 2 4
        # Each additional label is added to the current max.
        # Unless it overlaps with a previous label, in which case it matches.
        # Here top-right (3) overlaps bottom-left (2), so bottom-left is preserved.
        # Stich order = 1, 2, 3, 4. Cumulative = 1, 3, 3 (overlap), 7.
        top_left, bottom_left, top_right, bottom_right = 1, 3, 3, 7

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
        offsets = np.array([
            (0, 0),
            (0, 32),
            (32, 0),
            (32, 32),
        ])
        tx, ty = -4, -3
        for a in offsets:
            if a[0]:
                if a[1]:
                    a[0] += tx
                a[1] += ty
            elif a[1]:
                a[0] += tx
        # offsets must be positive
        offsets -= offsets.min(axis=0)
        result = stitch_labels_from_offsets(
            labels, offsets,
        )
        assert result.ndim == 3
        assert result.shape == (64 + abs(ty), 64 + abs(tx), 1)
        # check corner values.
        # Note: Stiching defined in x, y order.
        # 1 3
        # 2 4
        # Each additional label is added to the current max.
        # Unless it overlaps with a previous label, in which case it matches.
        # Here bottom-right (4) overlaps top-left (1), so top-left is preserved.
        # Stich order = 1, 2, 3, 4. Cumulative = 1, 3, 6, 1 (overlap).
        top_left, bottom_left, top_right, bottom_right = 1, 3, 6, 1

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
