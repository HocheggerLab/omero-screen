# ------------------------------------------------------------------------------
# Tests for cellclass.bin.create_dataset private helpers and run().
# ------------------------------------------------------------------------------

"""Tests for create_dataset._binary_mask, _to_uint8, _extract_roi, and run()."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from cellclass.bin.create_dataset import _binary_mask, _extract_roi, _to_uint8, run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROP: int = 50
CHANNELS: int = 2


# ---------------------------------------------------------------------------
# _binary_mask
# ---------------------------------------------------------------------------


class TestBinaryMask:
    """Tests for _binary_mask."""

    def test_centred_label_returns_mask_and_false(self) -> None:
        """Centre pixel is non-zero: returns a valid binary mask and non_central=False."""
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        mask[CROP // 2, CROP // 2] = 1
        mask[20:30, 20:30] = 1
        result, non_central = _binary_mask(mask)
        assert result is not None
        assert non_central is False
        assert result.dtype == np.uint8
        assert result[CROP // 2, CROP // 2] == 1

    def test_off_centre_single_label_returns_mask_and_true(self) -> None:
        """Centre pixel is zero but one unique label exists elsewhere: non_central=True."""
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        mask[5:10, 5:10] = 7  # away from centre
        result, non_central = _binary_mask(mask)
        assert result is not None
        assert non_central is True
        assert result[7, 7] == 1

    def test_all_zero_mask_returns_none(self) -> None:
        """All-zero mask has no identifiable cell: returns (None, False)."""
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        result, non_central = _binary_mask(mask)
        assert result is None
        assert non_central is False

    def test_multiple_off_centre_labels_returns_none(self) -> None:
        """Multiple unique non-zero labels away from centre: returns (None, False)."""
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        mask[5:8, 5:8] = 2
        mask[40:43, 40:43] = 3
        result, non_central = _binary_mask(mask)
        assert result is None
        assert non_central is False


# ---------------------------------------------------------------------------
# _to_uint8
# ---------------------------------------------------------------------------


class TestToUint8:
    """Tests for _to_uint8."""

    def test_output_dtype_is_uint8(self) -> None:
        """Output array must be dtype uint8."""
        rng = np.random.default_rng(1)
        img = rng.random((CROP, CROP, CHANNELS), dtype=np.float64)
        out = _to_uint8(img)
        assert out.dtype == np.uint8

    def test_output_values_in_range_1_to_255(self) -> None:
        """All values must lie in [1, 255] per the function contract."""
        rng = np.random.default_rng(2)
        img = rng.random((CROP, CROP, CHANNELS), dtype=np.float64)
        out = _to_uint8(img)
        assert int(out.min()) >= 1
        assert int(out.max()) <= 255

    def test_each_channel_normalised_independently(self) -> None:
        """Different channel ranges produce independent normalisation."""
        img = np.ones((CROP, CROP, 2), dtype=np.float64)
        img[..., 0] = np.linspace(0, 1, CROP * CROP).reshape(CROP, CROP)
        img[..., 1] = np.linspace(10, 100, CROP * CROP).reshape(CROP, CROP)
        out = _to_uint8(img)
        # Both channels should span almost the full [1, 255] range
        for c in range(2):
            channel_min = int(out[..., c].min())
            channel_max = int(out[..., c].max())
            assert channel_min >= 1
            assert channel_max <= 255
            assert channel_max > channel_min

    def test_constant_channel_does_not_crash(self) -> None:
        """A flat channel (pmin == pmax) must not raise ZeroDivisionError."""
        img = np.full((CROP, CROP, 1), 5.0, dtype=np.float64)
        out = _to_uint8(img)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# _extract_roi
# ---------------------------------------------------------------------------


class TestExtractRoi:
    """Tests for _extract_roi."""

    def _make_image_and_centred_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a (CROP, CROP, 2) uint8 image and centred binary mask."""
        rng = np.random.default_rng(3)
        img = rng.integers(1, 255, (CROP, CROP, CHANNELS), dtype=np.uint8)
        mask = np.zeros((CROP, CROP), dtype=np.uint8)
        half = CROP // 4
        c = CROP // 2
        mask[c - half : c + half, c - half : c + half] = 1
        return img, mask

    def test_output_shape_is_cyx(self) -> None:
        """ROI must be transposed to (C, Y, X) layout."""
        img, mask = self._make_image_and_centred_mask()
        roi, _, _ = _extract_roi(img, mask)
        assert roi.shape == (CHANNELS, CROP, CROP)

    def test_centred_mask_yields_zero_shifts(self) -> None:
        """A perfectly centred mask should produce shiftx == shifty == 0."""
        img, mask = self._make_image_and_centred_mask()
        _, shiftx, shifty = _extract_roi(img, mask)
        assert shiftx == 0
        assert shifty == 0

    def test_off_centre_mask_yields_nonzero_shift(self) -> None:
        """A mask placed in a corner must result in at least one non-zero shift."""
        rng = np.random.default_rng(4)
        img = rng.integers(1, 255, (CROP, CROP, CHANNELS), dtype=np.uint8)
        mask = np.zeros((CROP, CROP), dtype=np.uint8)
        mask[2:8, 2:8] = 1  # top-left corner
        _, shiftx, shifty = _extract_roi(img, mask)
        assert shiftx != 0 or shifty != 0


# ---------------------------------------------------------------------------
# run() end-to-end
# ---------------------------------------------------------------------------


class TestRun:
    """End-to-end tests for create_dataset.run()."""

    def test_npz_file_is_created(self, npz_file: Path) -> None:
        """run() must produce a .npz file on disk."""
        assert npz_file.exists()

    def test_npz_has_required_keys(self, npz_file: Path) -> None:
        """Loaded .npz must contain keys X, y_names, and channels."""
        data = np.load(npz_file, allow_pickle=True)
        assert "X" in data
        assert "y_names" in data
        assert "channels" in data

    def test_unassigned_crops_are_excluded(self, tmp_path: Path) -> None:
        """Labels in the ignore list must not appear in the output y_names."""
        rng = np.random.default_rng(5)
        n = 3
        images = [rng.random((CROP, CROP, CHANNELS), dtype=np.float32) for _ in range(n)]
        masks = [np.zeros((CROP, CROP), dtype=np.int32) for _ in range(n)]
        for m in masks:
            m[CROP // 2, CROP // 2] = 1
        d = {"data": (images, masks), "target": ["mitosis", "unassigned", "interphase"]}
        np.save(tmp_path / "data.npy", d)
        (tmp_path / "metadata.json").write_text(
            json.dumps({"user_data": {"channels": ["DAPI", "Tub"]}})
        )

        args = argparse.Namespace(
            dir=str(tmp_path),
            name="rois",
            out=None,
            ignore=["unassigned"],
            duplicates=False,
            channels=[],
            single_label=True,
        )
        run(args)

        data = np.load(tmp_path / "rois.npz", allow_pickle=True)
        y_names = list(data["y_names"])
        assert "unassigned" not in y_names

    def test_duplicate_crops_are_deduplicated(self, tmp_path: Path) -> None:
        """Saving the same .npy twice should not double the sample count."""
        rng = np.random.default_rng(6)
        img = rng.random((CROP, CROP, CHANNELS), dtype=np.float32)
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        mask[CROP // 2, CROP // 2] = 1
        d = {"data": ([img], [mask]), "target": ["mitosis"]}
        np.save(tmp_path / "a.npy", d)
        np.save(tmp_path / "b.npy", d)

        (tmp_path / "metadata.json").write_text(
            json.dumps({"user_data": {"channels": ["DAPI", "Tub"]}})
        )
        args = argparse.Namespace(
            dir=str(tmp_path),
            name="rois",
            out=None,
            ignore=["unassigned"],
            duplicates=False,
            channels=[],
            single_label=True,
        )
        run(args)

        data = np.load(tmp_path / "rois.npz", allow_pickle=True)
        # Only one unique crop should be stored
        assert len(data["y_names"]) == 1

    def test_channel_count_mismatch_raises_system_exit(self, tmp_path: Path) -> None:
        """A .npy file with wrong channel count must cause SystemExit(1)."""
        rng = np.random.default_rng(7)
        # Image has 3 channels but metadata declares 2
        img = rng.random((CROP, CROP, 3), dtype=np.float32)
        mask = np.zeros((CROP, CROP), dtype=np.int32)
        mask[CROP // 2, CROP // 2] = 1
        d = {"data": ([img], [mask]), "target": ["mitosis"]}
        np.save(tmp_path / "bad.npy", d)

        (tmp_path / "metadata.json").write_text(
            json.dumps({"user_data": {"channels": ["DAPI", "Tub"]}})
        )
        args = argparse.Namespace(
            dir=str(tmp_path),
            name="rois",
            out=None,
            ignore=["unassigned"],
            duplicates=False,
            channels=[],
            single_label=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            run(args)
        assert exc_info.value.code == 1
