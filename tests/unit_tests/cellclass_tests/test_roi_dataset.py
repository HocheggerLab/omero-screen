# ------------------------------------------------------------------------------
# Tests for ROIDataset from cellclass.datasets.
# ------------------------------------------------------------------------------

"""Tests for datasets.ROIDataset."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from cellclass.datasets import ROIDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SAMPLES: int = 5
NUM_CHANNELS: int = 2
CROP_SIZE: int = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n: int = NUM_SAMPLES,
    channels: int = NUM_CHANNELS,
    size: int = CROP_SIZE,
    transform: object = None,
) -> ROIDataset:
    """Return a small ROIDataset filled with deterministic random data.

    Args:
        n: Number of samples.
        channels: Number of image channels (C dimension).
        size: Spatial size in pixels (H == W).
        transform: Optional torchvision-style transform callable.

    Returns:
        Populated ROIDataset instance.

    """
    rng = np.random.default_rng(0)
    rois = rng.integers(1, 255, (n, channels, size, size), dtype=np.uint8)
    labels = np.arange(n, dtype=np.int_)
    return ROIDataset(rois, labels, transform=transform)  # type: ignore[arg-type]


class TestROIDataset:
    """Tests for ROIDataset."""

    def test_len_equals_number_of_samples(self) -> None:
        """__len__ must return the number of input samples."""
        ds = _make_dataset()
        assert len(ds) == NUM_SAMPLES

    def test_getitem_returns_tuple_of_two(self) -> None:
        """__getitem__ must return a 2-tuple (image_tensor, label_tensor)."""
        ds = _make_dataset()
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_image_tensor_shape_is_cyx(self) -> None:
        """Image tensor must have shape (C, H, W) matching the input array."""
        ds = _make_dataset()
        image, _ = ds[0]
        assert image.shape == torch.Size([NUM_CHANNELS, CROP_SIZE, CROP_SIZE])

    def test_label_tensor_is_scalar_integer(self) -> None:
        """Label tensor must be a zero-dimensional long tensor."""
        ds = _make_dataset()
        _, label = ds[0]
        assert label.shape == torch.Size([])
        assert label.dtype == torch.long

    def test_integer_labels_round_trip_correctly(self) -> None:
        """Labels must be recoverable without loss for indices 0, 1, 2."""
        ds = _make_dataset(n=3)
        for idx in range(3):
            _, label = ds[idx]
            assert int(label) == idx

    def test_transform_is_applied_to_image(self) -> None:
        """A transform that changes shape must be reflected in the returned tensor."""

        def double_channels(t: torch.Tensor) -> torch.Tensor:
            """Concatenate tensor with itself along channel axis."""
            return torch.cat([t, t], dim=0)

        ds = _make_dataset(transform=double_channels)
        image, _ = ds[0]
        assert image.shape[0] == NUM_CHANNELS * 2

    def test_no_transform_leaves_shape_unchanged(self) -> None:
        """Without a transform the image shape must match the raw array layout."""
        ds = _make_dataset(transform=None)
        image, _ = ds[0]
        assert image.shape == torch.Size([NUM_CHANNELS, CROP_SIZE, CROP_SIZE])
