# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""PyTorch Dataset classes for cell ROI image classification."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import from_numpy
from torch.utils.data import Dataset


class ROIDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):  # type: ignore[misc]
    """Dataset of image regions (ROIs) and integer class labels.

    Args:
        rois: Images array of shape (N, C, Y, X).
        labels: Integer class labels of shape (N,).
        transform: Optional transform applied to each ROI tensor.

    """

    def __init__(
        self,
        rois: npt.NDArray[np.uint8],
        labels: npt.NDArray[np.int_],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialise the dataset with ROI images and class labels."""
        self.rois = rois
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(len(self.labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ROI tensor and label tensor at the given index."""
        roi = from_numpy(self.rois[idx])
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.transform is not None:
            roi = self.transform(roi)
        return roi, label
