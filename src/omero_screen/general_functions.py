"""Module for common functions."""

from typing import Any

import numpy as np
import numpy.typing as npt
from skimage import exposure
from skimage.segmentation import clear_border


def scale_img(
    img: npt.NDArray[Any], percentile: tuple[float, float] = (1, 99)
) -> npt.NDArray[Any]:
    """Increase contrast by scaling image to exclude lowest and highest intensities.
    Args:
        img: Image
        percentile: Lower and upper range for intensities
    Returns:
        scaled image
    """
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))


def filter_segmentation(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Removes border objects and filters large and small objects from segmentation mask.
    Args:
        mask: unfiltered segmentation mask
    Returns:
        filtered segmentation mask
    """
    cleared: npt.NDArray[Any] = clear_border(mask, buffer_size=5)
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask
