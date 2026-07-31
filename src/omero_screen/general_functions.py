"""This module provides general utility functions for image processing tasks commonly used in OMERO screen workflows.

Functions:
    - scale_img: Scales image intensities to enhance contrast by excluding extreme percentiles.
    - filter_segmentation: Cleans segmentation masks by removing border objects and filtering by object size.

These functions are intended to support preprocessing and postprocessing steps in image analysis pipelines.
"""

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
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))  # type: ignore


def filter_segmentation(
    mask: npt.NDArray[Any],
    border: int = 5,
    foreground: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[Any]:
    """Removes border objects and filters large and small objects from segmentation mask.

    The size of the border can be specified. Use a negative size to skip removal of
    border objects.

    Optionally all foreground pixels can be provided. Objects overlapping
    with False pixels will be removed.
    This option overrides the border argument.

    Args:
        mask: unfiltered segmentation mask
        border: width of the border examined (negative to disable)
        foreground: array of bool, same shape as `mask`, optional.
            Image foreground mask. Objects in mask overlapping with
            False pixels of foreground will be removed. If defined, the
            argument border will be ignored.

    Returns:
        filtered segmentation mask
    """
    cleared: npt.NDArray[Any] = (
        mask
        if border < 0 and foreground is None
        else clear_border(
            mask,
            buffer_size=border,
            mask=foreground,
        )  # type: ignore[no-untyped-call]
    )
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask  # type: ignore[no-any-return]
