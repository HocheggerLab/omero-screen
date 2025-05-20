"""This module provides image aggregation and filtering utilities for 2D numpy arrays.

It includes functions for morphological operations, median and Gaussian filtering, and block-wise image processing. The main class, ImageAggregator, accumulates and aggregates image data from multiple images, supporting block-wise minimum aggregation and optional smoothing filters.

Functions:
    strel_disk(radius): Create a disk structuring element for morphological operations.
    median_filter(pixel_data, radius): Apply a median filter with a disk-shaped structuring element.
    gaussian_filter(pixel_data, sigma): Apply a Gaussian filter with edge artifact correction.
    block(shape, block_shape): Divide an image into labeled blocks for block-wise processing.
    fixup_scipy_ndimage_result(whatever_it_returned): Ensure scipy.ndimage results are returned as numpy arrays.

Classes:
    ImageAggregator: Accumulates and aggregates image data, with support for block-wise minimum and smoothing filters.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import scipy
import skimage


def strel_disk(radius: float) -> npt.NDArray[np.float64]:
    """Create a disk structuring element for morphological operations.

    Args:
        radius: radius of the disk

    Returns:
        np.ndarray: disk element
    """
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius**2
    strel = np.zeros(x.shape, dtype=np.float64)
    strel[x * x + y * y <= radius2] = 1
    return strel


def median_filter(
    pixel_data: npt.NDArray[Any], radius: float
) -> npt.NDArray[Any]:
    """Perform median filter with the given radius.

    Args:
        pixel_data: 2D pixels
        radius: radius of the disk

    Returns:
        np.ndarray: filtered data
    """
    filter_sigma = max(1, int(radius + 0.5))
    strel = strel_disk(filter_sigma)
    scale = 65535 / np.max(pixel_data)
    rescaled_pixel_data = pixel_data * scale
    rescaled_pixel_data = rescaled_pixel_data.astype(np.uint16)
    # Note: skimage is not typed
    output_pixels: npt.NDArray[Any] = skimage.filters.median(  # type: ignore[no-untyped-call]
        rescaled_pixel_data, strel, behavior="rank"
    )
    output_pixels = output_pixels / scale
    output_pixels = output_pixels.astype(pixel_data.dtype)
    return output_pixels


def gaussian_filter(
    pixel_data: npt.NDArray[Any], sigma: float
) -> npt.NDArray[Any]:
    """Perform gaussian filter with the given radius.

    Args:
        pixel_data: 2D pixels
        sigma: standard deviation of the Gaussian

    Returns:
        np.ndarray: filtered data
    """

    # Use the method to divide by the bleed over fraction
    # to remove edge artifacts
    def fn(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        im: npt.NDArray[Any] = scipy.ndimage.gaussian_filter(
            image, sigma, mode="constant", cval=0
        )
        return im

    mask = np.ones(pixel_data.shape)
    bleed_over = fn(mask)
    smoothed_image = fn(pixel_data)
    output_image: npt.NDArray[Any] = smoothed_image / (
        bleed_over + np.finfo(float).eps
    )
    return output_image


# https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py
# centrosome.cpmorphology.block
def block(
    shape: tuple[int, ...], block_shape: tuple[int, ...]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Create a labels image that divides the image into blocks.

    The idea here is to block-process an image by using SciPy label
    routines. This routine divides the image into blocks of a configurable
    dimension. The caller then calls scipy.ndimage functions to process
    each block as a labeled image. The block values can then be applied
    to the image via indexing. For instance:

    labels, indexes = block(image.shape, (60,60))
    minima = scind.minimum(image, labels, indexes)
    img2 = image - minima[labels]

    Args:
        shape: the shape of the image to be blocked
        block_shape: the shape of one block

    Returns:
        a labels matrix and the indexes of all labels generated
    """
    shape_array = np.array(shape)
    block_array = np.array(block_shape)
    i, j = np.mgrid[0 : shape_array[0], 0 : shape_array[1]]
    ijmax = (shape_array.astype(float) / block_array.astype(float)).astype(int)
    ijmax = np.maximum(ijmax, 1)
    multiplier = ijmax.astype(float) / shape_array.astype(float)
    i = (i * multiplier[0]).astype(int)
    j = (j * multiplier[1]).astype(int)
    labels = i * ijmax[1] + j
    indexes = np.array(list(range(np.prod(ijmax))))
    return labels, indexes


# https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py
# centrosome.cpmorphology.fixup_scipy_ndimage_result
def fixup_scipy_ndimage_result(whatever_it_returned: Any) -> npt.NDArray[Any]:
    """Convert a result from scipy.ndimage to a numpy array.

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scind.maximum(image, labels, [1]) returns a float
    but
    scind.maximum(image, labels, [1,2]) returns a list

    Args:
        whatever_it_returned: scipy result

    Returns:
        np.ndarray: result as an array
    """
    if getattr(whatever_it_returned, "__getitem__", False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])


class ImageAggregator:
    """Accumulates and aggregates image data from multiple 2D images, providing methods to retrieve the aggregate image and apply optional smoothing filters.

    This class is useful for combining a sequence of images (e.g., for illumination correction or averaging) and supports block-wise minimum aggregation for background correction. The aggregated image can be retrieved directly or after applying a median or Gaussian filter for smoothing.

    Attributes:
        __block_size (int): Size of the block region for block-wise minimum aggregation. If not strictly positive, block-wise aggregation is not used.
        __labels (Optional[np.ndarray]): Label matrix for block-wise processing.
        __indexes (Optional[np.ndarray]): Indexes for block-wise processing.
        __dirty (bool): Indicates if the cached image is out of date.
        __image_sum (Optional[np.ndarray]): Accumulated sum of images (or block-minimum images).
        __count (int): Number of images added.
        __cached_image (Optional[np.ndarray]): Cached result of the aggregated image.
    """

    def __init__(self, block_size: int = 0):
        """Create an instance.

        Args:
            block_size: size of the block region for the minimum; ignored if not strictly positive.
        """
        super().__init__()
        self.__block_size: int = block_size
        self.__labels: Optional[npt.NDArray[Any]] = None
        self.__indexes: Optional[npt.NDArray[Any]] = None
        self.__dirty: bool = False
        self.__image_sum: Optional[npt.NDArray[Any]] = None
        self.__count: int = 0
        self.__cached_image: Optional[npt.NDArray[Any]] = None

    def add_image(self, image: npt.NDArray[Any]) -> None:
        """Accumulate the data from the given image.

        Args:
            image: an instance of a 2D numpy array
        """
        self.__dirty = True
        # Optional aggregation of the minimum from blocks
        # See: https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/correctilluminationcalculate.py#L923
        if self.__count == 0:
            self.__image_sum = np.zeros(image.shape)
            if self.__block_size > 0:
                self.__labels, self.__indexes = block(
                    image.shape[:2], (self.__block_size, self.__block_size)
                )
        if self.__block_size > 0:
            minima = fixup_scipy_ndimage_result(
                scipy.ndimage.minimum(image, self.__labels, self.__indexes)
            )
            pixel_data = minima[self.__labels]
        else:
            pixel_data = image
        self.__image_sum = self.__image_sum + pixel_data
        self.__count = self.__count + 1

    def get_image(self) -> Optional[npt.NDArray[Any]]:
        """Get the aggregated image.

        Returns:
            np.ndarray: image data
        """
        if self.__dirty and self.__image_sum is not None:
            self.__cached_image = self.__image_sum / self.__count
            self.__dirty = False
        return self.__cached_image

    def get_median_image(self, radius: float) -> Optional[npt.NDArray[Any]]:
        """Get the aggregated image after smoothing with a median filter.

        Args:
            radius: radius of the disk

        Returns:
            np.ndarray: filtered data
        """
        im = self.get_image()
        return median_filter(im, radius) if im is not None else None

    def get_gaussian_image(self, sigma: float) -> Optional[npt.NDArray[Any]]:
        """Get the aggregated image after smoothing with a Gaussian filter.

        Args:
            sigma: standard deviation of the Gaussian

        Returns:
            np.ndarray: filtered data
        """
        im = self.get_image()
        return gaussian_filter(im, sigma) if im is not None else None

    def reset(self) -> None:
        """Reset the aggregator to its initial state, clearing all accumulated data and cached results.

        After calling this method, the aggregator will behave as if newly created, with no images added.
        """
        self.__labels = np.array(None)
        self.__indexes = None
        self.__dirty = False
        self.__image_sum = None
        self.__count = 0
        self.__cached_image = None
