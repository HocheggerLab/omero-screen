"""Module for creating a gallery of images."""

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon


def create_gallery(images: list[npt.NDArray[Any]], grid_size: int) -> Figure:
    """
    Generates a gallery figure of the images in a grid.
    Args:
        images: List of numpy image arrays
        grid_size: Edge length of the grid
    Returns:
        gallery figure
    """
    fig, axs = plt.subplots(
        grid_size, grid_size, figsize=(20, 20), facecolor="white"
    )
    axs = axs.reshape(grid_size, grid_size)  # Ensure axs is a 2D grid

    for idx, ax in enumerate(axs.flat):
        if idx < len(images):
            im = _create_image(images[idx])
            ax.imshow(im)
            # Create contours using first channel and assuming non-masked pixels are > 0
            if len(im.shape) == 3:
                im = im[:, :, 0]
            _, thresholded_image = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                # contour has shape n x 1 x 2
                p = Polygon(
                    contour.squeeze(axis=1), fc="none", ec="cyan", lw=1
                )
                ax.add_patch(p)
        ax.axis("off")

    return fig


def _create_image(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Generates an image (M, N) or (M, N, 3).
    Note: Multi-channel images use the first 3 channels as RBG.
    Args:
        image: Input grayscale image (single channel), or CYX multi-channel.
    Returns:
        Processed image.
    Raises:
        Exception: if the input shape is not 2D or 3D
    """
    # Ensure the image is CYX for processing
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        pass
    else:
        raise Exception(f"Unsupported image shape: {image.shape}")

    # Normalize the image to the range 0-255
    image_normalized = np.array(
        [
            cv2.normalize(
                x,
                np.zeros(image.shape[1:], dtype=np.uint8),
                0,
                255,
                cv2.NORM_MINMAX,
            )
            for x in image
        ]
    )

    s = image_normalized.shape
    if s[0] == 1:
        # single-channel
        return image_normalized[0]  # type: ignore[no-any-return]

    # multi-channel
    # Pad with a blank plane or crop to 3 channels
    if s[0] == 2:
        image_normalized = np.concatenate(
            [image_normalized, np.zeros((1, s[1], s[2]), dtype=np.uint8)]
        )
    else:
        image_normalized = image_normalized[0:3]
    # (C,Y,X) -> (M,N,3)
    return image_normalized.transpose((1, 2, 0))
