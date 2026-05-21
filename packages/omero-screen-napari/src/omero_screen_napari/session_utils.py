"""Utilities for loading and processing annotation session data from NPY files.

This module provides reusable functions for loading saved training data,
applying masks to image crops, and preparing images for display in napari.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omero_screen.config import get_logger

from omero_screen_napari.gallery_api import (
    draw_contours,
    fill_missing_channels,
)

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData

logger = get_logger(__name__)


def parse_npy_file(
    file_path: Path,
    omero_data: "OmeroData",
    user_data: "UserData | None" = None,
) -> None:
    """Load and parse a saved NPY file into OmeroData.

    Args:
        file_path: Path to the NPY file containing saved training data
        omero_data: OmeroData instance to populate with loaded data
        user_data: UserData instance for processing settings (channels, contour)

    Raises:
        FileNotFoundError: If the NPY file doesn't exist
        ValueError: If the NPY file has invalid structure
        Exception: For other loading errors

    Notes:
        Expected NPY structure:
        {
            "data": (crops, labels),  # tuple of (images, masks)
            "target": classes  # array of class labels
        }
    """
    if not file_path.exists():
        error_msg = f"NPY file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Load NPY data
        data = np.load(file_path, allow_pickle=True).item()

        # Validate structure
        if "data" not in data or "target" not in data:
            error_msg = (
                "Invalid NPY structure: missing 'data' or 'target' keys"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Populate OmeroData with crops and labels
        omero_data.selected_crops, omero_data.selected_labels = data["data"]
        omero_data.selected_classes = data["target"]

        logger.info(
            f"Loaded {len(omero_data.selected_crops)} crops from {file_path.name}"
        )

        # Apply masks and process images if user_data available
        if user_data is not None:
            if user_data.no_background:
                masked_images = apply_masks_to_crops(omero_data)
            else:
                masked_images = list(omero_data.selected_crops)

            # Resolve channels by position-packed RGB. Empties are
            # stripped at save time, so user_data.channels is the list
            # of non-empty names; fill_missing_channels then assigns
            # [R=ch0, G=ch1, B=ch2 or 0].
            channels_in_use = [ch for ch in user_data.channels if ch]
            n_selected = len(channels_in_use)
            n_in_crop = masked_images[0].shape[-1] if masked_images else 0
            channels_int: list[int] = []

            if n_in_crop == n_selected:
                # New format: crops already contain only the selected
                # channels in sequential order — use 0..n-1 for display.
                channels_int = list(range(n_selected))
                logger.info(
                    "Crops already channel-extracted (%d channels); using "
                    "sequential indices",
                    n_selected,
                )
            else:
                # Old format: crops contain all OMERO channels — resolve
                # names to indices via channel_data.
                for ch in channels_in_use:
                    try:
                        channels_int.append(int(ch))
                        continue
                    except ValueError:
                        pass
                    val = omero_data.channel_data.get(ch)
                    if val is not None:
                        try:
                            channels_int.append(int(float(val)))
                        except ValueError:
                            logger.warning(
                                f"Channel index '{val}' for '{ch}' is not a valid number."
                            )
                    else:
                        logger.warning(
                            f"Channel '{ch}' not found in channel_data map."
                        )

                if not channels_int and masked_images:
                    n_requested = len(channels_in_use)
                    if n_requested <= 1:
                        logger.info(
                            "Could not resolve channel names; averaging all "
                            "channels for grayscale display"
                        )
                    else:
                        num_available = (
                            masked_images[0].shape[2]
                            if len(masked_images[0].shape) > 2
                            else 1
                        )
                        channels_int = list(
                            range(min(n_requested, num_available))
                        )
                        logger.info(
                            f"Could not resolve channel names, using first "
                            f"{len(channels_int)} of {num_available} channels"
                        )

            if channels_int:
                processed_masked_images = [
                    fill_missing_channels(img, channels_int)
                    for img in masked_images
                ]
            else:
                # Grayscale fallback: average all channels → (H, W, 1)
                processed_masked_images = [
                    np.mean(img, axis=-1, keepdims=True).astype(img.dtype)
                    for img in masked_images
                ]

            # Add contours if requested
            if user_data.contour and omero_data.selected_labels is not None:
                processed_masked_images = [
                    draw_contours(img, label)
                    for img, label in zip(
                        processed_masked_images,
                        omero_data.selected_labels,
                        strict=False,
                    )
                ]

            omero_data.selected_images = processed_masked_images
            logger.info(f"Processed {len(omero_data.selected_images)} images")

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error loading NPY file: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def apply_masks_to_crops(
    omero_data: "OmeroData",
) -> list[np.ndarray[Any, Any]]:
    """Apply masks to image crops, nullifying pixels outside the mask.

    This function takes color images and corresponding binary masks, then sets
    all pixels outside the mask region to zero. This helps focus on the segmented
    region of interest.

    Args:
        omero_data: OmeroData instance containing selected_crops and selected_labels

    Returns:
        List of masked images with same shape as input crops

    Notes:
        - Images expected in shape (H, W, C) where C is number of channels
        - Masks expected in shape (H, W) as binary masks
        - Mask is expanded to match image channels before application
        - Returns empty list if crops or labels are None
    """
    masked_images = []

    if omero_data.selected_crops is None or omero_data.selected_labels is None:
        logger.warning("No crops or labels available for masking")
        return []

    for image, mask in zip(
        omero_data.selected_crops,
        omero_data.selected_labels,
        strict=False,
    ):
        # Ensure mask is 2D (H, W) before expanding to match image channels
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        # Ensure the mask is expanded to match the image's channels
        expanded_mask = (
            np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2) > 0
        )
        # Apply the expanded mask to the image
        masked_image = np.where(expanded_mask, image, 0)
        masked_images.append(masked_image)

    logger.debug(f"Applied masks to {len(masked_images)} images")
    return masked_images
