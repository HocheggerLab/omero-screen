"""Direct OMERO data loader for annotation sessions.

This module provides functionality to load cell crops directly from OMERO
for annotation without requiring welldata_widget pre-loading.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from omero_utils import omero_connect

from omero_screen_napari.gallery_api import (
    calculate_crop_coordinates,
    draw_contours,
    fill_missing_channels,
    pad_region,
)
from omero_screen_napari.session_utils import apply_masks_to_crops

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData

logger = get_logger(__name__)


def _parse_image_input(image_input: str, num_images: int) -> list[int]:
    """Parse an image input string into a list of image indices.

    Supports the same formats as the welldata widget:
    - ``"All"`` (case-insensitive) → all indices ``[0 .. num_images-1]``
    - ``"3-5"`` → ``[3, 4, 5]``
    - ``"0, 1, 2"`` → ``[0, 1, 2]``
    - ``"0"`` → ``[0]``

    Args:
        image_input: User-supplied image selection string.
        num_images: Total number of images in the well.

    Returns:
        Sorted list of 0-based image indices.

    Raises:
        ValueError: If the format is invalid or indices are out of range.
    """
    index = image_input.strip()

    if index.lower() == "all":
        return list(range(num_images))

    if not re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index):
        raise ValueError(
            f"Image input '{index}' doesn't match any of the expected "
            "patterns 'All', '0', '0, 1, 2', or '3-5'."
        )

    if "-" in index and "," not in index:
        start, end = map(int, index.split("-"))
        indices = list(range(start, end + 1))
    elif "," in index:
        indices = [int(x.strip()) for x in index.split(",")]
    else:
        indices = [int(index)]

    # Validate all indices are in range
    for idx in indices:
        if idx < 0 or idx >= num_images:
            raise ValueError(
                f"Image index {idx} out of range (well has {num_images} "
                f"images, use 0-{num_images - 1})."
            )

    return sorted(set(indices))


@omero_connect
def load_crops_from_omero(
    plate_id: int,
    well_id: str,
    image_input: str,
    classifier_name: str,
    omero_data: "OmeroData",
    user_data: "UserData",
    conn: BlitzGateway | None = None,
) -> tuple[bool, str]:
    """Load cell crops directly from OMERO for annotation.

    This bypasses welldata_widget by fetching only the needed image(s)
    and generating crops on-demand.

    Args:
        plate_id: OMERO plate ID
        well_id: Well position (e.g., "A1")
        image_input: Image selection string (e.g. "All", "0", "0, 1", "3-5")
        classifier_name: Name of classifier (for metadata)
        omero_data: OmeroData instance to populate
        user_data: UserData instance for settings
        conn: OMERO connection (injected by decorator)

    Returns:
        (success: bool, message: str)
    """
    if conn is None:
        return False, "OMERO connection failed"

    try:
        # Step 1: Load classifier metadata
        logger.info(f"Loading metadata for classifier: {classifier_name}")
        metadata_path = (
            Path.home()
            / "omeroscreen_trainingdata"
            / classifier_name
            / "metadata.json"
        )

        if not metadata_path.exists():
            return False, f"Metadata file not found: {metadata_path}"

        with metadata_path.open() as f:
            metadata = json.load(f)

        user_data_dict = metadata["user_data"]

        # Step 2: Fetch plate and well from OMERO
        logger.info(f"Fetching plate {plate_id} and well {well_id} from OMERO")
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            return False, f"Plate {plate_id} not found in OMERO"

        well = None
        for well_obj in plate.listChildren():
            if well_obj.getWellPos() == well_id:
                well = well_obj
                break

        if not well:
            return False, f"Well {well_id} not found in plate {plate_id}"

        # Step 3: Parse image input string into indices
        num_images = well.countWellSample()
        try:
            image_indices = _parse_image_input(image_input, num_images)
        except ValueError as exc:
            return False, str(exc)

        logger.info(
            f"Parsed image input '{image_input}' → indices {image_indices}"
        )

        # Get timepoint from metadata
        timepoint = user_data_dict.get("timepoint", 0)
        crop_size = user_data_dict["crop_size"]
        segmentation = user_data_dict.get("segmentation", "nucleus")

        # Step 3b: Load flatfield correction (once per plate)
        logger.info(f"Loading flatfield correction for plate {plate_id}")
        ff_success, flatfield = _load_flatfield_correction(conn, plate_id)

        # Step 4: Loop over images, accumulate crops
        all_crops: list[np.ndarray[Any, Any]] = []
        all_crop_labels: list[np.ndarray[Any, Any]] = []
        all_image_ids: list[int] = []

        for img_idx in image_indices:
            image = well.getImage(img_idx)
            if not image:
                logger.warning(f"Image at index {img_idx} not found, skipping")
                continue

            image_id = image.getId()
            logger.info(
                f"Processing image index {img_idx} (OMERO ID: {image_id})"
            )
            all_image_ids.append(image_id)

            # Load pixels
            pixels = image.getPrimaryPixels()
            size_c = image.getSizeC()
            size_y = image.getSizeY()
            size_x = image.getSizeX()

            image_array = np.zeros((size_y, size_x, size_c), dtype=np.float32)
            for c in range(size_c):
                plane = pixels.getPlane(0, c, timepoint)
                image_array[..., c] = np.array(plane)

            # Apply flatfield correction
            if ff_success and flatfield is not None:
                image_array = image_array / flatfield

            # Load segmentation masks
            success, masks = _load_segmentation_masks(
                conn, plate_id, image_id, timepoint
            )
            if not success:
                logger.warning(
                    f"Segmentation masks not found for image {image_id}, "
                    f"skipping: {masks}"
                )
                continue

            # Load centroids
            success, centroids = _load_centroids_from_cellview(
                plate_id, well_id, image_id, segmentation
            )
            if not success:
                logger.warning(
                    f"Centroids not found for image {image_id}, "
                    f"skipping: {centroids}"
                )
                continue

            logger.info(f"Image {image_id}: {len(centroids)} centroids found")

            # Generate crops
            crops, crop_labels = _generate_crops(
                image_array, masks, centroids, crop_size, segmentation
            )
            all_crops.extend(crops)
            all_crop_labels.extend(crop_labels)

        if not all_crops:
            return (
                False,
                "No crops could be generated from any of the selected images",
            )

        logger.info(
            f"Total crops across {len(image_indices)} image(s): {len(all_crops)}"
        )

        # Step 5: Limit to gallery-sized subset
        n_crops = metadata.get("n_crops", 0)
        if n_crops <= 0:
            rows = user_data_dict.get("rows", 0)
            columns = user_data_dict.get("columns", 0)
            if rows > 0 and columns > 0:
                n_crops = rows * columns
        if n_crops > 0 and len(all_crops) > n_crops:
            import random

            indices = random.sample(range(len(all_crops)), n_crops)
            all_crops = [all_crops[i] for i in indices]
            all_crop_labels = [all_crop_labels[i] for i in indices]
            logger.info(
                f"Selected {n_crops} random crops from {len(all_crops) + (len(indices) - n_crops)} available"
            )

        # Step 6: Populate omero_data
        omero_data.selected_crops = all_crops
        omero_data.selected_labels = all_crop_labels
        omero_data.plate_id = plate_id
        omero_data.well_pos_list = [well_id]
        omero_data.image_input = image_input
        omero_data.image_ids = all_image_ids

        # Initialize class assignments as "unassigned"
        omero_data.selected_classes = ["unassigned"] * len(all_crops)

        # Step 7: Apply masks only if no_background is enabled
        no_background = user_data_dict.get("no_background", True)
        if no_background:
            logger.info(
                f"Applying masks to {len(all_crops)} images (no_background=True)"
            )
            masked_images = apply_masks_to_crops(omero_data)
        else:
            logger.info(
                f"Keeping background for {len(all_crops)} images (no_background=False)"
            )
            masked_images = list(omero_data.selected_crops)

        # Convert channel names to indices, matching session_utils approach
        channels = user_data_dict.get("channels", [])
        channel_data = metadata.get("channel_data", {})
        channel_indices = []
        for ch in channels:
            try:
                channel_indices.append(int(ch))
                continue
            except (ValueError, TypeError):
                pass
            val = channel_data.get(ch)
            if val is not None:
                try:
                    channel_indices.append(int(float(val)))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Channel index '{val}' for '{ch}' is not a valid number."
                    )
            else:
                logger.warning(
                    f"Channel '{ch}' not found in channel_data map."
                )

        # If channel names could not be resolved, fall back gracefully
        if not channel_indices and masked_images:
            n_requested = len(channels)
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
                channel_indices = list(range(min(n_requested, num_available)))
                logger.info(
                    f"Could not resolve channel names, using first "
                    f"{len(channel_indices)} of {num_available} channels"
                )

        if channel_indices:
            processed_images = [
                fill_missing_channels(img, channel_indices)
                for img in masked_images
            ]
        else:
            processed_images = [
                np.mean(img, axis=-1, keepdims=True).astype(img.dtype)
                for img in masked_images
            ]
        logger.info(
            f"After channel processing: {len(processed_images)} images"
        )

        # Add contours if requested
        if user_data_dict.get("contour", False):
            processed_images = [
                draw_contours(img, lbl)
                for img, lbl in zip(
                    processed_images, all_crop_labels, strict=False
                )
            ]
            logger.info(
                f"After contour drawing: {len(processed_images)} images"
            )

        omero_data.selected_images = processed_images
        logger.info(
            f"Final: Set {len(omero_data.selected_images)} images in omero_data"
        )

        # Update user_data with loaded metadata
        user_data.populate_from_dict(user_data_dict)
        user_data.well = well_id

        logger.info(
            f"Successfully loaded {len(processed_images)} images for annotation"
        )
        return (
            True,
            f"Loaded {len(processed_images)} cells from images '{image_input}'",
        )

    except Exception as e:
        logger.exception(f"Failed to load crops from OMERO: {e}")
        return False, f"Error loading data: {str(e)}"


def _load_segmentation_masks(
    conn: BlitzGateway, plate_id: int, image_id: int, timepoint: int
) -> tuple[bool, Any]:
    """Load segmentation masks from OMERO dataset.

    Args:
        conn: OMERO connection
        plate_id: Plate ID
        image_id: Image ID
        timepoint: Timepoint index

    Returns:
        (success: bool, masks_array or error_message)
    """
    try:
        # Find dataset for this plate
        dataset_name = str(plate_id)

        # Search for dataset by name
        dataset = None
        for ds in conn.getObjects("Dataset"):
            if ds.getName() == dataset_name:
                dataset = ds
                break

        if not dataset:
            return False, f"Dataset for plate {plate_id} not found"

        # Find segmentation image
        seg_image_name = f"{image_id}_segmentation"
        seg_image = None
        for img in dataset.listChildren():
            if img.getName() == seg_image_name:
                seg_image = img
                break

        if not seg_image:
            return False, f"Segmentation masks not found for image {image_id}"

        # Load segmentation pixels
        pixels = seg_image.getPrimaryPixels()
        size_y = seg_image.getSizeY()
        size_x = seg_image.getSizeX()
        size_c = seg_image.getSizeC()  # Usually 2 channels: nucleus, cell

        # Load masks for the specific timepoint
        masks = np.zeros((size_y, size_x, size_c), dtype=np.int32)
        for c in range(size_c):
            plane = pixels.getPlane(0, c, timepoint)
            masks[..., c] = np.array(plane, dtype=np.int32)

        logger.info(f"Loaded segmentation masks with shape: {masks.shape}")
        return True, masks

    except Exception as e:
        logger.exception(f"Failed to load segmentation masks: {e}")
        return False, f"Error loading masks: {str(e)}"


def _load_flatfield_correction(
    conn: BlitzGateway, plate_id: int
) -> tuple[bool, Any]:
    """Load flatfield correction masks from OMERO dataset.

    Args:
        conn: OMERO connection
        plate_id: Plate ID

    Returns:
        (success: bool, flatfield_array or None)
    """
    try:
        # Find dataset for this plate
        dataset_name = str(plate_id)
        dataset = None
        for ds in conn.getObjects("Dataset"):
            if ds.getName() == dataset_name:
                dataset = ds
                break

        if not dataset:
            logger.warning(f"Dataset for plate {plate_id} not found")
            return False, None

        # Find flatfield image
        ff_image = None
        for img in dataset.listChildren():
            if "flatfield" in img.getName().lower():
                ff_image = img
                break

        if not ff_image:
            logger.warning("No flatfield correction image found")
            return False, None

        # Load flatfield pixels
        pixels = ff_image.getPrimaryPixels()
        size_y = ff_image.getSizeY()
        size_x = ff_image.getSizeX()
        size_c = ff_image.getSizeC()

        flatfield = np.zeros((size_y, size_x, size_c), dtype=np.float32)
        for c in range(size_c):
            plane = pixels.getPlane(0, c, 0)
            flatfield[..., c] = np.array(plane, dtype=np.float32)

        logger.info(
            f"Loaded flatfield correction with shape: {flatfield.shape}"
        )
        return True, flatfield

    except Exception as e:
        logger.warning(f"Failed to load flatfield correction: {e}")
        return False, None


def _load_centroids_from_cellview(
    plate_id: int, well_id: str, image_id: int, segmentation: str
) -> tuple[bool, Any]:
    """Load cell centroids from CellView database.

    Args:
        plate_id: Plate ID
        well_id: Well position (e.g., "A1")
        image_id: Image ID (actual OMERO image ID, not index)
        segmentation: "nucleus" or "cell"

    Returns:
        (success: bool, centroids_array or error_message)
    """
    try:
        from cellview.api import cellview_load_data

        # Load plate data from CellView (returns Pandas DataFrame)
        df, _ = cellview_load_data(plate_id)

        if df is None or df.empty:
            return False, f"No data found in CellView for plate {plate_id}"

        # Filter to specific well and image (Pandas syntax)
        filtered = df[(df["well"] == well_id) & (df["image_id"] == image_id)]

        if filtered.empty:
            return False, f"No data found for well {well_id}, image {image_id}"

        # Extract centroids based on segmentation type
        if segmentation == "nucleus":
            centroid_cols = ["centroid-0-nuc", "centroid-1-nuc"]
        else:
            centroid_cols = ["centroid-0-cell", "centroid-1-cell"]

        # Check if columns exist
        for col in centroid_cols:
            if col not in filtered.columns:
                return (
                    False,
                    f"Centroid column '{col}' not found in CellView data",
                )

        # Extract centroids as numpy array
        centroids = filtered[centroid_cols].to_numpy()

        logger.info(f"Loaded {len(centroids)} centroids from CellView")
        return True, centroids

    except ImportError:
        return False, "CellView package not available"
    except Exception as e:
        logger.exception(f"Failed to load centroids from CellView: {e}")
        return False, f"Error loading centroids: {str(e)}"


def _generate_crops(
    image: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    centroids: np.ndarray[Any, Any],
    crop_size: int,
    segmentation: str,
) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
    """Generate cropped images and label masks around centroids.

    Args:
        image: Full image array (H, W, C)
        labels: Full label masks (H, W, C) with channels [nucleus, cell]
        centroids: Array of centroids (N, 2) with [row, col]
        crop_size: Size of square crop
        segmentation: "nucleus" or "cell"

    Returns:
        Tuple of (crops_list, labels_list)
    """
    crops = []
    crop_labels = []

    # Select appropriate label channel
    if labels.ndim == 3 and labels.shape[-1] >= 2:
        channel_idx = 0 if segmentation == "nucleus" else 1
        label_array = labels[..., channel_idx]
    else:
        label_array = np.squeeze(labels)

    for centroid in centroids:
        row, col = int(centroid[0]), int(centroid[1])

        # Calculate crop bounds
        row_start, row_end = calculate_crop_coordinates(
            row, image.shape[0], crop_size
        )
        col_start, col_end = calculate_crop_coordinates(
            col, image.shape[1], crop_size
        )

        # Extract crop
        crop = image[row_start:row_end, col_start:col_end, :]
        label_crop = label_array[row_start:row_end, col_start:col_end]

        # Pad if needed
        if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
            crop, label_crop = pad_region(crop, label_crop, crop_size)

        # Only add if label is non-empty
        if np.any(label_crop):
            crops.append(crop)
            crop_labels.append(label_crop)

    logger.info(f"Generated {len(crops)} valid crops")
    return crops, crop_labels
