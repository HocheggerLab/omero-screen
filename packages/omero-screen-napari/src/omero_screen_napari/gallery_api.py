import datetime
import random
from ast import literal_eval
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, TypeVar

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from omero_utils import omero_connect
from skimage import exposure
from skimage.measure import find_contours, label, regionprops

from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData, get_dataset_id
from omero_screen_napari.utils import save_fig
from omero_screen_napari.welldata_api import well_image_parser

logger = get_logger(__name__)


def show_gallery(
    omero_data: OmeroData,
    user_data: UserData,
    classifier: bool = False,
    excluded_centroids: set[tuple[int, int, int]] | None = None,
) -> None:
    if user_data.reload or omero_data.cropped_images == []:
        cropped_image_parser = CroppedImageParser(
            omero_data, user_data, excluded_centroids=excluded_centroids
        )
        cropped_image_parser.parse_crops()

    random_image_parser = RandomImageParser(omero_data, user_data, classifier)
    random_image_parser.parse_random_images()

    # CroppedImageParser scales images to the range for the type using the mean channel image min/max.
    # Convert the random images to float64 for the gallery using a consistent range for each sample.
    if omero_data.selected_images and np.issubdtype(
        omero_data.selected_images[0].dtype, np.integer
    ):
        info = np.iinfo(omero_data.cropped_images[0].dtype)
        omero_data.selected_images = [
            exposure.rescale_intensity(
                x.astype(np.float64), in_range=(info.min, info.max)
            )  # type: ignore
            for x in omero_data.selected_images
        ]

    if not omero_data.selected_images:
        logger.warning("No images selected for gallery. Checking crops...")
        if not omero_data.cropped_images:
            logger.warning(
                "No crops available. Check segmentation/well parameters."
            )
        return

    gallery_parser = ParseGallery(omero_data, user_data)
    gallery_parser.plot_gallery()


# --------------------------Gallery Data Generators---------------------------


class CroppedImageParser:
    def __init__(
        self,
        omero_data: OmeroData,
        user_data: UserData,
        excluded_centroids: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._excluded_centroids: set[tuple[int, int, int]] = (
            excluded_centroids or set()
        )
        self._well_data: pl.DataFrame = pl.DataFrame()
        self._df: pl.DataFrame = pl.DataFrame()
        self._images: np.ndarray[Any, Any] = np.empty((0,))
        self._labels: np.ndarray[Any, Any] = np.empty((0,))
        self._ids: list[int] = []
        self._image_ids: list[int] = []
        self._centroids_row: list[int] = []
        self._centroids_col: list[int] = []
        self._cropped_images: list[np.ndarray[Any, Any]] = []
        self._cropped_labels: list[np.ndarray[Any, Any]] = []
        self._cell_meta: list[dict[str, Any]] = []
        self._current_image_id: int = 0

    def parse_crops(self) -> None:
        logger.info(
            "Parsing crops for well %s using segmentation %s",
            self._user_data.well,
            self._user_data.segmentation,
        )
        self._get_well_data()
        self._prepare_data_for_cropping()

        # Verify images are loaded
        if not self._image_ids:
            logger.warning(
                "No images found for well %s in metadata.",
                self._user_data.well,
            )
            return

        # Check if the first image ID is in the loaded images.
        # (Assuming if one is missing, likely all are, or at least the well isn't fully loaded)
        if self._image_ids[0] not in self._omero_data.image_ids:
            msg = (
                f"Images for well {self._user_data.well} are not loaded. "
                f"Please load the well data using the 'Well Data' widget first."
            )
            logger.error(msg)
            # We can't proceed without pixels
            return  # Or raise ValueError(msg) - return is safer for GUI to not crash hard

        self._crop_data()
        self._remove_duplicate_images()
        self._omero_data.cropped_images = self._cropped_images
        self._omero_data.cropped_labels = self._cropped_labels
        self._omero_data.cropped_cell_meta = self._cell_meta
        logger.info("Generated %d crops", len(self._cropped_images))

    def _get_well_data(self) -> None:
        schema = self._omero_data.plate_data.collect_schema()
        if len(schema) == 0:
            raise ValueError(
                f"Plate {self._omero_data.plate_id} has no data in CellView. "
                "Please import the plate into CellView before using the gallery."
            )
        self._well_data = self._omero_data.plate_data.filter(
            pl.col("well") == self._user_data.well
        ).collect()

    def _prepare_data_for_cropping(self) -> None:
        """Prepare separate lists for each image with its associated centroids and ids."""
        df = self._select_cellcycledata(self._well_data)
        df = self._select_classifierdata(df)

        # Filter: Only keep images that are actually loaded in omero_data
        loaded_ids = set(self._omero_data.image_ids)
        if not loaded_ids:
            logger.warning(
                "No images loaded in OmeroData. Cannot process crops."
            )
            df = df.filter(pl.col("image_id").is_in([]))  # Empty filter
        else:
            # Check for potential mismatch before filtering for validation logging
            metadata_ids = set(df["image_id"].unique().to_list())
            common_ids = metadata_ids.intersection(loaded_ids)

            if not common_ids:
                logger.warning(
                    "No intersection between requested well (%s) images and loaded images. Loaded IDs: %s...",
                    self._user_data.well,
                    list(loaded_ids)[:5],
                )
            elif len(common_ids) < len(metadata_ids):
                logger.info(
                    "Processing %d images (subset of well) that are loaded in memory.",
                    len(common_ids),
                )

            df = df.filter(pl.col("image_id").is_in(common_ids))

        # Check for duplicate indices
        self._check_duplicate_indices(df)

        self._df = df  # kept for per-image centroid lookup in _crop_data
        self._image_ids = df["image_id"].unique().to_list()
        # Sort image IDs to match the order in the OmeroData object
        self._image_ids.sort()
        logger.debug("Image IDs: %s", self._image_ids)

    def _get_centroid_columns(self) -> tuple[str, str, str]:
        if self._user_data.segmentation == "nucleus":
            return "centroid-0-nuc", "centroid-1-nuc", "label"
        return "centroid-0-cell", "centroid-1-cell", "Cyto_ID"

    def _select_image_data(self, image_id: int) -> None:
        try:
            index = self._omero_data.image_ids.index(image_id)
            self._images = self._omero_data.images[index]
            self._labels = self._omero_data.labels[index]
        except ValueError as e:
            logger.error(
                "Image ID %s not found in loaded images: %s", image_id, e
            )
            raise

    def _check_duplicate_indices(self, df: pl.DataFrame) -> None:
        c0, c1, _ = self._get_centroid_columns()
        # Check for exact duplicates where both coordinates match
        duplicates_count = df.select([c0, c1]).is_duplicated().sum()

        if duplicates_count > 0:
            logger.warning(
                "Found %d cells with identical coordinates (%s, %s). "
                "This might indicate segmentation issues but will not stop gallery generation.",
                duplicates_count,
                c0,
                c1,
            )

    def _select_cellcycledata(self, df: pl.DataFrame) -> pl.DataFrame:
        cellcycle = self._user_data.cellcycle
        if cellcycle == "All":
            return df

        # Check if column exists
        if "cell_cycle" not in df.columns:
            logger.error(
                "'cell_cycle' column not found in data. Available columns: %s",
                df.columns[:10],
            )
            raise ValueError(
                "Cell cycle data not found. Cannot filter by cell cycle."
            )

        unique_phases = df["cell_cycle"].unique().to_list()
        if cellcycle in unique_phases:
            return df.filter(df["cell_cycle"] == cellcycle)
        logger.error(
            "Invalid cell cycle phase: %s. Available phases: %s",
            cellcycle,
            unique_phases,
        )
        raise ValueError(
            f"Invalid cell cycle phase: {cellcycle}. Available phases in this well: {unique_phases}"
        )

    def _select_classifierdata(self, df: pl.DataFrame) -> pl.DataFrame:
        value = self._user_data.classifier_filter.strip()
        if not value:
            return df
        classifier_cols = [
            c for c in df.columns if c.startswith("classifier_")
        ]
        for col in classifier_cols:
            if value in df[col].unique().to_list():
                return df.filter(pl.col(col) == value)
        logger.warning(
            "Classifier filter '%s' not found in any classifier column: %s",
            value,
            classifier_cols,
        )
        return df

    def _select_centroids(
        self, df: pl.DataFrame
    ) -> tuple[list[int], list[int], list[int]]:
        c0, c1, id_col = self._get_centroid_columns()
        if self._user_data.segmentation == "nucleus":
            # labels may be average IDs (float64) or tuple of IDs. Convert strings if present.
            ids = df[id_col]
            if ids.dtype == pl.datatypes.String:
                ids = ids.map_elements(literal_eval, return_dtype=pl.Object)
            return (
                df[c0].to_list(),
                df[c1].to_list(),
                ids.to_list(),
            )
        unique_centroids_df = df.unique(subset=[c0, c1])
        return (
            unique_centroids_df[c0].to_list(),
            unique_centroids_df[c1].to_list(),
            unique_centroids_df[id_col].to_list(),
        )

    def _crop_data(self) -> None:
        for image_id in self._image_ids:
            if image_id not in self._omero_data.image_ids:
                logger.warning(
                    "Image ID %s not loaded in OmeroData. Skipping.", image_id
                )
                continue
            image_df = self._df.filter(pl.col("image_id") == image_id)
            # Exclude cells already used in a previous session for this classifier
            if self._excluded_centroids:
                c0, c1, _ = self._get_centroid_columns()
                excluded = self._excluded_centroids
                cur_id = image_id
                image_df = image_df.filter(
                    ~pl.struct([c0, c1]).map_elements(
                        lambda s, _c0=c0, _c1=c1, _id=cur_id, _ex=excluded: (
                            int(s[_c0]),
                            int(s[_c1]),
                            _id,
                        )
                        in _ex,
                        return_dtype=pl.Boolean,
                    )
                )
            self._centroids_row, self._centroids_col, self._ids = (
                self._select_centroids(image_df)
            )
            self._current_image_id = image_id
            self._process_image_for_cropping(image_id)

    def _process_image_for_cropping(self, image_id: int) -> None:
        """Checks if image and lable data are available and applies cropping and processing
        via _crop_and_process_image function
        """
        self._select_image_data(image_id)

        if self._images is None or self._labels is None:
            logger.error("No images or labels to use for cropping galleries")
            raise ValueError(
                "No images or labels to use for cropping galleries"
            )
        logger.debug(
            "Shape of selected images for cropping  is %s", self._images.shape
        )
        logger.debug(
            "Shape of selected labels for cropping  is %s", self._labels.shape
        )
        timepoint = self._user_data.timepoint
        if len(self._images.shape) == 5:
            current_data = self._images[timepoint, ...]
            current_labels_all = self._labels[timepoint, ...]
        elif len(self._images.shape) == 4:
            # Assuming (T, Y, X, C)
            current_data = self._images[timepoint, ...]
            current_labels_all = self._labels[timepoint, ...]
        else:
            current_data = self._images
            current_labels_all = self._labels

        # Scale the full image before cropping to ensure consistent intensity across all crops
        current_data = self._scale_full_image(current_data)

        # Select the correct label channel (0=Nucleus, 1=Cell)
        if current_labels_all.ndim == 3 and current_labels_all.shape[-1] >= 2:
            channel_idx = 0 if self._user_data.segmentation == "nucleus" else 1
            current_labels = current_labels_all[..., channel_idx]
        else:
            # Squeeze trailing dimension for single-channel labels (e.g. nucleus-only)
            current_labels = np.squeeze(current_labels_all)

        for row, col, obj_id in zip(
            self._centroids_row, self._centroids_col, self._ids, strict=False
        ):
            self._crop_and_process_image(
                current_data, current_labels, row, col, obj_id
            )
        crop_count = len(self._cropped_images)
        logger.info(
            "%d cropped images and labels have been generated for image %s",
            crop_count,
            image_id,
        )

    def _scale_full_image(
        self, image: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Scales the full image using OmeroData intensities or image min/max."""
        scaled_image = np.zeros_like(image, dtype=np.float32)
        n_channels = image.shape[-1]

        for i in range(n_channels):
            # i is the channel index
            if i in self._omero_data.intensities:
                min_i, max_i = self._omero_data.intensities[i]
                img_slice = image[..., i]
                range_i = max(max_i - min_i, 1)
                scaled = (img_slice - min_i) / range_i
                scaled = np.clip(scaled, 0, 1)
                scaled_image[..., i] = scaled
            else:
                # Fallback: scale relative to the *whole image* max, not local crop max
                img_slice = image[..., i]
                max_val = np.max(img_slice)
                if max_val > 0:
                    scaled_image[..., i] = img_slice / max_val
                else:
                    scaled_image[..., i] = img_slice
        return scaled_image

    def _crop_and_process_image(
        self,
        image: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        row: int,
        col: int,
        obj_id: int | list[int] | float,
    ) -> None:
        """Performs crop cleans mask, checks if mask is present before appending data to cropped_images and cropped_labels lists
        Uses helper functions crop_region, pad_region, erase_masks
        """
        cropped_image, cropped_label = crop_region(
            image, labels, row, col, self._user_data.crop_size
        )

        if cropped_image.shape != (
            self._user_data.crop_size,
            self._user_data.crop_size,
        ):
            cropped_image, cropped_label = pad_region(
                cropped_image, cropped_label, self._user_data.crop_size
            )

        # Support multiple labels in the mask
        for corrected_cropped_label in erase_masks(
            cropped_label.copy(), obj_id
        ):
            if np.any(
                corrected_cropped_label
            ):  # Check if the label is effectively empty
                self._cropped_images.append(cropped_image)
                self._cropped_labels.append(corrected_cropped_label)
                self._cell_meta.append(
                    {
                        "centroid_row": int(row),
                        "centroid_col": int(col),
                        "image_id": self._current_image_id,
                    }
                )

    def _remove_duplicate_images(self) -> None:
        """Remove duplicate images and their corresponding labels from the dataset."""
        unique_images = []
        unique_labels = []
        seen_images = set()
        initial_count = len(self._images)

        for image, unique_label in zip(
            self._images, self._labels, strict=False
        ):
            image_tuple = tuple(
                image.flatten()
            )  # Convert image to a hashable type
            if image_tuple not in seen_images:
                seen_images.add(image_tuple)
                unique_images.append(image)
                unique_labels.append(unique_label)

        self._images = np.array(unique_images)
        self._labels = np.array(unique_labels)

        # Calculate and log the number of removed images
        final_count = len(self._images)
        removed_count = initial_count - final_count
        logger.info("Removed %d duplicate images.", removed_count)


# helper functions for cropping images
def crop_region(
    current_data: np.ndarray[Any, Any],
    current_labels: np.ndarray[Any, Any],
    centroid_row: int,
    centroid_col: int,
    crop_size: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Crop a region around the centroid of a segmented object."""
    # Calculate crop coordinates
    crop_row_start, crop_row_end = calculate_crop_coordinates(
        centroid_row, current_data.shape[-3], crop_size
    )
    crop_col_start, crop_col_end = calculate_crop_coordinates(
        centroid_col, current_data.shape[-2], crop_size
    )

    # Crop the region from the image
    cropped_region = current_data[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end, :
    ]
    # Crop the corresponding region from the segmentation labels
    cropped_label = current_labels[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end
    ]

    return cropped_region, cropped_label


def calculate_crop_coordinates(
    centroid: int, max_length: int, crop_size: int
) -> tuple[int, int]:
    """Calculate start and end points for cropping around a centroid."""
    start = int(max(0, centroid - crop_size // 2))
    end = int(min(max_length, centroid + crop_size // 2))
    return start, end


def pad_region(
    cropped_region: np.ndarray[Any, Any],
    cropped_label: np.ndarray[Any, Any],
    crop_size: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Pad the cropped region and label to the desired crop size."""
    pad_row = crop_size - cropped_region.shape[0]
    pad_col = crop_size - cropped_region.shape[1]
    # Pad centrally
    cropped_region = np.pad(
        cropped_region,
        (
            (pad_row // 2, pad_row - pad_row // 2),
            (pad_col // 2, pad_col - pad_col // 2),
            (0, 0),
        ),
        mode="constant",
    )
    cropped_label = np.pad(
        cropped_label,
        (
            (pad_row // 2, pad_row - pad_row // 2),
            (pad_col // 2, pad_col - pad_col // 2),
        ),
        mode="constant",
    )
    return cropped_region, cropped_label


def erase_masks(
    cropped_label: np.ndarray[Any, Any],
    obj_id: int | list[int] | float,
    tol: int = 10,
) -> list[np.ndarray[Any, Any]]:
    """Extracts label regions from the cropped_label. Uses the ID. If this is iterable extract each ID in the iteration.
    If a scalar integer type then extract the ID. Otherwise ignore (legacy IDs could be a scalar float type)
    and extract labels within a distance tolerance of the centroid.

    Note that an integer type is detected by checking int(id) == id. This will not detect cases where multiple
    IDs have been averaged to a single integral value. This is a minority case and allows supporting legacy
    data with float IDs representing a single label.
    """
    if isinstance(obj_id, int | float) and int(obj_id) == obj_id:
        cropped_label[cropped_label != obj_id] = 0
        return [cropped_label]

    labels = []
    if isinstance(obj_id, Iterable):
        for i in obj_id:
            label_mask = cropped_label.copy()
            label_mask[cropped_label != i] = 0
            labels.append(label_mask)
        return labels

    center_row, center_col = np.array(cropped_label.shape) // 2

    unique_labels = np.unique(cropped_label)
    for unique_label in unique_labels:
        if unique_label == 0:  # Skip background
            continue

        binary_mask = cropped_label == unique_label

        if np.sum(binary_mask) == 0:  # Check for empty masks
            continue

        label_props = regionprops(label(binary_mask))  # type: ignore

        if len(label_props) == 1:
            cropped_centroid_row, cropped_centroid_col = label_props[
                0
            ].centroid

            # Using a small tolerance value for comparing centroids
            if (
                abs(cropped_centroid_row - center_row) <= tol
                and abs(cropped_centroid_col - center_col) <= tol
            ):
                label_mask = cropped_label.copy()
                label_mask[cropped_label != unique_label] = 0
                labels.append(label_mask)

    return labels


# --------------------------Select random images for gallery--------------------


class RandomImageParser:
    def __init__(
        self, omero_data: OmeroData, user_data: UserData, classifier: bool
    ) -> None:
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._classifier: bool = classifier
        self._chosen_indices: list[int] = []  # indices of images to be used
        self._random_images: list[np.ndarray[Any, Any]] = []
        self._random_labels: list[np.ndarray[Any, Any]] = []

    def parse_random_images(self) -> None:
        self._parse_random_index()
        self._parse_random_images()
        if self._classifier:
            self._omero_data.selected_cell_meta = [
                self._omero_data.cropped_cell_meta[i]
                for i in self._chosen_indices
                if i < len(self._omero_data.cropped_cell_meta)
            ]
        # self._check_identical_arrays()
        self._omero_data.cropped_images = self._remove_chosen_crops(
            self._omero_data.cropped_images
        )
        self._omero_data.cropped_labels = self._remove_chosen_crops(
            self._omero_data.cropped_labels
        )
        self._omero_data.cropped_cell_meta = self._remove_chosen_crops(
            self._omero_data.cropped_cell_meta
        )
        if self._user_data.no_background:
            self._apply_mask_to_images()

        # Convert channel names to indices
        channel_indices = []
        for channel_name in self._user_data.channels:
            val = self._omero_data.channel_data.get(channel_name)
            if val is not None:
                try:
                    channel_indices.append(
                        int(float(val))
                    )  # Handle '3.0' strings if any
                except ValueError:
                    logger.error(
                        "Channel index %s for %s is not a valid number.",
                        val,
                        channel_name,
                    )
            else:
                logger.warning(
                    "Channel %s not found in channel_data map.", channel_name
                )

        # selected_crops stores only the selected channels (no RGB padding) for training data
        if self._classifier:
            self._omero_data.selected_crops = [
                img[..., channel_indices] if channel_indices else img
                for img in self._random_images
            ]

        self._random_images = [
            fill_missing_channels(img, channel_indices)
            for img in self._random_images
        ]
        if self._user_data.contour:
            self._random_images = [
                draw_contours(img, label)
                for img, label in zip(
                    self._random_images, self._random_labels, strict=False
                )
            ]
        self._omero_data.selected_images = self._random_images
        self._omero_data.selected_labels = self._random_labels

    def _parse_random_index(self) -> None:
        """Select random index to be used to choose images the gallery from the croped images and labels."""
        if self._user_data.columns == 0 and self._user_data.rows == 0:
            self._chosen_indices = list(
                range(len(self._omero_data.cropped_images))
            )
        else:
            sample_size = min(
                self._user_data.columns * self._user_data.rows,
                len(self._omero_data.cropped_images),
            )
            self._chosen_indices = random.sample(
                range(len(self._omero_data.cropped_images)), sample_size
            )

    def _parse_random_images(self) -> None:
        """Use the random_indeces to select the images and labels to be used in the gallery."""
        self._random_images = [
            self._omero_data.cropped_images[i] for i in self._chosen_indices
        ]
        self._random_labels = [
            self._omero_data.cropped_labels[i] for i in self._chosen_indices
        ]

    def _check_identical_arrays(self) -> None:
        """Saftey check to ensure that gallery images have been chosen and that no identical arrays
        are present in the chosen cells.
        """
        if not self._random_images:
            logger.error("Slelection of crops for gallery has failed")
            raise ValueError("Slelection of crops for gallery has failed")
        # Check each array with every other array in the list
        n = len(self._random_images)
        for i in range(n):
            for j in range(
                i + 1, n
            ):  # Start from i+1 to avoid comparing the same array
                if np.array_equal(
                    self._random_images[i], self._random_images[j]
                ):
                    raise ValueError(
                        "There are identical arrays in the chosen cells. Please try again."
                    )
        return  # No identical arrays found

    _T = TypeVar("_T")

    def _remove_chosen_crops(self, array_list: list[_T]) -> list[_T]:
        """Removes the chosen images and labels from the cropped_images and cropped_labels lists."""
        return [
            item
            for index, item in enumerate(array_list)
            if index not in self._chosen_indices
        ]

    def _apply_mask_to_images(self) -> None:
        """Nullify pixels in color images that don't overlap with the corresponding masks.
        Images are expected to be in the shape of (H, W, C) and masks in the shape of (H, W).
        """
        masked_images = []
        for image, mask in zip(
            self._random_images, self._random_labels, strict=False
        ):  # type: ignore
            # Ensure mask is 2D (H, W) before expanding to match image channels
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            # Ensure the mask is expanded to match the image channels
            expanded_mask = (
                np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2) > 0
            )
            # Apply the expanded mask to the image
            masked_image = np.where(expanded_mask, image, 0)
            masked_images.append(masked_image)

        self._random_images = masked_images


# Helper functions for RandomImageParser


def fill_missing_channels(
    img: np.ndarray[Any, Any], channel_indices: list[int]
) -> np.ndarray[Any, Any]:
    """
    Select and rearrange image channels for display.

    Channel mapping:
    - 1 channel: Grayscale (H, W, 1)
    - 2 channels: RGB with Red=ch0, Green=ch1, Blue=0
    - 3+ channels: RGB with Red=ch0, Green=ch1, Blue=ch2

    Args:
        img: Input image with shape (H, W, C) where C can be 1, 2, 3, or more
        channel_indices: List of channel indices to extract

    Returns:
        Image with shape (H, W, 1) for single channel or (H, W, 3) for multi-channel
    """
    empty_image = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    ch_arrays = []

    # Extract requested channels
    for idx in channel_indices:
        if idx < img.shape[-1]:
            ch_arrays.append(img[..., idx])
        else:
            ch_arrays.append(empty_image)

    # Single channel: return as (H, W, 1) for grayscale display
    if len(ch_arrays) == 1:
        return ch_arrays[0][..., np.newaxis]

    # Map to RGB [Red, Green, Blue]
    if len(ch_arrays) >= 3:
        # 3+ channels: [ch0, ch1, ch2] -> RGB
        result_img = [ch_arrays[0], ch_arrays[1], ch_arrays[2]]
    elif len(ch_arrays) == 2:
        # 2 channels: [ch0, ch1] -> [Red=ch0, Green=ch1, Blue=0]
        result_img = [ch_arrays[0], ch_arrays[1], empty_image]
    else:
        # Fallback: all black (shouldn't happen with valid input)
        result_img = [empty_image, empty_image, empty_image]

    return np.stack(result_img, axis=-1)


def draw_contours(
    img: np.ndarray[Any, Any], label: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    channel_num = img.shape[-1]
    contours = find_contours(label, 0.5)  # type: ignore
    for contour in contours:
        for coords in contour:
            x, y = coords.astype(int)
            img[x, y] = [1] * channel_num  # White color
    return img


# --------------------------------Gallery Constructor--------------------------------


class ParseGallery:
    def __init__(
        self,
        omero_data: OmeroData,
        user_data: UserData,
        show_gallery: bool = True,
    ) -> None:
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._gallery_image: np.ndarray[Any, Any] = np.empty((0,))
        self._metadata: dict[str, str] = {}
        self._show_gallery = show_gallery

    def plot_gallery(self) -> Any:
        padding_height, padding_width = self._calculate_dynamic_padding()
        self._gallery_image = self._create_gallery_image(
            padding_height, padding_width
        )
        self._parse_metadata()
        return self._build_gallery()

    def _build_gallery(self) -> Any:
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(self._user_data.channels) == 1:
            ax.imshow(self._gallery_image[..., 0], cmap="gray_r")
        else:
            ax.imshow(self._gallery_image)
        metadata_str = ", ".join(
            [f"{key}: {value}" for key, value in self._metadata.items()]
        )
        channel_list = [
            channel for channel in self._user_data.channels if channel != ""
        ]
        classifier_str = self._user_data.classifier_filter.strip() or "None"
        ax.set_title(
            f"well: {self._user_data.well}\n{metadata_str}\nchannels: {', '.join(channel_list)}, cellcycle phase: {self._user_data.cellcycle}, classifier filter: {classifier_str}, timepoint: {self._user_data.timepoint}",
            fontsize=12,
            fontweight="bold",
        )
        plt.axis("off")
        # Add scale bar
        self._add_scale_bar(ax)
        logger.info("plotting gallery")
        if self._show_gallery:
            plt.show(block=False)
        return fig

    def _create_gallery_image(
        self, padding_height: int, padding_width: int
    ) -> np.ndarray[Any, Any]:
        if not self._omero_data.selected_images:
            raise ValueError("No images selected for gallery")

        # Verify all images have the same shape
        first_shape = self._omero_data.selected_images[0].shape
        for i, img in enumerate(self._omero_data.selected_images):
            if img.shape != first_shape:
                logger.error(
                    "Image %d has shape %s, expected %s",
                    i,
                    img.shape,
                    first_shape,
                )
                raise ValueError(
                    f"All gallery images must have the same shape. "
                    f"Image {i} has shape {img.shape}, expected {first_shape}"
                )

        img_height, img_width, img_channels = first_shape
        n_row, n_col = self._user_data.rows, self._user_data.columns

        # Adjust gallery dimensions to include border padding
        gallery_height = n_row * img_height + (n_row + 1) * padding_height
        gallery_width = n_col * img_width + (n_col + 1) * padding_width

        # Create an array filled with 1.0 (white background)
        gallery_image = np.full(
            (gallery_height, gallery_width, img_channels),
            fill_value=1.0,
            dtype=np.float64,
        )

        for row in range(n_row):
            for col in range(n_col):
                idx = row * n_col + col
                if idx >= len(self._omero_data.selected_images):
                    break
                # Adjust start positions to account for the border padding
                start_row = (
                    row * (img_height + padding_height) + padding_height
                )
                end_row = start_row + img_height
                start_col = col * (img_width + padding_width) + padding_width
                end_col = start_col + img_width
                gallery_image[start_row:end_row, start_col:end_col, :] = (
                    self._omero_data.selected_images[idx]
                )

        return gallery_image

    def _calculate_dynamic_padding(self) -> tuple[int, int]:
        img_height, img_width = self._omero_data.selected_images[0].shape[:2]

        padding_height = int(img_height * 0.02)
        padding_width = int(img_width * 0.02)
        return padding_height, padding_width

    def _parse_metadata(self) -> None:
        if self._omero_data.well_pos_list:
            index_number = self._omero_data.well_pos_list.index(
                self._user_data.well
            )
            self._metadata = self._omero_data.well_metadata_list[index_number]

    def _add_scale_bar(self, ax: Any) -> None:
        gallery_height, gallery_width, _ = self._gallery_image.shape
        physical_scale_bar_length = (
            10 if self._user_data.crop_size <= 30 else 25
        )  # in microns
        if self._omero_data.pixel_size:
            scale_bar_length_in_pixels = int(
                physical_scale_bar_length / self._omero_data.pixel_size[0]
            )
        else:
            scale_bar_length_in_pixels = 100  # default fallback

        bar_height = 1
        start_x = (
            gallery_width - scale_bar_length_in_pixels - gallery_width * 0.04
        )
        start_y = gallery_height - bar_height - gallery_width * 0.01
        color = "black" if len(self._user_data.channels) == 1 else "white"
        scale_bar = patches.Rectangle(
            (start_x, start_y),
            scale_bar_length_in_pixels,
            bar_height,
            linewidth=1,
            edgecolor=color,
            facecolor=color,
        )
        ax.add_patch(scale_bar)
        label_x = start_x + scale_bar_length_in_pixels / 2
        label_y = start_y - 0.5
        ax.text(
            label_x,
            label_y,
            f"{physical_scale_bar_length} µm",
            color=color,
            ha="center",
            va="bottom",
            fontsize=12,
        )


# -----------------------------Well Galleries-----------------------------------


@omero_connect
def run_gallery_parser(
    omero_data: OmeroData,
    user_data: UserData,
    well_input: list[str],
    galleries: int,
    conn: Optional[BlitzGateway] = None,
) -> None:
    if conn:
        try:
            gallery_path = _manage_path(omero_data)
            well_list = _get_wells(omero_data, well_input, conn)
            for well in well_list:
                try:
                    _save_gallery(
                        omero_data,
                        user_data,
                        well,
                        galleries,
                        gallery_path,
                        conn,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(e)
        except Exception as e:
            logger.error(e)
            raise e


def _manage_path(omero_data: OmeroData) -> Path:
    DEFAULT_PATH = Path.home() / "Omero-Screen-Galleries"
    DEFAULT_PATH.mkdir(exist_ok=True)
    experiment = f"{omero_data.plate_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_path = DEFAULT_PATH / experiment
    experiment_path.mkdir(exist_ok=True)
    print(f"Creating directory at {experiment_path}")
    return experiment_path


def _get_wells(
    omero_data: OmeroData, well_input: list[str], conn: BlitzGateway
) -> list[str]:
    plate = conn.getObject("Plate", omero_data.plate_id)
    well_list = [well.getWellPos() for well in plate.listChildren()]  # type: ignore
    if well_input == ["All"]:
        return well_list
    else:
        return [well for well in well_input if well in well_list]


def _save_gallery(
    omero_data: OmeroData,
    userdata: UserData,
    wellpos: str,
    galleries: int,
    path: Path,
    conn: BlitzGateway,
) -> None:
    _reset_data(omero_data, userdata, wellpos, conn)
    well_image_parser(omero_data, wellpos, conn)
    userdata.reload = False
    cropped_image_parser = CroppedImageParser(omero_data, userdata)
    cropped_image_parser.parse_crops()
    for i in range(galleries):
        random_image_parser = RandomImageParser(
            omero_data, userdata, classifier=False
        )
        random_image_parser.parse_random_images()
        gallery_parser = ParseGallery(omero_data, userdata, show_gallery=False)
        fig = gallery_parser.plot_gallery()
        save_fig(
            fig,
            path,
            f"{omero_data.plate_id}_{wellpos}_gallery_{i}",
            fig_extension="pdf",
        )


def _reset_data(
    omero_data: OmeroData, userdata: UserData, wellpos: str, conn: BlitzGateway
) -> None:
    omero_data.reset_well_and_image_data()
    # omero_data.cropped_images == [np.empty((0,))]
    # omero_data.cropped_labels == []
    omero_data.well_pos_list = [wellpos]
    userdata.well = wellpos
    if plate := conn.getObject("Plate", omero_data.plate_id):
        omero_data.plate = plate
    else:
        raise ValueError(f"Error connection to plate {omero_data.plate_id}")
    # Check if both project and dataset can be retrieved, and then assign dataset to omero_data.screen_dataset
    if dataset := conn.getObject(
        "Dataset",
        get_dataset_id(conn, omero_data.plate_id),
    ):
        omero_data.screen_dataset = dataset
    else:
        raise ValueError(
            f"Error connection to plate dataset {omero_data.plate_id}"
        )
