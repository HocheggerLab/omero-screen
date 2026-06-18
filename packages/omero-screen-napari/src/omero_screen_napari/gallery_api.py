import random
from typing import TYPE_CHECKING, Any, TypeVar

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from skimage import exposure
from skimage.measure import find_contours

if TYPE_CHECKING:
    import pandas as pd

# Re-exported for backward compatibility: these crop helpers now live in
# crop_pipeline (shared with the direct-load paths) but are still imported
# from gallery_api by direct_omero_loader, session_utils, and tests.
from omero_screen_napari.crop_pipeline import (  # noqa: F401
    CropPipeline,
    CropSourceError,
    WelldataSource,
    ZarrSource,
    calculate_crop_coordinates,
    crop_region,
    erase_masks,
    pad_region,
)
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData


def show_gallery(
    omero_data: OmeroData,
    user_data: UserData,
    classifier: bool = False,
    excluded_centroids: set[tuple[int, int, int]] | None = None,
) -> None:
    if user_data.reload or omero_data.cropped_images == []:
        parse_crops_into_omero_data(
            omero_data, user_data, excluded_centroids=excluded_centroids
        )

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


def parse_crops_into_omero_data(
    omero_data: OmeroData,
    user_data: UserData,
    excluded_centroids: set[tuple[int, int, int]] | None = None,
) -> None:
    """Generate the well's crop pool into ``omero_data.cropped_*``.

    Replaces the old ``CroppedImageParser``. Filters the well's CellView
    rows (cellcycle / classifier / loaded-image intersection / timepoint),
    picks a crop source (zarr stitched canvas when a per-channel contrast
    window is available, else the in-memory fields), and runs
    :class:`CropPipeline`. The downstream finalize step
    (:class:`RandomImageParser`) is unchanged.
    """
    logger.info(
        f"Parsing crops for well {user_data.well} using segmentation {user_data.segmentation}"
    )
    centroids = _filter_well_centroids(omero_data, user_data)
    if centroids.empty:
        logger.warning(
            f"No centroids for well {user_data.well} after filtering; no crops generated."
        )
        omero_data.cropped_images = []
        omero_data.cropped_labels = []
        omero_data.cropped_cell_meta = []
        return

    pipeline = CropPipeline(
        source=_make_gallery_source(omero_data, user_data),
        centroids_df=centroids,
        segmentation=user_data.segmentation,  # type: ignore[arg-type]
        crop_size=user_data.crop_size,
        timepoint=int(user_data.timepoint),
        excluded_centroids=excluded_centroids,
    )
    try:
        result = pipeline.run()
    except CropSourceError as exc:
        # Mirror the gallery's old per-image zarr→in-memory fallback, at
        # run granularity: if the zarr store can't serve a crop mid-run,
        # redo the whole well from the in-memory fields.
        logger.warning(
            f"Crop source failed mid-run ({exc}); retrying from in-memory fields"
        )
        result = CropPipeline(
            source=WelldataSource(omero_data),
            centroids_df=centroids,
            segmentation=user_data.segmentation,  # type: ignore[arg-type]
            crop_size=user_data.crop_size,
            timepoint=int(user_data.timepoint),
            excluded_centroids=excluded_centroids,
        ).run()

    omero_data.cropped_images = result.crops
    omero_data.cropped_labels = result.labels
    omero_data.cropped_cell_meta = result.cell_meta
    logger.info(f"Generated {len(result.crops)} crops")


def _make_gallery_source(
    omero_data: OmeroData, user_data: UserData
) -> WelldataSource | ZarrSource:
    """Pick the zarr stitched-canvas source when a contrast window is known.

    The gallery only ever used its zarr fast path when
    ``omero_data.intensities`` was populated (otherwise it fell back to the
    in-memory path to keep display contrast consistent). Preserve that: use
    :class:`ZarrSource` only when intensities are set and the well is built,
    else :class:`WelldataSource`.
    """
    intensities = omero_data.intensities
    if intensities:
        try:
            return ZarrSource(omero_data.plate_id, user_data.well, intensities)
        except CropSourceError as exc:
            logger.info(
                f"Zarr source unavailable ({exc}); using in-memory fields"
            )
    return WelldataSource(omero_data)


def _select_cellcycledata(df: pl.DataFrame, cellcycle: str) -> pl.DataFrame:
    """Filter ``df`` to one cell-cycle phase; raise on an invalid phase.

    ``"All"`` is a no-op. Behaviour matches the old
    ``CroppedImageParser._select_cellcycledata``.
    """
    if cellcycle == "All":
        return df
    if "cell_cycle" not in df.columns:
        logger.error(
            f"'cell_cycle' column not found in data. Available columns: {df.columns[:10]}"
        )
        raise ValueError(
            "Cell cycle data not found. Cannot filter by cell cycle."
        )
    unique_phases = df["cell_cycle"].unique().to_list()
    if cellcycle not in unique_phases:
        raise ValueError(
            f"Invalid cell cycle phase: {cellcycle}. "
            f"Available phases in this well: {unique_phases}"
        )
    return df.filter(df["cell_cycle"] == cellcycle)


def _select_classifierdata(
    df: pl.DataFrame, classifier_filter: str
) -> pl.DataFrame:
    """Filter ``df`` to rows where any ``classifier_*`` column equals the value.

    Empty / whitespace value is a no-op. Searches each classifier column in
    turn and filters on the first that contains the value; logs a warning and
    returns ``df`` unchanged when no column matches. Behaviour matches the old
    ``CroppedImageParser._select_classifierdata``.
    """
    value = classifier_filter.strip()
    if not value:
        return df
    classifier_cols = [c for c in df.columns if c.startswith("classifier_")]
    for col in classifier_cols:
        if value in df[col].unique().to_list():
            return df.filter(pl.col(col) == value)
    logger.warning(
        f"Classifier filter '{value}' not found in any classifier column: {classifier_cols}"
    )
    return df


def _filter_well_centroids(
    omero_data: OmeroData, user_data: UserData
) -> "pd.DataFrame":
    """Filter the plate's CellView rows down to the well's croppable cells.

    Ported faithfully from the old ``CroppedImageParser`` polars helpers
    (``_get_well_data`` / ``_select_cellcycledata`` / ``_select_classifierdata``
    / the loaded-image intersection / the timepoint filter), then converted
    to pandas for :class:`CropPipeline`. Behaviour — including raising on an
    invalid cell-cycle phase — is unchanged.
    """
    schema = omero_data.plate_data.collect_schema()
    if len(schema) == 0:
        raise ValueError(
            f"Plate {omero_data.plate_id} has no data in CellView. "
            "Please import the plate into CellView before using the gallery."
        )
    df = omero_data.plate_data.filter(
        pl.col("well") == user_data.well
    ).collect()
    df = _select_cellcycledata(df, user_data.cellcycle)
    df = _select_classifierdata(df, user_data.classifier_filter)

    # Keep only images actually loaded (image_ids is populated for both the
    # in-memory and zarr backends).
    loaded_ids = set(omero_data.image_ids)
    if not loaded_ids:
        logger.warning("No images loaded in OmeroData. Cannot process crops.")
        df = df.filter(pl.col("image_id").is_in([]))
    else:
        metadata_ids = set(df["image_id"].unique().to_list())
        common_ids = metadata_ids.intersection(loaded_ids)
        if not common_ids:
            logger.warning(
                f"No intersection between requested well ({user_data.well}) images and loaded images. Loaded IDs: {list(loaded_ids)[:5]}..."
            )
        elif len(common_ids) < len(metadata_ids):
            logger.info(
                f"Processing {len(common_ids):d} images (subset of well) that are loaded."
            )
        df = df.filter(pl.col("image_id").is_in(common_ids))

    # Live-cell time-lapse: keep only the requested timepoint's rows so crops
    # centre on the cell at that t (CellView stores one row per (cell, t)).
    if "timepoint" in df.columns:
        tp = int(user_data.timepoint)
        df_tp = df.filter(pl.col("timepoint") == tp)
        if df_tp.height > 0:
            df = df_tp
        else:
            logger.info(
                f"No rows for timepoint {tp:d} (well {user_data.well}); falling back to all timepoints"
            )

    return df.to_pandas()


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

        # Convert channel names to indices. Empties are stripped at the
        # gallery_widget layer, so ``user_data.channels`` only contains
        # resolvable names; ``fill_missing_channels`` then packs them
        # into RGB by list position (R=ch0, G=ch1, B=ch2 or 0).
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
                        f"Channel index {val} for {channel_name} is not a valid number."
                    )
            else:
                logger.warning(
                    f"Channel {channel_name} not found in channel_data map."
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

    # Extract requested channels. ``-1`` (or any negative / out-of-range
    # index) is the sentinel for "no channel in this RGB slot" — the
    # caller must NOT receive img[..., -1] (which would silently pick the
    # last channel and mis-colour the result).
    for idx in channel_indices:
        if 0 <= idx < img.shape[-1]:
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
        # Close any previously-open gallery figures before creating a new
        # one. Each Enter would otherwise leak a figure into pyplot's
        # global registry; on macOS the Cocoa/Qt backend segfaults after
        # ~10 accumulated windows.
        plt.close("all")
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
                    f"Image {i:d} has shape {img.shape}, expected {first_shape}"
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
