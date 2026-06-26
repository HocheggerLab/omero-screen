"""Module for image segmentation and feature extraction in high-content screening workflows using OMERO.

This module provides tools to segment microscopy images with nucleus and cell channels, apply flatfield correction,
and extract quantitative properties from labelled regions. It is designed to work with OMERO server objects and
supports multi-channel, multi-timepoint images. The segmentation leverages Cellpose models, and extracted features
are organized into pandas DataFrames for downstream analysis.

Main Components:
----------------
- Image: Handles image correction, segmentation (nucleus, cell, cytoplasm), and mask management. Integrates with OMERO objects and supports flatfield correction.
- ImageProperties: Extracts region properties (area, intensity, etc.) from segmented masks, merges features across channels, and compiles experiment metadata.

Key Features:
-------------
- Flatfield correction for each channel using provided masks.
- Segmentation using Cellpose models, with model selection based on cell line and magnification.
- Automatic mask upload and retrieval from OMERO datasets.
- Extraction of region properties (area, intensity, etc.) for nuclei, cells, and cytoplasm.
- Data organization into pandas DataFrames, including experiment and well metadata.
- Quality control metrics for each image channel.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ezomero import get_image
from loguru import logger
from omero.gateway import BlitzGateway, ImageWrapper, WellWrapper
from omero_utils.images import parse_mip, upload_masks
from omero_utils.stitching import assign_field_by_centroid
from pandas.api.types import is_integer_dtype
from skimage import measure

from omero_screen import default_config
from omero_screen.benchmarking import get_benchmark
from omero_screen.general_functions import filter_segmentation, scale_img
from omero_screen.image_classifier import ImageClassifier
from omero_screen.metadata_parser import MetadataParser, strip_role_suffix
from omero_screen.segmentation import (
    SegmentationModel,
    apply_gamma,
    apply_seg_profile,
)

# Identity columns: cheap, channel-independent, and required on every channel
# pass. ``label`` is the per-segment join key; ``centroid`` (expanded by
# regionprops to ``centroid-0``/``centroid-1``) gives position for stitched
# image_id assignment and cross-round alignment. They are always measured with
# the per-channel (intensity) group and are never given a channel token.
IDENTITY_FEATURES: tuple[str, ...] = ("label", "centroid")

# Default classification used ONLY when a config supplies a legacy *flat*
# feature list (no explicit intensity/morphology split). Features named here
# are treated as mask-only geometry; everything else as per-channel intensity.
# The preferred config form states the split explicitly (see
# ``normalize_featureset``), so this set is a backward-compatibility shim, not
# the primary mechanism.
MORPHOLOGY_FEATURES: frozenset[str] = frozenset(
    {
        "area",
        "area_convex",
        "equivalent_diameter_area",
        "axis_major_length",
        "axis_minor_length",
        "solidity",
        "eccentricity",
        "extent",
        "perimeter",
    }
)

# A feature configuration is either the explicit structured form
# ``{"intensity": [...], "morphology": [...]}`` or a legacy flat list.
FeatureConfig = list[str] | dict[str, list[str]]


def normalize_featureset(
    featurelist: FeatureConfig,
) -> tuple[list[str], list[str]]:
    """Split a feature configuration into per-channel and per-mask groups.

    The split is taken from the config data itself, not from a hard-coded
    source-side table, so a JSON author has full control without editing code.

    Two input forms are accepted:

    * **Structured** (preferred): ``{"intensity": [...], "morphology": [...]}``.
      Used verbatim.
    * **Legacy flat list**: classified against :data:`MORPHOLOGY_FEATURES`.

    The identity columns in :data:`IDENTITY_FEATURES` (``label``, ``centroid``)
    are always ensured in the intensity group — they are measured on every
    channel pass as the join key / position — and never in the morphology group.

    Args:
        featurelist: The feature configuration (structured dict or flat list).

    Returns:
        A ``(intensity, morphology)`` pair of feature-name lists. ``intensity``
        is measured per channel; ``morphology`` once per segment.
    """
    if isinstance(featurelist, dict):
        intensity = list(featurelist.get("intensity", []))
        morphology = list(featurelist.get("morphology", []))
    else:
        intensity = [f for f in featurelist if f not in MORPHOLOGY_FEATURES]
        morphology = [f for f in featurelist if f in MORPHOLOGY_FEATURES]
    # Identity columns belong to the per-channel group only.
    morphology = [f for f in morphology if f not in IDENTITY_FEATURES]
    intensity = [f for f in intensity if f not in IDENTITY_FEATURES]
    intensity = [*IDENTITY_FEATURES, *intensity]
    return intensity, morphology


class Image:
    """Generates corrected images and segmentation masks for microscopy data.

    This class handles flatfield correction, segmentation of nuclei and cell channels using Cellpose models, and management of segmentation masks.
    It stores corrected images and segmentation results for downstream analysis.

    Attributes:
        img_dict (dict[str, np.ndarray]): Dictionary mapping channel names to flatfield-corrected image arrays.
        n_mask (np.ndarray): Segmentation mask for nuclei.
        c_mask (np.ndarray or None): Segmentation mask for cells, if available.
        cyto_mask (np.ndarray or None): Segmentation mask for cytoplasm, if available.
        nuc_diameter (int): Estimated diameter of nuclei, used for segmentation.
        channels (dict): Channel metadata from the MetadataParser.
        well_pos (tuple): Well position in the plate.
        cell_line (str): Cell line name for the current well.

    Args:
        conn (BlitzGateway): OMERO server connection.
        well (WellWrapper): OMERO WellWrapper object for the current well.
        image_obj (ImageWrapper): OMERO ImageWrapper object for the image.
        metadata (MetadataParser): Metadata parser with channel and plate information.
        dataset_id (int): OMERO dataset ID.
        flatfield_dict (dict[str, np.ndarray]): Flatfield correction masks for each channel.
    """

    def __init__(
        self,
        conn: BlitzGateway,
        well: WellWrapper,
        image_obj: ImageWrapper,
        metadata: MetadataParser,
        dataset_id: int,
        flatfield_dict: dict[str, npt.NDArray[Any]],
        border: int = 5,
    ):
        """Initializes the Image object for segmentation and correction.

        Args:
            conn (BlitzGateway): OMERO server connection.
            well (WellWrapper): OMERO WellWrapper object for the current well.
            image_obj (ImageWrapper): OMERO ImageWrapper object for the image.
            metadata (MetadataParser): Metadata parser with channel and plate information.
            dataset_id (int): OMERO dataset ID.
            flatfield_dict (dict[str, np.ndarray]): Flatfield correction masks for each channel.
            border: Width of the border examined when filtering segmented objects (negative to disable).
        """
        self._conn = conn
        self._well = well
        self.omero_image = image_obj
        self._meta_data = metadata
        self.dataset_id = dataset_id
        self._flatfield_dict = flatfield_dict
        self._border = border

        self._bench = get_benchmark()
        self._get_metadata()
        self.nuc_diameter = (
            10  # default value for nuclei diameter for 10x images
        )
        with self._bench.stage("download"):
            self.img_dict = self._get_img_dict()
        self.n_mask, self.c_mask, self.cyto_mask = self._segmentation()

    def _get_metadata(self) -> None:
        """Extracts channel metadata, well position, and cell line information from the metadata parser."""
        self.channels = self._meta_data.channel_data
        self._nucleus_channel: str = self._meta_data.channel_roles["nucleus"]
        self._cell_channel: str | None = self._meta_data.channel_roles.get(
            "cell"
        )
        self.well_pos = self._well.getWellPos()
        self.cell_line = self._meta_data.well_conditions(self.well_pos)[
            "cell_line"
        ]

    def _get_img_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image.

        Returns:
            dict[str, npt.NDArray[Any]]: Dictionary mapping channel names to flatfield-corrected image arrays.
        """
        img_dict = {}
        image_id = self.omero_image.getId()
        if self.omero_image.getSizeZ() > 1:
            array = parse_mip(self._conn, image_id, self.dataset_id)
        else:
            _, array = get_image(self._conn, image_id)

        for ch, idx in self.channels.items():
            ch_idx = int(idx)
            if ch_idx >= array.shape[-1]:
                raise IndexError(
                    f"Channel '{ch}' has index {ch_idx}, but image {image_id} "
                    f"only has {array.shape[-1]} channels (valid indices: 0-{array.shape[-1] - 1}). "
                    f"Check your channel metadata."
                )
            if ch not in self._flatfield_dict:
                raise KeyError(
                    f"Channel '{ch}' not found in flatfield correction masks. "
                    f"Available channels: {list(self._flatfield_dict.keys())}. "
                    f"This usually means the flatfield masks were generated with different metadata."
                )
            img = array[..., ch_idx] / self._flatfield_dict[ch]
            # Reduce (tzyx) to (tyx)
            img = np.squeeze(img, axis=1)

            # # Convert back to original pixel type, clipping as necessary.
            # np.clip(img, out=img, a_min=0, a_max=np.iinfo(array.dtype).max)
            # img_dict[ch] = img.astype(array.dtype)

            # Use float image. When passed to scale_img this will scale to [0, 1] for cellpose.
            img_dict[ch] = img
        return img_dict

    def _segmentation(
        self,
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any] | None
    ]:
        """Performs segmentation of nuclei and cell channels, retrieving or generating masks as needed.

        This method checks if segmentation masks already exist in the OMERO dataset. If not, it performs segmentation using Cellpose models,
        generates the required masks, and uploads them to OMERO. It supports both nucleus-only and nucleus+cell segmentation workflows.

        Returns:
            tuple:
                n_mask (np.ndarray): Segmentation mask for nuclei.
                c_mask (np.ndarray or None): Segmentation mask for cells, if available.
                cyto_mask (np.ndarray or None): Segmentation mask for cytoplasm, if available.
        """
        # check if masks already exist
        image_name = f"{self.omero_image.getId()}_segmentation"
        dataset = self._conn.getObject("Dataset", self.dataset_id)
        n_mask, c_mask, cyto_mask = None, None, None
        for image in dataset.listChildren():
            if image.getName() == image_name:
                image_id = image.getId()
                logger.info(f"Segmentation masks found for image {image_id}")
                # masks is TZYXC
                _, masks = get_image(self._conn, image_id)
                if masks.shape[-1] == 2:
                    n_mask, c_mask = masks[..., 0], masks[..., 1]
                    cyto_mask = self._get_cyto(n_mask, c_mask)
                else:
                    n_mask = masks[..., 0]
                break  # stop the loop once the image is found
        if n_mask is None:
            with self._bench.stage("nucleus_segmentation"):
                n_mask = self._n_segmentation()
            if self._cell_channel is not None:
                with self._bench.stage("cell_segmentation"):
                    c_mask = self._c_segmentation()
                n_mask, c_mask = self._compact_mask(np.stack([n_mask, c_mask]))
                cyto_mask = self._get_cyto(n_mask, c_mask)
            else:
                n_mask = self._compact_mask(n_mask)

            upload_masks(
                self._conn,
                self.dataset_id,
                self.omero_image,
                n_mask,
                c_mask,
            )
        return n_mask, c_mask, cyto_mask

    def _get_cyto(
        self, n_mask: npt.NDArray[Any], c_mask: npt.NDArray[Any]
    ) -> npt.NDArray[Any] | None:
        """Substract nuclei mask from cell mask to get cytoplasm mask.

        Args:
            n_mask (npt.NDArray[Any]): Nuclei segmentation mask.
            c_mask (npt.NDArray[Any]): Cell segmentation mask.

        Returns:
            npt.NDArray[Any] | None: Cytoplasm segmentation mask.
        """
        overlap = (c_mask != 0) * (n_mask != 0)
        cyto_mask_binary = (c_mask != 0) * (overlap == 0)
        return c_mask * cyto_mask_binary  # type: ignore[no-any-return]

    def _n_segmentation(self) -> npt.NDArray[Any]:
        """Performs nuclei segmentation using Cellpose models.

        This method selects the appropriate Cellpose model based on the cell line and magnification,
        and performs segmentation on the DAPI channel.

        Returns:
            npt.NDArray[Any]: Segmentation mask for nuclei.
        """
        if "40X" in self.cell_line.upper():
            self.nuc_diameter = 100
        elif "20X" in self.cell_line.upper():
            self.nuc_diameter = 25
        else:
            self.nuc_diameter = 10

        model_name = default_config.MODEL_DICT.get("nuclei")
        if model_name is None:
            raise RuntimeError(
                "No nuclei segmentation model configured. "
                "Add a 'nuclei' entry to MODEL_DICT in your config."
            )

        segmentation_model = _get_segmentation_model(model_name)
        # Get the image array via the nuclei role (resolved by MetadataParser).
        if self._nucleus_channel not in self.img_dict:
            raise KeyError(
                f"Nuclei channel '{self._nucleus_channel}' not found in image data. "
                f"Available channels: {list(self.img_dict.keys())}. "
                f"Nucleus segmentation requires the channel resolved to role 'nucleus'."
            )
        img_array = self.img_dict[self._nucleus_channel]

        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(img_array, dtype=np.uint32)

        # Cellpose 3 requires scaling of nuclei models; cellpose 4 is scale independent
        if segmentation_model.get_type() == "cellpose3":
            diameter = self.nuc_diameter
            logger.info(f"Segmenting nuclei with diameter {diameter}")
        else:
            diameter = None

        profile = apply_seg_profile(self._nucleus_channel)
        gamma = profile.get("gamma")
        eval_kwargs: dict[str, Any] = {
            k: v
            for k, v in profile.items()
            if k in ("cellprob_threshold", "flow_threshold")
        }

        for t in range(img_array.shape[0]):
            # Select the image at the current timepoint
            img_t = img_array[t]

            # Prepare the image for segmentation
            scaled_t = scale_img(img_t)
            if gamma is not None:
                scaled_t = apply_gamma(scaled_t, gamma)
            scaled_img_t = np.stack([scaled_t])

            # Perform segmentation
            try:
                n_mask_array = segmentation_model.eval(
                    scaled_img_t,
                    diameter=diameter,
                    normalize=False,
                    **eval_kwargs,
                )
            except IndexError:
                logger.warning(
                    f"Nucleus segmentation failed for image {self.omero_image.getId()} (t={t:d}) — returning empty mask. This may indicate an issue with the image data or segmentation model."
                )
                n_mask_array = np.zeros(scaled_img_t.shape, dtype=np.uint8)
            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(
                n_mask_array, border=self._border
            )
        return segmentation_masks

    def _c_segmentation(self) -> npt.NDArray[Any]:
        """Perform cellpose segmentation using cell mask.

        This method uses the CellposeModel to segment the cell channel.

        Returns:
            npt.NDArray[Any]: Segmentation mask for cells.
        """
        model_name = get_cell_model(self.cell_line)
        if model_name is None:
            raise RuntimeError(
                f"Unknown model for cell line: {self.cell_line}"
            )
        segmentation_model = _get_segmentation_model(model_name)

        # Get the image arrays for the nuclei and cell role channels.
        if self._cell_channel is None:
            raise RuntimeError(
                "Cell segmentation called but no channel resolved to role 'cell'. "
                "This is an internal error — _segmentation() should have skipped "
                "the cell branch."
            )
        if self._nucleus_channel not in self.img_dict:
            raise KeyError(
                f"Nuclei channel '{self._nucleus_channel}' not found in image data. "
                f"Available channels: {list(self.img_dict.keys())}. "
                f"Cell segmentation requires both nuclei and cell role channels."
            )
        if self._cell_channel not in self.img_dict:
            raise KeyError(
                f"Cell channel '{self._cell_channel}' not found in image data. "
                f"Available channels: {list(self.img_dict.keys())}. "
                f"Cell segmentation requires a channel resolved to role 'cell' "
                f"(suffix '_cell' or legacy substring 'Tub')."
            )
        dapi_array = self.img_dict[self._nucleus_channel]
        tub_array = self.img_dict[self._cell_channel]

        # Check if the time dimension matches
        assert dapi_array.shape[0] == tub_array.shape[0], (
            "Time dimension mismatch between nuclei and cell role channels"
        )

        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(dapi_array, dtype=np.uint32)

        profile = apply_seg_profile(self._cell_channel)
        gamma = profile.get("gamma")
        eval_kwargs: dict[str, Any] = {
            k: v
            for k, v in profile.items()
            if k in ("cellprob_threshold", "flow_threshold")
        }

        # Process each timepoint
        for t in range(dapi_array.shape[0]):
            # Select the images at the current timepoint
            dapi_t = dapi_array[t]
            tub_t = tub_array[t]

            # Combine the 2 channel numpy array for cell segmentation with the nuclei channel
            tub_scaled = scale_img(tub_t)
            if gamma is not None:
                tub_scaled = apply_gamma(tub_scaled, gamma)
            comb_image_t = np.stack([tub_scaled, scale_img(dapi_t)])

            # Perform segmentation
            try:
                c_masks_array = segmentation_model.eval(
                    comb_image_t, normalize=False, **eval_kwargs
                )
            except IndexError:
                logger.warning(
                    f"Cell segmentation failed for image {self.omero_image.getId()} (t={t:d}) — returning empty mask. This may indicate an issue with the image data or segmentation model '{model_name}'."
                )
                c_masks_array = np.zeros_like(comb_image_t).astype(np.uint8)

            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(
                c_masks_array, border=self._border
            )
        return segmentation_masks

    def _compact_mask(self, mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compact the uint32 datatype to the smallest required to store all mask IDs.

        Args:
            mask (npt.NDArray[Any]): Segmentation mask.

        Returns:
            npt.NDArray[Any]: Compact segmentation mask.
        """
        m = mask.max()
        if m < 2**8:
            return mask.astype(np.uint8)
        if m < 2**16:
            return mask.astype(np.uint16)
        return mask


@lru_cache(maxsize=4)
def _get_segmentation_model(model_name: str) -> SegmentationModel:
    """Gets the segmentation model."""
    return SegmentationModel(model_name)


class _SyntheticOmeroImage:
    """Minimal stand-in for omero.gateway.ImageWrapper.

    ImageProperties only calls ``.getId()`` on the wrapper, so this
    suffices for the stitched-mode path where the well is treated as a
    single synthetic image (id = first OMERO image id in the well).
    """

    def __init__(self, image_id: int):
        self._id = image_id

    def getId(self) -> int:  # noqa: N802 — mirrors OMERO API
        return self._id


class StitchedWellImage:
    """Adapter exposing the Image surface that ImageProperties consumes.

    Built from a pre-stitched canvas + nucleus mask rather than from
    OMERO field reads. Stage 1 is nucleus-only, so ``c_mask`` and
    ``cyto_mask`` are always ``None``.

    Attributes mirror ``Image``:
        img_dict: channel name → (T, Y, X) array
        n_mask: (T, Y, X) nucleus labels
        c_mask: None
        cyto_mask: None
        omero_image: synthetic wrapper with ``getId()``
        well_pos: well position string
    """

    def __init__(
        self,
        stitched_img: npt.NDArray[Any],
        stitched_mask: npt.NDArray[Any],
        channels: dict[str, int],
        nucleus_channel: str,
        well_pos: str,
        synthetic_image_id: int,
        c_mask: npt.NDArray[Any] | None = None,
        cyto_mask: npt.NDArray[Any] | None = None,
        cell_channel: str | None = None,
        field_image_ids: list[int] | None = None,
        field_positions: list[tuple[float, float]] | None = None,
        tile_h: int | None = None,
        tile_w: int | None = None,
        stitch_params: dict[str, int] | None = None,
    ):
        """Initialise the stitched-well image adapter.

        Args:
            stitched_img: Stitched canvas of shape (T, Y, X, C).
            stitched_mask: Nucleus label mask of shape (T, Y, X).
            channels: Channel name → channel-axis index in stitched_img.
            nucleus_channel: Name of the nucleus channel.
            well_pos: Well position (e.g. "A1").
            synthetic_image_id: Synthetic OMERO image id for the well.
            c_mask: Optional cell mask of shape (T, Y, X).
            cyto_mask: Optional cytoplasm mask of shape (T, Y, X).
            cell_channel: Name of the cell channel (if cell segmentation
                was performed).
            field_image_ids: Per-field OMERO image ids aligned with
                ``field_positions``. When provided, ImageProperties
                resolves a per-row image_id by centroid lookup rather
                than broadcasting ``synthetic_image_id``.
            field_positions: Stage positions per field (same ordering
                as ``field_image_ids``).
            tile_h: Per-field tile height in pixels.
            tile_w: Per-field tile width in pixels.
            stitch_params: Stitching params dict with keys
                ``overlap_x``, ``overlap_y``, ``translate_x``,
                ``translate_y``.
        """
        self.img_dict: dict[str, npt.NDArray[Any]] = {
            ch: stitched_img[..., idx] for ch, idx in channels.items()
        }
        self.n_mask = stitched_mask
        self.c_mask: npt.NDArray[Any] | None = c_mask
        self.cyto_mask: npt.NDArray[Any] | None = cyto_mask
        self._nucleus_channel = nucleus_channel
        self._cell_channel: str | None = cell_channel
        self.well_pos = well_pos
        self.omero_image = _SyntheticOmeroImage(synthetic_image_id)
        self.field_image_ids = field_image_ids
        self.field_positions = field_positions
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stitch_params = stitch_params


def get_cell_model(
    cell_line: str,
    default_model: str | None = default_config.MODEL_DICT["U2OS"],
) -> str | None:
    """Gets the cell segmentation model for the specified cell line.

    If the cell line is not recognised the default model is returned.

    Args:
        cell_line: Cell line.
        default_model: The default model if the cell line is not recognised.

    Returns:
        model name
    """
    cell_line = cell_line.replace(
        " ", ""
    ).upper()  # remove spaces and make uppercase
    if "40X" in cell_line:
        logger.info("40x image detected, using 40x nuclei model")
        return "40x_Tub_H2B"
    elif "20X" in cell_line:
        logger.info("20x image detected, using 20x nuclei model")
        return "cyto"
    elif cell_line in default_config.MODEL_DICT:
        return default_config.MODEL_DICT[cell_line]

    # substring matching: cell line may be longer than the model key
    for k, v in default_config.MODEL_DICT.items():
        if k in cell_line:
            return v

    return default_model


class ImageProperties:
    """Extracts feature measurements from segmented nuclei, cells and cytoplasm and generates combined data frames.

    This class processes segmented masks to extract quantitative features from nuclei, cells, and cytoplasm.
    It combines measurements from different channels and generates a comprehensive DataFrame for downstream analysis.

    Attributes:
        image_df (pd.DataFrame): DataFrame containing feature measurements for all regions and channels.
        quality_df (pd.DataFrame): DataFrame containing quality control metrics for each channel.
        plate_name (str): Name of the plate.
        _cond_dict (dict): Experimental conditions for the current well.
        _well (WellWrapper): OMERO WellWrapper object for the current well.
        _well_id (int): OMERO Well ID.
        _image (Image): Image object containing segmentation masks and corrected images.
        _meta_data (MetadataParser): Metadata parser with channel and plate information.
        _overlay (pd.DataFrame): DataFrame linking nuclear IDs with cell IDs.
    """

    def __init__(
        self,
        well: WellWrapper,
        image_obj: Image,
        meta_data: MetadataParser,
        featurelist: FeatureConfig = default_config.FEATURELIST,
        image_classifier: None | list[ImageClassifier] = None,
    ):
        """Initializes the ImageProperties object for feature extraction and data aggregation.

        Args:
            well (WellWrapper): OMERO WellWrapper object for the current well.
            image_obj (Image): Image object containing segmentation masks and corrected images.
            meta_data (MetadataParser): Metadata parser with channel and plate information.
            featurelist: Feature configuration — structured ``{"intensity": [...],
                "morphology": [...]}`` or a legacy flat list. Defaults to
                ``default_config.FEATURELIST``.
            image_classifier (optional): Optional image classifier(s) for additional processing. Defaults to None.
        """
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._meta_data = meta_data

        # Assumes the well parent is the plate
        self.plate_name = well.getParent().getName()
        # Get the dict[str, Any] for the given well
        self._cond_dict = meta_data.well_conditions(well.getWellPos())
        self._overlay = self._overlay_mask()
        self.image_df = self._combine_channels(featurelist)
        self._add_background()
        self.quality_df = self._concat_quality_df()

        if image_classifier is not None:
            # Use cell mask if available, otherwise fall back to nucleus mask
            if image_obj.c_mask is not None:
                classifier_mask = image_obj.c_mask
            else:
                classifier_mask = image_obj.n_mask
                # Synthesize columns the classifier expects from cell/nucleus merge
                if "Cyto_ID" not in self.image_df.columns:
                    self.image_df["Cyto_ID"] = self.image_df["label"]
                if "centroid-0_x" not in self.image_df.columns:
                    self.image_df["centroid-0_x"] = self.image_df["centroid-0"]
                    self.image_df["centroid-1_x"] = self.image_df["centroid-1"]
            for cls in image_classifier:
                if cls.select_channels(image_obj.img_dict):
                    self.image_df = cls.process_images(
                        self.image_df, classifier_mask
                    )

    def _add_background(self) -> None:
        """Add per-channel background intensity columns to image_df.

        Background is estimated as the median intensity of pixels outside all
        cell masks (or nucleus masks when no cell mask is available). One value
        per channel per timepoint is broadcast to every row in that group.
        """
        ref_mask = (
            self._image.c_mask
            if self._image.c_mask is not None
            else self._image.n_mask
        )
        # ref_mask shape: (T, Y, X) after squeezing
        ref_mask_sq: npt.NDArray[Any] = np.squeeze(ref_mask)
        timepoints = self._image.img_dict[
            next(iter(self._image.img_dict))
        ].shape[0]

        for channel, img_array in self._image.img_dict.items():
            col = f"{channel}_background"
            bg_values: dict[int, float] = {}
            for t in range(timepoints):
                mask_t = ref_mask_sq if timepoints == 1 else ref_mask_sq[t]
                img_t: npt.NDArray[Any] = np.squeeze(img_array[t])
                background_pixels = img_t[mask_t == 0]
                bg_values[t] = float(
                    np.median(background_pixels)
                    if background_pixels.size > 0
                    else 0.0
                )
            self.image_df[col] = self.image_df["timepoint"].map(bg_values)

    def _overlay_mask(self) -> pd.DataFrame:
        """Links nuclear IDs with cell IDs.

        This method creates a DataFrame linking nuclear IDs with cell IDs.

        Returns:
            pd.DataFrame: DataFrame linking nuclear IDs with cell IDs.
        """
        if self._image.c_mask is None:
            return pd.DataFrame({"label": self._image.n_mask.flatten()})

        overlap = (self._image.c_mask != 0) * (self._image.n_mask != 0)
        stack = np.stack(
            [self._image.n_mask[overlap], self._image.c_mask[overlap]]
        )
        list_n_masks = stack[-2].tolist()
        list_masks = stack[-1].tolist()
        overlay_all = {
            list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))
        }
        return pd.DataFrame(
            list(overlay_all.items()), columns=["label", "Cyto_ID"]
        )

    def _combine_channels(self, featurelist: FeatureConfig) -> pd.DataFrame:
        """Combines feature measurements from different channels into a single DataFrame.

        This method processes the segmented masks for each channel and combines the measurements into a single DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing feature measurements for all regions and channels.
        """
        # The intensity/morphology split comes from the config (see
        # ``normalize_featureset``). Morphology is measured once per segment
        # (routed to the channel that segmented the mask, in ``_channel_data``);
        # intensity is measured for every channel.
        intensity, morphology = normalize_featureset(featurelist)
        if morphology and self._image.c_mask is not None:
            # Cell/cyto geometry is owned by the cell channel; fail loud if the
            # resolved cell-channel name is not among the channels we iterate,
            # otherwise cell geometry would silently never be measured.
            assert self._image._cell_channel in self._meta_data.channel_data, (
                f"Cell channel '{self._image._cell_channel}' not in "
                f"channels {list(self._meta_data.channel_data)}"
            )
        channel_data = [
            self._channel_data(channel, intensity, morphology)
            for channel in self._meta_data.channel_data
        ]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[
            :, ~props_data.columns.duplicated()
        ].copy()
        cond_list = [
            self.plate_name,
            self._meta_data.plate_id,
            self._well.getWellPos(),
            self._well_id,
        ]
        cond_list.extend(iter(self._cond_dict.values()))
        col_list = ["experiment", "plate_id", "well", "well_id"]
        col_list.extend(iter(self._cond_dict.keys()))
        col_list_edited = [entry.lower() for entry in col_list]
        edited_props_data[col_list_edited] = cond_list

        # Per-row image_id: in stitched mode, look up which field's
        # tile owns each cell's centroid so the row carries that
        # field's OMERO image_id (rather than a single synthetic id
        # for the whole well). Falls back to the (synthetic or real)
        # wrapper id when tile geometry is not present.
        field_image_ids = getattr(self._image, "field_image_ids", None)
        field_positions = getattr(self._image, "field_positions", None)
        tile_h = getattr(self._image, "tile_h", None)
        tile_w = getattr(self._image, "tile_w", None)
        stitch_params = getattr(self._image, "stitch_params", None) or {}
        if (
            field_image_ids is not None
            and field_positions is not None
            and tile_h is not None
            and tile_w is not None
        ):
            centroids = edited_props_data[
                ["centroid-0", "centroid-1"]
            ].to_numpy()
            field_idx = assign_field_by_centroid(
                centroids,
                field_positions,
                tile_h,
                tile_w,
                **stitch_params,
            )
            edited_props_data["image_id"] = np.asarray(field_image_ids)[
                field_idx
            ]
        else:
            edited_props_data["image_id"] = self._image.omero_image.getId()

        return edited_props_data.sort_values(by=["timepoint"]).reset_index(
            drop=True
        )

    def _channel_data(
        self, channel: str, intensity: list[str], morphology: list[str]
    ) -> pd.DataFrame:
        """Processes the segmented masks for a specific channel and combines the measurements into a single DataFrame.

        Intensity features are measured for this channel on every segment.
        Morphology (mask-only geometry) is measured only when ``channel`` is the
        channel that segmented the relevant mask — the nucleus channel for the
        nucleus segment, the cell channel for the cell/cyto segments — so each
        segment's geometry is computed exactly once across the channel loop.

        Args:
            channel: Channel name being processed.
            intensity: Per-channel intensity feature names (measured here).
            morphology: Mask-only geometry feature names (measured only on the
                owning segmentation channel).

        Returns:
            pd.DataFrame: DataFrame containing feature measurements for the given channel.
        """
        channel_token = self._feature_channel_token(channel)
        # Names that must be rendered channel-independent (``{feature}_{segment}``)
        # rather than per-channel, passed down to ``_edit_properties``.
        morphology_names = frozenset(morphology)
        # Nucleus geometry is owned by the nucleus channel.
        nucleus_features = intensity + (
            morphology if channel == self._image._nucleus_channel else []
        )
        nucleus_data = self._get_properties(
            self._image.n_mask,
            channel,
            channel_token,
            "nucleus",
            nucleus_features,
            morphology_names,
        )
        # merge channel data, outer merge combines all area columns into 1
        if self._image.c_mask is not None:
            nucleus_data = self._outer_merge(
                nucleus_data, self._overlay, "label"
            )
        if channel == self._image._nucleus_channel:
            # Build the integrated-intensity column using the actual nucleus
            # channel token, so cellcycle_analysis (parameterised by
            # ``nucleus_channel``) can find it for DNA-content normalisation.
            # Legacy DAPI plates keep the historical ``integrated_int_DAPI``
            # column name; non-DAPI plates get ``integrated_int_{channel}``.
            # ``area_nucleus`` is present because nucleus geometry is owned by
            # this channel (see ``nucleus_features`` above).
            nucleus_data[f"integrated_int_{channel_token}"] = (
                nucleus_data[f"intensity_mean_{channel_token}_nucleus"]
                * nucleus_data["area_nucleus"]
            )

        if (
            self._image.c_mask is not None
            and self._image.cyto_mask is not None
        ):
            # Cell and cyto geometry is owned by the cell channel.
            cell_features = intensity + (
                morphology if channel == self._image._cell_channel else []
            )
            cell_data = self._get_properties(
                self._image.c_mask,
                channel,
                channel_token,
                "cell",
                cell_features,
                morphology_names,
            )
            cyto_data = self._get_properties(
                self._image.cyto_mask,
                channel,
                channel_token,
                "cyto",
                cell_features,
                morphology_names,
            )
            merge_1 = self._outer_merge(
                cell_data, cyto_data, ["label", "timepoint"]
            )
            merge_1 = merge_1.rename(columns={"label": "Cyto_ID"})
            return self._outer_merge(
                nucleus_data, merge_1, ["Cyto_ID", "timepoint"]
            )
        else:
            return nucleus_data

    def _get_properties(
        self,
        segmentation_mask: npt.NDArray[Any],
        channel: str,
        channel_token: str,
        segment: str,
        featurelist: list[str],
        morphology_names: frozenset[str],
    ) -> pd.DataFrame:
        """Measure selected features for each segmented cell in given channel.

        Args:
            segmentation_mask: Mask array used as the region label image.
            channel: Original channel name (used to index ``img_dict``).
            channel_token: Token to embed in feature column names. ``"DAPI"`` for
                the nuclei role (canonical), otherwise ``strip_role_suffix(channel)``.
            segment: Segment label (``nucleus`` / ``cell`` / ``cyto``).
            featurelist: List of regionprops features to extract.
            morphology_names: Feature names to render channel-independent
                (``{feature}_{segment}``); the rest get a channel token.

        Returns:
            pd.DataFrame: DataFrame containing feature measurements for the given channel.
        """
        timepoints = self._image.img_dict[channel].shape[0]
        # squeezing [t]z
        label = np.squeeze(segmentation_mask).astype(np.int64)

        if timepoints > 1:
            data_list = []
            for t in range(timepoints):
                props = measure.regionprops_table(  # type: ignore[no-untyped-call]
                    label[t],
                    # squeezing z
                    np.squeeze(self._image.img_dict[channel][t]),
                    properties=featurelist,
                )
                data = pd.DataFrame(props)
                feature_dict = self._edit_properties(
                    channel_token, segment, featurelist, morphology_names
                )
                data = data.rename(columns=feature_dict)
                data["timepoint"] = t  # Add timepoint for all channels
                data_list.append(data)
            combined_data = pd.concat(data_list, axis=0, ignore_index=True)
            return combined_data.sort_values(
                by=["timepoint", "label"]
            ).reset_index(drop=True)
        else:
            props = measure.regionprops_table(  # type: ignore[no-untyped-call]
                label,
                # squeezing tz
                np.squeeze(self._image.img_dict[channel]),
                properties=featurelist,
            )
            data = pd.DataFrame(props)
            feature_dict = self._edit_properties(
                channel_token, segment, featurelist, morphology_names
            )
            data = data.rename(columns=feature_dict)
            data["timepoint"] = 0  # Add timepoint 0 for single timepoint data
            return data.sort_values(by=["label"]).reset_index(drop=True)

    def _feature_channel_token(self, channel: str) -> str:
        """Token used to name feature columns for a given channel.

        Returns the suffix-stripped channel name for every channel, including
        the nucleus role. Legacy plates whose nucleus channel is named ``DAPI``
        therefore continue to produce ``intensity_mean_DAPI_nucleus`` /
        ``integrated_int_DAPI``; non-DAPI plates produce columns named after
        the actual fluorophore (e.g. ``intensity_mean_H2B_RFP_nucleus``).
        """
        return strip_role_suffix(channel)

    @staticmethod
    def _edit_properties(
        channel_token: str,
        segment: str,
        featurelist: list[str],
        morphology_names: frozenset[str],
    ) -> dict[str, str]:
        """Build the rename map from regionprops column names to feature column names.

        Classification is by feature name, driven by the config-derived
        ``morphology_names`` (not a hard-coded table):

        * identity features (:data:`IDENTITY_FEATURES`) are left untouched —
          ``label`` is the join key; ``centroid`` is expanded by regionprops to
          ``centroid-0``/``centroid-1``, which are channel-independent.
        * morphology features (those in ``morphology_names``) are named
          ``{feature}_{segment}`` — channel-independent geometry.
        * everything else is a per-channel intensity feature, named
          ``{feature}_{channel_token}_{segment}``.

        Args:
            channel_token: Token used in the feature column name (canonical ``DAPI``
                for the nuclei role; suffix-stripped channel name otherwise).
            segment: Segment label (``nucleus`` / ``cell`` / ``cyto``).
            featurelist: List of regionprops feature names.
            morphology_names: Feature names to render channel-independent.

        Returns:
            dict[str, str]: Dictionary mapping regionprops column names to their
            final feature column names.
        """
        feature_dict: dict[str, str] = {}
        for feature in featurelist:
            if feature in IDENTITY_FEATURES:
                continue
            if feature in morphology_names:
                feature_dict[feature] = f"{feature}_{segment}"
            else:
                feature_dict[feature] = f"{feature}_{channel_token}_{segment}"
        return feature_dict

    def _outer_merge(
        self, df1: pd.DataFrame, df2: pd.DataFrame, on: list[str] | str
    ) -> pd.DataFrame:
        """Perform an outer-join merge on the two pandas dataframes. NA rows are removed and integer columns are restored.

        This method performs an outer-join merge on the two pandas DataFrames and removes NA rows.

        Returns:
            pd.DataFrame: Merged DataFrame with integer columns restored.
        """
        df = pd.merge(df1, df2, how="outer", on=on).dropna(axis=0, how="any")
        # Outer-join merge will create columns that support NA. This changes int columns to float.
        # After dropping all the NA rows restore the int columns.
        for c in df1.columns:
            if is_integer_dtype(df1[c].dtype) and not is_integer_dtype(
                df[c].dtype
            ):
                df[c] = df[c].astype(df1[c].dtype)
        for c in df2.columns:
            if is_integer_dtype(df2[c].dtype) and not is_integer_dtype(
                df[c].dtype
            ):
                df[c] = df[c].astype(df2[c].dtype)
        return df

    def _set_quality_df(
        self, channel: str, corr_img: npt.NDArray[Any]
    ) -> pd.DataFrame:
        """Generates df for image quality control saving the median intensity of the image.

        This method generates a DataFrame for image quality control by saving the median intensity of the image.

        Returns:
            pd.DataFrame: DataFrame containing quality control metrics for the given channel.
        """
        return pd.DataFrame(
            {
                "experiment": [self.plate_name],
                "plate_id": [self._meta_data.plate_id],
                "position": [self._image.well_pos],
                "image_id": [self._image.omero_image.getId()],
                "channel": [channel],
                "intensity_median": [np.median(corr_img)],
            }
        )

    def _concat_quality_df(self) -> pd.DataFrame:
        """Concatenate quality dfs for all channels in _corr_img_dict.

        This method concatenates the quality DataFrames for all channels in the _corr_img_dict.

        Returns:
            pd.DataFrame: Concatenated DataFrame containing quality control metrics for all channels.
        """
        df_list = [
            self._set_quality_df(channel, image)
            for channel, image in self._image.img_dict.items()
        ]
        return pd.concat(df_list)
