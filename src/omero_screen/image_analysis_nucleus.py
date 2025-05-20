"""Module for Nucleus Channel Image Segmentation and Feature Extraction.

This module provides classes and functions to segment images containing a nucleus channel,
apply flatfield correction, and extract quantitative properties from the segmented regions.
It is designed for use with OMERO image data, supporting multi-channel, multi-timepoint,
and multi-well plate imaging experiments.

Classes:
    NucImage:
        Handles image correction (flatfield), segmentation of nuclei using Cellpose,
        and management of segmentation masks for OMERO images.

    NucImageProperties:
        Extracts and aggregates region properties (features) from segmented nuclei,
        producing pandas DataFrames suitable for downstream analysis and quality control.

Typical Workflow:
    1. Initialize a `NucImage` object with OMERO connection, well, image, metadata,
       dataset ID, and flatfield correction data.
    2. The `NucImage` object corrects the image, segments nuclei, and stores masks.
    3. Initialize a `NucImageProperties` object with the well, `NucImage` object,
       and metadata to extract features and quality metrics.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from cellpose import models
from ezomero import get_image
from omero.gateway import BlitzGateway, ImageWrapper, WellWrapper
from omero_utils.images import parse_mip, upload_masks
from skimage import measure

from omero_screen import default_config
from omero_screen.config import getenv_as_bool
from omero_screen.general_functions import filter_segmentation, scale_img
from omero_screen.metadata_parser import MetadataParser

logger = logging.getLogger("omero-screen")


class NucImage:
    """Generates corrected images and segmentation masks for nucleus channel images.

    This class handles flatfield correction, segmentation of nuclei using Cellpose,
    and management of segmentation masks for OMERO images. It supports multi-channel
    and multi-timepoint images, and stores both the corrected images and the
    segmentation masks for downstream analysis.

    Attributes:
        _conn (BlitzGateway): OMERO connection object.
        _well (WellWrapper): OMERO WellWrapper object for the current well.
        omero_image (ImageWrapper): OMERO ImageWrapper object for the image.
        _meta_data (MetadataParser): Metadata parser for extracting channel and well info.
        dataset_id (int): OMERO dataset ID.
        _flatfield_dict (dict[str, np.ndarray]): Flatfield correction arrays for each channel.
        channels (dict): Channel metadata from the metadata parser.
        well_pos (Any): Well position identifier.
        cell_line (str): Cell line name or identifier.
        img_dict (dict[str, np.ndarray]): Corrected images for each channel.
        n_mask (np.ndarray): Segmentation mask array for nuclei.

    Methods:
        _get_metadata():
            Extracts channel and well metadata from the metadata parser.

        _get_img_dict() -> dict[str, np.ndarray]:
            Applies flatfield correction and returns a dictionary of corrected images.

        _segmentation() -> np.ndarray:
            Retrieves or computes the segmentation mask for the image.

        _n_segmentation() -> np.ndarray:
            Runs Cellpose nucleus segmentation for each timepoint.

        _compact_mask(mask: np.ndarray) -> np.ndarray:
            Compacts the mask datatype to the smallest required type.
    """

    def __init__(
        self,
        conn: BlitzGateway,
        well: WellWrapper,
        image_obj: ImageWrapper,
        metadata: MetadataParser,
        dataset_id: int,
        flatfield_dict: dict[str, npt.NDArray[Any]],
    ):
        """Initialize the NucImage object.

        This method initializes the NucImage object with the given parameters.

        Args:
            conn: BlitzGateway object for OMERO connection.
            well: OMERO WellWrapper object for the current well.
            image_obj: OMERO ImageWrapper object for the image.
            metadata: MetadataParser object for extracting channel and well info.
            dataset_id: OMERO dataset ID.
            flatfield_dict: Dictionary of flatfield correction arrays for each channel.
        """
        self._conn = conn
        self._well = well
        self.omero_image = image_obj
        self._meta_data = metadata
        self.dataset_id = dataset_id
        self._flatfield_dict = flatfield_dict

        self._get_metadata()
        self.img_dict = self._get_img_dict()
        self.n_mask = self._segmentation()

    def _get_metadata(self) -> None:
        """Extracts channel and well metadata from the metadata parser.

        This method extracts the channel and well metadata from the metadata parser and stores it in the NucImage object.
        """
        self.channels = self._meta_data.channel_data
        self.well_pos = self._well.getWellPos()
        self.cell_line = self._meta_data.well_conditions(self.well_pos)[
            "cell_line"
        ]

    def _get_img_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image.

        This method divides the image array with the flatfield correction mask and returns a dictionary of corrected images.
        """
        img_dict = {}
        image_id = self.omero_image.getId()
        if self.omero_image.getSizeZ() > 1:
            array = parse_mip(self._conn, image_id, self.dataset_id)
        else:
            _, array = get_image(self._conn, image_id)

        for ch, idx in self.channels.items():
            img = array[..., int(idx)] / self._flatfield_dict[ch]
            # Reduce (tzyx) to (tyx)
            img = np.squeeze(img, axis=1)

            # # Convert back to original pixel type, clipping as necessary.
            # np.clip(img, out=img, a_min=0, a_max=np.iinfo(array.dtype).max)
            # img_dict[ch] = img.astype(array.dtype)

            # Use float image. When passed to scale_img this will scale to [0, 1] for cellpose.
            img_dict[ch] = img
        return img_dict

    def _segmentation(self) -> npt.NDArray[Any]:
        """Retrieve or compute the segmentation mask for the image.

        This method retrieves the segmentation mask for the image from the dataset if it exists, otherwise it computes the segmentation mask using Cellpose.
        """
        image_name = f"{self.omero_image.getId()}_segmentation"
        dataset = self._conn.getObject("Dataset", self.dataset_id)
        n_mask = None
        for image in dataset.listChildren():
            if image.getName() == image_name:
                image_id = image.getId()
                logger.info("Segmentation masks found for image %s", image_id)
                # masks is TZYXC
                _, masks = get_image(self._conn, image_id)
                n_mask = masks[..., 0]
                break  # stop the loop once the image is found
        if n_mask is None:
            n_mask = self._n_segmentation()

            upload_masks(
                self._conn,
                self.dataset_id,
                self.omero_image,
                n_mask,
            )
        return n_mask

    def _n_segmentation(self) -> npt.NDArray[Any]:
        """Segment nuclei using Cellpose models.

        This method segments nuclei using Cellpose models.
        """
        if "40X" in self.cell_line.upper():
            self.nuc_diameter = 100
        elif "20X" in self.cell_line.upper():
            self.nuc_diameter = 25
        else:
            self.nuc_diameter = 10
        segmentation_model = models.CellposeModel(
            gpu=getenv_as_bool("GPU", default=torch.cuda.is_available()),
            model_type=default_config.MODEL_DICT["nuclei"],
        )
        # Get the image array
        img_array = self.img_dict["DAPI"]

        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(img_array, dtype=np.uint32)

        for t in range(img_array.shape[0]):
            # Select the image at the current timepoint
            img_t = img_array[t]

            # Prepare the image for segmentation
            scaled_img_t = scale_img(img_t)

            # Perform segmentation
            n_channels = [[0, 0]]
            logger.info(
                "Segmenting nuclei with diameter %s", self.nuc_diameter
            )
            try:
                n_mask_array, n_flows, n_styles = segmentation_model.eval(
                    scaled_img_t,
                    channels=n_channels,
                    diameter=self.nuc_diameter,
                    normalize=False,
                )
            except IndexError:
                n_mask_array = np.zeros_like(scaled_img_t).astype(np.uint8)
            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(n_mask_array)
        return segmentation_masks

    def _compact_mask(self, mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compact the uint32 datatype to the smallest required to store all mask IDs.

        This method compacts the uint32 datatype to the smallest required to store all mask IDs.
        """
        m = mask.max()
        if m < 2**8:
            return mask.astype(np.uint8)
        if m < 2**16:
            return mask.astype(np.uint16)
        return mask


class NucImageProperties:
    """Extracts feature measurements from segmented nuclei and generates combined data frames.

    This class extracts quantitative features from segmented nuclei in OMERO images,
    aggregates them into pandas DataFrames, and provides quality control metrics.
    It supports multi-channel and multi-timepoint images, and is designed for
    downstream analysis and quality control in high-content screening experiments.

    Attributes:
        _well (WellWrapper): OMERO WellWrapper object for the current well.
        _well_id (int): OMERO well ID.
        _image (NucImage): NucImage object containing corrected images and segmentation masks.
        _meta_data (MetadataParser): Metadata parser for extracting channel and well info.
        plate_name (str): Name of the plate.
        _cond_dict (dict): Experimental condition metadata for the well.
        image_df (pd.DataFrame): DataFrame with extracted features for all nuclei.
        quality_df (pd.DataFrame): DataFrame with image quality control metrics.

    Methods:
        _combine_channels(featurelist: list[str]) -> pd.DataFrame:
            Combines feature measurements from all channels into a single DataFrame.

        _channel_data(channel: str, featurelist: list[str]) -> pd.DataFrame:
            Extracts features for a single channel and returns as a DataFrame.

        _get_properties(segmentation_mask, channel, segment, featurelist) -> pd.DataFrame:
            Measures selected features for each segmented nucleus in the given channel.

        _edit_properties(channel: str, segment: str, featurelist: list[str]) -> dict[str, str]:
            Renames feature columns to include channel and segment information.

        _set_quality_df(channel: str, corr_img: np.ndarray) -> pd.DataFrame:
            Generates a DataFrame for image quality control, saving the median intensity.

        _concat_quality_df() -> pd.DataFrame:
            Concatenates quality control DataFrames for all channels.
    """

    def __init__(
        self,
        well: WellWrapper,
        image_obj: NucImage,
        meta_data: MetadataParser,
        featurelist: list[str] = default_config.FEATURELIST,
    ):
        """Initialize the NucImageProperties object.

        This method initializes the NucImageProperties object with the given parameters.

        Args:
            well: OMERO WellWrapper object for the current well.
            image_obj: NucImage object containing corrected images and segmentation masks.
            meta_data: MetadataParser object for extracting channel and well info.
            featurelist: List of features to extract from the segmented nuclei.
        """
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._meta_data = meta_data

        self.plate_name = meta_data.plate.getName()
        # Get the dict[str, Any] for the given well
        self._cond_dict = meta_data.well_conditions(well.getWellPos())
        self.image_df = self._combine_channels(featurelist)
        self.quality_df = self._concat_quality_df()

    def _combine_channels(self, featurelist: list[str]) -> pd.DataFrame:
        """Combine feature measurements from all channels into a single DataFrame.

        This method processes the segmented masks for each channel and combines the measurements into a single DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing feature measurements for all regions and channels.
        """
        channel_data = [
            self._channel_data(channel, featurelist)
            for channel in self._meta_data.channel_data
        ]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[
            :, ~props_data.columns.duplicated()
        ].copy()
        cond_list = [
            self.plate_name,
            self._meta_data.plate.getId(),
            self._well.getWellPos(),
            self._well_id,
            self._image.omero_image.getId(),
        ]
        cond_list.extend(iter(self._cond_dict.values()))
        col_list = ["experiment", "plate_id", "well", "well_id", "image_id"]
        col_list.extend(iter(self._cond_dict.keys()))
        col_list_edited = [entry.lower() for entry in col_list]
        edited_props_data[col_list_edited] = cond_list

        return edited_props_data.sort_values(by=["timepoint"]).reset_index(
            drop=True
        )

    def _channel_data(
        self, channel: str, featurelist: list[str]
    ) -> pd.DataFrame:
        """Processes the segmented masks for a specific channel and combines the measurements into a single DataFrame.

        This method extracts quantitative features from the segmented masks for a given channel and combines them with the overlay DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing feature measurements for the given channel.
        """
        nucleus_data = self._get_properties(
            self._image.n_mask, channel, "nucleus", featurelist
        )
        if channel == "DAPI":
            nucleus_data["integrated_int_DAPI"] = (
                nucleus_data["intensity_mean_DAPI_nucleus"]
                * nucleus_data["area_nucleus"]
            )
        return nucleus_data

    def _get_properties(
        self,
        segmentation_mask: npt.NDArray[Any],
        channel: str,
        segment: str,
        featurelist: list[str],
    ) -> pd.DataFrame:
        """Measure selected features for each segmented cell in given channel.

        This method measures the selected features for each segmented cell in the given channel.

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
                    channel, segment, featurelist
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
            feature_dict = self._edit_properties(channel, segment, featurelist)
            data = data.rename(columns=feature_dict)
            data["timepoint"] = 0  # Add timepoint 0 for single timepoint data
            return data.sort_values(by=["label"]).reset_index(drop=True)

    @staticmethod
    def _edit_properties(
        channel: str, segment: str, featurelist: list[str]
    ) -> dict[str, str]:
        """Edit the properties of the features.

        This method edits the properties of the features to be used in the DataFrame.

        Returns:
            dict[str, str]: Dictionary mapping feature names to their edited names.
        """
        feature_dict = {
            feature: f"{feature}_{channel}_{segment}"
            for feature in featurelist[2:]
        }
        feature_dict["area"] = (
            f"area_{segment}"  # the area is the same for each channel
        )
        return feature_dict

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
                "plate_id": [self._meta_data.plate.getId()],
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
