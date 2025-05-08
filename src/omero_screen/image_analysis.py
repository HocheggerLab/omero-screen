#!/usr/bin/env python
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

from omero_screen import default_config
from omero_screen.config import getenv_as_bool
from omero_screen.general_functions import filter_segmentation, scale_img
from omero_screen.metadata_parser import MetadataParser

logger = logging.getLogger("omero-screen")


class Image:
    """
    Generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask and cyto_mask arrays.
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
        self._conn = conn
        self._well = well
        self.omero_image = image_obj
        self._meta_data = metadata
        self.dataset_id = dataset_id
        self._flatfield_dict = flatfield_dict

        self._get_metadata()
        self.nuc_diameter = (
            10  # default value for nuclei diameter for 10x images
        )
        self.img_dict = self._get_img_dict()

        self.n_mask, self.c_mask, self.cyto_mask = self._segmentation()

    def _get_metadata(self) -> None:
        self.channels = self._meta_data.channel_data
        self.well_pos = self._well.getWellPos()
        self.cell_line = self._meta_data.well_conditions(self.well_pos)[
            "cell_line"
        ]

    def _get_img_dict(self) -> dict[str, npt.NDArray[Any]]:
        """divide image_array with flatfield correction mask and return dictionary "channel_name": corrected image"""
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

    def _segmentation(
        self,
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any] | None
    ]:
        # check if masks already exist
        image_name = f"{self.omero_image.getId()}_segmentation"
        dataset = self._conn.getObject("Dataset", self.dataset_id)
        image_id = None
        for image in dataset.listChildren():
            if image.getName() == image_name:
                image_id = image.getId()
                logger.info("Segmentation masks found for image %s", image_id)
                # masks is TZYXC
                _, masks = get_image(self._conn, image_id)
                if "Tub" in self.channels:
                    self.n_mask, self.c_mask = masks[..., 0], masks[..., 1]
                    self.cyto_mask = self._get_cyto()
                else:
                    self.n_mask = masks[..., 0]
                    self.c_mask = None
                    self.cyto_mask = None
                break  # stop the loop once the image is found
        if image_id is None:
            n_mask = self._n_segmentation()
            if "Tub" in self.channels:
                c_mask = self._c_segmentation()
                self.n_mask, self.c_mask = self._compact_mask(
                    np.stack([n_mask, c_mask])
                )
                self.cyto_mask = self._get_cyto()
            else:
                self.n_mask = self._compact_mask(n_mask)
                self.c_mask = None
                self.cyto_mask = None

            upload_masks(
                self._conn,
                self.dataset_id,
                self.omero_image,
                self.n_mask,
                self.c_mask,
            )
        return self.n_mask, self.c_mask, self.cyto_mask

    def _get_cyto(self) -> npt.NDArray[Any] | None:
        """substract nuclei mask from cell mask to get cytoplasm mask"""
        if self.c_mask is None:
            return None
        overlap = (self.c_mask != 0) * (self.n_mask != 0)
        cyto_mask_binary = (self.c_mask != 0) * (overlap == 0)
        return self.c_mask * cyto_mask_binary

    def _n_segmentation(self) -> npt.NDArray[Any]:
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

    def _c_segmentation(self) -> npt.NDArray[Any]:
        """Perform cellpose segmentation using cell mask"""
        segmentation_model = models.CellposeModel(
            gpu=getenv_as_bool("GPU", default=torch.cuda.is_available()),
            model_type=self._get_models(),
        )
        c_channels = [[2, 1]]

        # Get the image arrays for DAPI and Tubulin channels
        dapi_array = self.img_dict["DAPI"]
        tub_array = self.img_dict["Tub"]

        # Check if the time dimension matches
        assert dapi_array.shape[0] == tub_array.shape[0], (
            "Time dimension mismatch between DAPI and Tubulin channels"
        )

        # Initialize an array to store the segmentation masks
        segmentation_masks = np.zeros_like(dapi_array, dtype=np.uint32)

        # Process each timepoint
        for t in range(dapi_array.shape[0]):
            # Select the images at the current timepoint
            dapi_t = dapi_array[t]
            tub_t = tub_array[t]

            # Combine the 2 channel numpy array for cell segmentation with the nuclei channel
            comb_image_t = scale_img(np.dstack([dapi_t, tub_t]))

            # Perform segmentation
            try:
                c_masks_array, c_flows, c_styles = segmentation_model.eval(
                    comb_image_t, channels=c_channels, normalize=False
                )
            except IndexError:
                c_masks_array = np.zeros_like(comb_image_t).astype(np.uint8)

            # Store the segmentation mask in the corresponding timepoint
            segmentation_masks[t] = filter_segmentation(c_masks_array)
        return segmentation_masks

    def _get_models(self) -> str:
        """Matches well with cell line and gets model_path for cell line from plate_layout.
        Returns:
            path to model
        """
        cell_line = self.cell_line.replace(
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
        else:
            return default_config.MODEL_DICT["U2OS"]

    def _compact_mask(self, mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compact the uint32 datatype to the smallest required to store all mask IDs"""
        m = mask.max()
        if m < 2**8:
            return mask.astype(np.uint8)
        if m < 2**16:
            return mask.astype(np.uint16)
        return mask


class ImageProperties:
    """
    Extracts feature measurements from segmented nuclei, cells and cytoplasm
    and generates combined data frames.
    """

    def __init__(
        self,
        well: WellWrapper,
        image_obj: Image,
        meta_data: MetadataParser,
        featurelist: list[str] = default_config.FEATURELIST,
        image_classifier: None = None,
    ):
        self._well = well
        self._well_id = well.getId()
        self._image = image_obj
        self._meta_data = meta_data
        self.image_df = pd.DataFrame()
        self.quality_df = pd.DataFrame()

    #     self.plate_name = meta_data.plate.getName()
    # # TODO: add method to get the dict[str, Any] for the given well id
    #     self._cond_dict = image_obj._meta_data.well_conditions(self._well_id)
    #     self._overlay = self._overlay_mask()
    #     self.image_df = self._combine_channels(featurelist)
    #     self.quality_df = self._concat_quality_df()

    #     if image_classifier is not None:
    #         for cls in image_classifier:
    #             if cls.select_channels(image_obj.img_dict):
    #                 self.image_df = cls.process_images(
    #                     self.image_df, image_obj.c_mask
    #                 )

    # def _overlay_mask(self) -> pd.DataFrame:
    #     """Links nuclear IDs with cell IDs"""
    #     if self._image.c_mask is None:
    #         return pd.DataFrame({"label": self._image.n_mask.flatten()})

    #     overlap = (self._image.c_mask != 0) * (self._image.n_mask != 0)
    #     list_n_masks = np.stack(
    #         [self._image.n_mask[overlap], self._image.c_mask[overlap]]
    #     )[-2].tolist()
    #     list_masks = np.stack(
    #         [self._image.n_mask[overlap], self._image.c_mask[overlap]]
    #     )[-1].tolist()
    #     overlay_all = {
    #         list_n_masks[i]: list_masks[i] for i in range(len(list_n_masks))
    #     }
    #     return pd.DataFrame(
    #         list(overlay_all.items()), columns=["label", "Cyto_ID"]
    #     )

    # @staticmethod
    # def _edit_properties(channel, segment, featurelist):
    #     """generates a dictionary with"""
    #     feature_dict = {
    #         feature: f"{feature}_{channel}_{segment}"
    #         for feature in featurelist[2:]
    #     }
    #     feature_dict["area"] = (
    #         f"area_{segment}"  # the area is the same for each channel
    #     )
    #     return feature_dict

    # def _get_properties(
    #     self, segmentation_mask, channel, segment, featurelist
    # ):
    #     """Measure selected features for each segmented cell in given channel"""
    #     timepoints = self._image.img_dict[channel].shape[0]
    #     label = np.squeeze(segmentation_mask).astype(np.int64)

    #     if timepoints > 1:
    #         data_list = []
    #         for t in range(timepoints):
    #             props = measure.regionprops_table(
    #                 label[t],
    #                 np.squeeze(self._image.img_dict[channel][t]),
    #                 properties=featurelist,
    #             )
    #             data = pd.DataFrame(props)
    #             feature_dict = self._edit_properties(
    #                 channel, segment, featurelist
    #             )
    #             data = data.rename(columns=feature_dict)
    #             data["timepoint"] = t  # Add timepoint for all channels
    #             data_list.append(data)
    #         combined_data = pd.concat(data_list, axis=0, ignore_index=True)
    #         return combined_data.sort_values(
    #             by=["timepoint", "label"]
    #         ).reset_index(drop=True)
    #     else:
    #         props = measure.regionprops_table(
    #             label,
    #             np.squeeze(self._image.img_dict[channel]),
    #             properties=featurelist,
    #         )
    #         data = pd.DataFrame(props)
    #         feature_dict = self._edit_properties(channel, segment, featurelist)
    #         data = data.rename(columns=feature_dict)
    #         data["timepoint"] = 0  # Add timepoint 0 for single timepoint data
    #         return data.sort_values(by=["label"]).reset_index(drop=True)

    # def _channel_data(self, channel, featurelist):
    #     nucleus_data = self._get_properties(
    #         self._image.n_mask, channel, "nucleus", featurelist
    #     )
    #     # merge channel data, outer merge combines all area columns into 1
    #     if self._image.c_mask is not None:
    #         nucleus_data = self._outer_merge(
    #             nucleus_data, self._overlay, "label"
    #         )
    #     if channel == "DAPI":
    #         nucleus_data["integrated_int_DAPI"] = (
    #             nucleus_data["intensity_mean_DAPI_nucleus"]
    #             * nucleus_data["area_nucleus"]
    #         )

    #     if self._image.c_mask is not None:
    #         cell_data = self._get_properties(
    #             self._image.c_mask, channel, "cell", featurelist
    #         )
    #         cyto_data = self._get_properties(
    #             self._image.cyto_mask, channel, "cyto", featurelist
    #         )
    #         merge_1 = self._outer_merge(
    #             cell_data, cyto_data, ["label", "timepoint"]
    #         )
    #         merge_1 = merge_1.rename(columns={"label": "Cyto_ID"})
    #         return self._outer_merge(
    #             nucleus_data, merge_1, ["Cyto_ID", "timepoint"]
    #         )
    #     else:
    #         return nucleus_data

    # def _outer_merge(self, df1, df2, on):
    #     """Perform an outer-join merge on the two pandas dataframes. NA rows are removed and integer columns are restored."""
    #     df = pd.merge(df1, df2, how="outer", on=on).dropna(axis=0, how="any")
    #     # Outer-join merge will create columns that support NA. This changes int columns to float.
    #     # After dropping all the NA rows restore the int columns.
    #     for c in df1.columns:
    #         if is_integer_dtype(df1[c].dtype) and not is_integer_dtype(
    #             df[c].dtype
    #         ):
    #             df[c] = df[c].astype(df1[c].dtype)
    #     for c in df2.columns:
    #         if is_integer_dtype(df2[c].dtype) and not is_integer_dtype(
    #             df[c].dtype
    #         ):
    #             df[c] = df[c].astype(df2[c].dtype)
    #     return df

    # def _combine_channels(self, featurelist):
    #     channel_data = [
    #         self._channel_data(channel, featurelist)
    #         for channel in self._meta_data.channels
    #     ]
    #     props_data = pd.concat(channel_data, axis=1, join="inner")
    #     edited_props_data = props_data.loc[
    #         :, ~props_data.columns.duplicated()
    #     ].copy()
    #     cond_list = [
    #         self.plate_name,
    #         self._meta_data.plate_obj.getId(),
    #         self._well.getWellPos(),
    #         self._well_id,
    #         self._image.omero_image.getId(),
    #     ]
    #     cond_list.extend(iter(self._cond_dict.values()))
    #     col_list = ["experiment", "plate_id", "well", "well_id", "image_id"]
    #     col_list.extend(iter(self._cond_dict.keys()))
    #     col_list_edited = [entry.lower() for entry in col_list]
    #     edited_props_data[col_list_edited] = cond_list

    #     return edited_props_data.sort_values(by=["timepoint"]).reset_index(
    #         drop=True
    #     )

    # def _set_quality_df(self, channel, corr_img):
    #     """generates df for image quality control saving the median intensity of the image"""
    #     return pd.DataFrame(
    #         {
    #             "experiment": [self.plate_name],
    #             "plate_id": [self._meta_data.plate_obj.getId()],
    #             "position": [self._image.well_pos],
    #             "image_id": [self._image.omero_image.getId()],
    #             "channel": [channel],
    #             "intensity_median": [np.median(corr_img)],
    #         }
    #     )

    # def _concat_quality_df(self) -> pd.DataFrame:
    #     """Concatenate quality dfs for all channels in _corr_img_dict"""
    #     df_list = [
    #         self._set_quality_df(channel, image)
    #         for channel, image in self._image.img_dict.items()
    #     ]
    #     return pd.concat(df_list)
