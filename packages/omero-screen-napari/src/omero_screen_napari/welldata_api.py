"""
This module contains the classes and functions to parse the plate, well, image and label data from the Omero Plate
and collect them in the omero_data data clasee to be used by the napari viewer.
"""

import random
import re
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import omero
import polars as pl
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    ImageWrapper,
    MapAnnotationWrapper,
    PlateWrapper,
    WellWrapper,
    _OriginalFileWrapper,
)
from omero_screen.config import get_logger
from omero_utils import omero_connect
from qtpy.QtWidgets import QMessageBox
from tqdm import tqdm

from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import (
    omero_data,
    reset_omero_data,
)
from omero_screen_napari.utils import correct_channel_order

# Initialize logger with the module's name
logger = get_logger(__name__)


get_image: Callable = get_image  # allows type hints for get_image


@omero_connect
def parse_omero_data(
    omero_data: OmeroData,
    plate_id: str,
    well_pos: str,
    image_input: str,
    conn: Optional[BlitzGateway] = None,
) -> None:
    """
    This functions controls the flow of the data
    via the seperate parser classes to the omero_data class that stores omero plate metadata
    to be supplied to the napari viewer.
    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        plate_id (int): Omero Screen Plate ID number provided by user via welldata_widget
        well_list (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        image_input (str): String Describing the combinations of images to be supplied for each well
        The images options are 'All'; 0-3; 1, 3, 4.
        conn (BlitzGateway): BlitzGateway connection
    """
    plate_number = int(plate_id)
    if conn is not None:
        try:
            parse_plate_data(
                omero_data, plate_number, well_pos, image_input, conn
            )
            # clear well and image data to avoid appending to existing data
            omero_data.reset_well_and_image_data()
            for well_pos in omero_data.well_pos_list:
                logger.info(f"Processing well {well_pos}")  # noqa: G004
                well_image_parser(omero_data, well_pos, conn)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error parsing plate data: %s\n%s",
                e,
                "".join(traceback.format_exception(None, e, e.__traceback__)),
            )
            if not hasattr(sys, "ps1") and "pytest" not in sys.modules:
                # Show QMessageBox only in GUI mode
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setText(str(e))
                msg_box.setWindowTitle("Error")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()


def parse_plate_data(
    omero_data: OmeroData,
    plate_id: int,
    well_pos: str,
    image_input: str,
    conn: BlitzGateway,
) -> None:
    """
    Function that combines the different parser classes that are responsible to process userinput
    and plate data and collect data to populate the plate related fields in the omero_data data class.

    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        plate_id (int): Omero Screen Plate ID number provided by user via welldata_widget
        well_pos (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        image_input (str):  String Describing the combinations of images to be supplied for each well
                            The images options are 'All'; 0-3; 1, 3, 4.
        conn (BlitzGateway): BlitzGateway connection
    """
    # check if plate-ID already supplied, if it is then skip PlateHandler
    if plate_id != omero_data.plate_id:
        reset_omero_data()
        omero_data.plate_id = plate_id
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn
        )
        user_input.parse_data()
        logger.info("Loaded data for plate with ID %s", plate_id)
        csv_parser = CsvFileParser(omero_data)
        csv_parser.parse_csv()
        channel_parser = ChannelDataParser(omero_data)
        channel_parser.parse_channel_data()
        flatfield_parser = FlatfieldMaskParser(omero_data, conn)
        flatfield_parser.parse_flatfieldmask()
        scale_intensity_parser = ScaleIntensityParser(omero_data)
        scale_intensity_parser.parse_intensities()
        pixel_size_parser = PixelSizeParser(omero_data)
        pixel_size_parser.parse_pixel_size_values()

    else:
        logger.info(
            "Plate data for plate %s already exists. Skip reload.", plate_id
        )
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn
        )
        user_input.parse_data()


def well_image_parser(
    omero_data: OmeroData, well_pos: str, conn: BlitzGateway
) -> None:
    """
    Function that combines the different parser classes that are responsible to process well, image and label data.
    The data is collected to populate the well/image related fields in the omero_data data class.

    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        well_pos (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        conn (BlitzGateway): BlitzGateway connection
    """
    well_data_parser = WellDataParser(omero_data, well_pos)
    well_data_parser.parse_well()
    if well := well_data_parser._well:
        image_parser = ImageParser(omero_data, well, conn)
        image_parser.parse_images_and_labels()


# -----------------------------------------------USER INPUT -----------------------------------------------------
class UserInput:
    """
    Class to handle user input for plate data retrieval from Omero.
    The class checks if the plate exists in Omero, parses the image number, well positions and image index
    and stores them in the omero_data class.
    public methods:
    parse_data: Orchestrate the data parsing process.
    private methods:
    _check_plate_id: Verifies that plate_id input points to an existing plate in Omero.
    _parse_image_number: Infers the number of images per well in the first well of the plate.
    _well_data_parser: Parses the well data input string and checks if the well position inputs are valid.
    _image_index_parser: Parses the image index input string and checks if the image index inputs are valid.
    """

    def __init__(
        self,
        omero_data: OmeroData,
        plate_id: int,
        well_pos: str,
        image_input: str,
        conn: BlitzGateway,
    ):
        self._omero_data: OmeroData = omero_data
        self._plate_id: int = plate_id
        self._well_pos: str = well_pos
        self._images: str = image_input
        self._conn: BlitzGateway = conn
        self._plate: Optional[PlateWrapper] = None  # added by _check_plate_id
        self._image_number: int = 0  # added by _parse_image_number
        self._well_pos_list: list[str] = []  # added by _well_data_parser

    def parse_data(self) -> None:
        self._check_plate_id()
        self._load_dataset()
        self._parse_image_number()
        self._well_data_parser()
        self._omero_data.well_pos_list = self._well_pos_list
        self._image_index_parser()
        self._omero_data.image_index = self._image_index

    def _check_plate_id(self) -> None:
        """
        Verifies that plate_id input points to an existing plate in Omero.
        Raises:
            ValueError: If the plate does not exist in Omero.
        """
        self._plate = self._conn.getObject("Plate", self._plate_id)
        if not self._plate:
            logger.error("Plate with ID %s does not exist.", self._plate_id)
            raise ValueError(f"Plate with ID {self._plate_id} does not exist.")
        else:
            self._omero_data.plate = self._plate

    def _load_dataset(self) -> None:
        """
        Looks for a data set that matches the plate name in the screen project
        with id supplied by omero_data.project_id via the env variable.
        if it finds the data set it assigns it to the omero_data object
        otherwise it will throw an error becuase the program cant proceed without
        flatfielmasks and segmentationlabels stored in the data set.

        Raises:
            ValueError: If the plate has not been assigned a dataset.
        """
        project = self._conn.getObject("Project", self._omero_data.project_id)
        if not project:
            logger.error(
                "Project for screen with ID %s not found.",
                self._omero_data.project_id,
            )
            raise ValueError(
                f"Project for screen with ID {self._omero_data.project_id} not found."
            )
        if dataset := self._conn.getObject(
            "Dataset",
            attributes={"name": str(self._omero_data.plate_id)},
            opts={"project": project.getId()},
        ):
            self._omero_data.screen_dataset = dataset
        else:
            logger.error(
                "The plate %s has not been assigned a dataset.",
                omero_data.plate_name,
            )
            raise ValueError(
                f"The plate {omero_data.plate_name} has not been assigned a dataset."
            )

    def _parse_image_number(self) -> None:
        """
        Infers the number of images per well in the first well of the plate.
        This presumes that each well has the same number of images.
        Raises:
            ValueError: If no plate is found.
        """
        if self._plate:
            first_well = list(self._plate.listChildren())[0]
            self._image_number = len(list(first_well.listChildren()))
        else:
            logger.error("No plate found, unable to parse image number.")
            raise ValueError("No plate found, unable to parse image number.")

    def _well_data_parser(self) -> None:
        """
        Parses the well data input string and checks if the well position inputs are valid.
        Raises:
            ValueError: If the well input format is invalid.
        """
        pattern = re.compile(r"^[A-H][1-9]|1[0-2]$")
        self._well_pos_list = [
            item.strip() for item in self._well_pos.split(",")
        ]
        for item in self._well_pos_list:
            if not pattern.match(item):
                logger.error(
                    "Invalid well input format: '%s'. Must be A-H followed by 1-12.",
                    item,
                )
                raise ValueError(
                    f"Invalid well input format: '{item}'. Must be A-H followed by 1-12."
                )

    def _image_index_parser(self) -> None:
        """
        Parses the image index input string and checks if the image index inputs are valid.
        Raises:
            ValueError: If the image index input format is invalid.
        """
        self._omero_data.image_input = (
            self._images
        )  # store this for use in saving training data
        index = self._images

        if not (
            index.lower() == "all"
            or re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index)
        ):
            logger.error(
                "Image input '%s' doesn't match any of the expected patterns 'All, 1-3, 1'.",
                index,
            )
            raise ValueError(
                f"Image input '{index}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )

        if index.lower() == "all":
            self._image_index = list(
                range(self._image_number)
            )  # Assuming well count is inclusive and starts at 1
        elif "-" in index:
            # Handle range, e.g., '1-3'
            start, end = map(int, index.split("-"))
            self._image_index = list(range(start, end + 1))
        elif "," in index:
            # Handle list, e.g., '1, 4, 5'
            self._image_index = list(map(int, index.split(", ")))
        else:
            # Handle single number, e.g., '1'
            self._image_index = [int(index)]


# -----------------------------------------------PLATE DATA -----------------------------------------------------

# classes to parse the plate data from the Omero Plate: csv file, channel data, flatfield mask, pixel size,
# image intensity and pixels size

# -----------------------------------------------CSV FILE -----------------------------------------------------


class CsvFileParser:
    """
    Class to handle csv file retrieval and processing from Omero Plate
    public methods:
    parse_csv: Orchestrate the csv file handling. Check if csv file is available, if not download it.
    and make it available to the omero_data class as a polars LazyFrame object.
    private methods:
    _csv_available: checks if any csv file in the target directory contains the string self.plate_id in its name.
    _get_csv_file: gets the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
    _download_csv: downloads the csv file from the Omero Plate. If _cc.csv the file name will be
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_id: int = omero_data.plate_id
        self._plate = omero_data.plate
        self._data_path: Path = omero_data.data_path
        self._csv_file_path: Optional[Path] = None
        self._file_name: Optional[str] = None
        self._original_file: Optional[_OriginalFileWrapper] = None

    def parse_csv(self) -> None:
        """Orchestrate the csv file handling. Check if csv file is available, if not download it.
        and make it available to the omero_data class as a polars LazyFrame object.

        """
        if not self._csv_available():
            logger.info("Downloading csv file from Omero")  # noqa: G004
            self._get_csv_file()
            self._download_csv()
        else:  # csv file already exists
            logger.info(
                "CSV file already exists in local directory. Skipping download."
            )
        if self._csv_file_path:
            omero_data.plate_data = pl.scan_csv(self._csv_file_path)
            self._omero_data.csv_path = self._csv_file_path
        else:
            logger.error("No csv file assigned by CsvParser.")
            raise ValueError("No csv file by CsvParser")

    # helper functions for csv handling

    def _csv_available(self) -> bool:
        """
        Check if any csv file in the directory contains the string self.plate_id in its name.
        """
        # make the directory to store the csv file if it does not exist
        self._data_path.mkdir(exist_ok=True)
        for file in self._data_path.iterdir():
            if file.is_file() and str(self._plate_id) in file.name:
                self._csv_file_path = file
                return True
        return False

    def _get_csv_file(self) -> None:
        """
        Get the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
        """
        self._original_file = None
        file_anns = self._plate.listAnnotations()
        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper):
                name = ann.getFile().getName()
                if name and name.endswith("_cc.csv"):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    break  # Prioritize and stop searching if _cc.csv is found
                elif (
                    name
                    and name.endswith("final_data.csv")
                    and self._original_file is None
                ):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    # Don't break to continue searching for a _cc.csv file

        if self._original_file is None:
            logger.error("No suitable csv file found for the plate.")
            raise ValueError("No suitable csv file found for the plate.")

    def _download_csv(self) -> None:
        """
        Download the csv file from the Omero Plate. If _cc.csv the file name will be
        {plate_id}_cc.csv, otherwise {plate_id}_data.csv
        """
        if self._file_name and self._original_file:
            saved_name = "{}_{}".format(
                self._plate_id, self._file_name.split("_")[-1]
            )
            self._csv_file_path = self._data_path / saved_name
            with open(self._csv_file_path, "wb") as file_on_disk:
                for chunk in self._original_file.asFileObj():
                    file_on_disk.write(chunk)
        else:
            logger.error("Problem with parsing csv file.")
            raise ValueError("Problem with parsing csv file.")


# -----------------------------------------------CHANNEL DATA -----------------------------------------------------


class ChannelDataParser:
    """
    Class to handle channel data retrieval and processing from Omero Plate.
    public method:
    parse_channel_data: Coordinates the retrieval of channel data from Omero Plate
    private methods:
    _get_map_ann: Get channel information from Omero map annotations or raise exception if no appropriate annotation is found.
    _tidy_up_channel_data: Helper method that processes the channel data to a dictionary
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data: OmeroData = omero_data
        self._plate: PlateWrapper = omero_data.plate
        self._plate_id: int = omero_data.plate_id

    def parse_channel_data(self) -> None:
        """Process channel data to dictionary {channel_name: channel_index}
        and add it to viewer_data; raise exception if no channel data found."""
        self._get_map_ann()
        self._tidy_up_channel_data()
        self._omero_data.channel_data = self._channel_data
        logger.debug(
            "Channel data: %s loaded to omero_data.channel_data",
            self._channel_data,
        )  # noqa: G004

    def _get_map_ann(self) -> None:
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """

        annotations = self._plate.listAnnotations()

        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]

        if not map_annotations:
            logger.error(
                "No MapAnnotations found for plate %s.", self._plate_id
            )  # noqa: G004
            raise ValueError(
                f"No MapAnnotations found for plate {self._plate_id}."
            )

        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for key, _value in ann_value:
                if key.lower() in ["dapi", "hoechst", "rfp"]:
                    self._map_annotations = ann_value
                    return  # Return the first matching map annotation's value

        # If no matching annotation is found, raise a specific exception
        logger.error(
            "No DAPI or Hoechst channel information found for plate %s.",
            self._plate_id,
        )  # noqa: G004
        raise ValueError(
            f"No DAPI or Hoechst channel information found for platplate {self._omero_data.plate_id}."
        )

    def _tidy_up_channel_data(self) -> None:
        """
        Convert channel map annotation from list of tuples to dictionary.
        Eliminate spaces and swap Hoechst to DAPI if necessary.
        """
        channel_data = dict(self._map_annotations)
        sorted_channel_data = dict(
            sorted(channel_data.items(), key=lambda item: item[1])
        )
        self._channel_data = {
            key.strip(): value for key, value in sorted_channel_data.items()
        }
        if "Hoechst" in self._channel_data:
            self._channel_data["DAPI"] = self._channel_data.pop("Hoechst")
        if "RFP" in self._channel_data:
            self._channel_data["DAPI"] = self._channel_data.pop("RFP")
        # check if one of the channels is DAPI otherwise raise exception


# -----------------------------------------------FLATFIELD CORRECTION-----------------------------------------------------


class FlatfieldMaskParser:
    """
    Class to handle flatfield mask retrieval and processing from Omero Plate.
    public method:
    parse_flatfieldmask: Coordinates the retrieval of flatfield mask from Omero Plate
    private methods:
    _load_dataset: Looks for a data set that matches the plate name in the screen project
    _get_flatfieldmask: Gets flatfieldmasks from project linked to screen
    _get_map_ann: Get channel information from Omero map annotations or raise exception if no appropriate annotation is found.
    _check_flatfieldmask: Checks if the mappings with the plate channel date are consistent.
    """

    def __init__(self, omero_data: OmeroData, conn: BlitzGateway):
        self._omero_data: OmeroData = omero_data
        self._conn: BlitzGateway = conn

    def parse_flatfieldmask(self) -> None:
        """
        Coordinates the retrieval of flatfield mask from Omero Plate
        """
        self._get_flatfieldmask()
        self._get_map_ann()
        self._check_flatfieldmask()
        omero_data.flatfield_masks = self._flatfield_array

    def _get_flatfieldmask(self) -> None:
        """Gets flatfieldmasks from project linked to screen"""

        flatfield_mask_name = f"{self._omero_data.plate_id}_flatfield_masks"
        logger.debug("Flatfield mask name: %s", flatfield_mask_name)  # noqa: G004
        flatfield_mask_found = False
        for image in self._omero_data.screen_dataset.listChildren():
            image_name = image.getName()
            if flatfield_mask_name == image_name:
                flatfield_mask_found = True
                flatfield_mask_id = image.getId()
                self._flatfield_obj, self._flatfield_array = get_image(
                    self._conn, flatfield_mask_id
                )
                break  # Exit the loop once the flatfield mask is found
        if not flatfield_mask_found:
            logger.error(
                "No flatfieldmasks found in dataset %s.",
                self._omero_data.screen_dataset,
            )  # noqa: G004
            raise ValueError(
                "No flatfieldmasks found in dataset %s.",
                self._omero_data.screen_dataset,
            )

    def _get_map_ann(self) -> None:
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """
        annotations = self._flatfield_obj.listAnnotations()
        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]
        if not map_annotations:
            logger.error(
                "No Flatfield Mask Channel info found in datset %s.",
                self._omero_data.screen_dataset,
            )  # noqa: G004
            raise ValueError(
                f"No Flatfield Mask Channel info found in datset {self._omero_data.screen_dataset}."
            )
        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for _key, value in ann_value:
                if value.lower().strip() in ["dapi", "hoechst", "rfp"]:
                    self._flatfield_channels = dict(ann_value)
                    logger.debug(
                        "Flatfield mask channels: %s", self._flatfield_channels
                    )
                    return
        # If no matching annotation is found, raise a specific exception
        logger.error(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )  # noqa: G004
        raise ValueError(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )

    def _check_flatfieldmask(self) -> None:
        """
        Checks if the mappings with the plate channel date are consistent.
        self.flatfield_channels are {'channel_0' : 'DAPI}
        omero_data.channel_data are {'DAPI' : 0}

        Raises :
            InconsistencyError: If the mappings are inconsistent.
        """

        class InconsistencyError(ValueError):
            pass

        logger.debug(
            "Flatfield channels: %s, channel data %s",
            self._flatfield_channels.values(),
            omero_data.channel_data.keys(),
        )  # noqa: G004
        channel_check = True
        for key, value in self._flatfield_channels.items():
            # Extract the number from key, e.g., 'channel_1' -> '1'
            channel_number = key.split("_")[-1]

            # Find the channel name that corresponds to this number in channels
            expected_name = None
            for name, num in self._omero_data.channel_data.items():
                if num == channel_number:
                    expected_name = name
                    break
            # Check if the expected name matches the name in flatfield_channels
            if expected_name != value:
                channel_check = False
                break

        if not channel_check:
            error_message = "Inconsistency found: flatfield_mask and plate_map have different channels"
            logger.error(error_message)
            raise InconsistencyError(error_message)


# -----------------------------------------------SCALE INTENSITY -----------------------------------------------------


class ScaleIntensityParser:
    """
    The class extracts the caling values for image contrasts to display the different channels in napari.
    To compare intesities across different wells a single global contrasting value is set. This is based on the
    the min and max values for each channel. These data are extracted from the polars LazyFrame object stored in
    omero_data.plate_data.
    public method:
    parse_intensities: Parse the intensities from the plate data and store them in the omero_data object.
    private methods:
    _set_keyword: Set the keyword based on the presence of "_cell" or "_nucleus" in the dataframe columns.
    _get_values: Filter the dataframe to only include columns with the channel keyword.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_data: pl.LazyFrame = omero_data.plate_data
        self._keyword: Optional[str] = None
        self._intensities: Optional[dict[int, tuple[int, int]]] = None

    def parse_intensities(self) -> None:
        """
        Parse the intensities from the plate data and store them in the omero_data object.
        Raises:
            ValueError: when intesities cannot be collected from the plate csv data.
        """
        self._set_keyword()
        self._get_values()
        if self._intensities:
            self._omero_data.intensities = self._intensities  # type: ignore
        else:
            logger.error("Problem with loading intensities to scale channels.")
            raise ValueError(
                "Problem with loading intensities to scale channels."
            )

    def _set_keyword(self) -> None:
        """
        Set the keyword based on the presence of "_cell" or "_nucleus" in the dataframe columns.
        Raises:
            ValueError: when the dataframe does not contain the channel keywords.
            ValueError: when neither "_cell" nor "_nucleus" is present in the dataframe columns.
        """
        if not hasattr(self, "_plate_data") or self._plate_data is None:
            raise ValueError("Dataframe 'plate_data' does not exist.")

        # Initialize keyword to None
        self._keyword = None

        # Check for presence of "_cell" and "_nucleus" in the list of strings
        has_cell = any("_cell" in s for s in self._plate_data.columns)
        has_nucleus = any("_nucleus" in s for s in self._plate_data.columns)

        # Set keyword based on presence of "_cell" or "_nucleus"
        if has_cell:
            self._keyword = "_cell"
        elif has_nucleus:
            self._keyword = "_nucleus"
        else:
            # If neither is found, raise an exception
            raise ValueError(
                "Neither '_cell' nor '_nucleus' is present in the dataframe columns."
            )

    def _get_values(self) -> None:
        """
        Filter the dataframe to only include columns with the channel keyword.
        """
        intensity_dict = {}
        for channel, channel_value in self._omero_data.channel_data.items():
            cols = (
                f"intensity_max_{channel}{self._keyword}",
                f"intensity_min_{channel}{self._keyword}",
            )
            for col in cols:
                if col not in self._plate_data.columns:
                    logger.error("Column '%s' not found in DataFrame.", col)
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

            max_value = (
                self._plate_data.select(pl.col(cols[0])).mean().collect()
            )
            min_value = (
                self._plate_data.select(pl.col(cols[1])).min().collect()
            )

            intensity_dict[int(channel_value)] = (
                int(min_value[0, 0]),
                int(max_value[0, 0]),
            )

        self._intensities = intensity_dict


# -----------------------------------------------PIXEL SIZE -----------------------------------------------------


class PixelSizeParser:
    """
    The class extracts the pixel size from the plate metadata. For this purpose it uses the first well and image
    to extract the pixel size in X and Y dimensions. The pixel size is then stored in the omero_data class.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._random_wells: Optional[list[WellWrapper]] = None
        self._random_images: Optional[list[ImageWrapper]] = None
        self._pixel_size: Optional[tuple[float, float]] = None

    def parse_pixel_size_values(self) -> None:
        self._check_wells_and_images()
        self._check_pixel_values()
        if self._pixel_size:
            self._omero_data.pixel_size = self._pixel_size
        else:
            logger.error("Problem with loading pixel size values.")
            raise ValueError("Problem with loading pixel size values.")

    def _check_wells_and_images(self) -> None:
        """
        Check if any wells are found in the plate and if each well has images.
        Extract the first image of two random wells.
        """
        wells = list(self._plate.listChildren())
        if not wells:
            logger.debug("No wells found in the plate, raising ValueError.")
            raise ValueError("No wells found in the plate.")
        self._random_wells = random.sample(wells, 2)
        image_list = []
        for i, well in enumerate(self._random_wells):
            try:
                image = well.getImage(0)  # Attempt to get the first image
                image_list.append(image)
            except Exception as e:  # Catches any exception raised by getImage(0)  # noqa: BLE001
                logger.error("Unable to retrieve image from well %s: %s", i, e)
                raise ValueError(
                    f"Unable to retrieve image from well {i}: {e}"
                ) from e

        self._random_images = image_list

    def _get_pixel_values(self, image: ImageWrapper) -> tuple[float, float]:
        """
        Get pixel values from a single OMERO image object.
        Returns a tuple of floats for x, y values.
        """
        x_size = image.getPixelSizeX()
        y_size = image.getPixelSizeY()

        # Check if either x_size or y_size is None before rounding
        if x_size is None or y_size is None:
            logger.error(
                "No pixel data found for the image, raising ValueError."
            )
            raise ValueError("No pixel data found for the image.")

        # Since both values are not None, proceed to round them
        return (round(x_size, 1), round(y_size, 1))

    def _check_pixel_values(self) -> None:
        """
        Check if pixel values from the two random images are identical and not 0
        and load the pixel size tuple to the omero_data class.
        """
        if self._random_images:
            pixel_well_1 = self._get_pixel_values(self._random_images[0])
            pixel_well_2 = self._get_pixel_values(self._random_images[1])
            if 0 in pixel_well_1 + pixel_well_2:
                logger.error("One of the pixel sizes is 0")
                raise ValueError("One of the pixel sizes is 0")
            elif pixel_well_1 == pixel_well_2:
                self._pixel_size = pixel_well_1
            else:
                logger.error("Pixel sizes are not identical between wells")
                raise ValueError("Pixel sizes are not identical between wells")
        else:
            logger.error("No images found in the wells")
            raise ValueError("No images found in the wells")


# -----------------------------------------------Well Data-----------------------------------------------------


class WellDataParser:
    """
    Class to handle well data retrieval and processing from Omero Plate.
    Well metadata, csv data and imaged are checked and retrieved for each well
    and added to the omaro_data data class for further usage by the Napari Viewer.
    """

    def __init__(
        self,
        omero_data: OmeroData,
        well_pos: str,
    ):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._well_pos = well_pos
        self._well: Optional[WellWrapper] = None
        self._well_id: Optional[int] = None
        self._metadata: Optional[dict[str, Any]] = None
        self._well_ifdata: Optional[pl.DataFrame] = None

    def parse_well(self) -> None:
        self._parse_well_object()
        if self._well and self._well_id:
            self._omero_data.well_list.append(self._well)
            self._omero_data.well_id_list.append(int(self._well_id))
        else:
            logger.error("Problem with loading well %s data.", self._well_pos)
            raise ValueError(
                f"Problem with loading well {self._well_pos} data."
            )
        self._get_well_metadata()
        if self._metadata:
            self._omero_data.well_metadata_list.append(self._metadata)
        else:
            logger.error("No metadata found for well %s.", self._well_pos)
            raise ValueError(f"No metadata found for well {self._well_pos}.")
        self._load_well_csvdata()
        if self._well_ifdata is not None:
            self._omero_data.well_ifdata = pl.concat(
                [self._omero_data.well_ifdata, self._well_ifdata]
            )
        else:
            logger.error("No well csv data found for well %s.", self._well_pos)
            raise ValueError(
                f"No well csv data found for well {self._well_pos}."
            )

    def _parse_well_object(self) -> None:
        """
        Check if the well object exists in the plate and retrieve it.
        Raises:
            ValueError: _description_
        """
        well_found = False
        for well in self._plate.listChildren():
            if well.getWellPos() == self._well_pos:
                self._well = well
                self._well_id = well.getId()
                logger.info("Well %s retrieved", self._well_pos)
                well_found = True
                break
        if not well_found:
            logger.error(
                "Well with position %s does not exist.", self._well_pos
            )  # Raise an error if the well was not found  # Raise an error if the well was not found
            raise ValueError(
                f"Well with position {self._well_pos} does not exist."
            )

    def _get_well_metadata(self) -> None:
        """
        Get metadata for the well from the map annotations.
        The map annotations have to include a dictionary with a key "cell_line" to be valid.
        Raises:
            ValueError: If the well object is not found.
        """
        map_ann = None
        if not self._well:
            raise ValueError(f"Well at pos {self._well_pos} not found")
        for ann in self._well.listAnnotations():
            if ann and ann.getValue():
                map_ann = dict(ann.getValue())
                if "cell_line" in map_ann:
                    self._metadata = map_ann
                    break
                else:
                    logger.error(
                        "No metadata with cell line name found for well %s",
                        self._well_pos,
                    )
                    raise ValueError(
                        f"No metadata with cell line name found for well {self._well_pos}"
                    )
        if not self._metadata:
            logger.error(
                "No map annotations found for well %s", self._well_pos
            )
            raise ValueError(
                f"No map annotations found for well {self._well_pos}"
            )

    def _load_well_csvdata(self) -> None:
        """
        Load the csv data for the well using the polars lazyframe to
        filter out the required data for the well.
        """
        df = self._omero_data.plate_data
        self._well_ifdata = df.filter(
            pl.col("well") == self._well_pos
        ).collect()


# -----------------------------------------------Image Data-----------------------------------------------------
class ImageParser:
    """
    Class to handle image data retrieval and processing for a given well.
    Each image is loaded using the ezomero get_image function and then flatfield corrected and scaled.
    The indivudauk images and image names are stored in two seperat lists. The image list
    with the numpy array is finally convereted to a numpy array of the dimension
    number of images, 1080, 1080, number of channels and stored in the omero_data class as omero_data.images.
    The image ids are added to the image_ids list in the omero_data class
    """

    def __init__(
        self,
        omero_data: OmeroData,
        well: WellWrapper,
        conn: BlitzGateway,
    ):
        self._omero_data: OmeroData = omero_data
        self._well: WellWrapper = well
        self._conn: BlitzGateway = conn
        self._image_index: list[int] = self._omero_data.image_index
        self._images: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self._image_ids: list[int] = []
        self._image_arrays: list[np.ndarray[Any, np.dtype[Any]]] = []
        self._label_arrays: list[np.ndarray[Any, np.dtype[Any]]] = []

    def parse_images_and_labels(self) -> None:
        self._parse_images()
        self._parse_labels()
        logger.info("Images and labels added to omero_data")

    def _parse_images(self) -> None:
        """
        Collects images for the well and adds them to the omero_data object.
        If the omero_data object already has images, the new images are concatenated to the existing images.
        Otherwise, the new images are added to the omero_data object.
        """
        self._collect_images()

        # Determine the axes to squeeze based on the number of timepoints
        squeeze_axes = (2,) if self._image_arrays[0].shape[0] > 1 else (1, 2)

        # Stack and squeeze the new images
        new_images = np.squeeze(
            np.stack(self._image_arrays, axis=0), axis=squeeze_axes
        )
        if self._omero_data.images.shape == (0,):
            self._omero_data.images = new_images
            self._omero_data.image_ids = self._image_ids
            logger.debug(
                "Images loaded to empty omero_data.images, new shape: %s",
                self._omero_data.images.shape,
            )
        else:
            self._omero_data.images = np.concatenate(
                (self._omero_data.images, new_images), axis=0
            )
            self._omero_data.image_ids.extend(self._image_ids)

        logger.debug(
            "Images shape after processing: %s", self._omero_data.images.shape
        )

    def _parse_labels(self) -> None:
        """
        Collects labels for the well and adds them to the omero_data object.
        If the omero_data object already has labels, the new labels are concatenated to the existing labels.
        Otherwise, the new labels are added to the omero_data object.
        """
        self._collect_labels()
        if self._omero_data.labels.size == 0:
            logger.debug("Labels shape: %s", self._label_arrays[0].shape)
            self._omero_data.labels = np.stack(self._label_arrays, axis=0)

            logger.debug(
                "Labels loaded to empty omero_data.labels, new shape: %s",
                self._omero_data.labels.shape,
            )
        else:
            self._omero_data.labels = np.concatenate(
                (self._omero_data.labels, self._label_arrays), axis=0
            )
            logger.debug(
                "Labels added to omero_data.labels, new shape: %s",
                self._omero_data.labels.shape,
            )

    def _collect_images(self) -> None:
        """
        Collects images and image arrays for the well and adds the to the image_ids and image_arrays list.
        """

        logger.info("Collecting images for well %s", self._well.getWellPos())
        for index in tqdm(self._image_index):
            if image := self._well.getImage(index):
                if mip_id := self.check_mip(image):
                    logger.info(
                        "Image with index %s has Z-stacks, looking for MIP with id %s.",
                        index,
                        mip_id,
                    )
                    _, image_array = get_image(self._conn, int(mip_id))
                else:
                    _, image_array = get_image(self._conn, image.getId())
                flatfield_corrected_image = self._flatfield_correct_image(
                    image_array
                )
                logger.debug(
                    "Image shape: %s", flatfield_corrected_image.shape
                )
                self._image_arrays.append(flatfield_corrected_image)
                self._image_ids.append(image.getId())
            else:
                logger.error(
                    "Image with index %s not found in well %s",
                    index,
                    self._well.getWellPos(),
                )
                raise ValueError(
                    f"Image with index {index} not found in well {self._well.getWellPos()}"
                )

    def check_mip(self, image: ImageWrapper) -> Optional[str]:
        if map_anns := image.listAnnotations(
            ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
        ):
            for ann in map_anns:
                ann_values = dict(ann.getValue())
                for item in ann_values.values():
                    print(item)
                    if isinstance(item, str) and "mip" in item:
                        return item.split("_")[-1]
        if (
            image.getSizeZ() > 1
        ):  # check if image has z-stacks even though no mip is assigned
            raise ValueError(
                f"Image with index {self._image_index} has Z-stacks but no MIP assigned."
            )
        return None

    def _flatfield_correct_image(
        self, image_array: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Corrects the image array using the flatfield mask for the well.

        Args:
            image_array (np.ndarray): Image array to be corrected.

        Returns:
           np.ndarray: Corrected image array.
        """
        corrected_array = image_array / self._omero_data.flatfield_masks
        if len(corrected_array.shape) == 2:
            corrected_array = corrected_array[..., np.newaxis]
        logger.debug("Corrected image shape: %s", corrected_array.shape)
        return corrected_array  # type: ignore

    def _check_label_data(self) -> None:
        label_names = [
            int(image.getName().split("_")[0])
            for image in self._omero_data.screen_dataset.listChildren()
        ]
        logger.debug("Label names: %s", label_names)
        all_images_in_labels = all(
            name in label_names for name in self._image_ids
        )
        if not all_images_in_labels:
            logger.error("Available label images do not match loaded images")
            raise ValueError(
                "Available label images do not match loaded images"
            )
        else:
            logger.debug("All label images found")

    def _collect_labels(self) -> None:
        label_names = [f"{name}_segmentation" for name in self._image_ids]

        relevant_label_data = [
            label_data
            for label_data in self._omero_data.screen_dataset.listChildren()
            if any(
                label_name in label_data.getName()
                for label_name in label_names
            )
        ]

        for label_data in relevant_label_data:
            _, label_array = get_image(self._conn, label_data.getId())
            if label_array.shape[-1] == 2:
                corrected_label_array = correct_channel_order(label_array)
                self._label_arrays.append(corrected_label_array.squeeze())
            else:
                # Create a tuple of axes to squeeze, excluding the last axis
                shape = label_array.shape
                axes_to_squeeze = tuple(
                    i for i in range(len(shape) - 1) if shape[i] == 1
                )
                self._label_arrays.append(
                    np.squeeze(label_array, axis=axes_to_squeeze)
                )
