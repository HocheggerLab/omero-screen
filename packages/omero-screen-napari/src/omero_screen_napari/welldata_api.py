"""
This module contains the classes and functions to parse the plate, well, image and label data from the Omero Plate
and collect them in the omero_data data clasee to be used by the napari viewer.
"""

import os
import random
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional, cast

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from cellview.db.clean_up import (
    deep_clean_db,
)
from cellview.db.db import CellViewDB
from cellview.importers.import_functions import import_data
from cellview.utils.state import CellViewStateCore, clean_agg_data
from cellview.utils.ui import CellViewUI
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    ImageWrapper,
    MapAnnotationWrapper,
    PlateWrapper,
    WellWrapper,
)
from omero_screen.config import get_logger
from omero_screen.constants import OmeroScreenNS
from omero_utils import omero_connect
from omero_utils.attachments import get_file_attachments
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)
from tqdm import tqdm

from omero_screen_napari.omero_data import OmeroData, get_dataset_id
from omero_screen_napari.omero_data_singleton import (
    omero_data,
    reset_omero_data,
)
from omero_screen_napari.omero_image import get_image
from omero_screen_napari.utils import correct_channel_order

# Initialize logger with the module's name
logger = get_logger(__name__)


@omero_connect
def parse_omero_data(
    omero_data: OmeroData,
    plate_id: str,
    well_pos: str,
    image_input: str,
    time: str = "All",
    conn: Optional[BlitzGateway] = None,
    options: Optional[list[str]] = None,
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
    if options is None:
        options = []
    if conn is not None:
        try:
            parse_plate_data(
                omero_data,
                plate_number,
                well_pos,
                image_input,
                conn,
                time=time,
                options=options,
            )
            # clear well and image data to avoid appending to existing data
            omero_data.reset_well_and_image_data()
            for well_pos in omero_data.well_pos_list:
                logger.info(f"Processing well {well_pos}")  # noqa: G004
                well_image_parser(omero_data, well_pos, conn, options)
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
    time: str = "All",
    options: Optional[list[str]] = None,
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
        if options is None:
            options = []
        reset_omero_data()
        omero_data.plate_id = plate_id
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn, time=time
        )
        user_input.parse_data()
        logger.info("Loaded data for plate with ID %s", plate_id)
        cellview_parser = CellViewParser(omero_data, options)
        cellview_parser.parse_data()
        channel_parser = ChannelDataParser(omero_data)
        channel_parser.parse_channel_data()
        flatfield_parser = FlatfieldMaskParser(omero_data, conn)
        flatfield_parser.parse_flatfieldmask()
        # Skip intensity parsing if cellview was skipped (aligned plates)
        if "skip_cellview" not in options:
            scale_intensity_parser = ScaleIntensityParser(omero_data)
            scale_intensity_parser.parse_intensities()
        else:
            logger.info(
                "Skipping intensity parsing (skip_cellview option set)"
            )
        pixel_size_parser = PixelSizeParser(omero_data)
        pixel_size_parser.parse_pixel_size_values()

    else:
        logger.info(
            "Plate data for plate %s already exists. Skip reload.", plate_id
        )
        user_input = UserInput(
            omero_data, plate_id, well_pos, image_input, conn, time=time
        )
        user_input.parse_data()


def well_image_parser(
    omero_data: OmeroData,
    well_pos: str,
    conn: BlitzGateway,
    options: Optional[list[str]] = None,
) -> None:
    """
    Function that combines the different parser classes that are responsible to process well, image and label data.
    The data is collected to populate the well/image related fields in the omero_data data class.

    Args:
        omero_data (OmeroData): OmeroData class object that receives the parsed data
        well_pos (str): List of wells provided by welldata_widget (e.g 'A1, A2')
        conn (BlitzGateway): BlitzGateway connection
        options (Optional[list[str]]): Options to control parsing behavior
    """
    if options is None:
        options = []
    well_data_parser = WellDataParser(omero_data, well_pos, options)
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
        time: str = "All",
    ):
        self._omero_data: OmeroData = omero_data
        self._plate_id: int = plate_id
        self._well_pos: str = well_pos
        self._images: str = image_input
        self._conn: BlitzGateway = conn
        self._plate: Optional[PlateWrapper] = None  # added by _check_plate_id
        self._image_number: int = 0  # added by _parse_image_number
        self._well_pos_list: list[str] = []  # added by _well_data_parser
        self._time: str = time
        self._start: Optional[tuple[int, ...]] = None
        self._length: Optional[tuple[int, ...]] = None

    def parse_data(self) -> None:
        self._check_plate_id()
        self._load_dataset()
        self._parse_image_number()
        self._well_data_parser()
        self._omero_data.well_pos_list = self._well_pos_list
        self._image_index_parser()
        self._omero_data.image_index = self._image_index
        self._image_time_parser()

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
        Looks for a data set using the plate annotation.
        if it finds the data set it assigns it to the omero_data object
        otherwise it will throw an error because the program can't proceed without
        flatfield masks and segmentation labels stored in the data set.

        Raises:
            ValueError: If the plate has not been assigned a dataset.
        """
        if dataset := self._conn.getObject(
            "Dataset",
            get_dataset_id(self._conn, self._omero_data.plate_id),
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

    def _image_time_parser(self) -> None:
        """
        Parses the image time string and checks if the image crop inputs are valid.
        Raises:
            ValueError: If the image time input format is invalid.
        """
        time = self._time

        if not (time.lower() == "all" or re.match(r"^\d+(-\d+)?$", time)):
            logger.error(
                "Image time '%s' doesn't match any of the expected patterns 'All, 1-3, 1'.",
                time,
            )
            raise ValueError(
                f"Image time '{time}' doesn't match any of the expected patterns 'All, 1-3, 1'."
            )

        if time.lower() == "all":
            # Ignore and default to all time points
            self._omero_data.crop_start = ()
            self._omero_data.crop_length = ()
            return

        if "-" in time:
            # Handle range, e.g., '1-3'
            start, end = map(int, time.split("-"))
        else:
            # Handle single number, e.g., '1'
            start = int(time)
            end = start

        if end < start:
            logger.error("Invalid time range: %s-%s.", start, end)
            raise ValueError(f"Invalid time range: {start}-{end}.")

        # Get image dimensions
        if self._plate:
            first_well = list(self._plate.listChildren())[0]
            image = first_well.getImage(0)
            xyzct = [
                image.getSizeX(),
                image.getSizeY(),
                image.getSizeZ(),
                image.getSizeC(),
                image.getSizeT(),
            ]
        else:
            logger.error("No plate found, unable to parse image time.")
            raise ValueError("No plate found, unable to parse image time.")
        # Validate. Change start to zero-based indexing.
        if end > xyzct[0]:
            logger.error("Invalid end time: %s > %s.", end, xyzct[0])
            raise ValueError(f"Invalid end time: {end} > {xyzct[0]}.")
        if start < 1:
            logger.error("Invalid start time: %s < 1.", start)
            raise ValueError(f"Invalid start time: {start} < 1.")

        start -= 1
        length = end - start
        # XYZCT format
        self._omero_data.crop_start = (0, 0, 0, 0, start)
        self._omero_data.crop_length = (
            xyzct[0],
            xyzct[1],
            xyzct[2],
            xyzct[3],
            length,
        )


# -----------------------------------------------PLATE DATA -----------------------------------------------------

# classes to parse the plate data from the Omero Plate: csv file, channel data, flatfield mask, pixel size,
# image intensity and pixels size

# -----------------------------------------------CSV FILE -----------------------------------------------------


# -----------------------------------------------CELLVIEW DATA -----------------------------------------------------


class CellViewParser:
    """
    Class to handle data retrieval from CellView database.
    public methods:
    parse_data: Orchestrate the data retrieval.
    """

    def __init__(
        self, omero_data: OmeroData, options: Optional[list[str]] = None
    ):
        self._omero_data = omero_data
        self._plate_id: int = omero_data.plate_id
        self._db: Optional[CellViewDB] = None
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._options: list[str] = options if options is not None else []

    def parse_data(self) -> None:
        """
        Orchestrate the data retrieval from CellView database.
        Automatically creates database and imports plate data if needed.
        """
        # Skip cellview loading if option is set (e.g., for aligned plates where data already loaded)
        if "skip_cellview" in self._options:
            logger.info(
                "Skipping cellview data loading for plate %s (option set)",
                self._plate_id,
            )
            return

        logger.info(
            "Loading data from CellView database for plate %s", self._plate_id
        )
        try:
            # Step 1: Get or create database connection
            self._conn = self._get_or_create_database()

            # Step 2: Check if plate data exists
            if not self._check_plate_in_database():
                logger.info(
                    f"Plate {self._plate_id} not found in database. Starting import process..."
                )
                self._show_message(
                    f"Plate {self._plate_id} not found in database.\nStarting import process...",
                    "info",
                )

                # AT START: Clear any dangling records from previous failed runs
                try:
                    if self._db is not None and self._conn is not None:
                        deep_clean_db(self._db, self._conn)
                except Exception as clean_err:
                    logger.warning(f"Initial cleanup failed: {clean_err}")

                # Step 3: Import plate data interactively
                self._import_plate_data()

            # Step 4: Load data from database
            from cellview.exporters.db_to_polars import export_polars_lf

            logger.info(
                f"Loading plate {self._plate_id} data from database..."
            )
            lf, _ = export_polars_lf(self._plate_id, self._conn)

            # Verify data was loaded
            try:
                if lf.limit(1).collect().is_empty():
                    logger.error(
                        "No data found for plate %s in CellView database after import.",
                        self._plate_id,
                    )
                    raise ValueError(
                        f"No data found for plate {self._plate_id} in CellView database after import."
                    )
            except ValueError:
                # Re-raise intended verification failure
                raise
            except Exception as verify_error:
                # If collection fails for other reasons (e.g. DuckDB error), log but continue
                logger.warning(
                    f"Could not verify data loading due to unexpected error: {verify_error}"
                )

            self._omero_data.plate_data = lf
            logger.info("Data loaded successfully from CellView.")

        except Exception as e:
            logger.error("Error loading data from CellView: %s", e)
            raise ValueError(f"Error loading data from CellView: {e}") from e

    # ================== Database Management Methods ==================

    def _check_database_exists(self) -> bool:
        """
        Check if the cellview database file exists at the default location.
        Returns True if found, False otherwise.
        """
        db_path = Path.home() / "cellview_data" / "cellview.duckdb"
        exists = db_path.exists()
        if exists:
            logger.info(f"CellView database found at: {db_path}")
        else:
            logger.info(f"CellView database not found at: {db_path}")
        return exists

    def _create_database(self) -> None:
        """
        Initialize a new CellViewDB instance.
        Creates database and schema automatically.
        """
        logger.info("Creating new CellView database...")
        self._db = CellViewDB()
        self._conn = self._db.connect()
        logger.info("CellView database created successfully.")
        self._show_message("Created new CellView database.", "info")

    def _get_or_create_database(self) -> duckdb.DuckDBPyConnection:
        """
        Get existing database connection or create new database.
        Returns the database connection.
        """
        if not self._check_database_exists():
            self._create_database()
        else:
            self._db = CellViewDB()
            self._conn = self._db.connect()
            logger.info("Connected to existing CellView database.")

        if self._conn is None:
            raise ValueError("Failed to create or connect to database.")

        return self._conn

    # ================== Plate Data Check Methods ==================

    def _check_plate_in_database(self) -> bool:
        """
        Check if the plate_id exists in the database.
        Returns True if plate data already imported, False otherwise.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        query = """
            SELECT COUNT(*) as count
            FROM repeats
            WHERE plate_id = ?
        """
        result = self._conn.execute(query, [self._plate_id]).fetchone()
        exists = result[0] > 0 if result else False

        if exists:
            logger.info(f"Plate {self._plate_id} found in database.")
        else:
            logger.info(f"Plate {self._plate_id} not found in database.")

        return exists

    def _get_plate_metadata(self) -> Optional[dict[str, Any]]:
        """
        Retrieve metadata for the plate if it exists in the database.
        Returns dict with metadata or None if plate not found.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        query = """
            SELECT
                r.plate_id,
                r.date,
                r.lab_member,
                r.channel_0,
                r.channel_1,
                r.channel_2,
                r.channel_3,
                e.experiment_name,
                p.project_name,
                COUNT(m.measurement_id) as num_measurements
            FROM repeats r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN projects p ON e.project_id = p.project_id
            LEFT JOIN measurements m ON r.repeat_id =
                (SELECT repeat_id FROM repeats WHERE plate_id = ?)
            WHERE r.plate_id = ?
            GROUP BY r.plate_id, r.date, r.lab_member,
                     r.channel_0, r.channel_1, r.channel_2, r.channel_3,
                     e.experiment_name, p.project_name
        """
        result = self._conn.execute(
            query, [self._plate_id, self._plate_id]
        ).fetchone()

        if result:
            return {
                "plate_id": result[0],
                "date": result[1],
                "lab_member": result[2],
                "channels": {
                    "channel_0": result[3],
                    "channel_1": result[4],
                    "channel_2": result[5],
                    "channel_3": result[6],
                },
                "experiment_name": result[7],
                "project_name": result[8],
                "num_measurements": result[9],
            }

        return None

    # ================== OMERO CSV Retrieval Methods ==================

    def _get_csv_from_omero(self) -> str:
        """
        Retrieve CSV file from OMERO plate attachments.
        Downloads to temporary location and returns path.
        Raises ValueError if no CSV found.
        """
        logger.info(
            f"Checking for CSV attachments on plate {self._plate_id}..."
        )

        # Get plate object
        plate = self._omero_data.plate

        # Get CSV file annotations for the plate
        file_anns = get_file_attachments(plate, ".csv")

        if not file_anns:
            raise ValueError(
                f"No CSV file attachments found for plate {self._plate_id}. "
                "Please ensure the plate has a CSV file attached."
            )

        # Look for CSV files in priority order
        csv_priority = ["agg_data.csv", "final_data_cc.csv", "final_data.csv"]

        selected_ann = None
        for priority_name in csv_priority:
            for ann in file_anns:
                filename = ann.getFile().getName().lower()
                if priority_name in filename:
                    selected_ann = ann
                    break
            if selected_ann:
                break

        # If no priority match, take the first CSV file
        if not selected_ann:
            selected_ann = file_anns[0]

        # Download CSV to temporary location
        csv_filename = selected_ann.getFile().getName()
        logger.info(f"Downloading CSV: {csv_filename}")

        # Create temp file and download
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, csv_filename)

        # Download file using the same pattern as parse_csv_data
        original_file = selected_ann.getFile()
        with open(csv_path, "wb") as file_on_disk:
            for chunk in original_file.asFileObj():
                file_on_disk.write(chunk)

        if not os.path.exists(csv_path):
            raise ValueError(f"Failed to download CSV file: {csv_filename}")

        file_size = os.path.getsize(csv_path) / 1024  # KB
        logger.info(f"Downloaded CSV: {csv_filename} ({file_size:.1f} KB)")

        return csv_path

    def _check_csv_available(self) -> tuple[bool, str]:
        """
        Check if CSV is available in OMERO.
        Returns (True, csv_path) if found, (False, error_message) if not.
        """
        try:
            csv_path = self._get_csv_from_omero()
            return True, csv_path
        except Exception as e:
            logger.error(f"Error checking for CSV: {e}")
            return False, str(e)

    # ================== Project Selection Methods ==================

    def _get_existing_projects(self) -> list[dict[str, Any]]:
        """
        Query database to get all existing projects.
        Returns list of dicts with project_id, project_name, description.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        query = """
            SELECT project_id, project_name, description
            FROM projects
            ORDER BY project_name
        """
        results = self._conn.execute(query).fetchall()

        projects = [
            {
                "project_id": row[0],
                "project_name": row[1],
                "description": row[2] or "",
            }
            for row in results
        ]

        logger.info(f"Found {len(projects)} existing projects in database.")
        return projects

    def _create_project(self, name: str, description: str = "") -> int:
        """
        Create new project in database.
        Returns the new project_id.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        try:
            query = """
                INSERT INTO projects (project_name, description)
                VALUES (?, ?)
                RETURNING project_id
            """
            result = self._conn.execute(query, [name, description]).fetchone()

            if result is None:
                raise ValueError("Failed to create project - no ID returned.")

            project_id = result[0]
            logger.info(f"Created new project: {name} (ID: {project_id})")
            return cast(int, project_id)

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            if "UNIQUE constraint" in str(e):
                raise ValueError(
                    f"Project '{name}' already exists. Please choose a different name."
                ) from e
            raise

    def _show_project_selection_dialog(
        self, detected_project_name: Optional[str] = None
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Show dialog for selecting existing project or creating new one.
        Returns (project_id, project_name) for existing, or (None, new_name) for new.
        Returns (None, None) if cancelled.
        """
        projects = self._get_existing_projects()

        dialog = QDialog()
        dialog.setWindowTitle("Select or Create Project")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Show detected project name if available
        if detected_project_name:
            detected_label = QLabel(
                f"Detected from OMERO: <b>{detected_project_name}</b>"
            )
            layout.addWidget(detected_label)
            layout.addSpacing(10)

        # Radio buttons for selection mode
        button_group = QButtonGroup()

        # Existing project option
        if projects:
            existing_radio = QRadioButton("Use existing project")
            existing_radio.setChecked(True)
            button_group.addButton(existing_radio)
            layout.addWidget(existing_radio)

            project_combo = QComboBox()
            for proj in projects:
                display_text = proj["project_name"]
                if proj["description"]:
                    display_text += f" - {proj['description']}"
                project_combo.addItem(display_text, proj["project_id"])

            # Select detected project if it exists
            if detected_project_name:
                for i, proj in enumerate(projects):
                    if proj["project_name"] == detected_project_name:
                        project_combo.setCurrentIndex(i)
                        break

            layout.addWidget(project_combo)
            layout.addSpacing(10)
        else:
            no_projects_label = QLabel("<i>No existing projects found</i>")
            layout.addWidget(no_projects_label)
            layout.addSpacing(10)
            project_combo = None
            existing_radio = None

        # New project option
        new_radio = QRadioButton("Create new project")
        if not projects:
            new_radio.setChecked(True)
        button_group.addButton(new_radio)
        layout.addWidget(new_radio)

        form_layout = QFormLayout()
        name_edit = QLineEdit()
        if detected_project_name:
            name_edit.setText(detected_project_name)
        form_layout.addRow("Project name:", name_edit)

        desc_edit = QTextEdit()
        desc_edit.setMaximumHeight(60)
        form_layout.addRow("Description:", desc_edit)

        layout.addLayout(form_layout)

        # Enable/disable fields based on radio selection
        def update_fields() -> None:
            is_new = new_radio.isChecked()
            name_edit.setEnabled(is_new)
            desc_edit.setEnabled(is_new)
            if project_combo:
                project_combo.setEnabled(not is_new)

        if existing_radio:
            existing_radio.toggled.connect(update_fields)
        new_radio.toggled.connect(update_fields)
        update_fields()

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addSpacing(20)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        result = dialog.exec_()

        if result == QDialog.Accepted:
            if new_radio.isChecked():
                # Creating new project
                new_name = name_edit.text().strip()
                if not new_name:
                    self._show_message(
                        "Project name cannot be empty.", "error"
                    )
                    return None, None
                return None, new_name
            else:
                # Using existing project
                if project_combo:
                    project_id = project_combo.currentData()
                    project_name = projects[project_combo.currentIndex()][
                        "project_name"
                    ]
                    return cast(int, project_id), project_name
                else:
                    return None, None
        else:
            return None, None

    # ================== Experiment Selection Methods ==================

    def _get_existing_experiments(
        self, project_id: int
    ) -> list[dict[str, Any]]:
        """
        Query database to get all experiments for a project.
        Returns list of dicts with experiment_id, experiment_name, description.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        query = """
            SELECT experiment_id, experiment_name, description
            FROM experiments
            WHERE project_id = ?
            ORDER BY experiment_name
        """
        results = self._conn.execute(query, [project_id]).fetchall()

        experiments = [
            {
                "experiment_id": row[0],
                "experiment_name": row[1],
                "description": row[2] or "",
            }
            for row in results
        ]

        logger.info(
            f"Found {len(experiments)} existing experiments for project {project_id}."
        )
        return experiments

    def _create_experiment(
        self, project_id: int, name: str, description: str = ""
    ) -> int:
        """
        Create new experiment in database.
        Returns the new experiment_id.
        """
        if self._conn is None:
            raise ValueError("Database connection not established.")

        try:
            query = """
                INSERT INTO experiments (project_id, experiment_name, description)
                VALUES (?, ?, ?)
                RETURNING experiment_id
            """
            result = self._conn.execute(
                query, [project_id, name, description]
            ).fetchone()

            if result is None:
                raise ValueError(
                    "Failed to create experiment - no ID returned."
                )

            experiment_id = result[0]
            logger.info(
                f"Created new experiment: {name} (ID: {experiment_id})"
            )
            return cast(int, experiment_id)

        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise

    def _show_experiment_selection_dialog(
        self, project_id: int, detected_experiment_name: Optional[str] = None
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Show dialog for selecting existing experiment or creating new one.
        Returns (experiment_id, experiment_name) for existing, or (None, new_name) for new.
        Returns (None, None) if cancelled.
        """
        experiments = self._get_existing_experiments(project_id)

        # Get project name for display
        project_query = (
            "SELECT project_name FROM projects WHERE project_id = ?"
        )
        if self._conn is None:
            raise ValueError("Database connection not established")
        project_name_row = self._conn.execute(
            project_query, [project_id]
        ).fetchone()
        if project_name_row is None:
            raise ValueError(f"Project {project_id} not found")
        project_name = project_name_row[0]

        dialog = QDialog()
        dialog.setWindowTitle("Select or Create Experiment")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Show project context
        project_label = QLabel(f"Project: <b>{project_name}</b>")
        layout.addWidget(project_label)
        layout.addSpacing(10)

        # Show detected experiment name if available
        if detected_experiment_name:
            detected_label = QLabel(
                f"Detected from OMERO: <b>{detected_experiment_name}</b>"
            )
            layout.addWidget(detected_label)
            layout.addSpacing(10)

        # Radio buttons for selection mode
        button_group = QButtonGroup()

        # Existing experiment option
        if experiments:
            existing_radio = QRadioButton("Use existing experiment")
            existing_radio.setChecked(True)
            button_group.addButton(existing_radio)
            layout.addWidget(existing_radio)

            exp_combo = QComboBox()
            for exp in experiments:
                display_text = exp["experiment_name"]
                if exp["description"]:
                    display_text += f" - {exp['description']}"
                exp_combo.addItem(display_text, exp["experiment_id"])

            # Select detected experiment if it exists
            if detected_experiment_name:
                for i, exp in enumerate(experiments):
                    if exp["experiment_name"] == detected_experiment_name:
                        exp_combo.setCurrentIndex(i)
                        break

            layout.addWidget(exp_combo)
            layout.addSpacing(10)
        else:
            no_exp_label = QLabel(
                "<i>No existing experiments in this project</i>"
            )
            layout.addWidget(no_exp_label)
            layout.addSpacing(10)
            exp_combo = None
            existing_radio = None

        # New experiment option
        new_radio = QRadioButton("Create new experiment")
        if not experiments:
            new_radio.setChecked(True)
        button_group.addButton(new_radio)
        layout.addWidget(new_radio)

        form_layout = QFormLayout()
        name_edit = QLineEdit()
        if detected_experiment_name:
            name_edit.setText(detected_experiment_name)
        form_layout.addRow("Experiment name:", name_edit)

        desc_edit = QTextEdit()
        desc_edit.setMaximumHeight(60)
        form_layout.addRow("Description:", desc_edit)

        layout.addLayout(form_layout)

        # Enable/disable fields based on radio selection
        def update_fields() -> None:
            is_new = new_radio.isChecked()
            name_edit.setEnabled(is_new)
            desc_edit.setEnabled(is_new)
            if exp_combo:
                exp_combo.setEnabled(not is_new)

        if existing_radio:
            existing_radio.toggled.connect(update_fields)
        new_radio.toggled.connect(update_fields)
        update_fields()

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addSpacing(20)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        result = dialog.exec_()

        if result == QDialog.Accepted:
            if new_radio.isChecked():
                # Creating new experiment
                new_name = name_edit.text().strip()
                if not new_name:
                    self._show_message(
                        "Experiment name cannot be empty.", "error"
                    )
                    return None, None
                return None, new_name
            else:
                # Using existing experiment
                if exp_combo:
                    exp_id = exp_combo.currentData()
                    exp_name = experiments[exp_combo.currentIndex()][
                        "experiment_name"
                    ]
                    return exp_id, exp_name
                else:
                    return None, None
        else:
            return None, None

    # ================== Metadata Detection and Confirmation Methods ==================

    def _detect_metadata_from_omero(self) -> dict[str, Any]:
        """
        Detect metadata from OMERO plate annotations.
        Returns dict with detected project, experiment, date, lab_member, etc.
        Channels are automatically detected from CSV by cellview.
        """
        metadata: dict[str, Any] = {}

        # Get plate object
        plate = self._omero_data.plate

        # Look for map annotations
        for ann in plate.listAnnotations():
            if isinstance(ann, MapAnnotationWrapper):
                ann_dict = dict(ann.getValue())

                # Look for project/experiment names
                if "project" in ann_dict:
                    metadata["project_name"] = ann_dict["project"]
                if "experiment" in ann_dict:
                    metadata["experiment_name"] = ann_dict["experiment"]
                if "lab_member" in ann_dict:
                    metadata["lab_member"] = ann_dict["lab_member"]
                if "date" in ann_dict:
                    metadata["date"] = ann_dict["date"]

        # Get plate name as fallback for project
        if "project_name" not in metadata:
            plate_name = plate.getName()
            # Try to extract project from plate name (common pattern)
            metadata["project_name"] = (
                plate_name.split("_")[0] if "_" in plate_name else plate_name
            )

        # Get current date as fallback
        if "date" not in metadata:
            from datetime import datetime

            metadata["date"] = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Detected metadata from OMERO: {metadata}")
        return metadata

    def _show_repeat_metadata_dialog(
        self, detected_metadata: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Show dialog to confirm/edit plate (repeat) metadata.
        Returns dict with confirmed metadata or None if cancelled.
        Channels are automatically detected from CSV by cellview.
        """
        dialog = QDialog()
        dialog.setWindowTitle("Confirm Plate Metadata")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout()

        info_label = QLabel(
            "Please confirm or edit the metadata for this plate acquisition:"
        )
        layout.addWidget(info_label)
        layout.addSpacing(10)

        form_layout = QFormLayout()

        # Plate ID (read-only)
        plate_id_label = QLabel(str(self._plate_id))
        form_layout.addRow("Plate ID:", plate_id_label)

        # Date
        date_edit = QLineEdit(detected_metadata.get("date", ""))
        form_layout.addRow("Date (YYYY-MM-DD):", date_edit)

        # Lab member
        lab_member_edit = QLineEdit(detected_metadata.get("lab_member", ""))
        form_layout.addRow("Lab Member:", lab_member_edit)

        layout.addLayout(form_layout)

        # Info about automatic detection
        auto_detect_label = QLabel(
            "<i>Note: Channels will be automatically detected from the CSV file.</i>"
        )
        layout.addWidget(auto_detect_label)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addSpacing(20)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Collect metadata
            confirmed_metadata = {
                "plate_id": self._plate_id,
                "date": date_edit.text().strip(),
                "lab_member": lab_member_edit.text().strip(),
            }

            return confirmed_metadata
        else:
            return None

    # ================== Import Orchestration Methods ==================

    def _cleanup_orphaned_records(self) -> None:
        """
        Clean up orphaned projects and experiments after failed import.
        This removes any projects without experiments and experiments without repeats.
        """
        try:
            if self._conn is None:
                return

            # Perform deep cleanup to handle broken references and multi-level orphans
            if self._db is not None and self._conn is not None:
                deep_clean_db(self._db, self._conn)

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def _perform_import(
        self,
        csv_path: str,
        project_id: int,
        experiment_id: int,
        repeat_metadata: dict[str, Any],
    ) -> bool:
        """
        Perform the actual import using cellview's import pipeline.
        Returns True on success, False on failure.
        Cleans up orphaned records on failure.
        """
        progress = None
        try:
            # Load CSV data first (as pandas DataFrame for cellview)
            logger.info(f"Loading CSV data from: {csv_path}")
            df = pd.read_csv(csv_path)

            # Apply cleaning if this is an aggregated data file
            # Check if the filename is agg_data.csv
            csv_filename = Path(csv_path).name.lower()
            if csv_filename == "agg_data.csv":
                logger.info(
                    "Detected agg_data.csv - applying data cleaning for cyclicIF"
                )
                df = clean_agg_data(df)
                logger.info(
                    f"Cleaned dataframe: {len(df)} rows, {len(df.columns)} columns"
                )

            # Create a CellViewUI instance for the state (required parameter)
            ui = CellViewUI()

            # Extract channel names from DataFrame columns
            # Columns like 'intensity_mean_DAPI_nucleus', 'intensity_mean_EdU_nucleus', etc.
            # We need to map DAPI, EdU, etc. to channel_0, channel_1, etc.
            # However, for now, let's try to get them from self._omero_data.channel_data if available
            # or infer them from the dataframe columns.

            channels: dict[str, str | None] = {}
            # Defaults
            channels["channel_0"] = "DAPI"  # Nearly always present
            channels["channel_1"] = None
            channels["channel_2"] = None
            channels["channel_3"] = None

            # Attempt to get from OMERO metadata first (most reliable)
            if self._omero_data and self._omero_data.channel_data:
                # channel_data is {name: index}, e.g. {'DAPI': 0, 'EdU': 1}
                for name, idx in self._omero_data.channel_data.items():
                    key = f"channel_{idx}"
                    if key in channels:
                        channels[key] = name

            # Create state object (explicitly passing channels)
            state = CellViewStateCore(
                ui=ui,
                csv_path=Path(csv_path),
                df=df,
                plate_id=self._plate_id,
                project_name=None,
                experiment_name=None,
                project_id=project_id,
                experiment_id=experiment_id,
                repeat_id=None,
                condition_id_map=None,
                lab_member=repeat_metadata.get("lab_member", ""),
                date=repeat_metadata.get("date", ""),
                channel_0=channels["channel_0"],
                channel_1=channels["channel_1"],
                channel_2=channels["channel_2"],
                channel_3=channels["channel_3"],
                db_conn=self._conn,
            )

            # Show progress dialog
            progress = QProgressDialog(
                f"Importing plate {self._plate_id}...", "Cancel", 0, 0
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            # Perform import using cellview's import function
            logger.info("Starting cellview import process...")
            if self._db is not None:
                import_data(self._db, state, self._conn)
            else:
                raise ValueError("Database object is None")

            if progress:
                progress.close()

            logger.info("Import completed successfully!")
            return True

        except Exception as e:
            if progress:
                progress.close()

            logger.error(f"Error during import: {e}")
            logger.info("Cleaning up orphaned database records...")

            # Cleanup any orphaned projects/experiments created during failed import
            self._cleanup_orphaned_records()

            self._show_message(
                f"Import failed: {str(e)}\n\nOrphaned database records have been cleaned up.",
                "error",
            )
            return False

    def _interactive_import(self, csv_path: str) -> bool:
        """
        Guide user through interactive import process.
        Returns True if import successful, False if cancelled.
        Cleans up orphaned records if user cancels or import fails.
        """
        try:
            # 1. Detect metadata from OMERO
            detected_metadata = self._detect_metadata_from_omero()
            detected_project = detected_metadata.get("project_name")
            detected_experiment = detected_metadata.get("experiment_name")

            # 2. Show project selection dialog
            project_id, project_name = self._show_project_selection_dialog(
                detected_project
            )
            if project_name is None:  # User cancelled
                logger.info("Import cancelled by user at project selection.")
                return False

            # 3. If new project, create it now
            if project_id is None:
                project_description = detected_metadata.get(
                    "project_description", ""
                )
                project_id = self._create_project(
                    project_name, project_description
                )

            # 4. Show experiment selection dialog
            experiment_id, experiment_name = (
                self._show_experiment_selection_dialog(
                    project_id, detected_experiment
                )
            )
            if experiment_name is None:  # User cancelled
                logger.info(
                    "Import cancelled by user at experiment selection."
                )
                # Clean up the project if it was just created and has no experiments
                self._cleanup_orphaned_records()
                return False

            # 5. If new experiment, create it now
            if experiment_id is None:
                experiment_description = detected_metadata.get(
                    "experiment_description", ""
                )
                experiment_id = self._create_experiment(
                    project_id, experiment_name, experiment_description
                )

            # 6. Show repeat metadata dialog
            repeat_metadata = self._show_repeat_metadata_dialog(
                detected_metadata
            )
            if repeat_metadata is None:  # User cancelled
                logger.info(
                    "Import cancelled by user at metadata confirmation."
                )
                # Clean up orphaned experiment/project if they were just created
                self._cleanup_orphaned_records()
                return False

            # 7. Perform the actual import (handles its own cleanup on failure)
            success = self._perform_import(
                csv_path, project_id, experiment_id, repeat_metadata
            )

            return success

        except Exception as e:
            logger.error(f"Error during interactive import: {e}")
            # Clean up any orphaned records
            self._cleanup_orphaned_records()
            self._show_message(f"Import error: {str(e)}", "error")
            return False

    def _import_plate_data(self) -> None:
        """
        High-level method that orchestrates the full import workflow.
        """
        try:
            # Step 1: Get CSV from OMERO
            self._show_message(
                "Checking for CSV attachment in OMERO...", "info"
            )
            csv_path = self._get_csv_from_omero()
            file_size = os.path.getsize(csv_path) / 1024  # KB
            self._show_message(
                f"Found CSV: {os.path.basename(csv_path)} ({file_size:.1f} KB)",
                "info",
            )

            # Step 2: Interactive import with dialogs
            success = self._interactive_import(csv_path)

            if not success:
                self._show_message("Import cancelled or failed.", "warning")
                return

            self._show_message("Successfully imported plate data!", "success")

        except Exception as e:
            logger.error(f"Error during import: {e}")
            self._show_message(f"Import failed: {str(e)}", "error")
            raise

    # ================== UI Helper Methods ==================

    def _show_message(self, message: str, msg_type: str = "info") -> None:
        """
        Show message to user via QMessageBox.
        Types: "info", "warning", "error", "success"
        """
        msg_box = QMessageBox()

        if msg_type == "info":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Information")
        elif msg_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
        elif msg_type == "error":
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
        elif msg_type == "success":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Success")

        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Only show in GUI mode
        if not hasattr(sys, "ps1") and "pytest" not in sys.modules:
            msg_box.exec_()
        else:
            logger.info(f"[{msg_type.upper()}] {message}")


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

        annotations = self._plate.listAnnotations(ns=OmeroScreenNS.METADATA)

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
        from omero_screen.metadata_parser import resolve_channel_roles
        from omero_utils.message import ChannelAnnotationError

        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Accept any map annotation whose keys contain a resolvable nucleus
            # role (alias, ``_nucleus`` suffix, or legacy DAPI).
            keys = {k.strip(): 0 for k, _ in ann_value}
            try:
                resolve_channel_roles(keys)
            except ChannelAnnotationError:
                continue
            self._map_annotations = ann_value
            return

        logger.error(
            "No nucleus channel information found for plate %s.",
            self._plate_id,
        )  # noqa: G004
        raise ValueError(
            f"No nucleus channel information found for plate {self._omero_data.plate_id}."
        )

    def _tidy_up_channel_data(self) -> None:
        """
        Convert channel map annotation from list of tuples to dictionary.
        The nucleus-role channel is renamed to 'DAPI' so legacy display
        widgets continue to work; the original name is preserved separately
        for reference.
        """
        from omero_screen.metadata_parser import resolve_channel_roles

        channel_data = dict(self._map_annotations)
        sorted_channel_data = dict(
            sorted(channel_data.items(), key=lambda item: item[1])
        )
        self._channel_data = {
            key.strip(): value for key, value in sorted_channel_data.items()
        }
        roles = resolve_channel_roles(dict.fromkeys(self._channel_data, 0))
        nucleus_name = roles["nucleus"]
        if nucleus_name != "DAPI":
            self._channel_data["DAPI"] = self._channel_data.pop(nucleus_name)


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
                self._flatfield_obj = self._conn.getObject(
                    "Image", flatfield_mask_id
                )
                self._flatfield_array = get_image(
                    self._conn,
                    flatfield_mask_id,
                    tag=self._omero_data.plate_id,
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
        annotations = self._flatfield_obj.listAnnotations(
            ns=OmeroScreenNS.METADATA
        )
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
    The class extracts the scaling values for image contrasts to display the different channels in napari.
    To compare intensities across different wells a single global contrasting value is set. This is based on the
    the min and max values for each channel. These data are extracted from the polars LazyFrame object stored in
    omero_data.plate_data.
    public method:
    parse_intensities: Parse the intensities from the plate data and store them in the omero_data object.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_data: pl.LazyFrame = omero_data.plate_data
        self._intensities: Optional[dict[int, tuple[int, int]]] = None

    def parse_intensities(self) -> None:
        """
        Parse the intensities from the plate data and store them in the omero_data object.
        Raises:
            ValueError: when intensities cannot be collected from the plate csv data.
        """
        self._get_values()
        if self._intensities:
            self._omero_data.intensities = self._intensities  # type: ignore
        else:
            logger.error("Problem with loading intensities to scale channels.")
            raise ValueError(
                "Problem with loading intensities to scale channels."
            )

    def _get_values(self) -> None:
        """
        Filter the dataframe to include columns with the channel keyword.
        Prioritizes "_cell" suffix if available, falls back to "_nucleus".
        Raises ValueError if "_nucleus" columns are missing or contain only nulls.
        """
        if not hasattr(self, "_plate_data") or self._plate_data is None:
            raise ValueError("Dataframe 'plate_data' does not exist.")

        intensity_dict = {}
        columns = self._plate_data.collect_schema().names()

        for channel, channel_value in self._omero_data.channel_data.items():
            # Define potential column sets
            cell_cols = (
                f"intensity_max_{channel}_cell",
                f"intensity_min_{channel}_cell",
            )
            nucleus_cols = (
                f"intensity_max_{channel}_nucleus",
                f"intensity_min_{channel}_nucleus",
            )

            target_cols = None
            # Try cell columns first
            if all(col in columns for col in cell_cols):
                # Check if cell columns have non-null values
                max_value = (
                    self._plate_data.select(pl.col(cell_cols[0]))
                    .mean()
                    .collect()
                )
                if max_value[0, 0] is not None:
                    target_cols = cell_cols
                else:
                    logger.debug(
                        "Cell intensity columns for channel '%s' exist but contain only nulls, falling back to nucleus columns.",
                        channel,
                    )

            # Fallback to nucleus columns if cell columns don't work
            if target_cols is None:
                if all(col in columns for col in nucleus_cols):
                    target_cols = nucleus_cols
                else:
                    logger.error(
                        "Neither cell nor nucleus intensity columns found for channel '%s'. Expected %s or %s.",
                        channel,
                        cell_cols,
                        nucleus_cols,
                    )
                    raise ValueError(
                        f"Neither cell nor nucleus intensity columns found for channel '{channel}'."
                    )

            # Extract values from the chosen columns
            max_value = (
                self._plate_data.select(pl.col(target_cols[0]))
                .mean()
                .collect()
            )
            min_value = (
                self._plate_data.select(pl.col(target_cols[1])).min().collect()
            )

            # Verify values are not None
            if max_value[0, 0] is None or min_value[0, 0] is None:
                logger.error(
                    "Intensity values for channel '%s' are null even in %s columns.",
                    channel,
                    "nucleus" if target_cols == nucleus_cols else "cell",
                )
                raise ValueError(
                    f"Intensity values for channel '{channel}' contain only nulls."
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
        options: Optional[list[str]] = None,
    ):
        self._omero_data = omero_data
        self._plate = omero_data.plate
        self._well_pos = well_pos
        self._well: Optional[WellWrapper] = None
        self._well_id: Optional[int] = None
        self._metadata: Optional[dict[str, Any]] = None
        self._well_ifdata: Optional[pl.DataFrame] = None
        self._options: list[str] = options if options is not None else []

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

        # Skip loading CSV data if cellview was skipped (aligned plates)
        if "skip_cellview" not in self._options:
            self._load_well_csvdata()
            if self._well_ifdata is not None:
                self._omero_data.well_ifdata = pl.concat(
                    [self._omero_data.well_ifdata, self._well_ifdata]
                )
            else:
                logger.error(
                    "No well csv data found for well %s.", self._well_pos
                )
                raise ValueError(
                    f"No well csv data found for well {self._well_pos}."
                )
        else:
            logger.info(
                "Skipping well CSV data loading (skip_cellview option set)"
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
        for ann in self._well.listAnnotations(ns=OmeroScreenNS.METADATA):
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
        start, length = _get_crop(self._omero_data)
        tstart, tend = None, None
        if start and length:
            logger.info("Using crop: %s - %s", start, length)
            # Crop supports only time
            tstart, tend = start[-1], start[-1] + length[-1]

        for index in tqdm(self._image_index):
            if image := self._well.getImage(index):
                if mip_id := self.check_mip(image):
                    logger.info(
                        "Image with index %s has Z-stacks, looking for MIP with id %s.",
                        index,
                        mip_id,
                    )
                    image_array = get_image(
                        self._conn,
                        int(mip_id),
                        start=tstart,
                        end=tend,
                        tag=self._omero_data.plate_id,
                    )
                else:
                    image_array = get_image(
                        self._conn,
                        image.getId(),
                        start=tstart,
                        end=tend,
                        tag=self._omero_data.plate_id,
                    )
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
        annotations = image.listAnnotations(ns=OmeroScreenNS.METADATA)
        if map_anns := [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]:
            for ann in map_anns:
                ann_values = dict(ann.getValue())
                for k, v in ann_values.items():
                    if k == "MIP":
                        return (
                            str(v.split(":")[-1])
                            if v.find(":") >= 0
                            else str(v)
                        )
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
        start, length = _get_crop(self._omero_data)
        tstart, tend = None, None
        if start and length:
            # Crop supports only time
            tstart, tend = start[-1], start[-1] + length[-1]

        dataset_children = list(self._omero_data.screen_dataset.listChildren())

        for img_id in self._image_ids:
            expected_name = f"{img_id}_segmentation"

            # 1. Try to find an exact match first (avoids ambiguity with suffixes like 'segmentation1')
            exact_matches = [
                child
                for child in dataset_children
                if child.getName() == expected_name
            ]

            if exact_matches:
                # Found exact match, use it silently
                label_data = exact_matches[0]
            else:
                # 2. Fallback: Find matching label by substring
                candidates = [
                    child
                    for child in dataset_children
                    if expected_name in child.getName()
                ]

                if not candidates:
                    logger.warning(f"No label found for image {img_id}")
                    # Skip to next image (stack logic is per-image anyway in this loop structure?
                    # Wait, _label_arrays is just a list. If we skip, it will be shorter.
                    # As noted before, we should really insert a blank, but let's stick to current behavior
                    # which works for the user (just avoiding the warning).
                    continue

                # If multiple candidates, pick the first one and warn
                if len(candidates) > 1:
                    logger.warning(
                        f"Multiple labels found for image {img_id}: {[c.getName() for c in candidates]}. Using {candidates[0].getName()}"
                    )

                label_data = candidates[0]

            try:
                label_array = get_image(
                    self._conn,
                    label_data.getId(),
                    start=tstart,
                    end=tend,
                    tag=self._omero_data.plate_id,
                )
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
            except Exception as e:
                logger.error(
                    f"Failed to load label {label_data.getName()}: {e}"
                )


def _get_crop(
    omero_data: OmeroData,
) -> tuple[Optional[tuple[int, ...]], Optional[tuple[int, ...]]]:
    if omero_data.crop_start:
        return omero_data.crop_start, omero_data.crop_length
    return None, None


@omero_connect
def get_plate_alignments(
    plate_id: int,
    sample_alignments: bool = False,
    conn: Optional[BlitzGateway] = None,
) -> pd.DataFrame:
    """Get the alignments for the plate.

    Reads alignment CSV file attached to the plate in OMERO.
    The CSV contains translation data to align images from multiple plates.

    Args:
        plate_id: The plate ID
        sample_alignments: Set to True to obtain alignments for each well sample
        conn: The BlitzGateway connection

    Returns:
        DataFrame with columns: plate, well, x, y, [image_id if sample_alignments]

    Raises:
        Exception: if plate doesn't exist or alignment CSV not found
    """
    if conn is None:
        raise ValueError("Connection is required")
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise Exception(f"Plate:{plate_id}")
    filename = "sample_alignment.csv" if sample_alignments else "alignment.csv"
    original_file = None
    for ann in plate.listAnnotations():
        if isinstance(ann, FileAnnotationWrapper):
            fn = ann.getFile().getName()
            if fn and fn.lower() == filename:
                original_file = ann.getFile()
                break
    df = None
    if original_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = os.path.join(temp_dir, original_file.getName())
            with open(tmp_path, "wb") as file_on_disk:
                for chunk in original_file.asFileObj():
                    file_on_disk.write(chunk)
            logger.info("Parsing CSV alignment file: %s", filename)
            df = pd.read_csv(tmp_path)
    if df is None:
        raise Exception(
            f"Plate {plate_id} is missing alignment result data: {filename}"
        )
    return df
