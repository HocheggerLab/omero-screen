"""Parse and manage metadata for OMERO plates.

This module provides functionality to extract and validate metadata for OMERO plates.
It first checks for an attached Excel file on the plate. If found, metadata is parsed
from the Excel file; otherwise, it is extracted from existing plate annotations.

- Channel data is added to the plate as annotations.
- Well data is added as key-value annotations to each well.
- All parsed metadata is stored in a dataclass.

If no valid metadata is found, the program exits with an error.
"""

from collections import Counter
from typing import Any

from omero.gateway import BlitzGateway, FileAnnotationWrapper, PlateWrapper
from omero_utils.attachments import (
    delete_excel_attachment,
    get_file_attachments,
    parse_excel_data,
)
from omero_utils.map_anns import (
    add_map_annotations,
    delete_map_annotations,
    parse_annotations,
)
from omero_utils.message import (
    ChannelAnnotationError,
    ExcelParsingError,
    MetadataValidationError,
    PlateNotFoundError,
    WellAnnotationError,
    log_success,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from omero_screen.config import get_logger

logger = get_logger(__name__)
console = Console()

SUCCESS_STYLE = "bold cyan"


# --------------------Metadata Parser--------------------
class MetadataParser:
    """Parses and manages channel and well metadata for an OMERO plate.

    This class extracts metadata from an OMERO plate, either from an attached Excel file or from existing plate annotations. It validates, normalizes, and stores channel and well metadata, and provides methods to annotate the plate and its wells accordingly.

    Attributes:
        conn (BlitzGateway): The OMERO connection object.
        plate_id (int): The ID of the plate to parse.
        plate (PlateWrapper): The OMERO plate object.
        excel_file (bool): Whether an Excel file was found and used for metadata.
        channel_data (dict[str, str]): Channel metadata, mapping channel names to indices.
        well_data (dict[str, Any]): Well metadata, mapping annotation keys to lists of values.
        pixel_size (float): Pixel size in micrometers, determined from the first image.
    """

    def __init__(self, conn: BlitzGateway, plate_id: int):
        """Initializes the MetadataParser with an OMERO connection and plate ID.

        Args:
            conn (BlitzGateway): The OMERO connection object.
            plate_id (int): The ID of the plate to parse metadata from.
        """
        self.conn: BlitzGateway = conn
        self.plate_id: int = plate_id
        self.plate: PlateWrapper = self._check_plate()
        self.excel_file: bool = False
        self.channel_data: dict[str, str] = {}
        self.well_data: dict[str, Any] = {}
        self.pixel_size: float = 0

    def _check_plate(self) -> PlateWrapper:
        """Retrieve and validate the OMERO plate object for the given plate ID.

        Returns:
            PlateWrapper: The validated OMERO plate object corresponding to the provided plate ID.

        Raises:
            PlateNotFoundError: If no plate with the specified ID exists in OMERO.
        """
        plate = self.conn.getObject("Plate", self.plate_id)
        if plate is None:
            raise PlateNotFoundError(
                f"A plate with id {self.plate_id} was not found!", logger
            )
        assert isinstance(plate, PlateWrapper)
        log_success(
            SUCCESS_STYLE, f"Found plate with id {self.plate_id}", logger
        )
        return plate

    def manage_metadata(self) -> None:
        """Parse, validate, and apply metadata for the OMERO plate.

        This method orchestrates the metadata management workflow:
        - Parses metadata from an attached Excel file or from plate annotations.
        - Validates the extracted metadata.
        - Retrieves pixel size information from the first image.
        - If an Excel file was used, applies channel and well annotations to the plate and deletes the Excel file.
        - Logs the outcome of the metadata management process.
        """
        self._parse_metadata()  # checks for excel file or well data and pulls channel and well data into self.channel_data and self.well_data dictionaries
        self._validate_metadata()
        self._get_pixel_size()
        if self.excel_file:  # if excel file is found, add channel and well annotations to plate and delete excel file
            self._add_channel_annotations(self.channel_data)
            self._add_well_annotations(self.well_data)
            delete_excel_attachment(self.conn, self.plate)
            # Refresh plate after deletion of annotations
            self.plate = self.conn.getObject("Plate", self.plate_id)
            log_success(
                SUCCESS_STYLE,
                f"Metadata parsed from Excel file and transferred to plate {self.plate_id}",
                logger,
            )
        else:
            log_success(
                SUCCESS_STYLE,
                f"Metadata parsed from plate {self.plate_id}",
                logger,
            )

    def _parse_metadata(self) -> None:
        """Extract channel and well metadata from the OMERO plate.

        Attempts to parse metadata from an attached Excel file. If no Excel file is found,
        falls back to extracting metadata from existing plate and well annotations.
        Populates self.channel_data and self.well_data with the parsed results.

        Raises:
            ExcelParsingError: If there are issues parsing the Excel file.
            ChannelAnnotationError: If there are issues with channel annotations.
            WellAnnotationError: If there are issues with well annotations.
        """
        assert self.plate
        if file_annotations := self._check_excel_file():
            log_success(
                SUCCESS_STYLE,
                f"Found Excel file attachment on plate {self.plate_id}",
                logger,
            )
            try:
                self.channel_data, self.well_data = self._load_data_from_excel(
                    file_annotations
                )
            except Exception as e:
                raise ExcelParsingError(
                    f"Failed to parse Excel file: {str(e)}", logger
                ) from e
        else:
            try:
                self.channel_data = self._parse_channel_annotations()
                self.well_data = self._parse_well_annotations()
            except (ChannelAnnotationError, WellAnnotationError) as e:
                if isinstance(e, ChannelAnnotationError):
                    raise ChannelAnnotationError(
                        f"Failed to parse channel annotations: {str(e)}",
                        logger,
                    ) from e
                else:
                    raise WellAnnotationError(
                        f"Failed to parse well annotations: {str(e)}", logger
                    ) from e

    def _check_excel_file(self) -> FileAnnotationWrapper | None:
        """Check for an Excel file attachment on the OMERO plate.

        Returns:
            FileAnnotationWrapper | None: The Excel file annotation if exactly one is found; otherwise, None.

        Raises:
            ExcelParsingError: If multiple Excel file attachments are found on the plate.
        """
        # Plate is already validated in __init__
        file_annotations = get_file_attachments(self.plate, ".xlsx")
        if file_annotations and len(file_annotations) == 1:
            self.excel_file = True
            return file_annotations[0]
        elif file_annotations and len(file_annotations) > 1:
            raise ExcelParsingError(
                "Multiple Excel files found on plate", logger
            )
        else:
            return None

    def _load_data_from_excel(
        self, file_annotations: FileAnnotationWrapper
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Parse channel and well metadata from an attached Excel file.

        Reads the provided Excel file annotation and extracts channel and well metadata
        from the expected sheets ('Sheet1' for channels, 'Sheet2' for wells).

        Args:
            file_annotations (FileAnnotationWrapper): The Excel file annotation to parse.

        Returns:
            tuple[dict[str, str], dict[str, Any]]: A tuple containing:
                - channel_data: Dictionary mapping channel names to indices.
                - well_data: Dictionary mapping annotation keys to lists of values.

        Raises:
            ExcelParsingError: If the Excel file format is invalid or missing required sheets.
        """
        meta_data = parse_excel_data(file_annotations)
        if not meta_data or list(meta_data.keys()) != ["Sheet1", "Sheet2"]:
            raise ExcelParsingError(
                "Invalid excel file format - expected Sheet1 and Sheet2",
                logger,
            )

        channel_data = {
            meta_data["Sheet1"]["Channels"][i]: str(
                meta_data["Sheet1"]["Index"][i]
            )
            for i in range(len(meta_data["Sheet1"]["Channels"]))
        }

        well_data = {
            str(k): v
            for k, v in meta_data["Sheet2"].to_dict(orient="list").items()
        }

        return channel_data, well_data

    def _add_channel_annotations(self, channel_data: dict[str, str]) -> None:
        """Replace existing channel annotations on the plate with new channel data.

        Deletes any preexisting map annotations from the plate and adds the provided
        channel metadata as new map annotations.

        Args:
            channel_data (dict[str, str]): Dictionary mapping channel names to indices to be added as annotations.
        """
        delete_map_annotations(self.conn, self.plate)
        add_map_annotations(self.conn, self.plate, channel_data)

    def _add_well_annotations(self, well_data: dict[str, Any]) -> None:
        """Replace existing well annotations with new metadata for each well in the plate.

        Iterates over all wells in the plate, deletes any preexisting map annotations,
        and adds new annotations based on the provided well metadata.

        Args:
            well_data (dict[str, Any]): Dictionary mapping annotation keys to lists of values for each well.
        """
        for well in self.plate.listChildren():
            delete_map_annotations(self.conn, well)
            well_name = well.getWellPos()
            well_index = well_data["Well"].index(well_name)
            well_meta_data = {
                key: values[well_index]
                for key, values in well_data.items()
                if key
                != "Well"  # Skip the Well key since we don't need it in annotations
            }
            add_map_annotations(self.conn, well, well_meta_data)

    def _parse_channel_annotations(self) -> dict[str, str]:
        """Extract channel annotations from the plate and return as a dictionary.

        Parses map annotations from the plate and returns a dictionary mapping channel names
        to their indices. Raises an error if no channel annotations are found.

        Returns:
            dict[str, str]: Dictionary mapping channel names to their indices.

        Raises:
            ChannelAnnotationError: If no channel annotations are found on the plate.
        """
        annotations: dict[str, str] = parse_annotations(self.plate)
        if len(annotations):
            return annotations
        else:
            raise ChannelAnnotationError(
                "No channel annotations found on plate", logger
            )

    def _parse_well_annotations(self) -> dict[str, Any]:
        """Extract well annotations from the plate and return as a dictionary.

        Iterates over all wells in the plate, collecting map annotations for each well.
        Returns a dictionary where each key is an annotation key and the value is a list
        of values for that key across all wells. Also includes a 'Well' key with the well positions.

        Returns:
            dict[str, Any]: Dictionary mapping annotation keys to lists of values for each well, including a 'Well' key for well positions.

        Raises:
            WellAnnotationError: If any well is missing annotations.
        """
        well_data: dict[str, list[Any]] = {"Well": []}

        for well in self.plate.listChildren():
            well_pos = well.getWellPos()
            well_data["Well"].append(well_pos)

            well_annotation = parse_annotations(well)
            if not well_annotation:
                raise WellAnnotationError(
                    f"No well annotations found for well {well_pos}", logger
                )

            # For each key in the well's annotations, ensure it exists in well_data
            # and append the value to its list
            for key, value in well_annotation.items():
                if key not in well_data:
                    well_data[key] = []
                well_data[key].append(value)
        return well_data

    def _create_two_column_table(
        self, title: str, col1_name: str, col2_name: str
    ) -> Table:
        """Create and return a Rich table with two columns and a title.

        Args:
            title (str): The title of the table.
            col1_name (str): Name of the first column.
            col2_name (str): Name of the second column.

        Returns:
            Table: A Rich Table object with the specified columns and title.
        """
        table = Table(title=title)
        table.add_column(col1_name, style="cyan")
        table.add_column(col2_name, style="green")
        return table

    def _display_metadata(self) -> None:
        """Display parsed channel and well metadata using Rich tables.

        Formats and prints channel and well metadata in visually appealing tables using the Rich library.
        Channel data is shown as a two-column table, and well data is summarized with unique values and counts.
        """
        # Create and populate channel table
        channel_table = self._create_two_column_table(
            "Channel Information", "Channel Name", "Index"
        )
        for channel, index in self.channel_data.items():
            channel_table.add_row(channel, str(index))

        # Create and populate well summary table
        well_table = self._create_two_column_table(
            "Well Data Summary", "Key", "Unique Values"
        )
        well_table.add_column("Count", style="yellow")

        for key, values in self.well_data.items():
            if key != "Well":  # Skip the Well column as it's too verbose
                unique_values = Counter(values)
                well_table.add_row(
                    key,
                    ", ".join(str(k) for k in unique_values),
                    ", ".join(str(v) for v in unique_values.values()),
                )

        # Display the tables in panels
        console.print(
            Panel(channel_table, title="Channel Data", border_style="cyan")
        )
        console.print(
            Panel(well_table, title="Well Data Summary", border_style="cyan")
        )

    def _validate_metadata(self) -> None:
        """Validate the structure and content of parsed metadata.

        Runs a series of validation checks on channel and well metadata, collecting all errors.
        If any validation errors are found, raises a MetadataValidationError with details.
        Also displays the metadata if validation passes.

        Raises:
            MetadataValidationError: If any validation errors are detected in the metadata.
        """
        errors = []

        # Collect errors from all validation steps
        errors.extend(self._validate_metadata_structure())
        errors.extend(self._validate_channel_data())
        errors.extend(self._validate_well_data())

        # If any errors were found, raise them all together
        if errors:
            if len(errors) == 1:
                raise MetadataValidationError(
                    errors[0],
                    logger,
                )
            else:
                raise MetadataValidationError(
                    "Multiple validation errors found:\n"
                    + "\n".join(f"- {error}" for error in errors),
                    logger,
                )

        log_success(
            SUCCESS_STYLE,
            f"Metadata validation passed for plate {self.plate_id}",
            logger,
        )

        self._display_metadata()

    def _validate_metadata_structure(self) -> list[str]:
        """Check the basic structure and types of the parsed metadata.

        Validates that channel and well metadata are present and have the correct types.
        Returns a list of error messages for any structural issues found.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        if not self.channel_data:
            errors.append("No channel data found")
        if not self.well_data:
            errors.append("No well data found")

        if not isinstance(self.channel_data, dict) or not all(
            isinstance(k, str) for k in self.channel_data
        ):
            errors.append("Channel data must be a dictionary with string keys")
        if not all(isinstance(v, str) for v in self.channel_data.values()):
            errors.append(
                "Channel data must be a dictionary with string values"
            )

        return errors

    def _validate_channel_data(self) -> list[str]:
        """Validate and normalize channel metadata, ensuring a nuclei channel is present.

        Checks that the channel data includes at least one nuclei channel (DAPI, Hoechst, DNA, or RFP),
        and normalizes the channel name to 'DAPI' if found. Returns a list of error messages for any issues.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        nuclei_channels = {"dapi", "hoechst", "dna", "rfp"}
        found_nuclei = False
        channel_data_normalized = {}

        # First pass: normalize keys and check for nuclei channels
        for key, value in self.channel_data.items():
            normalized_key = key.lower()
            if normalized_key in nuclei_channels:
                found_nuclei = True
                channel_data_normalized["DAPI"] = value
            else:
                channel_data_normalized[key] = value

        if not found_nuclei:
            errors.append(
                "At least one nuclei channel (DAPI/Hoechst/DNA/RFP) is required"
            )
        else:
            self.channel_data = channel_data_normalized

        return errors

    def _validate_well_data(self) -> list[str]:
        """Validate the well data structure and content.

        Returns:
            list[str]: List of error messages, empty if no errors
        """
        errors = []
        # Check required keys exist
        required_keys = {"Well", "cell_line"}
        if missing_keys := required_keys - self.well_data.keys():
            errors.append(
                f"Missing required keys in well data: {', '.join(missing_keys)}"
            )

        if non_list_keys := [
            key
            for key, value in self.well_data.items()
            if not isinstance(value, list)
        ]:
            errors.append(
                f"Values must be lists for all keys. Non-list values found for: {', '.join(non_list_keys)}"
            )

        # Check all lists have the same length
        list_lengths = {
            key: len(value) for key, value in self.well_data.items()
        }
        if len(set(list_lengths.values())) > 1:
            # Create a message showing the different lengths
            length_info = [
                f"{key}: {length}" for key, length in list_lengths.items()
            ]
            errors.append(
                f"All well data lists must have the same length. Found: {', '.join(length_info)}"
            )

        # Validate well positions match actual wells in plate
        errors.extend(self._validate_well_positions())

        return errors

    def _validate_well_positions(self) -> list[str]:
        """Check that well positions in metadata match the actual wells in the plate.

        Compares the well positions listed in the metadata with those present in the plate.
        Returns a list of error messages for any missing or extra wells.

        Returns:
            list[str]: A list of error messages. The list is empty if no errors are found.
        """
        errors = []
        # Get actual well positions from the plate
        actual_wells = [
            well.getWellPos() for well in self.plate.listChildren()
        ]

        # Get well positions from metadata
        metadata_wells = self.well_data["Well"]

        # Check for missing and extra wells
        s1 = set(actual_wells)
        s2 = set(metadata_wells)
        if s1 != s2:
            missing_wells = s1 - s2
            extra_wells = s2 - s1
            if len(missing_wells):
                errors.append(
                    f"Missing wells in metadata: {', '.join(sorted(missing_wells))}"
                )
            if len(extra_wells):
                errors.append(
                    f"Extra wells in metadata: {', '.join(sorted(extra_wells))}"
                )

        # Here the plate wells and the metadata have the same well position names.
        # The order does not matter as the dictionary list under 'Well' is used to index
        # into the well values for each key (see method: well_conditions)

        return errors

    def _get_first_image(self) -> Any:
        """Retrieve the first image from the first well of the plate.

        Accesses the first well and its first well sample to obtain the associated image.
        Raises an error if no wells or images are found.

        Returns:
            Any: The first image object from the first well.

        Raises:
            MetadataValidationError: If no wells or images are found in the plate.
        """
        # Get the first well
        first_well = next(self.plate.listChildren(), None)
        if not first_well:
            raise MetadataValidationError("No wells found in plate", logger)

        # Get the first well sample from the well
        first_well_sample = next(first_well.listChildren(), None)
        if not first_well_sample:
            raise MetadataValidationError(
                "No images found in first well", logger
            )

        if first_image := first_well_sample.getImage():
            return first_image
        else:
            raise MetadataValidationError(
                "Could not get image from well sample", logger
            )

    def _get_pixel_size(self) -> None:
        """Determine the pixel size in micrometers from the first image of the first well.

        Retrieves the pixel size (X and Y) from the primary pixels of the first image.
        Validates that both pixel sizes are present and equal. Sets self.pixel_size to the value.

        Raises:
            MetadataValidationError: If pixel size information is missing, inconsistent, or cannot be determined.
        """
        first_image = self._get_first_image()

        # Get the pixel size from the image's pixels
        pixels = first_image.getPrimaryPixels()
        if not pixels:
            raise MetadataValidationError(
                "No pixel information found in image", logger
            )

        # Get the physical size in micrometers
        pixel_size_x = round(float(pixels.getPhysicalSizeX().getValue()), 1)
        pixel_size_y = round(float(pixels.getPhysicalSizeY().getValue()), 1)
        logger.debug(
            "Pixel size x: %s, Pixel size y: %s", pixel_size_x, pixel_size_y
        )
        # Validate that we have valid pixel sizes
        if pixel_size_x is None or pixel_size_y is None:
            raise MetadataValidationError(
                "Could not determine pixel size from image", logger
            )

        if pixel_size_x != pixel_size_y:
            raise MetadataValidationError(
                f"Pixel size x ({pixel_size_x}) and y ({pixel_size_y}) are not the same",
                logger,
            )

        log_success(
            SUCCESS_STYLE,
            f"The images in plate {self.plate_id} have a pixel size of {pixel_size_x} micrometers",
            logger,
        )
        self.pixel_size = pixel_size_x

    def well_conditions(self, well_id: str) -> dict[str, Any]:
        """Get the conditions for the specified well position (e.g. A1).

        See: WellWrapper.getWellPos().

        Returns:
            dict[str, Any]: Dictionary with annotations
        """
        idx = self.well_data["Well"].index(well_id)
        return {k: v[idx] for k, v in self.well_data.items() if k != "Well"}
