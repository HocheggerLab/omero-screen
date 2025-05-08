"""Module for parsing metadata from OMERO.
First the plate is scanned for an Excel file attachment.
If found, the metadata is parsed from the Excel file.
Channel data is added to the plate and Well data is added
as key value annotations to each well.
plate data and metadata are then stored in a dataclass.
If no Excel file is found, the metadata is parsed from the plate data.
If metadata is not found, the program exits with an error.
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
    """Class to parse channel and well metadata from a plate.
    and store the data in a dataclass.
    """

    def __init__(self, conn: BlitzGateway, plate_id: int):
        self.conn: BlitzGateway = conn
        self.plate_id: int = plate_id
        self.plate: PlateWrapper = self._check_plate()
        self.excel_file: bool = False
        self.channel_data: dict[str, str] = {}
        self.well_data: dict[str, Any] = {}
        self.pixel_size: float = 0

    def _check_plate(self) -> PlateWrapper:
        """Get the plate, validating it exists first.

        Returns:
            PlateWrapper: The validated plate object

        Raises:
            PlateNotFoundError: If the plate with the given ID doesn't exist
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
        """Manage the metadata for the plate."""
        self._parse_metadata()  # checks for excel file or well data and pulls channel and well data into self.channel_data and self.well_data dictionaries
        self._validate_metadata()
        self._get_pixel_size()
        if self.excel_file:  # if excel file is found, add channel and well annotations to plate and delete excel file
            self._add_channel_annotations(self.channel_data)
            self._add_well_annotations(self.well_data)
            delete_excel_attachment(self.conn, self.plate)
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
        """Parse the metadata from the plate.

        Returns:
            PlateMetadata: A dataclass containing the parsed metadata

        Raises:
            ExcelParsingError: If there are issues parsing the Excel file
            ChannelAnnotationError: If there are issues with channel annotations
            WellAnnotationError: If there are issues with well annotations
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
        """Check if the plate has an Excel file attachment."""
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
        """Load the data from the Excel file."""

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
        """Delete preexisting annotations and add the channel annotations to the plate."""
        delete_map_annotations(self.conn, self.plate)
        add_map_annotations(self.conn, self.plate, channel_data)

    def _add_well_annotations(self, well_data: dict[str, Any]) -> None:
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
        """Parse the channel annotations from the plate.

        Returns:
            dict[str, str]: Dictionary mapping channel names to their indices

        Raises:
            ChannelAnnotationError: If no channel annotations are found or if values are not integers
        """
        if annotations := parse_annotations(self.plate):
            # Validate and convert values to integers
            return annotations  # type: ignore
        else:
            raise ChannelAnnotationError(
                "No channel annotations found on plate", logger
            )

    def _parse_well_annotations(self) -> dict[str, Any]:
        """Parse the well annotations from the plate.

        Returns a dictionary where each key is an annotation key and the value
        is a list of values for that key across all wells. Also includes a 'Well'
        key with the well positions.

        Returns:
            dict[str, Any]: Dictionary with annotation keys mapping to lists of values
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
        """Create a table with two columns.

        Args:
            title: The title of the table
            col1_name: Name of the first column
            col2_name: Name of the second column

        Returns:
            Table: A Rich table with two columns
        """
        table = Table(title=title)
        table.add_column(col1_name, style="cyan")
        table.add_column(col2_name, style="green")
        return table

    def _display_metadata(self) -> None:
        """Display the metadata in a nicely formatted way using Rich."""
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
        """Validate the metadata.

        Collects all validation errors and reports them together.
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
        """Validate the basic structure and types of the metadata.

        Returns:
            list[str]: List of error messages, empty if no errors
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
        """Make sure the channel data contain a nuclei channel
        and normalize the channel names to DAPI.

        Returns:
            list[str]: List of error messages, empty if no errors
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

    def _validate_well_positions(self) -> list[str]:
        """Validate that the well positions in the metadata match the actual wells in the plate.

        Returns:
            list[str]: List of error messages, empty if no errors
        """
        errors = []
        # Get actual well positions from the plate
        actual_wells = [
            well.getWellPos() for well in self.plate.listChildren()
        ]

        # Get well positions from metadata
        metadata_wells = self.well_data["Well"]

        # Check for missing wells
        if len(actual_wells) != len(metadata_wells):
            if len(actual_wells) > len(metadata_wells):
                missing_wells = set(actual_wells) - set(metadata_wells)
                errors.append(
                    f"Missing wells in metadata: {', '.join(sorted(missing_wells))}"
                )
            else:
                extra_wells = set(metadata_wells) - set(actual_wells)
                errors.append(
                    f"Extra wells in metadata: {', '.join(sorted(extra_wells))}"
                )

        # Check well order
        if actual_wells != metadata_wells:
            # Collect all mismatches
            mismatches: list[str] = []
            mismatches.extend(
                f"position {i + 1}: expected {actual}, found {metadata}"
                for i, (actual, metadata) in enumerate(
                    zip(actual_wells, metadata_wells, strict=False)
                )
                if actual != metadata
            )
            errors.append(f"Well order mismatches at {', '.join(mismatches)}")

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

    def _get_first_image(self) -> Any:
        """Get the first image from the first well of the plate.

        Returns:
            Any: The first image from the first well

        Raises:
            MetadataValidationError: If no wells or images are found
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
        """Get the pixel size in micrometers from the first image of the first well.
        Raises:
            MetadataValidationError: If no images are found or if pixel size cannot be determined
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
        d = {k: v[idx] for k, v in self.well_data.items() if k != "Well"}
        return d
