"""Module for parsing metadata from.
First the plate is scanned for an Excel file attachment.
If found, the metadata is parsed from the Excel file.
Channel data is added to the plate and Well data is added
as key value annotations to each well.
plate data and metadat are then stored in a dataclass
If no Excel file is found, the metadata is parsed from the plate data.
if metadata is not found, the program extists with an error.
"""

from dataclasses import dataclass
from typing import Any, TypeVar

from omero.gateway import BlitzGateway, FileAnnotationWrapper, PlateWrapper
from rich.console import Console
from rich.panel import Panel

from omero_screen.config import setup_logging
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

logger = setup_logging("omero_screen")
console = Console()

T = TypeVar("T")


# --------------------Dataclass to store metadata--------------------
@dataclass
class PlateMetadata:
    """Data class to store plate metadata."""

    channels: dict[str, int]
    well_inputs: dict[str, Any]
    pixel_size: float


# --------------------Metadata Parser--------------------
class MetadataParser:
    """Class to parse channel and well metadata from a plate.
    and store the data in a dataclass.
    """

    def __init__(self, conn: BlitzGateway, plate_id: int):
        self.conn: BlitzGateway = conn
        self.plate_id: int = plate_id
        self.plate: PlateWrapper = self._check_plate()
        self.channel_data: dict[str, int] = {}
        self.well_data: dict[str, Any] = {}

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
                f"A plate with id {self.plate_id} was not found!"
            )
        assert isinstance(plate, PlateWrapper)
        return plate

    def manage_metadata(self) -> None:
        """Manage the metadata for the plate."""
        self._parse_metadata()
        self._validate_metadata()

    # --------------------Metadata Parsing--------------------

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
            try:
                self.channel_data, self.well_data = self._load_data_from_excel(
                    file_annotations
                )
                self._add_channel_annotations(self.channel_data)
                self._add_well_annotations(self.well_data)
                delete_excel_attachment(self.conn, self.plate)
            except ExcelParsingError as e:
                raise ExcelParsingError(
                    f"Failed to parse Excel file: {str(e)}"
                ) from e
        else:
            try:
                self.channel_data = self._parse_channel_annotations()
                self.well_data = self._parse_well_annotations()
            except (ChannelAnnotationError, WellAnnotationError) as e:
                if isinstance(e, ChannelAnnotationError):
                    raise ChannelAnnotationError(
                        f"Failed to parse channel annotations: {str(e)}"
                    ) from e
                else:
                    raise WellAnnotationError(
                        f"Failed to parse well annotations: {str(e)}"
                    ) from e

    def _check_excel_file(self) -> FileAnnotationWrapper | None:
        """Check if the plate has an Excel file attachment."""
        # Plate is already validated in __init__
        file_annotations = get_file_attachments(self.plate, ".xlsx")
        if file_annotations and len(file_annotations) == 1:
            return file_annotations[0]
        elif file_annotations and len(file_annotations) > 1:
            raise ExcelParsingError("Multiple Excel files found on plate")
        else:
            return None

    def _load_data_from_excel(
        self, file_annotations: FileAnnotationWrapper
    ) -> tuple[dict[str, int], dict[str, Any]]:
        """Load the data from the Excel file."""
        meta_data = parse_excel_data(file_annotations)
        if meta_data and list(meta_data.keys()) == ["Sheet1", "Sheet2"]:
            channel_data = {
                str(k): v
                for k, v in meta_data["Sheet1"].to_dict(orient="list").items()
            }
            well_data = {
                str(k): v
                for k, v in meta_data["Sheet2"].to_dict(orient="list").items()
            }
            return channel_data, well_data
        else:
            raise ExcelParsingError(
                "Invalid excel file format - expected Sheet1 and Sheet2"
            )

    def _add_channel_annotations(self, channel_data: dict[str, int]) -> None:
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

    def _parse_channel_annotations(self) -> dict[str, int]:
        """Parse the channel annotations from the plate.

        Returns:
            dict[str, int]: Dictionary mapping channel names to their indices

        Raises:
            ChannelAnnotationError: If no channel annotations are found
        """
        annotations = parse_annotations(self.plate)
        if not annotations:
            raise ChannelAnnotationError(
                "No channel annotations found on plate"
            )
        return annotations

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
            if well_annotation:
                # For each key in the well's annotations, ensure it exists in well_data
                # and append the value to its list
                for key, value in well_annotation.items():
                    if key not in well_data:
                        well_data[key] = []
                    well_data[key].append(value)
            else:
                raise WellAnnotationError(
                    f"No well annotations found for well {well_pos}"
                )

        return well_data

    # --------------------Metadata Validation--------------------

    def _validate_metadata(self) -> None:
        """Validate the metadata."""
        self._validate_metadata_structure()
        # Check for nuclei channel and normalize to DAPI
        self._validate_channel_data()
        self._validate_well_data()

    def _validate_metadata_structure(self) -> None:
        """Validate the basic structure and types of the metadata."""
        if not self.channel_data:
            raise MetadataValidationError("No channel data found")
        if not self.well_data:
            raise MetadataValidationError("No well data found")

        if not isinstance(self.channel_data, dict) or not all(
            isinstance(k, str) for k in self.channel_data
        ):
            raise MetadataValidationError(
                "Channel data must be a dictionary with string keys"
            )
        if not all(isinstance(v, int) for v in self.channel_data.values()):
            raise MetadataValidationError(
                "Channel data must be a dictionary with integer values"
            )

    def _validate_channel_data(self) -> None:
        """Make sure the channel data contain a nuclei channels
        and normalize the channel names to DAPI.
        """
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
            raise MetadataValidationError(
                "At least one nuclei channel (DAPI/Hoechst/DNA/RFP) is required"
            )

        self.channel_data = channel_data_normalized

    def _validate_well_data(self) -> None:
        """Validate the well data structure and content.

        Ensures:
        - Required keys ("Well" and "cell_line") exist
        - All values are lists
        - All lists have the same length (matching number of wells)

        Raises:
            MetadataValidationError: If any validation check fails
        """
        # Check required keys exist
        required_keys = {"Well", "cell_line"}
        missing_keys = required_keys - self.well_data.keys()
        if missing_keys:
            raise MetadataValidationError(
                f"Missing required keys in well data: {', '.join(missing_keys)}"
            )

        # Check all values are lists
        non_list_keys = [
            key
            for key, value in self.well_data.items()
            if not isinstance(value, list)
        ]
        if non_list_keys:
            raise MetadataValidationError(
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
            raise MetadataValidationError(
                f"All well data lists must have the same length. Found: {', '.join(length_info)}"
            )


# --------------------Custom Exception Classes--------------------


class PlateNotFoundError(Exception):
    """Raised when a plate is not found"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Plate Not Found:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )


class ExcelParsingError(Exception):
    """Raised when there are issues parsing the Excel file"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Excel Parsing Error:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )


class ChannelAnnotationError(Exception):
    """Raised when there are issues with channel annotations"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Channel Annotation Error:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )


class WellAnnotationError(Exception):
    """Raised when there are issues with well annotations"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Well Annotation Error:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )


class MetadataValidationError(Exception):
    """Raised when parsed data doesn't meet requirements"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Metadata Validation Error:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )


class MetadataParsingError(Exception):
    """Raised when no data could be parsed from the Excel file"""

    def __init__(self, message: str):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Metadata Parsing Error:[/red]\n{message}",
                title="Error",
                border_style="red",
            )
        )
