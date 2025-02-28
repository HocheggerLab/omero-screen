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
from typing import Any, Union

import pandas as pd
from omero.gateway import BlitzGateway, FileAnnotationWrapper, PlateWrapper
from omero_utils.attachments import get_file_attachments, parse_excel_data
from omero_utils.map_anns import parse_annotations
from pydantic.v1 import BaseModel as PydanticBaseModel
from pydantic.v1 import Field, validator
from rich.console import Console
from rich.panel import Panel

from omero_screen.config import setup_logging

logger = setup_logging("omero_screen")
console = Console()


# --------------------Dataclass to store metadata--------------------
@dataclass
class PlateMetadata:
    """Data class to store plate metadata."""

    channels: dict[str, int]
    well_inputs: dict[str, Any]
    pixel_size: float


# --------------------Pydantic Models--------------------


class ChannelData(PydanticBaseModel):  # type: ignore
    """Model for validating channel data from Sheet1"""

    Channels: str
    Index: int

    @validator("Channels")  # type: ignore[misc]
    def validate_channel_name(cls, v: str) -> str:
        """Ensure channel name is valid and standardize format"""
        v = v.upper().strip()
        return v

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ChannelData":
        """Create ChannelData from a row of the DataFrame"""
        return cls(**row)


class WellData(PydanticBaseModel):  # type: ignore
    """Model for validating well data from Sheet2"""

    Well: str
    metadata: dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict
    )

    @validator("Well")  # type: ignore[misc]
    def validate_well_format(cls, v: str) -> str:
        """Ensure well follows format like 'A1', 'B12', etc."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Well must be a non-empty string")
        if not (len(v) >= 2 and v[0].isalpha() and v[1:].isdigit()):
            raise ValueError("Well must be in format like 'A1', 'B12', etc.")
        return v.strip()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "WellData":
        """Create WellData from a row of the DataFrame"""
        well = row.pop("Well")  # Extract Well field
        return cls(
            Well=well, metadata=row
        )  # All other fields go into metadata


class ExcelMetadata(PydanticBaseModel):  # type: ignore
    """Model for complete Excel metadata validation"""

    sheet1: list[ChannelData]
    sheet2: list[WellData]

    @validator("sheet1")  # type: ignore[misc]
    def validate_required_channels(
        cls, channels: list[ChannelData]
    ) -> list[ChannelData]:
        """Ensure at least one nuclei channel (DAPI/HOECHST/RFP) exists"""
        nuclei_channels = {"DAPI", "HOECHST", "RFP"}
        if not any(ch.Channels.upper() in nuclei_channels for ch in channels):
            raise ValueError(
                "At least one nuclei channel (DAPI/HOECHST/RFP) is required"
            )
        return channels

    @validator("sheet1")  # type: ignore[misc]
    def validate_unique_indices(
        cls, channels: list[ChannelData]
    ) -> list[ChannelData]:
        """Ensure channel indices are unique"""
        indices = [ch.Index for ch in channels]
        if len(indices) != len(set(indices)):
            raise ValueError("Channel indices must be unique")
        return channels


# --------------------Metadata Parser--------------------
class MetadataParser:
    """Class to parse channel and well metadata from a plate.
    and store the data in a dataclass.
    """

    def __init__(
        self, conn: BlitzGateway, plate_id: int, console: Console = console
    ):
        self.conn: BlitzGateway = conn
        self.plate_id: int = plate_id
        self.plate: PlateWrapper | None = self.conn.getObject(
            "Plate", self.plate_id
        )
        self.console: Console = console

    def parse_metadata(self) -> None:
        """Parse the metadata from the plate."""
        self._check_plate()
        if file_annotations := self._check_excel_file():
            excel_data = parse_excel_data(file_annotations)
            validated_data = self._validate_excel_data(excel_data)
            channel_data = self._format_channel_data(validated_data)  # type: ignore[unused-ignore] # noqa: F841
            well_data = self._format_well_data(validated_data)  # type: ignore[unused-ignore] # noqa: F841

        # if self._check_plate_annotations() and self._check_well_annotations():
        #     channel_data = self._parse_plate_data()
        #     well_data = self._parse_well_data()

    def _check_plate(self) -> bool:
        """Check if the plate exists."""
        conn_object = self.conn.getObject("Plate", self.plate_id)
        if conn_object is None:
            raise PlateNotFoundError(
                f"A plate with id {self.plate_id} was not found!",
                console=self.console,
            )
        else:
            return True

    def _check_excel_file(self) -> FileAnnotationWrapper | None:
        """Check if the plate has an Excel file attachment."""
        assert self.plate is not None
        file_annotations = get_file_attachments(self.plate, ".xlsx")
        if file_annotations and len(file_annotations) == 1:
            return file_annotations[0]
        elif file_annotations and len(file_annotations) > 1:
            raise MetadataParsingError("Multiple Excel files found on plate")
        else:
            return None

    def _validate_excel_data(
        self, excel_data: dict[str, pd.DataFrame] | None
    ) -> ExcelMetadata:
        """Validate Excel metadata structure and content using Pydantic models"""
        if excel_data is None:
            raise MetadataValidationError(
                "No data could be parsed from the Excel file"
            )

        try:
            # Convert DataFrames to list of dicts for Pydantic validation
            sheet1_data = excel_data["Sheet1"].to_dict("records")
            sheet2_data = excel_data["Sheet2"].to_dict("records")

            # Validate using our Pydantic model
            validated_data = ExcelMetadata(
                sheet1=[ChannelData.from_row(row) for row in sheet1_data],
                sheet2=[WellData.from_row(row) for row in sheet2_data],
            )
            return validated_data

        except KeyError as e:
            raise MetadataValidationError(
                f"Missing required sheet: {e}"
            ) from e
        except ValueError as e:
            raise MetadataValidationError(f"Validation error: {e}") from e

    def _format_channel_data(
        self, excel_data: dict[str, pd.DataFrame]
    ) -> dict[str, int]:
        """Format the excel data for the plate.
        the channel dataframe is turned into a dict[str, int],
        e.g. {DAPI: 0}
        """
        return dict[str, int](
            excel_data["Sheet1"].set_index("Channels")["Index"].to_dict()
        )

    def _format_well_data(
        self, excel_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Format the excel data for the plate.
        the well dataframe is turned into a dict[str, Any],
        e.g. {A1: {cell_line: "RPE-1", condition: "ctr"}}
        """
        return {
            str(k): v
            for k, v in excel_data["Sheet2"]
            .set_index("Well")
            .to_dict(orient="index")
            .items()
        }

    # --------------------Plate and Well Annotation Parsing--------------------

    def _parse_plate_annotations(self) -> dict[str, Any]:
        """Parse channel annotations from the plate.

        Returns:
            dict[str, Any]: Dictionary of channel annotations with integer indices

        Raises:
            MetadataValidationError: If channel indices cannot be converted to integers
            MetadataValidationError: If no nuclei channel (DAPI/HOECHST/RFP) is found
        """
        assert self.plate
        annotations = parse_annotations(self.plate)

        # Check for required nuclei channels
        nuclei_channels = {"DAPI", "HOECHST", "RFP"}
        if not any(
            channel.upper() in nuclei_channels for channel in annotations
        ):
            raise MetadataValidationError(
                "At least one nuclei channel (DAPI/HOECHST/RFP) is required"
            )

        # Convert channel indices to integers
        channel_data = {}
        for channel, index in annotations.items():
            try:
                channel_data[channel] = int(index)
            except (ValueError, TypeError) as e:
                raise MetadataValidationError(
                    f"Channel index for {channel} must be an integer. Got: {index}"
                ) from e

        return channel_data

    def _parse_well_annotations(self) -> dict[str, str]:
        """Check if the well has annotations."""
        assert self.plate
        well_annotations = {}
        for well in self.plate.listChildren():
            well_name = well.getWellPos()
            well_annotations[well_name] = well.getAnnotation()
        return well_annotations


# --------------------Custom Exception Classes--------------------


class PlateNotFoundError(Exception):
    """Raised when a plate is not found"""

    def __init__(self, message: str, console: Console = console):
        super().__init__(message)
        console.print(
            Panel.fit(
                f"[red]Plate Not Found:[/red]\n{message}",
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
