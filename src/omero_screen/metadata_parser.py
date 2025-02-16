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
from typing import Any

import pandas as pd
from omero.gateway import BlitzGateway, FileAnnotationWrapper, PlateWrapper
from omero_utils.attachments import get_named_file_attachment, parse_excel_data
from rich.console import Console
from rich.panel import Panel

from omero_screen.config import setup_logging

logger = setup_logging("omero_screen")
console = Console()


@dataclass
class PlateMetadata:
    """Data class to store plate metadata."""

    channels: dict[str, int]
    well_inputs: dict[str, Any]
    pixel_size: float


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
            excel_data = self._parse_excel_file(file_annotations)
            if self._validate_excel_data(excel_data):
                channel_data = self._format_channel_data(excel_data)  # noqa: F841
                well_data = self._format_well_data(excel_data)  # noqa: F841
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
        if self.plate:
            return get_named_file_attachment(self.plate, "metadata.xlsx")
        return None

    def _parse_excel_file(
        self, file_annotation: FileAnnotationWrapper
    ) -> dict[str, pd.DataFrame]:
        """Parse the metadata from the Excel file.
        returns a dictionary exceldata with keys Sheet1 and Sheet2, with the dataframes
        of the respective sheets.
        """
        exel_data: dict[str, pd.DataFrame] | None = parse_excel_data(
            file_annotation
        )
        if exel_data is None:
            raise MetadataParsingError(
                "No data could be parsed from the Excel file"
            )
        return exel_data

    def _validate_excel_data(
        self, excel_data: dict[str, pd.DataFrame]
    ) -> bool:
        """
        Validate Excel metadata structure and content.

        Requirements:
        - Must have Sheet1 and Sheet2
        - Sheet1 must have 'Channels' and 'Index' columns
        - Sheet1 'Channels' must contain at least one of: DAPI, HOECHST, RFP (case insensitive)
        - Sheet2 must have 'cell_line' column
        """
        if (
            not excel_data
            or "Sheet1" not in excel_data
            or "Sheet2" not in excel_data
        ):
            logger.debug("excel_data keys check %s", excel_data.keys())
            raise MetadataValidationError(
                "Missing required sheets: Sheet1 and Sheet2"
            )

        sheet1, sheet2 = excel_data["Sheet1"], excel_data["Sheet2"]

        # Check Sheet1 requirements Channels and Index
        if not {"Channels", "Index"}.issubset(sheet1.columns):
            logger.debug("Sheet1 columns check %s", sheet1.head())
            raise MetadataValidationError(
                "Sheet1 missing required columns: Channels and/or Index"
            )
        # Check Sheet1 requirements chanel for nuclei segmentation
        if (
            not sheet1["Channels"]
            .str.contains(r"DAPI|Hoechst|RFP", case=False)
            .any()
        ):
            logger.debug("Sheet1 columns check %s", sheet1.head())
            raise MetadataValidationError(
                "Sheet1 missing required columns:missing nuclei channel"
            )

        # Check Sheet2 requirements
        if "cell_line" not in sheet2.columns:
            logger.debug("Sheet2 columns check %s", sheet2.head())
            raise MetadataValidationError(
                "Sheet2 missing required column: cell_line"
            )

        return True

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

    def _check_plate_annotations(self) -> bool:
        """Check if the plate has annotations."""
        if self.plate is None or not hasattr(self.plate, "getAnnotations"):
            return False
        return self.plate.getAnnotations() is not None

    def _check_well_annotations(self) -> bool:
        """Check if the well has annotations."""
        if self.plate is None or not hasattr(self.plate, "getWellAnnotations"):
            return False
        return self.plate.getWellAnnotations() is not None


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
