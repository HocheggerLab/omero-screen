import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from omero_screen.config import get_logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cellview.utils.error_classes import (
    DataError,
    DBError,
    FileError,
    StateError,
)

# Initialize logger with the module's name
logger = get_logger(__name__)


@dataclass
class CellViewState:
    """Singleton state manager for CellView application.

    This class maintains the state of various IDs and foreign keys
    that need to be tracked across different operations in the application.
    """

    # Class attributes with default values
    csv_path: Optional[Path] = None
    df: Optional[pd.DataFrame] = None
    plate_id: Optional[int] = None
    project_id: Optional[int] = None
    experiment_id: Optional[int] = None
    repeat_id: Optional[int] = None
    condition_id_map: Optional[dict[str, int]] = None
    lab_member: Optional[str] = None
    date: Optional[str] = None
    channel_0: Optional[str] = None
    channel_1: Optional[str] = None
    channel_2: Optional[str] = None
    channel_3: Optional[str] = None
    db_conn: Optional[duckdb.DuckDBPyConnection] = (
        None  # Add database connection
    )

    # Singleton instance
    _instance = None

    def __init__(self) -> None:
        """Initialize a new instance."""
        self.console = Console()
        self.logger = get_logger(__name__)

    @classmethod
    def get_instance(
        cls, args: Optional[argparse.Namespace] = None
    ) -> "CellViewState":
        """Get the singleton instance of CellViewState."""
        if cls._instance is None:
            # Create new instance
            instance = cls()
            cls._instance = instance

        # Initialize with args if provided
        if args is not None:
            cls._instance.csv_path = args.csv
            try:
                cls._instance.df = pd.read_csv(args.csv)
                cls._instance.plate_id = cls._instance.get_plate_id()
                cls._instance.date = cls._instance.extract_date_from_filename(
                    args.csv.name
                )
                channels = cls._instance.get_channels()
                cls._instance.channel_0 = channels[0] if channels else None
                cls._instance.channel_1 = (
                    channels[1] if len(channels) > 1 else None
                )
                cls._instance.channel_2 = (
                    channels[2] if len(channels) > 2 else None
                )
                cls._instance.channel_3 = (
                    channels[3] if len(channels) > 3 else None
                )
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as err:
                raise DataError(
                    f"Error reading CSV file: {err}",
                    context={"csv_path": str(args.csv)},
                ) from err
            except KeyError as err:
                raise DataError(
                    f"Required column missing in CSV: {err}",
                    context={
                        "csv_path": str(args.csv),
                        "missing_column": str(err),
                    },
                ) from err
            except ValueError as err:
                raise DataError(
                    f"Error processing data: {err}",
                    context={"csv_path": str(args.csv)},
                ) from err
            except FileNotFoundError as err:
                raise FileError(
                    "CSV file not found",
                    file_path=args.csv,
                    context={"csv_path": str(args.csv)},
                ) from err
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the state to default values."""
        cls._instance = None
        cls._instance = cls()
        cls._instance.console = Console()
        cls._instance.logger = get_logger(__name__)
        cls._instance.df = None
        cls._instance.plate_id = None
        cls._instance.project_id = None
        cls._instance.experiment_id = None
        cls._instance.repeat_id = None
        cls._instance.condition_id_map = None
        cls._instance.lab_member = None
        cls._instance.date = None
        cls._instance.channel_0 = None
        cls._instance.channel_1 = None
        cls._instance.channel_2 = None
        cls._instance.channel_3 = None

    def load_csv(self, csv_path: Path) -> None:
        """Load a CSV file into the state."""
        try:
            self.csv_path = csv_path
            self.df = pd.read_csv(csv_path)

            # Check required columns
            required_columns = ["plate_id", "well", "cell_line"]
            if missing_columns := [
                col for col in required_columns if col not in self.df.columns
            ]:
                raise DataError(
                    f"Missing required columns: {', '.join(missing_columns)}",
                    context={
                        "csv_path": str(csv_path),
                        "missing_columns": missing_columns,
                        "available_columns": list(self.df.columns),
                    },
                )

            self.console.print(
                Panel(
                    Text(f"Loaded CSV file: {csv_path}", style="green"),
                    title="CSV Load",
                    border_style="green",
                )
            )
        except pd.errors.EmptyDataError as err:
            raise DataError(
                "CSV file is empty",
                context={"csv_path": str(csv_path)},
            ) from err
        except pd.errors.ParserError as err:
            raise DataError(
                "Failed to parse CSV file",
                context={
                    "csv_path": str(csv_path),
                    "error": str(err),
                },
            ) from err
        except FileNotFoundError as err:
            raise FileError(
                "CSV file not found",
                file_path=csv_path,
                context={"csv_path": str(csv_path)},
            ) from err

    def get_plate_id(self) -> int:
        """Get the plate ID from the loaded DataFrame."""
        if self.df is None:
            raise StateError(
                "Cannot get plate ID: no DataFrame loaded",
                context={"current_state": self.get_state_dict()},
            )

        try:
            plate_ids = self.df["plate_id"].unique()
            if len(plate_ids) > 1:
                raise DataError(
                    "Multiple plates found in the CSV file",
                    context={
                        "plate_ids": list(plate_ids),
                        "csv_path": str(self.csv_path),
                    },
                )
            elif len(plate_ids) == 0:
                raise DataError(
                    "No plate ID found in CSV",
                    context={"csv_path": str(self.csv_path)},
                )

            self.plate_id = int(plate_ids[0])
            return self.plate_id
        except KeyError as err:
            raise DataError(
                "plate_id column not found in CSV",
                context={
                    "csv_path": str(self.csv_path),
                    "available_columns": list(self.df.columns),
                },
            ) from err

    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract a date in YYMMDD format from a filename and convert to YYYY-MM-DD.
        If no date is found, returns the current date in YYYY-MM-DD format.

        Args:
            filename: The filename to extract the date from

        Returns:
            The date string in YYYY-MM-DD format
        """
        # Pattern looks for 6 digits where first two start with 2 (for 20s decade)
        pattern = r"2[0-9][01][0-9][0-3][0-9]"
        match = re.search(pattern, filename)
        if not match:
            # Return current date in YYYY-MM-DD format
            return datetime.now().strftime("%Y-%m-%d")

        date_str = match[0]
        # Convert YYMMDD to YYYY-MM-DD
        yy = date_str[:2]
        mm = date_str[2:4]
        dd = date_str[4:6]
        return f"20{yy}-{mm}-{dd}"  # Assuming dates are in the 2000s

    def get_channels(self) -> list[str]:
        """Get the channels from the CSV file."""
        if self.df is None:
            return []
        pattern = re.compile(
            r"intensity_(?:max|min|mean)_([A-Za-z0-9]+)_[a-z]+"
        )
        names = self.df.columns.tolist()
        seen = set()
        markers = []
        for name in names:
            if match := pattern.match(name):
                marker = match[1]
                if marker not in seen:
                    markers.append(marker)
                    seen.add(marker)
        return markers

    def get_state_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the current state."""
        return {
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "df_loaded": self.df is not None,
            "plate_id": self.plate_id,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "repeat_id": self.repeat_id,
        }

    # -----------------methods to prepare for measurements import-----------------

    def prepare_for_measurements(self) -> None:
        """Prepare the dataframe for measurements import using the methods below."""
        meas_cols = self._find_measurement_cols()
        self.df = self._trim_df(meas_cols)
        channels = self._get_channel_list()
        self._validate_channels(channels)
        self._rename_channel_columns(channels)
        self._rename_centroid_cols()
        self._optimize_measurement_types()
        self._set_classifier()

    def _find_measurement_cols(self) -> list[str]:
        """Find columns that are likely to be measurements by identifying columns that vary within images."""
        assert isinstance(self.df, pd.DataFrame)

        # Count unique values per image for each column
        nunique_per_well = self.df.groupby("image_id").nunique()

        # For each column, find cols that have more than 1 unique value
        variable_cols = nunique_per_well.columns[
            nunique_per_well.max() > 1
        ].tolist()

        # Filter out Unnamed columns
        measurement_cols = [
            col for col in variable_cols if "Unnamed" not in col
        ]

        self.logger.debug("Found measurement columns: %s", measurement_cols)
        return measurement_cols

    def _trim_df(self, measurement_cols: list[str]) -> pd.DataFrame:
        """Trim the df to the measurement columns plus required identification columns.

        Keeps measurement columns along with well, image_id, and timepoint columns
        for proper data identification. Explicitly excludes plate_id.
        """
        if self.df is None:
            raise StateError("No dataframe loaded in state")

        # Required columns for identification
        required_cols = ["well", "image_id", "timepoint"]

        # Add required columns that exist in the dataframe
        cols_to_keep = measurement_cols.copy()
        for col in required_cols:
            if col in self.df.columns:
                cols_to_keep.append(col)
            else:
                raise StateError(
                    f"Required column {col} not found in dataframe",
                    context={"available_columns": self.df.columns.tolist()},
                )

        # Explicitly exclude plate_id
        cols_to_keep = [
            col
            for col in cols_to_keep
            if col not in ["plate_id", "integrated_int_DAPI"]
        ]

        # Return trimmed dataframe with only needed columns
        return self.df[cols_to_keep]

    def _get_channel_list(self) -> list[str]:
        """Set up channels for each measurement column.

        Identifies all unique channels from the measurement columns.
        The pattern is 'intensity_<measure_type>_<channel>_<location>'
        where measure_type is max, min, mean
        and location is nucleus, cell, cyto

        Returns:
            list[str]: List of unique channel names found in measurements in order of appearance
        """
        if self.df is None:
            return []

        # Use a list to maintain order of first appearance instead of a set
        channels = []

        for col in self.df.columns:
            if match := re.match(
                r"intensity_(?:max|min|mean)_([A-Za-z0-9]+)_(?:nucleus|cell|cyto)",
                col,
            ):
                channel = match[1]
                if channel not in channels:
                    channels.append(channel)

        self.logger.debug("Found channels: %s", channels)
        return channels

    def _validate_channels(self, channels: list[str]) -> None:
        """Compare the channels extracted from the df with the channels in the state."""
        # Get non-None state channels
        state_channels = [
            ch
            for ch in [
                self.channel_0,
                self.channel_1,
                self.channel_2,
                self.channel_3,
            ]
            if ch is not None
        ]

        # Check if we have enough extracted channels
        if len(state_channels) > len(channels):
            self.logger.warning("Not enough channels in data to match state")
            return

        # Check if channels match at each position
        if channels != state_channels:
            raise StateError(
                "Channels do not match",
                context={
                    "channels": channels,
                    "state_channels": state_channels,
                },
            )

    def _rename_channel_columns(self, channels: list[str]) -> None:
        """
        Renames non-DAPI channel names in DataFrame columns to ch1, ch2, etc.
        DAPI is left unchanged.
        """
        assert isinstance(self.df, pd.DataFrame)
        non_dapi_channels: list[str] = [ch for ch in channels if ch != "DAPI"]
        channel_map: dict[str, str] = {
            ch: f"ch{i + 1}" for i, ch in enumerate(non_dapi_channels)
        }

        # Create new column names using simple string replacement
        new_columns: list[str] = []
        for col in self.df.columns:
            new_col = col
            for original, replacement in channel_map.items():
                # Use direct string replacement instead of regex with word boundaries
                # which may not work as expected in all contexts
                if original in new_col:
                    new_col = new_col.replace(original, replacement)
            new_columns.append(new_col)

        # Rename columns
        self.df.rename(
            columns=dict(zip(self.df.columns, new_columns, strict=False)),
            inplace=True,
        )

    def _rename_centroid_cols(self) -> None:
        """Rename centroid columns to have consistent names.

        - Always add -nuc to centroid-0 and centroid-1
        - If centroid-0_x or centroid-1_x exist, rename to centroid-0-cell and centroid-1-cell
        - Drop any centroid-*_y columns
        """
        assert isinstance(self.df, pd.DataFrame)

        rename_map: dict[str, str] = {}

        # Rename base nucleus centroids
        if "centroid-0" in self.df.columns:
            rename_map["centroid-0"] = "centroid-0-nuc"
        if "centroid-1" in self.df.columns:
            rename_map["centroid-1"] = "centroid-1-nuc"

        # Rename cell centroids if present
        if "centroid-0_x" in self.df.columns:
            rename_map["centroid-0_x"] = "centroid-0-cell"
        if "centroid-1_x" in self.df.columns:
            rename_map["centroid-1_x"] = "centroid-1-cell"

        # Drop y-centroid columns if present
        drop_cols: list[str] = [
            col for col in self.df.columns if col.endswith("_y")
        ]

        # Apply renaming
        self.df.rename(columns=rename_map, inplace=True)

        # Drop _y columns
        self.df.drop(columns=drop_cols, inplace=True, errors="ignore")

    def _optimize_measurement_types(self) -> None:
        """Optimize data types for measurement columns.

        Converts numeric columns to appropriate types:
        - For columns with median > 10 and all values in 0-65535 range, converts to uint16
        - For other numeric columns, converts to float32
        """
        assert isinstance(self.df, pd.DataFrame)

        for col in self.df.columns:
            # Skip if column is not numeric
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            # Skip if column is already optimized
            if self.df[col].dtype in ["uint16", "float32"]:
                continue

            # Check if values are in uint16 range and median > 10
            if (
                (self.df[col] >= 0).all()
                and (self.df[col] <= 65535).all()
                and self.df[col].median() > 10
            ):
                self.df[col] = self.df[col].astype("uint16")
            else:
                # Convert to float32 for other numeric columns
                self.df[col] = self.df[col].astype("float32")

    def _set_classifier(self) -> None:
        """Set the classifier field in the repeats table based on the first column name."""
        assert isinstance(self.df, pd.DataFrame)

        # Check if we have a repeat_id stored
        if not self.repeat_id:
            raise StateError(
                "Cannot set classifier: no repeat_id stored in state",
                context={"state": self.__dict__},
            )

        # Check if we have a database connection
        if not self.db_conn:
            raise StateError(
                "Cannot set classifier: no database connection available",
                context={"state": self.__dict__},
            )

        # Get the first column name
        first_col = self.df.columns[0]
        if first_col != "label":
            # Update the classifier field in the repeats table using the stored connection
            try:
                self.db_conn.execute(
                    """
                    UPDATE repeats
                    SET classifier = ?
                    WHERE repeat_id = ?
                    """,
                    [first_col, self.repeat_id],
                )

                # Inform the user about the classifier
                self.console.print(
                    f"Found classifier column: [green]{first_col}[/green]"
                )

            except duckdb.Error as err:
                raise DBError(
                    "Failed to update classifier in repeats table",
                    context={
                        "repeat_id": self.repeat_id,
                        "classifier": first_col,
                        "error": str(err),
                    },
                ) from err

            self.df.rename(columns={first_col: "classifier"}, inplace=True)
        else:
            self.console.print(
                "No classifier column found in the file", style="yellow"
            )
