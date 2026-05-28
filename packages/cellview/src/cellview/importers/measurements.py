"""Module for importing single cell measurements into CellView.

This module provides a class for managing single cell measurements import operations
to populate the measurements table.
"""

from typing import Optional

import duckdb
import pandas as pd
from rich.console import Console

from cellview.utils.error_classes import MeasurementError
from cellview.utils.state import CellViewState, CellViewStateCore
from omero_screen.config import get_logger

logger = get_logger(__name__)

# Trackastra nucleus-tracking columns added by omero-screen when --track is
# enabled. Migrated onto legacy measurements tables by
# ``MeasurementsManager._ensure_dynamic_columns_exist``.
_TRACK_COLUMNS = frozenset(
    {"track_id", "track_id_raw", "parent_track_id", "parent_track_id_raw"}
)


class MeasurementsManager:
    """Class for managing single cell measurements import operations.

    Attributes:
        db_conn: The DuckDB connection.
        console: The console.
        state: The CellView state.
        logger: The logger.
    """

    def __init__(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        state: Optional[CellViewStateCore] = None,
    ) -> None:
        """Initialize the MeasurementsManager.

        Args:
            db_conn: The DuckDB connection.
            state: The CellView state instance (optional, falls back to singleton if not provided).
        """
        self.db_conn: duckdb.DuckDBPyConnection = db_conn
        self.console = Console()
        # Support both dependency injection and backward compatibility with singleton
        self.state = (
            state if state is not None else CellViewState.get_instance()
        )
        self.logger = get_logger(__name__)

    def import_measurements(self) -> None:
        """Import measurements from the state dataframe into the database.

        Raises:
            MeasurementError: If the state is not valid.
        """
        self.state.prepare_for_measurements()
        # Validate state
        self._validate_state()
        # We know df is not None because _validate_state was called
        assert self.state.df is not None
        assert self.state.condition_id_map is not None

        # Get measurement columns
        measurement_cols = self._get_measurement_columns(self.state.df)

        # Prepare and insert measurements
        self._bulk_insert_measurements(measurement_cols)

        self.console.print("[green]Successfully imported measurements[/green]")

    def _validate_state(self) -> None:
        """Validate that the state has the required data.

        Raises:
            MeasurementError: If the state is not valid.
        """
        if self.state.df is None:
            raise MeasurementError("No data available in state")
        if not self.state.condition_id_map:
            raise MeasurementError("No condition_id map available in state")

    def _get_measurement_columns(self, df: pd.DataFrame) -> list[str]:
        """Get the list of columns to insert into the measurements table.

        Excludes well column as it is used for condition_id lookup but not stored
        in the measurements table. Both image_id and timepoint are required columns
        in the measurements table.

        Args:
            df: The dataframe to get the measurement columns from.

        Returns:
            The list of measurement columns.
        """
        # Sanitise column names: replace spaces with underscores
        df.columns = [col.replace(" ", "_") for col in df.columns]

        # Columns to exclude from measurements table
        exclude_cols = {
            "well",
            "experiment",
            "plate_id",
            "well_id",
            "cell_line",
            "si",
            "stimulus",
            "hours",
            "antibody",
            "antibody_1",
            "antibody_2",
            "antibody_3",
        }
        return [col for col in df.columns if col not in exclude_cols]

    def _bulk_insert_measurements(self, measurement_cols: list[str]) -> None:
        """Bulk insert measurements into the database using DuckDB's COPY command.

        Args:
            measurement_cols: List of measurement columns to insert

        Raises:
            MeasurementError: If any wells don't have corresponding condition_ids
        """
        # Ensure DataFrame and condition_id_map exist
        if self.state.df is None:
            raise MeasurementError("No DataFrame available in state")
        if self.state.condition_id_map is None:
            raise MeasurementError("No condition_id_map available in state")
        self.logger.debug("measurment_cols: %s", measurement_cols)

        # Add any missing dynamic columns to the measurements table
        self._ensure_dynamic_columns_exist(measurement_cols)

        # Ensure we are working on a copy to avoid SettingWithCopyWarning
        if self.state.df._is_view:
            self.state.df = self.state.df.copy()

        # Add condition_id to the state's DataFrame
        self.state.df["condition_id"] = self.state.df["well"].map(
            self.state.condition_id_map
        )

        # Check for any NaN values in condition_id
        if self.state.df["condition_id"].isna().any():
            missing_wells = (
                self.state.df[self.state.df["condition_id"].isna()]["well"]
                .unique()
                .tolist()
            )
            raise MeasurementError(
                "Found wells without corresponding condition_ids",
                context={
                    "missing_wells": missing_wells,
                    "available_wells": list(
                        self.state.condition_id_map.keys()
                    ),
                },
            )

        # Reorder columns to match database schema
        columns = ["condition_id"] + measurement_cols
        self.state.df = self.state.df[columns]

        # Remove duplicate columns
        self.state.df = self.state.df.loc[
            :, ~self.state.df.columns.duplicated()
        ]

        # Convert label column to string representation
        if "label" in self.state.df.columns:
            self.state.df["label"] = self.state.df["label"].astype(str)

        self.logger.info("df columns: %s", self.state.df.columns)
        # Bulk insert using DuckDB's COPY FROM
        # Register the DataFrame as a DuckDB table
        try:
            self.db_conn.register("temp_df", self.state.df)
            sql_columns = ", ".join(
                f'"{col}"' for col in self.state.df.columns
            )
            query = f"""
                INSERT INTO measurements ({sql_columns})
                SELECT {sql_columns} FROM temp_df
            """
            self.db_conn.execute(query)

        except Exception as err:
            raise MeasurementError(
                "Failed to import measurements into database"
            ) from err

    def _ensure_dynamic_columns_exist(
        self, measurement_cols: list[str]
    ) -> None:
        """Dynamically add missing intensity and classifier columns to the measurements table.

        ``intensity_*`` columns are added as FLOAT; ``classifier_*`` columns
        are added as TEXT.

        Args:
            measurement_cols: List of measurement columns from the dataframe

        Raises:
            MeasurementError: If unable to add columns to database
        """
        try:
            # Get current table columns
            result = self.db_conn.execute(
                "PRAGMA table_info(measurements)"
            ).fetchall()
            existing_columns = {
                row[1] for row in result
            }  # row[1] is column name

            self.logger.debug(
                "Existing columns from PRAGMA table_info: %s", existing_columns
            )

            # Collect columns to add with their SQL types
            columns_to_add: list[tuple[str, str]] = []
            for col in measurement_cols:
                if col in existing_columns:
                    continue
                if col.startswith("intensity_"):
                    if not self._validate_intensity_column_name(col):
                        raise MeasurementError(
                            f"Invalid column name format: {col}"
                        )
                    columns_to_add.append((col, "FLOAT"))
                elif col.startswith("classifier_"):
                    if not self._validate_classifier_column_name(col):
                        raise MeasurementError(
                            f"Invalid classifier column name format: {col}"
                        )
                    columns_to_add.append((col, "TEXT"))
                elif col.endswith("_background"):
                    if not self._validate_background_column_name(col):
                        raise MeasurementError(
                            f"Invalid background column name format: {col}"
                        )
                    columns_to_add.append((col, "FLOAT"))
                elif col in _TRACK_COLUMNS:
                    # Trackastra nucleus-tracking columns (track_id,
                    # track_id_raw, parent_track_id, parent_track_id_raw)
                    # added at import time for plates that opted into
                    # --track. The static schema also declares them on
                    # fresh DBs; this branch migrates legacy DBs.
                    columns_to_add.append((col, "INTEGER"))

            # Add missing columns
            for col, dtype in columns_to_add:
                try:
                    self.logger.info(
                        "Adding missing column to measurements table: %s (%s)",
                        col,
                        dtype,
                    )
                    self.db_conn.execute(
                        f'ALTER TABLE measurements ADD COLUMN "{col}" {dtype}'
                    )
                except duckdb.CatalogException as e:
                    if "already exists" in str(e):
                        self.logger.warning(
                            "Column %s already exists (likely from previous partial import), continuing",
                            col,
                        )
                    else:
                        raise

        except Exception as err:
            raise MeasurementError(
                f"Failed to add dynamic columns to measurements table: {err}"
            ) from err

    def _validate_intensity_column_name(self, column_name: str) -> bool:
        """Validate that an intensity column name is safe for SQL DDL operations.

        Args:
            column_name: The column name to validate

        Returns:
            True if the column name is valid, False otherwise
        """
        import re

        pattern = r"^intensity_[\w\-\(\)\.]+$"
        return bool(re.match(pattern, column_name))

    def _validate_classifier_column_name(self, column_name: str) -> bool:
        """Validate that a classifier column name is safe for SQL DDL operations.

        Args:
            column_name: The column name to validate

        Returns:
            True if the column name is valid, False otherwise
        """
        import re

        pattern = r"^classifier_[\w\-]+$"
        return bool(re.match(pattern, column_name))

    def _validate_background_column_name(self, column_name: str) -> bool:
        """Validate that a background column name is safe for SQL DDL operations.

        Args:
            column_name: The column name to validate

        Returns:
            True if the column name is valid, False otherwise
        """
        import re

        pattern = r"^[\w\-]+_background$"
        return bool(re.match(pattern, column_name))


def import_measurements(
    conn: duckdb.DuckDBPyConnection, state: Optional[CellViewStateCore] = None
) -> None:
    """Instantiate a MeasurementsManager and import measurements.

    Args:
        conn: The DuckDB connection.
        state: The CellView state instance (optional, falls back to singleton if not provided).
    """
    measurements_manager = MeasurementsManager(conn, state)
    measurements_manager.import_measurements()
