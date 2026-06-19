"""Module for exporting data from CellView to a pandas DataFrame.

This module provides a class for exporting data from CellView to a pandas DataFrame.
"""

import duckdb
import pandas as pd

from cellview.utils.ui import CellViewUI


class PlateParser:
    """Class for parsing plate data from the database into a pandas DataFrame.

    Attributes:
        conn: The active DuckDB connection.
        ui: The CellView UI.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """Initialize the PlateParser with an active database connection.

        Args:
            conn: An active DuckDB connection

        """
        self.conn = conn
        self.ui = CellViewUI()

    def _get_condition_variables(
        self, plate_id: int
    ) -> tuple[pd.DataFrame, list[str]]:
        """Get condition variables as separate columns and return variable names.

        Args:
            plate_id: The ID of the plate to get variables for.

        Returns:
            A tuple containing:
                - A pandas DataFrame with condition variables as columns.
                - A list of unique variable names.

        """
        query = """
        SELECT
            c.well,
            c.well_id,
            c.cell_line,
            c.antibody,
            c.antibody_1,
            c.antibody_2,
            c.antibody_3,
            cv.variable_name,
            cv.variable_value
        FROM repeats r
        JOIN conditions c ON r.repeat_id = c.repeat_id
        LEFT JOIN condition_variables cv ON c.condition_id = cv.condition_id
        WHERE r.plate_id = ?
        """
        df = self.conn.execute(query, [plate_id]).df()

        variable_names = []
        if "variable_name" in df.columns:
            variable_names = [
                v
                for v in df["variable_name"].dropna().unique().tolist()
                if v is not None
            ]

        self.ui.info(f"Unique variables: {variable_names}")

        if (
            not df.empty
            and "variable_name" in df.columns
            and "variable_value" in df.columns
        ):
            # First, get the base DataFrame with unique wells
            base_cols = [
                "well",
                "well_id",
                "cell_line",
                "antibody",
                "antibody_1",
                "antibody_2",
                "antibody_3",
            ]
            df_base = df[base_cols].drop_duplicates()

            # Then, pivot the variables
            df_vars = df.pivot(
                index=["well", "well_id"],
                columns="variable_name",
                values="variable_value",
            ).reset_index()

            # Merge the variables back with the base DataFrame
            df = pd.merge(df_base, df_vars, on=["well", "well_id"], how="left")

            return df, variable_names

        return pd.DataFrame(), variable_names

    def _get_measurements(self, plate_id: int) -> pd.DataFrame:
        """Get measurements for a plate.

        Args:
            plate_id: The ID of the plate to get measurements for.

        Returns:
            A pandas DataFrame with measurements.
        """
        # Get available columns in the measurements table
        table_info = self.conn.execute(
            "PRAGMA table_info(measurements)"
        ).fetchdf()
        available_measurement_cols = set(table_info["name"].tolist())

        # Always include columns from other tables
        always_include = [
            "r.plate_id",
            "r.repeat_id",
            "c.well",
            "c.well_id",
            "r.channel_0",
            "r.channel_1",
            "r.channel_2",
            "r.channel_3",
            "r.nucleus_channel",
            "e.experiment_name",
        ]

        # Build the list of measurement columns to select (all available columns from measurements table)
        measurement_cols = []
        for col_name in available_measurement_cols:
            # Skip primary key and foreign key columns that are handled by joins
            if col_name not in ["measurement_id", "condition_id"]:
                # Handle columns with special characters in their names
                if any(char in col_name for char in ["-", " "]):
                    measurement_cols.append(f'm."{col_name}"')
                else:
                    measurement_cols.append(f"m.{col_name}")

        # Combine always_include columns with all measurement columns
        select_cols = always_include + measurement_cols
        select_clause = ",\n            ".join(select_cols)
        query = f"""
        SELECT
            {select_clause}
        FROM repeats r
        JOIN conditions c ON r.repeat_id = c.repeat_id
        JOIN measurements m ON c.condition_id = m.condition_id
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE r.plate_id = ?
        ORDER BY c.well, r.repeat_id, m.measurement_id
        """
        df = self.conn.execute(query, [plate_id]).df()

        # Keep channel name columns for reference - they show what channels were used
        # All measurement columns are now included automatically, no need for renaming
        return df

    def build_df(self, plate_id: int) -> tuple[pd.DataFrame, list[str]]:
        """Get the final tidy DataFrame for a plate.

        Args:
            plate_id: The ID of the plate to collect data for.

        Returns:
            A tidy pandas DataFrame with all measurements and well conditions.

        """
        # Get condition variables as separate columns and variable names
        conditions_df, variable_names = self._get_condition_variables(plate_id)
        # Get measurements
        measurements_df = self._get_measurements(plate_id)
        if measurements_df.empty:
            self.ui.error(f"No measurements found for plate {plate_id}")
            return pd.DataFrame(), variable_names
        # Guard against measurement readouts (e.g. ``*_background``) that may
        # have leaked into condition_variables on import in older DBs. The
        # measurements table is the source of truth, so drop any condition-side
        # column that also exists in measurements (other than the join keys);
        # otherwise the merge below would disambiguate them with ``_x``/``_y``
        # suffixes.
        join_keys = {"well", "well_id"}
        overlap = [
            col
            for col in conditions_df.columns
            if col in measurements_df.columns and col not in join_keys
        ]
        if overlap:
            conditions_df = conditions_df.drop(columns=overlap)
            variable_names = [v for v in variable_names if v not in overlap]
        # Merge measurements with condition variables
        df = pd.merge(
            measurements_df, conditions_df, on=["well", "well_id"], how="left"
        )
        self.ui.info(
            f"Retrieved DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        return df, variable_names


def export_pandas_df(
    plate_id: int, conn: duckdb.DuckDBPyConnection
) -> tuple[pd.DataFrame, list[str]]:
    """Export a plate as a DataFrame.

    The canonical ``*_DAPI_*`` measurement columns in the DB are rehydrated
    to use the actual nucleus fluorophore name (e.g. ``Hoechst``,
    ``H2B_RFP``) as recorded on ``repeats.nucleus_channel`` for this plate.
    Legacy plates (``nucleus_channel == 'DAPI'``) emerge unchanged.

    Args:
        plate_id: The ID of the plate to export.
        conn: The active DuckDB connection.

    Returns:
        A tuple containing:
            - A pandas DataFrame with the plate data.
            - A list of unique variable names.

    """
    from cellview.utils.nucleus_channel import rehydrate_dapi_to_nucleus

    parser = PlateParser(conn)
    df, variable_names = parser.build_df(plate_id)
    df.rename(columns={"experiment_name": "experiment"}, inplace=True)
    df = df.dropna(axis=1, how="all")

    # All rows in a single-plate export share the same nucleus_channel.
    # Defensive fallback to 'DAPI' if the column is absent or empty.
    if "nucleus_channel" in df.columns and not df["nucleus_channel"].empty:
        nucleus_channel = df["nucleus_channel"].iloc[0] or "DAPI"
        df = rehydrate_dapi_to_nucleus(df, str(nucleus_channel))

    return df, variable_names
