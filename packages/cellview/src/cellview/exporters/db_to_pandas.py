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
        query = """
        SELECT
            r.plate_id,
            r.repeat_id,
            c.well,
            c.well_id,
            m.measurement_id,
            m.condition_id,
            m.image_id,
            m.timepoint,
            m.classifier,
            m.cell_cycle,
            m.cell_cycle_detailed,
            m.label,
            m.area_nucleus,
            m."centroid-0-nuc",
            m."centroid-1-nuc",
            m.integrated_int_DAPI_norm,
            m.intensity_mean_ch2_nucleus,
            m.intensity_mean_ch3_nucleus_norm,
            m.area_cell,
            m."centroid-0-cell",
            m."centroid-1-cell",
            r.channel_0,
            r.channel_1,
            r.channel_2,
            r.channel_3
        FROM repeats r
        JOIN conditions c ON r.repeat_id = c.repeat_id
        JOIN measurements m ON c.condition_id = m.condition_id
        WHERE r.plate_id = ?
        ORDER BY c.well, r.repeat_id, m.measurement_id
        """
        df = self.conn.execute(query, [plate_id]).df()

        # Get the channel names for this plate
        channel_names = (
            df[["channel_0", "channel_1", "channel_2", "channel_3"]]
            .iloc[0]
            .to_dict()
        )

        # Rename the columns based on channel names
        column_mapping = {
            "intensity_mean_ch2_nucleus": f"intensity_mean_{channel_names['channel_2']}_nucleus",
            "intensity_mean_ch3_nucleus_norm": f"intensity_mean_{channel_names['channel_3']}_nucleus_norm",
        }

        # Drop the channel name columns and rename the measurement columns
        df = df.drop(
            columns=["channel_0", "channel_1", "channel_2", "channel_3"]
        )
        df = df.rename(columns=column_mapping)

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

    Args:
        plate_id: The ID of the plate to export.
        conn: The active DuckDB connection.

    Returns:
        A tuple containing:
            - A pandas DataFrame with the plate data.
            - A list of unique variable names.

    """
    parser = PlateParser(conn)
    df, variable_names = parser.build_df(plate_id)

    # Drop any columns that contain NaN values
    df = df.dropna(axis=1, how="all")

    return df, variable_names
