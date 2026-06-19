"""Module for exporting data from CellView to a polars LazyFrame.

This module provides functions for exporting data from CellView directly to a Polars LazyFrame,
bypassing Pandas to avoid compatibility issues.
"""

import duckdb
import polars as pl
from loguru import logger

from cellview.utils.ui import CellViewUI


class PlateParserPolars:
    """Class for parsing plate data from the database into a Polars DataFrame.

    Attributes:
        conn: The active DuckDB connection.
        ui: The CellView UI.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """Initialize the PlateParserPolars with an active database connection.

        Args:
            conn: An active DuckDB connection
        """
        self.conn = conn
        self.ui = CellViewUI()

    def _get_condition_variables(
        self, plate_id: int, well: str | None = None
    ) -> tuple[pl.DataFrame, list[str]]:
        """Get condition variables as separate columns and return variable names.

        Args:
            plate_id: The ID of the plate to get variables for.
            well: Optional well label (e.g. ``"D1"``) to restrict the query
                to a single well. ``None`` returns all wells on the plate.

        Returns:
            A tuple containing:
                - A Polars DataFrame with condition variables as columns.
                - A list of unique variable names.
        """
        params: list[object] = [plate_id]
        where = "WHERE r.plate_id = ?"
        if well is not None:
            where += " AND c.well = ?"
            params.append(well)
        query = f"""
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
        {where}
        """
        # Load directly to Polars
        df = self.conn.execute(query, params).pl()

        variable_names = []
        if "variable_name" in df.columns:
            # Filter unique non-null variable names
            variable_names = (
                df["variable_name"].drop_nulls().unique().to_list()
            )

        logger.info(f"Unique variables: {variable_names}")

        if (
            not df.is_empty()
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
            df_base = df.select(base_cols).unique()

            # Then, pivot the variables
            # Polars pivot syntax: pivot(values, index, columns)
            # Note: pivot is eager in Polars usually.
            df_vars = df.pivot(
                values="variable_value",
                index=["well", "well_id"],
                on="variable_name",
                aggregate_function="first",  # Should be unique per well/id/name tuple anyway
            )

            # Merge the variables back with the base DataFrame
            # Polars join
            df = df_base.join(df_vars, on=["well", "well_id"], how="left")

            return df, variable_names

        return pl.DataFrame(), variable_names

    def _get_measurements(
        self,
        plate_id: int,
        well: str | None = None,
        timepoint: int | None = None,
    ) -> pl.DataFrame:
        """Get measurements for a plate.

        Args:
            plate_id: The ID of the plate to get measurements for.
            well: Optional well label (e.g. ``"D1"``) to restrict the query.
            timepoint: Optional timepoint to restrict the query.

        Returns:
            A Polars DataFrame with measurements.
        """
        # Get available columns in the measurements table
        # We can fetchdf() here for metadata as it's small/safe
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
        params: list[object] = [plate_id]
        where = "WHERE r.plate_id = ?"
        if well is not None:
            where += " AND c.well = ?"
            params.append(well)
        if timepoint is not None and "timepoint" in available_measurement_cols:
            where += " AND m.timepoint = ?"
            params.append(int(timepoint))
        query = f"""
        SELECT
            {select_clause}
        FROM repeats r
        JOIN conditions c ON r.repeat_id = c.repeat_id
        JOIN measurements m ON c.condition_id = m.condition_id
        JOIN experiments e ON r.experiment_id = e.experiment_id
        {where}
        ORDER BY c.well, r.repeat_id, m.measurement_id
        """
        # Execute and convert directly to Polars
        df = self.conn.execute(query, params).pl()

        # Debug logging
        if "cell_cycle" in df.columns:
            uniques = df["cell_cycle"].unique().to_list()
            logger.debug(
                f"Loaded measurements for plate {plate_id}. 'cell_cycle' uniques: {uniques}"
            )
        else:
            logger.debug(
                "cell_cycle column MISSING from measurements query result!"
            )

        return df

    def build_df(
        self,
        plate_id: int,
        well: str | None = None,
        timepoint: int | None = None,
    ) -> tuple[pl.DataFrame, list[str]]:
        """Get the final tidy DataFrame for a plate.

        Args:
            plate_id: The ID of the plate to collect data for.
            well: Optional well label to restrict the query (SQL pushdown).
            timepoint: Optional timepoint to restrict the query (SQL pushdown).

        Returns:
            A tidy Polars DataFrame with all measurements and well conditions.

        """
        # Get condition variables as separate columns and variable names
        conditions_df, variable_names = self._get_condition_variables(
            plate_id, well=well
        )
        # Get measurements
        measurements_df = self._get_measurements(
            plate_id, well=well, timepoint=timepoint
        )

        if measurements_df.is_empty():
            logger.error(f"No measurements found for plate {plate_id}")
            return pl.DataFrame(), variable_names

        # Guard against measurement readouts (e.g. ``*_background``) that may
        # have leaked into condition_variables on import in older DBs. The
        # measurements table is the source of truth, so drop any condition-side
        # column that also exists in measurements (other than the join keys);
        # otherwise the join below would disambiguate them with a ``_right``
        # suffix.
        join_keys = {"well", "well_id"}
        overlap = [
            col
            for col in conditions_df.columns
            if col in measurements_df.columns and col not in join_keys
        ]
        if overlap:
            conditions_df = conditions_df.drop(overlap)
            variable_names = [v for v in variable_names if v not in overlap]

        # Merge measurements with condition variables
        # Using left join similar to original logic
        df = measurements_df.join(
            conditions_df, on=["well", "well_id"], how="left"
        )

        logger.info(
            f"Retrieved DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        return df, variable_names


def _build_polars_rehydrate_map(
    columns: list[str], channel: str
) -> dict[str, str]:
    """Build the ``DAPI`` → ``{channel}`` rename map for a Polars DataFrame.

    Mirrors :func:`cellview.utils.nucleus_channel.rehydrate_dapi_to_nucleus`
    but operates on Polars' column-list interface so we can keep the
    Polars exporter free of pandas. No-op when ``channel == "DAPI"``.
    """
    if channel == "DAPI":
        return {}
    rename_map: dict[str, str] = {}
    for col in columns:
        new_col: str | None = None
        for stat in ("min", "mean", "max"):
            for segment in ("nucleus", "cell", "cyto"):
                if col == f"intensity_{stat}_DAPI_{segment}":
                    new_col = f"intensity_{stat}_{channel}_{segment}"
                    break
            if new_col is not None:
                break
        if new_col is None and col == "integrated_int_DAPI":
            new_col = f"integrated_int_{channel}"
        if new_col is None and col == "integrated_int_DAPI_norm":
            new_col = f"integrated_int_{channel}_norm"
        if new_col is not None and new_col not in columns:
            rename_map[col] = new_col
    return rename_map


def export_polars_lf(
    plate_id: int,
    conn: duckdb.DuckDBPyConnection,
    well: str | None = None,
    timepoint: int | None = None,
) -> tuple[pl.LazyFrame, list[str]]:
    """Export a plate as a Polars LazyFrame.

    The canonical ``*_DAPI_*`` measurement columns in the DB are rehydrated
    to use the actual nucleus fluorophore name (e.g. ``Hoechst``,
    ``H2B_RFP``) as recorded on ``repeats.nucleus_channel`` for this plate.

    Args:
        plate_id: The ID of the plate to export.
        conn: The active DuckDB connection.
        well: Optional well label (e.g. ``"D1"``) to push down into the SQL
            ``WHERE`` clause. Cuts the materialised row count from a full
            plate to a single well — useful for callers that only need
            one well (e.g. the napari training-data loader).
        timepoint: Optional timepoint to push down into the SQL ``WHERE``
            clause (matched against ``measurements.timepoint``).

    Returns:
        A tuple containing:
            - A Polars LazyFrame with the plate data.
            - A list of unique variable names.

    """
    parser = PlateParserPolars(conn)
    df, variable_names = parser.build_df(
        plate_id, well=well, timepoint=timepoint
    )

    if "experiment_name" in df.columns:
        df = df.rename({"experiment_name": "experiment"})

    # All rows in a single-plate export share the same nucleus_channel.
    if not df.is_empty() and "nucleus_channel" in df.columns:
        nucleus_channel = df["nucleus_channel"][0] or "DAPI"
        rename_map = _build_polars_rehydrate_map(
            df.columns, str(nucleus_channel)
        )
        if rename_map:
            df = df.rename(rename_map)

    # Return as LazyFrame
    return df.lazy(), variable_names
