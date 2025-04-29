import duckdb
import pandas as pd
from omero_screen.config import get_logger
from rich.console import Console

from cellview.utils.error_classes import MeasurementError
from cellview.utils.state import CellViewState

# Initialize logger with the module's name
logger = get_logger(__name__)


class MeasurementsManager:
    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        self.db_conn: duckdb.DuckDBPyConnection = db_conn
        self.console = Console()
        self.state = CellViewState.get_instance()
        self.logger = get_logger(__name__)

    def import_measurements(self) -> None:
        """Import measurements from the state dataframe into the database."""
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
        """Validate that the state has the required data."""
        if self.state.df is None:
            raise MeasurementError("No data available in state")
        if not self.state.condition_id_map:
            raise MeasurementError("No condition_id map available in state")

    def _get_measurement_columns(self, df: pd.DataFrame) -> list[str]:
        """Get the list of columns to insert into the measurements table.

        Excludes well column as it is used for condition_id lookup but not stored
        in the measurements table. Both image_id and timepoint are required columns
        in the measurements table.
        """
        return [col for col in df.columns if col != "well"]

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

    # # ----------------helper functions for _bulk_insert_measurements----------------
    # def _prepare_insert_statement(self, columns: list[str]) -> str:
    #     """Prepare the SQL INSERT statement for measurements."""
    #     placeholders = ", ".join(["?"] * len(columns))
    #     return f"""
    #     INSERT INTO measurements ({", ".join(columns)})
    #     VALUES ({placeholders})
    #     """

    # def _insert_measurement_row(
    #     self,
    #     row: pd.Series,  # type: ignore
    #     condition_id_map: dict[str, int],
    #     sql: str,
    # ) -> None:
    #     """Insert a single row of measurements into the database."""
    #     well = row["well"]
    #     condition_id = condition_id_map.get(well)

    #     if not condition_id:
    #         self.console.print(
    #             f"[yellow]Warning: No condition_id found for well {well}[/yellow]"
    #         )
    #         return

    #     # We know df is not None because _validate_state was called
    #     assert self.state.df is not None
    #     measurement_cols = self._get_measurement_columns(self.state.df)
    #     values = [condition_id] + [row[col] for col in measurement_cols]

    #     try:
    #         self.db_conn.execute(sql, values)
    #     except duckdb.Error as err:
    #         self.logger.error(
    #             "Failed to insert measurement for well %s: %s",
    #             well,
    #             str(err),
    #         )
    #         raise

    # from rich.table import Table

    # def _display_final_data(self) -> None:
    #     """Display a summary table of the imported data showing plate, project, experiment, and measurement information."""
    #     try:
    #         result = self._fetch_summary_data()
    #         if not result:
    #             self.console.print("[yellow]No data found to display[/yellow]")
    #             return

    #         variable_conditions = self._fetch_variable_conditions(result[3])
    #         measurements = self._fetch_measurements(result[3])

    #         summary_table = self._build_summary_table(
    #             result, variable_conditions
    #         )
    #         self.console.print()
    #         self.console.print(summary_table)

    #         if measurements:
    #             measurements_table = self._build_measurements_table(
    #                 measurements
    #             )
    #             self.console.print()
    #             self.console.print(measurements_table)
    #         else:
    #             self.console.print(
    #                 "[yellow]No measurements found for the first well[/yellow]"
    #             )

    #     except duckdb.Error as err:
    #         self.logger.error("Failed to display final data: %s", str(err))
    #         self.console.print("[red]Failed to display summary data[/red]")

    # # ----------------reassembling data from database for final display----------------

    # def _fetch_summary_data(self) -> Optional[Any]:
    #     return self.db_conn.execute(
    #         """
    #         SELECT
    #             r.plate_id,
    #             p.project_name,
    #             e.experiment_name,
    #             c.well,
    #             c.cell_line,
    #             c.antibody,
    #             c.antibody_1,
    #             c.antibody_2,
    #             c.antibody_3,
    #             r.channel_0,
    #             r.channel_1,
    #             r.channel_2,
    #             r.channel_3
    #         FROM repeats r
    #         JOIN experiments e ON r.experiment_id = e.experiment_id
    #         JOIN projects p ON e.project_id = p.project_id
    #         JOIN conditions c ON r.repeat_id = c.repeat_id
    #         WHERE r.repeat_id = ?
    #         LIMIT 1
    #         """,
    #         [self.state.repeat_id],
    #     ).fetchone()

    # def _fetch_variable_conditions(
    #     self, well: str
    # ) -> list[tuple[str, str | int | float]]:
    #     return cast(
    #         list[tuple[str, str | int | float]],
    #         self.db_conn.execute(
    #             """
    #         SELECT variable_name, variable_value
    #         FROM condition_variables cv
    #         JOIN conditions c ON cv.condition_id = c.condition_id
    #         WHERE c.well = ?
    #         """,
    #             [well],
    #         ).fetchall(),
    #     )

    # def _fetch_measurements(self, well: str) -> Any:
    #     return self.db_conn.execute(
    #         """
    #         SELECT
    #             m.image_id,
    #             m.timepoint,
    #             m.label,
    #             m.classifier,
    #             m.cell_cycle,
    #             m.cell_cycle_detailed,
    #             m.area_nucleus,
    #             m."centroid-0-nuc",
    #             m."centroid-1-nuc",
    #             m.intensity_min_DAPI_nucleus,
    #             m.intensity_mean_DAPI_nucleus,
    #             m.intensity_max_DAPI_nucleus,
    #             m.integrated_int_DAPI_norm,
    #             m.intensity_min_ch1_nucleus,
    #             m.intensity_mean_ch1_nucleus,
    #             m.intensity_max_ch1_nucleus,
    #             m.intensity_min_ch2_nucleus,
    #             m.intensity_mean_ch2_nucleus,
    #             m.intensity_max_ch2_nucleus,
    #             m.intensity_min_ch3_nucleus,
    #             m.intensity_mean_ch3_nucleus,
    #             m.intensity_max_ch3_nucleus,
    #             m.intensity_mean_ch3_nucleus_norm,
    #             m.Cyto_ID,
    #             m.area_cell,
    #             m."centroid-0-cell",
    #             m."centroid-1-cell",
    #             m.intensity_min_DAPI_cell,
    #             m.intensity_mean_DAPI_cell,
    #             m.intensity_max_DAPI_cell,
    #             m.intensity_min_ch1_cell,
    #             m.intensity_mean_ch1_cell,
    #             m.intensity_max_ch1_cell,
    #             m.intensity_min_ch2_cell,
    #             m.intensity_mean_ch2_cell,
    #             m.intensity_max_ch2_cell,
    #             m.intensity_min_ch3_cell,
    #             m.intensity_mean_ch3_cell,
    #             m.intensity_max_ch3_cell,
    #             m.area_cyto,
    #             m.intensity_min_DAPI_cyto,
    #             m.intensity_mean_DAPI_cyto,
    #             m.intensity_max_DAPI_cyto,
    #             m.intensity_min_ch1_cyto,
    #             m.intensity_mean_ch1_cyto,
    #             m.intensity_max_ch1_cyto,
    #             m.intensity_min_ch2_cyto,
    #             m.intensity_mean_ch2_cyto,
    #             m.intensity_max_ch2_cyto,
    #             m.intensity_min_ch3_cyto,
    #             m.intensity_mean_ch3_cyto,
    #             m.intensity_max_ch3_cyto
    #         FROM measurements m
    #         JOIN conditions c ON m.condition_id = c.condition_id
    #         WHERE c.well = ?
    #         LIMIT 5
    #         """,
    #         [well],
    #     ).fetchall()

    # def _build_summary_table(
    #     self,
    #     result: Any,
    #     variable_conditions: list[tuple[str, str | int | float]],
    # ) -> Table:
    #     table = Table(title="Import Summary")
    #     table.add_column("Field", style="cyan")
    #     table.add_column("Value", style="green")

    #     table.add_row("Plate ID", str(result[0]))
    #     table.add_row("Project", result[1])
    #     table.add_row("Experiment", result[2])
    #     table.add_row("First Well", result[3])
    #     table.add_row("Cell Line", result[4])

    #     for i, label in enumerate(
    #         ["Antibody", "Antibody 1", "Antibody 2", "Antibody 3"]
    #     ):
    #         if result[5 + i]:
    #             table.add_row(label, result[5 + i])

    #     for i in range(4):
    #         if result[9 + i]:
    #             table.add_row(f"Channel {i}", result[9 + i])

    #     if variable_conditions:
    #         table.add_row("", "")
    #         table.add_row("[bold]Variable Conditions[/bold]", "")
    #         for name, value in variable_conditions:
    #             table.add_row(name, str(value))

    #     return table

    # def _build_measurements_table(self, measurements: list[Any]) -> Table:
    #     measurements_table = Table(title="First 5 Measurements")
    #     measurements_table.add_column("Measurement Type", style="cyan")
    #     for i in range(5):
    #         measurements_table.add_column(
    #             f"Measurement {i + 1}", justify="right", style="green"
    #         )

    #     for m_type, get_value in self._measurement_types():
    #         row = [m_type] + [get_value(m) for m in measurements]
    #         measurements_table.add_row(*row)

    #     return measurements_table

    # @staticmethod
    # def _measurement_types() -> list[tuple[str, Callable[[Any], str]]]:
    #     return [
    #         ("Image ID", lambda m: str(m[0])),
    #         ("Timepoint", lambda m: str(m[1])),
    #         ("Label", lambda m: str(m[2])),
    #         ("Classifier", lambda m: str(m[3]) if m[3] else "N/A"),
    #         ("Cell Cycle", lambda m: str(m[4]) if m[4] else "N/A"),
    #         ("Cell Cycle Detailed", lambda m: str(m[5]) if m[5] else "N/A"),
    #         ("Nucleus Area", lambda m: f"{m[6]:.2f}"),
    #         ("Nucleus Centroid X", lambda m: f"{m[7]:.2f}"),
    #         ("Nucleus Centroid Y", lambda m: f"{m[8]:.2f}"),
    #         ("Nucleus DAPI Min", lambda m: f"{m[9]:.2f}"),
    #         ("Nucleus DAPI Mean", lambda m: f"{m[10]:.2f}"),
    #         ("Nucleus DAPI Max", lambda m: f"{m[11]:.2f}"),
    #         (
    #             "Nucleus DAPI Integrated",
    #             lambda m: f"{m[12]:.2f}" if m[12] else "N/A",
    #         ),
    #         ("Nucleus Ch1 Min", lambda m: f"{m[13]:.2f}" if m[13] else "N/A"),
    #         ("Nucleus Ch1 Mean", lambda m: f"{m[14]:.2f}" if m[14] else "N/A"),
    #         ("Nucleus Ch1 Max", lambda m: f"{m[15]:.2f}" if m[15] else "N/A"),
    #         ("Nucleus Ch2 Min", lambda m: f"{m[16]:.2f}" if m[16] else "N/A"),
    #         ("Nucleus Ch2 Mean", lambda m: f"{m[17]:.2f}" if m[17] else "N/A"),
    #         ("Nucleus Ch2 Max", lambda m: f"{m[18]:.2f}" if m[18] else "N/A"),
    #         ("Nucleus Ch3 Min", lambda m: f"{m[19]:.2f}" if m[19] else "N/A"),
    #         ("Nucleus Ch3 Mean", lambda m: f"{m[20]:.2f}" if m[20] else "N/A"),
    #         ("Nucleus Ch3 Max", lambda m: f"{m[21]:.2f}" if m[21] else "N/A"),
    #         (
    #             "Nucleus Ch3 Mean Norm",
    #             lambda m: f"{m[22]:.2f}" if m[22] else "N/A",
    #         ),
    #         ("Cyto ID", lambda m: str(m[23]) if m[23] else "N/A"),
    #         ("Cell Area", lambda m: f"{m[24]:.2f}" if m[24] else "N/A"),
    #         ("Cell Centroid X", lambda m: f"{m[25]:.2f}" if m[25] else "N/A"),
    #         ("Cell Centroid Y", lambda m: f"{m[26]:.2f}" if m[26] else "N/A"),
    #         ("Cell DAPI Min", lambda m: f"{m[27]:.2f}" if m[27] else "N/A"),
    #         ("Cell DAPI Mean", lambda m: f"{m[28]:.2f}" if m[28] else "N/A"),
    #         ("Cell DAPI Max", lambda m: f"{m[29]:.2f}" if m[29] else "N/A"),
    #         ("Cell Ch1 Min", lambda m: f"{m[30]:.2f}" if m[30] else "N/A"),
    #         ("Cell Ch1 Mean", lambda m: f"{m[31]:.2f}" if m[31] else "N/A"),
    #         ("Cell Ch1 Max", lambda m: f"{m[32]:.2f}" if m[32] else "N/A"),
    #         ("Cell Ch2 Min", lambda m: f"{m[33]:.2f}" if m[33] else "N/A"),
    #         ("Cell Ch2 Mean", lambda m: f"{m[34]:.2f}" if m[34] else "N/A"),
    #         ("Cell Ch2 Max", lambda m: f"{m[35]:.2f}" if m[35] else "N/A"),
    #         ("Cell Ch3 Min", lambda m: f"{m[36]:.2f}" if m[36] else "N/A"),
    #         ("Cell Ch3 Mean", lambda m: f"{m[37]:.2f}" if m[37] else "N/A"),
    #         ("Cell Ch3 Max", lambda m: f"{m[38]:.2f}" if m[38] else "N/A"),
    #         ("Cytoplasm Area", lambda m: f"{m[39]:.2f}" if m[39] else "N/A"),
    #         (
    #             "Cytoplasm DAPI Min",
    #             lambda m: f"{m[40]:.2f}" if m[40] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm DAPI Mean",
    #             lambda m: f"{m[41]:.2f}" if m[41] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm DAPI Max",
    #             lambda m: f"{m[42]:.2f}" if m[42] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch1 Min",
    #             lambda m: f"{m[43]:.2f}" if m[43] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch1 Mean",
    #             lambda m: f"{m[44]:.2f}" if m[44] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch1 Max",
    #             lambda m: f"{m[45]:.2f}" if m[45] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch2 Min",
    #             lambda m: f"{m[46]:.2f}" if m[46] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch2 Mean",
    #             lambda m: f"{m[47]:.2f}" if m[47] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch2 Max",
    #             lambda m: f"{m[48]:.2f}" if m[48] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch3 Min",
    #             lambda m: f"{m[49]:.2f}" if m[49] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch3 Mean",
    #             lambda m: f"{m[50]:.2f}" if m[50] else "N/A",
    #         ),
    #         (
    #             "Cytoplasm Ch3 Max",
    #             lambda m: f"{m[51]:.2f}" if m[51] else "N/A",
    #         ),
    #     ]


def import_measurements(conn: duckdb.DuckDBPyConnection) -> None:
    """Import measurements from the state dataframe into the database."""
    measurements_manager = MeasurementsManager(conn)
    measurements_manager.import_measurements()
    # measurements_manager._display_final_data()
