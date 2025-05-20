"""Module for parsing well level experimental conditions from omero screen csv files."""

import logging

import duckdb
import pandas as pd
from omero_screen.config import get_logger
from rich.console import Console

from cellview.utils.error_classes import DataError
from cellview.utils.state import CellViewState

# Initialize logger with the module's name
logger = get_logger(__name__)
SUCCESS_STYLE = "bold green"


class ConditionManager:
    """Class for parsing well level experimental conditions from omero screen csv files.

    Attributes:
        db_conn: The DuckDB connection.
        state: The CellView state.
        logger: The logger.
        console: The console.
        per_well_constant_cols: The columns that are constant per well.
    """

    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize the ConditionManager.

        Args:
            db_conn: The DuckDB connection.
        """
        self.db_conn: duckdb.DuckDBPyConnection = db_conn
        self.state: CellViewState = CellViewState.get_instance()
        self.logger: logging.Logger = get_logger(__name__)
        self.logger.debug(
            "State initialized with repeat_id: %s", self.state.repeat_id
        )
        self.console = Console()
        self.per_well_constant_cols: list[str] = [
            "well",
            "well_id",
            "cell_line",
        ]

    def populate_condition_variables(
        self, condition_id_map: dict[str, int]
    ) -> None:
        """Populate the condition_variables table with per-well information.

        Args:
            condition_id_map: The map of well to condition_id.
        """
        assert isinstance(self.state.df, pd.DataFrame)
        variable_cols = self._identify_variable_columns()
        self._populate_condition_variables_table(
            variable_cols, condition_id_map
        )

    def _check_antibodies(self) -> None:
        """Check if channels are annotated as wavelengths.

        If they are, the df needs to contain corresponding antibody columns.

        Raises:
            DataError: If the antibody column is missing.
        """
        assert isinstance(self.state.df, pd.DataFrame)
        # Create a list of channel values and their positions
        channels = [
            (self.state.channel_1, 1),
            (self.state.channel_2, 2),
            (self.state.channel_3, 3),
        ]

        # Count how many channels are integers (wavelengths)
        wavelength_channels = [
            (val, idx) for val, idx in channels if isinstance(val, int)
        ]
        num_wavelengths = len(wavelength_channels)

        if num_wavelengths == 0:
            self.logger.info("No channel is set as wavelength")
            return

        # For single wavelength, check for 'antibody' column
        if num_wavelengths == 1:
            if "antibody" not in self.state.df.columns:
                channel_number = wavelength_channels[0][1]
                raise DataError(
                    f"Antibody column is missing for channel {channel_number}"
                )
            self.per_well_constant_cols.append("antibody")

        # For multiple wavelengths, check for 'antibody_X' columns
        else:
            for _, channel_number in wavelength_channels:
                antibody_col = f"antibody_{channel_number}"
                if antibody_col not in self.state.df.columns:
                    raise DataError(
                        f"Antibody column '{antibody_col}' is missing for channel {channel_number}"
                    )
                self.per_well_constant_cols.append(antibody_col)

    def _populate_conditions_table(
        self, conditions_dict: dict[str, dict[str, str | int | float]]
    ) -> None:
        """Populate the conditions table with per-well information.

        Args:
            conditions_dict: The dictionary of conditions.

        Raises:
            DataError: If the conditions table cannot be populated.
        """
        try:
            # Get the current repeat_id from state
            repeat_id = self.state.repeat_id

            # Prepare the data for insertion
            for well, conditions in conditions_dict.items():
                # Extract the values we need
                well_id = conditions["well_id"]
                cell_line = conditions["cell_line"]

                # Handle antibody columns
                antibody = conditions.get("antibody")
                antibody_1 = conditions.get("antibody_1")
                antibody_2 = conditions.get("antibody_2")
                antibody_3 = conditions.get("antibody_3")

                # Insert into conditions table
                self.db_conn.execute(
                    """
                    INSERT INTO conditions (
                        repeat_id, well, well_id, cell_line,
                        antibody, antibody_1, antibody_2, antibody_3
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        repeat_id,
                        well,
                        well_id,
                        cell_line,
                        antibody,
                        antibody_1,
                        antibody_2,
                        antibody_3,
                    ),
                )

            # Commit the transaction to ensure the records are actually inserted
            self.db_conn.commit()

            self.logger.info(
                "Successfully populated conditions table with %d wells",
                len(conditions_dict),
            )

        except duckdb.Error as e:
            raise DataError(
                f"Failed to populate conditions table: {str(e)}"
            ) from e

    def define_per_well_conditions(self) -> dict[str, int]:
        """Define per-well conditions and populate the database.

        Returns:
            The dictionary of conditions.
        """
        assert isinstance(self.state.df, pd.DataFrame)
        self._check_antibodies()
        raw_dict = (
            self.state.df.groupby("well")[self.per_well_constant_cols]
            .first()
            .to_dict("index")
        )
        # Convert to correct types
        conditions_dict: dict[str, dict[str, str | int | float]] = {
            str(k): {str(k2): v2 for k2, v2 in v.items()}
            for k, v in raw_dict.items()
        }

        self._populate_conditions_table(conditions_dict)

        # Get condition IDs for each well
        result = self.db_conn.execute(
            "SELECT well, condition_id FROM conditions WHERE repeat_id = ?",
            (self.state.repeat_id,),
        ).fetchall()

        # Create a dictionary with well as key and condition_id as value
        condition_id_map = {row[0]: row[1] for row in result}
        self.logger.debug("Final condition_id_map: %s", condition_id_map)
        self.state.condition_id_map = condition_id_map
        return condition_id_map

    def _identify_variable_columns(self) -> list[str]:
        """Identify variable columns in the dataframe.

        Returns:
            The list of variable columns.
        """
        assert isinstance(self.state.df, pd.DataFrame)

        # Find columns that are constant per well
        per_well_constant_cols = (
            self.state.df.groupby("well")
            .nunique()
            .eq(1)
            .all()
            .pipe(lambda x: x[x].index)
            .tolist()
        )

        # Exclude metadata columns
        exclude_cols = {
            "experiment",
            "image_id",
            "well_id",
            "plate_id",
            "cell_line",
            "timepoint",
        }
        well_variable = [
            col for col in per_well_constant_cols if col not in exclude_cols
        ]

        self.logger.info("Well variable: %s", well_variable)
        return well_variable

    def _populate_condition_variables_table(
        self, variable_cols: list[str], condition_id_map: dict[str, int]
    ) -> None:
        """Populate the condition_variables table with per-well information.

        Args:
            variable_cols: The list of variable columns.
            condition_id_map: The map of well to condition_id.
        """
        try:
            assert isinstance(self.state.df, pd.DataFrame)
            # Create a dictionary where keys are wells and values are dictionaries of variable values
            well_variables = (
                self.state.df.groupby("well")[variable_cols]
                .first()
                .to_dict("index")
            )

            for well, variables in well_variables.items():
                condition_id = condition_id_map[str(well)]

                # Insert each variable-value pair for this well
                for variable_name, variable_value in variables.items():
                    self.db_conn.execute(
                        """
                        INSERT INTO condition_variables (
                            condition_id, variable_name, variable_value
                        ) VALUES (?, ?, ?)
                        """,
                        (condition_id, variable_name, str(variable_value)),
                    )

            self.logger.info(
                "Successfully populated condition_variables table with %d variables",
                sum(
                    len(variable_conditions)
                    for variable_conditions in well_variables.values()
                ),
            )

        except duckdb.Error as e:
            raise DataError(
                f"Failed to populate condition_variables table: {str(e)}"
            ) from e

    # def display_well_conditions(self) -> None:
    #     """Display a rich table of well conditions."""
    #     # First check if we have any antibody columns
    #     check_antibody_query = """
    #     SELECT
    #         CASE WHEN COUNT(antibody) > 0 OR COUNT(antibody_1) > 0 OR
    #              COUNT(antibody_2) > 0 OR COUNT(antibody_3) > 0 THEN 1 ELSE 0 END as has_antibodies
    #     FROM conditions
    #     WHERE repeat_id = ?
    #     """
    #     result = self.db_conn.execute(
    #         check_antibody_query, (self.state.repeat_id,)
    #     ).fetchone()
    #     has_antibodies = bool(result[0]) if result else False

    #     # Query to get all condition information
    #     query = """
    #     SELECT c.well, c.cell_line, c.antibody, c.antibody_1, c.antibody_2, c.antibody_3,
    #            cv.variable_name, cv.variable_value
    #     FROM conditions c
    #     LEFT JOIN condition_variables cv ON c.condition_id = cv.condition_id
    #     WHERE c.repeat_id = ?
    #     ORDER BY c.well, cv.variable_name
    #     """

    #     results = self.db_conn.execute(
    #         query, (self.state.repeat_id,)
    #     ).fetchall()

    #     # Create a dictionary to organize the results
    #     well_data = {}
    #     for row in results:
    #         well = row[0]
    #         if well not in well_data:
    #             well_data[well] = {
    #                 "cell_line": row[1],
    #                 "antibody": row[2],
    #                 "antibody_1": row[3],
    #                 "antibody_2": row[4],
    #                 "antibody_3": row[5],
    #                 "variables": {},
    #             }
    #         if row[6]:  # variable_name
    #             well_data[well]["variables"][row[6]] = row[7]

    #     # Get all unique variable names for columns
    #     all_variables = set()
    #     for data in well_data.values():
    #         all_variables.update(data["variables"].keys())

    #     # Create the table
    #     table = Table(title="Well Conditions")
    #     table.add_column("Well", style="cyan")
    #     table.add_column("Cell Line", style="magenta")
    #     if has_antibodies:
    #         table.add_column("Antibodies", style="green")
    #     for var in sorted(all_variables):
    #         table.add_column(var, style="yellow")

    #     # Add rows
    #     for well, data in sorted(well_data.items()):
    #         row = [well, data["cell_line"]]

    #         if has_antibodies:
    #             antibodies = []
    #             if data["antibody"] and pd.notna(data["antibody"]):
    #                 antibodies.append(f"Ab: {data['antibody']}")
    #             if data["antibody_1"] and pd.notna(data["antibody_1"]):
    #                 antibodies.append(f"Ab1: {data['antibody_1']}")
    #             if data["antibody_2"] and pd.notna(data["antibody_2"]):
    #                 antibodies.append(f"Ab2: {data['antibody_2']}")
    #             if data["antibody_3"] and pd.notna(data["antibody_3"]):
    #                 antibodies.append(f"Ab3: {data['antibody_3']}")
    #             row.append("\n".join(antibodies) if antibodies else "None")

    #         # Add variable values in order
    #         row.extend(
    #             data["variables"].get(var, "None")
    #             for var in sorted(all_variables)
    #         )
    #         table.add_row(*row)

    #     self.console.print(table)


def import_conditions(db_conn: duckdb.DuckDBPyConnection) -> None:
    """Function that instantiates a ConditionManager and populates the conditions table.

    Args:
        db_conn: The DuckDB connection.
    """
    condition_manager = ConditionManager(db_conn)
    condition_id_map = condition_manager.define_per_well_conditions()
    condition_manager.populate_condition_variables(condition_id_map)
