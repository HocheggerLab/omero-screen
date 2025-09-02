"""Module for importing repeat data into CellView.

This module provides a class for managing repeat selection and creation operations.
"""

import io
from typing import Optional, cast

import duckdb
from rich.console import Console

from cellview.utils.error_classes import DataError, DBError, StateError
from cellview.utils.state import CellViewState, CellViewStateCore
from omero_screen.config import get_logger

# Initialize logger with the module's name
logger = get_logger(__name__)


class RepeatsManager:
    """Manages repeat selection and creation operations.

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
        """Initialize the RepeatsManager.

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

    def _fetch_existing_repeats(
        self,
    ) -> list[tuple[int, int, int, str, str, str, str, str, str]]:
        """Fetch all existing repeats for the current experiment from the database."""
        if not self.state.experiment_id:
            raise StateError(
                "No experiment selected",
                context={"current_state": self.state.__dict__},
            )

        try:
            result = self.db_conn.execute(
                """
                SELECT repeat_id, experiment_id, plate_id, date, lab_member, channel_0, channel_1, channel_2, channel_3
                FROM repeats
                WHERE experiment_id = ?
                ORDER BY repeat_id
                """,
                [self.state.experiment_id],
            ).fetchall()
            return cast(
                list[tuple[int, int, int, str, str, str, str, str, str]],
                result,
            )
        except duckdb.Error as err:
            raise DBError(
                "Failed to fetch repeats from database",
                context={
                    "experiment_id": self.state.experiment_id,
                    "error": str(err),
                },
            ) from err

    def _check_plate_duplicate(
        self,
        repeats: list[tuple[int, int, int, str, str, str, str, str, str]],
    ) -> None:
        """Check if the current plate_id already exists in the database and raise an error if it does.

        Args:
            repeats: A list of tuples containing the repeat ID, experiment ID, plate ID, date, lab member, channel 0, channel 1, channel 2, and channel 3.
        """
        for (
            _,
            _,
            plate_id,
            *_,
        ) in repeats:  # Skip repeat_id and experiment_id to get plate_id
            if plate_id == self.state.plate_id:
                raise DataError(
                    "Duplicate Plate ID Found!",
                    context={
                        "plate_id": self.state.plate_id,
                        "message": "This plate ID already exists in the database for this experiment.",
                        "help": "Please check the table above for existing entries and try again with a different plate ID.",
                    },
                )

    def _fetch_experiment_name(self) -> str:
        """Fetch the name of the current experiment.

        Returns:
            The name of the experiment.
        """
        if not self.state.experiment_id:
            raise StateError(
                "No experiment selected",
                context={"current_state": self.state.__dict__},
            )

        try:
            result = self.db_conn.execute(
                "SELECT experiment_name FROM experiments WHERE experiment_id = ?",
                [self.state.experiment_id],
            ).fetchone()

            if result is None:
                raise DBError(
                    "Experiment not found",
                    context={
                        "experiment_id": self.state.experiment_id,
                    },
                )
            return cast(str, result[0])
        except duckdb.Error as err:
            raise DBError(
                "Failed to fetch experiment name",
                context={
                    "experiment_id": self.state.experiment_id,
                    "error": str(err),
                },
            ) from err

    def _create_new_repeat(self) -> int:
        """Create a new repeat and return its ID.

        Returns:
            The ID of the new repeat.
        """
        if not self.state.experiment_id:
            raise StateError(
                "No experiment selected",
                context={"current_state": self.state.__dict__},
            )

        try:
            self.db_conn.execute(
                """
                INSERT INTO repeats (experiment_id, plate_id, date, lab_member, channel_0, channel_1, channel_2, channel_3)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.state.experiment_id,
                    self.state.plate_id,
                    self.state.date,
                    self.state.lab_member,
                    self.state.channel_0,
                    self.state.channel_1,
                    self.state.channel_2,
                    self.state.channel_3,
                ),
            )
            result = self.db_conn.execute(
                "SELECT currval('repeat_id_seq')"
            ).fetchone()
            if result is None:
                raise DBError(
                    "Failed to get ID of newly created repeat",
                    context={
                        "plate_id": self.state.plate_id,
                        "experiment_id": self.state.experiment_id,
                    },
                )
            return cast(int, result[0])
        except duckdb.Error as err:
            raise DBError(
                "Failed to create new repeat",
                context={
                    "plate_id": self.state.plate_id,
                    "experiment_id": self.state.experiment_id,
                    "error": str(err),
                },
            ) from err

    def create_new_repeat(self) -> None:
        """Create a new repeat and return its ID.

        Raises:
            DataError: If the plate ID is not provided.
        """
        if not self.state.plate_id:
            if self.state.df is not None:
                buf = io.StringIO()
                self.state.df.info(buf=buf)
                df_info = buf.getvalue()
            else:
                df_info = "No DataFrame loaded"

            raise DataError(
                "No plate ID provided in CSV file",
                context={
                    "current_state": self.state.__dict__,
                    "dataframe_info": df_info,
                },
            )

        self._check_plate_duplicate(self._fetch_existing_repeats())
        repeat_id = self._create_new_repeat()
        self.state.repeat_id = repeat_id


def create_new_repeat(
    db_conn: duckdb.DuckDBPyConnection,
    state: Optional[CellViewStateCore] = None,
) -> None:
    """Function that creates a RepeatsManager instance and calls its main method.

    Args:
        db_conn: The DuckDB connection.
        state: The CellView state instance (optional, falls back to singleton if not provided).
    """
    manager = RepeatsManager(db_conn, state)
    manager.create_new_repeat()
