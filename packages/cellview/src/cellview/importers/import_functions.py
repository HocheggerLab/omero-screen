"""Module for importing data into CellView.

This module combined all parsers and importers into a single function.
Data are imported either via a path to a csv file or via a file that has been
attached to an omero plate
"""

import contextlib
from typing import Optional, Union

import duckdb

from cellview.db.clean_up import clean_up_db
from cellview.db.db import CellViewDB
from cellview.db.display import display_plate_summary
from cellview.importers.conditions import import_conditions
from cellview.importers.experiments import select_or_create_experiment
from cellview.importers.measurements import import_measurements
from cellview.importers.projects import select_or_create_project
from cellview.importers.repeats import create_new_repeat
from cellview.utils.state import CellViewState, CellViewStateCore
from cellview.utils.ui import CellViewUI
from omero_screen.config import get_logger

# Initialize logger with the module's name
logger = get_logger(__name__)
ui = CellViewUI()


def import_data(
    db: CellViewDB,
    state: Union[CellViewState, CellViewStateCore],
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> int:
    """Import data from CSV files into the database.

    Args:
        db: The CellView database.
        state: The CellView state (can be singleton or dependency-injectable version).
        conn: The DuckDB connection.

    Returns:
        The exit code.
    """
    if conn is None:
        conn = db.connect()

    # Start transaction for the entire import process
    # Start transaction for the entire import process
    with contextlib.suppress(Exception):
        conn.begin()

    try:
        # Set the database connection in the state
        state.db_conn = conn

        # Always pass the state to each importer function for dependency injection
        # ... logic ...
        if isinstance(state, CellViewStateCore):
            state_for_importers = state
        else:
            # ... logic ...
            state_for_importers = CellViewStateCore()
            # Copy all attributes from singleton to the new state
            for attr in [
                "csv_path",
                "df",
                "plate_id",
                "project_name",
                "experiment_name",
                "project_id",
                "experiment_id",
                "repeat_id",
                "condition_id_map",
                "lab_member",
                "date",
                "channel_0",
                "channel_1",
                "channel_2",
                "channel_3",
                "db_conn",
            ]:
                if hasattr(state, attr):
                    setattr(state_for_importers, attr, getattr(state, attr))

        select_or_create_project(conn, state_for_importers)
        select_or_create_experiment(conn, state_for_importers)
        create_new_repeat(conn, state_for_importers)
        import_conditions(conn, state_for_importers)
        import_measurements(conn, state_for_importers)
        assert state.plate_id is not None

        display_plate_summary(state.plate_id, conn)

        # Commit if everything worked
        # Commit if everything worked
        with contextlib.suppress(Exception):
            conn.commit()

    except Exception as e:
        # Improved error logging
        import traceback

        logger.error(f"Import failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if hasattr(
            e, "context"
        ):  # Check if it's our wrapper DBError/DataError
            logger.error(f"Error Context: {e.context}")

        # Rollback on failure
        try:
            conn.rollback()
        except Exception as rollback_err:
            logger.warning(f"Rollback failed: {rollback_err}")

        try:
            clean_up_db(db, conn)
        except Exception as cleanup_err:
            logger.error(f"Cleanup also failed: {str(cleanup_err)}")
        # Always re-raise to allow the caller (CLI or GUI) to handle reporting
        raise e
    return 0
