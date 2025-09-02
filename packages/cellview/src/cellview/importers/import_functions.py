"""Module for importing data into CellView.

This module combined all parsers and importers into a single function.
Data are imported either via a path to a csv file or via a file that has been
attached to an omero plate
"""

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
from cellview.utils.error_classes import CellViewError
from cellview.utils.state import CellViewState, CellViewStateCore
from cellview.utils.ui import CellViewUI

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
    try:
        # Set the database connection in the state
        state.db_conn = conn

        # Always pass the state to each importer function for dependency injection
        # Convert singleton to CellViewStateCore if needed
        if isinstance(state, CellViewStateCore):
            state_for_importers = state
        else:
            # Convert singleton state to CellViewStateCore for dependency injection
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
    except Exception as e:
        clean_up_db(db, conn)
        if isinstance(e, CellViewError):
            e.display()
            return 1  # Return error code without re-raising
        raise e
    # finally:
    #     conn.close()
    return 0
