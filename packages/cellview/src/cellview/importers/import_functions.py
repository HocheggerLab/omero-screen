"""Module for importing data into CellView.

This module combined all parsers and importers into a single function.
Data are imported either via a path to a csv file or via a file that has been
attached to an omero plate
"""

from typing import Optional

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
from cellview.utils.state import CellViewState
from cellview.utils.ui import CellViewUI

ui = CellViewUI()


def import_data(
    db: CellViewDB,
    state: CellViewState,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> int:
    """Import data from CSV files into the database.

    Args:
        db: The CellView database.
        state: The CellView state.
        conn: The DuckDB connection.

    Returns:
        The exit code.
    """
    if conn is None:
        conn = db.connect()
    try:
        # Set the database connection in the state
        state.db_conn = conn

        select_or_create_project(conn)
        select_or_create_experiment(conn)
        create_new_repeat(conn)
        import_conditions(conn)
        import_measurements(conn)
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
