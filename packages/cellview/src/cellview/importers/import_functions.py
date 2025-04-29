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


def import_from_csv(db: CellViewDB, state: CellViewState) -> int:
    """Import data from CSV files into the database."""
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
    finally:
        conn.close()
    return 0
