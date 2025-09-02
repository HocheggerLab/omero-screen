"""Main entry point for the CellView application.

This function parses command-line arguments and performs actions such as:
- Importing data from CSV files or by plate ID
- Cleaning up the database
- Displaying plate summaries or project lists
- Deleting measurements by plate ID
It manages the database connection and delegates tasks based on the provided arguments.
"""

from cellview.db.clean_up import clean_up_db, del_measurements_by_plate_id
from cellview.db.db import CellViewDB
from cellview.db.display import (
    display_experiment,
    display_plate_summary,
    display_projects,
    display_single_project,
)
from cellview.db.edit import edit_experiment, edit_project
from cellview.importers import import_data
from cellview.utils.error_classes import CellViewError
from cellview.utils.state import CellViewState, create_cellview_state

from .cli import parse_args


def main_with_dependency_injection() -> None:
    """Main entry point for CellView application using dependency injection.

    This is the new, preferred way to run CellView that uses dependency injection
    instead of singleton pattern for better testability and thread safety.
    """
    conn = None
    try:
        args = parse_args()
        db = CellViewDB(args.db)
        conn = db.connect()

        if args.csv or args.plate_id:
            # Create state using dependency injection
            state = create_cellview_state(args)
            import_data(db, state)
        if args.clean:
            clean_up_db(db, conn)
        if args.plate:
            display_plate_summary(args.plate, conn)
        if args.projects:
            display_projects(conn)
        if args.project:
            display_single_project(conn, args.project)
        if args.experiment:
            display_experiment(conn, args.experiment)
        if args.edit_project:
            edit_project(args.edit_project, conn)
        if args.edit_experiment:
            edit_experiment(args.edit_experiment, conn)
        if args.delete_plate:
            del_measurements_by_plate_id(db, conn, args.delete_plate)
    except CellViewError as e:
        e.display()
        import sys

        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    """Main entry point for the CellView application (legacy singleton version).

    This function maintains the original singleton-based approach for backward
    compatibility. For new code, prefer main_with_dependency_injection().
    """
    conn = None
    try:
        args = parse_args()
        db = CellViewDB(args.db)
        conn = db.connect()
        if args.csv:
            state = CellViewState.get_instance(args)
            import_data(db, state)
        if args.plate_id:
            state = CellViewState.get_instance(args)
            import_data(db, state)
        if args.clean:
            clean_up_db(db, conn)
        if args.plate:
            display_plate_summary(args.plate, conn)
        if args.projects:
            display_projects(conn)
        if args.project:
            display_single_project(conn, args.project)
        if args.experiment:
            display_experiment(conn, args.experiment)
        if args.edit_project:
            edit_project(args.edit_project, conn)
        if args.edit_experiment:
            edit_experiment(args.edit_experiment, conn)
        if args.delete_plate:
            del_measurements_by_plate_id(db, conn, args.delete_plate)
    except CellViewError as e:
        e.display()
        import sys

        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main_with_dependency_injection()
