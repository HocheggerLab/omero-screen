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

        if args.csv:
            state = create_cellview_state(args)
            import_data(db, state)
        elif args.plate_id:
            # Handle multiple plate IDs
            if isinstance(args.plate_id, list) and len(args.plate_id) > 1:
                # Validate plates belong to same screen
                temp_state = create_cellview_state(args)
                temp_state.validate_plates_same_screen(args.plate_id)

                temp_state.console.print(
                    f"[bold cyan]Importing {len(args.plate_id)} plates...[/bold cyan]"
                )

                import argparse

                for pid in args.plate_id:
                    temp_state.console.print(
                        f"\n[bold green]Importing plate {pid}...[/bold green]"
                    )
                    # Create a new args object for this plate
                    new_args_dict = vars(args).copy()
                    new_args_dict["plate_id"] = pid
                    new_args = argparse.Namespace(**new_args_dict)

                    state = create_cellview_state(new_args)
                    import_data(db, state)
            else:
                # Single plate ID (either int or list of 1 int)
                if isinstance(args.plate_id, list):
                    args.plate_id = args.plate_id[0]

                state = create_cellview_state(args)
                import_data(db, state)
        elif args.screen_id:
            # Create a temporary state to fetch plates
            temp_state = create_cellview_state(args)
            plate_ids = temp_state.get_plates_from_screen(args.screen_id)

            temp_state.console.print(
                f"[bold cyan]Found {len(plate_ids)} plates in screen {args.screen_id}[/bold cyan]"
            )

            import argparse

            for plate_id in plate_ids:
                temp_state.console.print(
                    f"\n[bold green]Importing plate {plate_id}...[/bold green]"
                )
                # Create a new args object for this plate
                new_args_dict = vars(args).copy()
                new_args_dict["plate_id"] = plate_id
                new_args_dict["screen_id"] = None
                new_args = argparse.Namespace(**new_args_dict)

                state = create_cellview_state(new_args)
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
            # Handle multiple plate IDs
            if isinstance(args.plate_id, list) and len(args.plate_id) > 1:
                # Validate plates belong to same screen
                # We can use a temporary state for validation
                temp_state = CellViewState.get_instance(args)
                temp_state.validate_plates_same_screen(args.plate_id)

                print(f"Importing {len(args.plate_id)} plates...")

                import argparse

                for pid in args.plate_id:
                    print(f"\nImporting plate {pid}...")
                    # Create a new args object for this plate
                    new_args_dict = vars(args).copy()
                    new_args_dict["plate_id"] = pid
                    new_args = argparse.Namespace(**new_args_dict)

                    state = CellViewState.get_instance(new_args)
                    import_data(db, state)
            else:
                # Single plate ID (either int or list of 1 int)
                if isinstance(args.plate_id, list):
                    args.plate_id = args.plate_id[0]

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
