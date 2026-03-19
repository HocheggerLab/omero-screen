"""Main entry point for the CellView application.

Dispatches CLI subcommands to their handler functions.
"""

import argparse

from cellview.db.clean_up import clean_up_db, del_measurements_by_plate_id
from cellview.db.db import CellViewDB
from cellview.db.display import (
    display_experiment,
    display_plate_summary,
    display_projects,
    display_single_project,
)
from cellview.db.edit import edit_experiment, edit_project
from cellview.exporters.db_to_pandas import export_pandas_df
from cellview.importers import import_data
from cellview.utils.error_classes import CellViewError
from cellview.utils.state import create_cellview_state

from .cli import get_parser, parse_args


def _handle_import(args: argparse.Namespace, db: CellViewDB) -> None:
    """Handle all import subcommands.

    Args:
        args: Parsed CLI arguments.
        db: CellView database instance.
    """
    if args.import_command == "csv":
        state_args = argparse.Namespace(csv=args.path, plate_id=None)
        state = create_cellview_state(state_args)
        import_data(db, state)

    elif args.import_command == "plate":
        if len(args.ids) > 1:
            state_args = argparse.Namespace(csv=None, plate_id=args.ids[0])
            temp_state = create_cellview_state(state_args)
            temp_state.validate_plates_same_screen(args.ids)
            temp_state.console.print(
                f"[bold cyan]Importing {len(args.ids)} plates...[/bold cyan]"
            )
            for pid in args.ids:
                temp_state.console.print(
                    f"\n[bold green]Importing plate {pid}...[/bold green]"
                )
                plate_args = argparse.Namespace(csv=None, plate_id=pid)
                state = create_cellview_state(plate_args)
                import_data(db, state)
        else:
            state_args = argparse.Namespace(csv=None, plate_id=args.ids[0])
            state = create_cellview_state(state_args)
            import_data(db, state)

    elif args.import_command == "screen":
        state_args = argparse.Namespace(csv=None, plate_id=None)
        temp_state = create_cellview_state(state_args)
        plate_ids = temp_state.get_plates_from_screen(args.id)
        temp_state.console.print(
            f"[bold cyan]Found {len(plate_ids)} plates "
            f"in screen {args.id}[/bold cyan]"
        )
        for plate_id in plate_ids:
            temp_state.console.print(
                f"\n[bold green]Importing plate {plate_id}...[/bold green]"
            )
            plate_args = argparse.Namespace(csv=None, plate_id=plate_id)
            state = create_cellview_state(plate_args)
            import_data(db, state)

    else:
        get_parser().parse_args(["import", "--help"])


def _handle_explore(args: argparse.Namespace) -> None:
    """Handle the explore subcommand.

    Args:
        args: Parsed CLI arguments.
    """
    from cellview.explore._cli import launch_explore, show_available_templates

    if args.list_templates:
        show_available_templates()
        return

    experiment: str | int | None = None
    if args.experiment:
        try:
            experiment = int(args.experiment)
        except ValueError:
            experiment = args.experiment

    launch_explore(
        plate_ids=args.plate_ids or None,
        experiment=experiment,
        template=args.template,
        fresh=args.fresh,
        no_napari=args.no_napari,
    )


def main() -> None:
    """Main entry point for CellView application.

    Dispatches to the appropriate handler based on the CLI subcommand.
    """
    conn = None
    try:
        args = parse_args()

        # Commands that don't need a DB connection
        if args.command == "explore":
            _handle_explore(args)
            return

        if args.command is None:
            get_parser().print_help()
            return

        # All remaining commands need a DB connection
        db = CellViewDB(args.db)
        conn = db.connect()

        match args.command:
            case "projects":
                display_projects(conn)

            case "project":
                display_single_project(conn, args.id)

            case "experiment":
                display_experiment(conn, args.id)

            case "plate":
                display_plate_summary(args.id, conn)

            case "import":
                _handle_import(args, db)

            case "edit":
                if args.edit_command == "project":
                    edit_project(args.id, conn)
                elif args.edit_command == "experiment":
                    edit_experiment(args.id, conn)
                else:
                    get_parser().parse_args(["edit", "--help"])

            case "export":
                df, variable_names = export_pandas_df(args.id, conn)
                print(
                    f"Exported plate {args.id}: "
                    f"{len(df)} rows, variables: {variable_names}"
                )

            case "delete":
                if args.delete_command == "plate":
                    del_measurements_by_plate_id(db, conn, args.id)
                    clean_up_db(db, conn)
                else:
                    get_parser().parse_args(["delete", "--help"])

            case "clean":
                clean_up_db(db, conn)

    except CellViewError as e:
        e.display()
        import sys

        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
