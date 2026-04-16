"""Main entry point for the CellView application.

Dispatches CLI subcommands to their handler functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

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


def _parse_plate_ids(raw: list[str]) -> list[int]:
    """Parse plate IDs from CLI arguments.

    Accepts plain integers (``3602 3603``) or a notebook-style name
    (``plates_3602_3603_3604`` / ``plate_3602``).  A single string
    argument that starts with ``plate`` is treated as a notebook name
    and its embedded integers are extracted.

    Args:
        raw: Raw string arguments from the CLI.

    Returns:
        Sorted list of plate IDs.

    Raises:
        SystemExit: If parsing fails.
    """
    import sys

    from cellview.utils.ui import ui

    # Single argument that looks like a notebook name
    if len(raw) == 1 and not raw[0].isdigit():
        name = raw[0].removesuffix(".ipynb").removeprefix("explore_")
        parts = name.split("_")
        ids = [int(p) for p in parts if p.isdigit()]
        if not ids:
            ui.error(f"Cannot extract plate IDs from '{raw[0]}'")
            sys.exit(1)
        return sorted(ids)

    # Plain integers
    try:
        return sorted(int(x) for x in raw)
    except ValueError:
        ui.error(f"Invalid plate ID(s): {raw}")
        sys.exit(1)


def _handle_explore(args: argparse.Namespace) -> None:
    """Handle the explore subcommand.

    Args:
        args: Parsed CLI arguments.
    """
    from cellview.explore._cli import (
        explore_json_command,
        launch_explore,
        show_available_templates,
    )

    if args.list_templates:
        show_available_templates()
        return

    experiment: str | int | None = None
    if args.experiment:
        try:
            experiment = int(args.experiment)
        except ValueError:
            experiment = args.experiment

    plate_ids = _parse_plate_ids(args.plate_ids) if args.plate_ids else None

    if getattr(args, "json", False):
        explore_json_command(plate_ids=plate_ids, experiment=experiment)
        return

    launch_explore(
        plate_ids=plate_ids,
        experiment=experiment,
        template=args.template,
        fresh=args.fresh,
        no_napari=args.no_napari,
        code=args.code,
    )


def _handle_template(
    args: argparse.Namespace, conn: duckdb.DuckDBPyConnection
) -> None:
    """Handle the template subcommand.

    Args:
        args: Parsed CLI arguments.
        conn: Active DuckDB connection.
    """
    import sys

    from cellview.db.templates import (
        delete_template,
        get_template_record,
        upsert_template,
    )
    from cellview.explore._template_registry import (
        _extract_description,
        _fmt_from_path,
        list_templates_from_db,
        sync_filesystem_to_db,
    )
    from cellview.utils.ui import ui

    sub = getattr(args, "template_command", None)

    if sub == "list" or sub is None:
        # Auto-sync built-in templates so a fresh DB always has something to show
        sync_filesystem_to_db(conn)
        templates = list_templates_from_db(conn)
        if not templates:
            ui.warning("No templates registered. Run: cellview template sync")
            return
        ui.header("Registered templates")
        for t in templates:
            marker = (
                " [dim](file missing)[/dim]" if t.source == "db-only" else ""
            )
            desc = f" — {t.description}" if t.description else ""
            ui.info(f"  {t.name:20s} [{t.fmt:7s}] [{t.source}]{desc}{marker}")
        return

    if sub == "sync":
        n = sync_filesystem_to_db(conn)
        ui.success(f"Synced {n} template(s) to the database.")
        return

    if sub == "add":
        path = args.path.expanduser().resolve()
        if not path.exists():
            ui.error(f"File not found: {path}")
            sys.exit(1)
        name = args.name or path.stem
        fmt = _fmt_from_path(path)
        description = args.description or _extract_description(path) or None
        upsert_template(
            conn, name=name, path=path, fmt=fmt, description=description
        )
        ui.success(f"Registered template '{name}' ({fmt}) from {path}")
        return

    if sub == "remove":
        removed = delete_template(conn, args.name)
        if removed:
            ui.success(f"Removed template '{args.name}' from the database.")
        else:
            ui.warning(f"Template '{args.name}' not found in the database.")
        return

    if sub == "show":
        sync_filesystem_to_db(conn)
        rec = get_template_record(conn, args.name)
        if rec is None:
            ui.error(f"Template '{args.name}' not found.")
            sys.exit(1)
        p = Path(rec.path)
        exists_tag = "exists" if p.exists() else "MISSING"
        ui.header(f"Template: {rec.name}")
        ui.info(f"  Format      : {rec.format}")
        ui.info(f"  Path        : {rec.path} [{exists_tag}]")
        ui.info(f"  Description : {rec.description or '(none)'}")
        ui.info(f"  DB id       : {rec.template_id}")
        if rec.parent_template_id:
            ui.info(f"  Derived from: template_id {rec.parent_template_id}")
        return

    get_parser().parse_args(["template", "--help"])


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

            case "template":
                _handle_template(args, conn)

    except CellViewError as e:
        e.display()
        import sys

        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
