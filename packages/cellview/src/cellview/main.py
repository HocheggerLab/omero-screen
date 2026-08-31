"""Handlers and entry point for the CellView application.

The command surface itself lives in :mod:`cellview.cli`. This module holds the
handlers that do the work, taking explicit values rather than a parser
namespace, so they can be called and tested without going through Click.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import duckdb

from cellview.importers import import_data
from cellview.utils.error_classes import CellViewError, DBError
from cellview.utils.state import create_cellview_state

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cellview.cli import Context


def _resolve_import_target(
    project_id: int | None,
    experiment_id: int | None,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[int | None, int | None]:
    """Resolve ``--project`` / ``--experiment`` into validated IDs.

    Both are optional. Validating up front means a typo fails before any
    plate is touched, rather than part-way through a multi-plate import.

    Args:
        project_id: Project ID given on the command line, if any.
        experiment_id: Experiment ID given on the command line, if any.
        conn: Active DuckDB connection.

    Returns:
        ``(project_id, experiment_id)``, either of which may be None.
        An experiment given on its own resolves its parent project too,
        so the pair is always internally consistent.

    Raises:
        DBError: If an ID does not exist, or the experiment does not
            belong to the given project.
    """
    if project_id is not None:
        row = conn.execute(
            "SELECT project_id FROM projects WHERE project_id = ?",
            [project_id],
        ).fetchone()
        if row is None:
            raise DBError(
                f"Project {project_id} does not exist.",
                {"hint": "Run 'cellview projects' to list available projects"},
                show_traceback=False,
            )

    if experiment_id is not None:
        row = conn.execute(
            "SELECT project_id FROM experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if row is None:
            raise DBError(
                f"Experiment {experiment_id} does not exist.",
                {
                    "hint": "Run 'cellview project <id>' to list its experiments"
                },
                show_traceback=False,
            )
        parent_project_id = row[0]
        if project_id is not None and parent_project_id != project_id:
            raise DBError(
                f"Experiment {experiment_id} belongs to project "
                f"{parent_project_id}, not project {project_id}.",
                {"hint": "Drop --project, or pass the matching project ID"},
                show_traceback=False,
            )
        # An experiment implies its project, so --project is redundant.
        project_id = parent_project_id

    return project_id, experiment_id


def _state_args(
    *,
    csv: Path | None = None,
    plate_id: int | None = None,
    nucleus_channel: str | None,
    project_id: int | None,
    experiment_id: int | None,
) -> argparse.Namespace:
    """Build the Namespace consumed by ``create_cellview_state``.

    ``create_cellview_state`` takes a Namespace as its internal data-transfer
    object. That is a state-construction contract rather than a CLI parsing
    one, so it is unaffected by the move to Click and left as is.
    """
    return argparse.Namespace(
        csv=csv,
        plate_id=plate_id,
        nucleus_channel=nucleus_channel,
        project_id=project_id,
        experiment_id=experiment_id,
    )


def handle_import_csv(
    ctx: Context,
    *,
    path: Path,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import measurements from a CSV file."""
    project_id, experiment_id = _resolve_import_target(
        project, experiment, ctx.conn
    )
    state = create_cellview_state(
        _state_args(
            csv=path,
            nucleus_channel=nucleus_channel,
            project_id=project_id,
            experiment_id=experiment_id,
        )
    )
    import_data(ctx.db, state)


def handle_import_plate(
    ctx: Context,
    *,
    ids: list[int],
    interactive: bool,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import one or more plates by ID."""
    project_id, experiment_id = _resolve_import_target(
        project, experiment, ctx.conn
    )

    def args_for(plate_id: int) -> argparse.Namespace:
        return _state_args(
            plate_id=plate_id,
            nucleus_channel=nucleus_channel,
            project_id=project_id,
            experiment_id=experiment_id,
        )

    if len(ids) > 1:
        temp_state = create_cellview_state(args_for(ids[0]))
        temp_state.validate_plates_same_screen(ids)
        temp_state.console.print(
            f"[bold cyan]Importing {len(ids)} plates...[/bold cyan]"
        )
        for pid in ids:
            temp_state.console.print(
                f"\n[bold green]Importing plate {pid}...[/bold green]"
            )
            import_data(ctx.db, create_cellview_state(args_for(pid)))
    else:
        import_data(ctx.db, create_cellview_state(args_for(ids[0])))


def handle_import_screen(
    ctx: Context,
    *,
    screen_id: int,
    interactive: bool,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import every plate belonging to a screen."""
    project_id, experiment_id = _resolve_import_target(
        project, experiment, ctx.conn
    )

    def args_for(plate_id: int | None) -> argparse.Namespace:
        return _state_args(
            plate_id=plate_id,
            nucleus_channel=nucleus_channel,
            project_id=project_id,
            experiment_id=experiment_id,
        )

    temp_state = create_cellview_state(args_for(None))
    plate_ids = temp_state.get_plates_from_screen(screen_id)
    temp_state.console.print(
        f"[bold cyan]Found {len(plate_ids)} plates "
        f"in screen {screen_id}[/bold cyan]"
    )
    for plate_id in plate_ids:
        temp_state.console.print(
            f"\n[bold green]Importing plate {plate_id}...[/bold green]"
        )
        import_data(ctx.db, create_cellview_state(args_for(plate_id)))


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


def handle_explore(
    *,
    plate_ids: list[str],
    experiment: str | None,
    template: str,
    fresh: bool,
    no_napari: bool,
    code: bool,
    json_output: bool,
) -> None:
    """Launch or describe an exploration notebook."""
    from cellview.explore._cli import explore_json_command, launch_explore

    target: str | int | None = None
    if experiment:
        try:
            target = int(experiment)
        except ValueError:
            target = experiment

    ids = _parse_plate_ids(plate_ids) if plate_ids else None

    if json_output:
        explore_json_command(plate_ids=ids, experiment=target)
        return

    launch_explore(
        plate_ids=ids,
        experiment=target,
        template=template,
        fresh=fresh,
        no_napari=no_napari,
        code=code,
    )


def handle_template_list(conn: duckdb.DuckDBPyConnection) -> None:
    """List registered templates, syncing built-ins first."""
    from cellview.explore._template_registry import (
        list_templates_from_db,
        sync_filesystem_to_db,
    )
    from cellview.utils.ui import ui

    # Auto-sync built-in templates so a fresh DB always has something to show
    sync_filesystem_to_db(conn)
    templates = list_templates_from_db(conn)
    if not templates:
        ui.warning("No templates registered. Run: cellview template sync")
        return
    ui.header("Registered templates")
    for t in templates:
        marker = " [dim](file missing)[/dim]" if t.source == "db-only" else ""
        desc = f" — {t.description}" if t.description else ""
        ui.info(f"  {t.name:20s} [{t.fmt:7s}] [{t.source}]{desc}{marker}")


def handle_template_sync(conn: duckdb.DuckDBPyConnection) -> None:
    """Register every template discovered on the filesystem."""
    from cellview.explore._template_registry import sync_filesystem_to_db
    from cellview.utils.ui import ui

    n = sync_filesystem_to_db(conn)
    ui.success(f"Synced {n} template(s) to the database.")


def handle_template_add(
    conn: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    name: str | None,
    description: str | None,
) -> None:
    """Register a template file in the database."""
    from cellview.db.templates import upsert_template
    from cellview.explore._template_registry import (
        _extract_description,
        _fmt_from_path,
    )
    from cellview.utils.ui import ui

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        ui.error(f"File not found: {resolved}")
        sys.exit(1)
    template_name = name or resolved.stem
    fmt = _fmt_from_path(resolved)
    desc = description or _extract_description(resolved) or None
    upsert_template(
        conn, name=template_name, path=resolved, fmt=fmt, description=desc
    )
    ui.success(
        f"Registered template '{template_name}' ({fmt}) from {resolved}"
    )


def handle_template_remove(
    conn: duckdb.DuckDBPyConnection, *, name: str
) -> None:
    """Remove a template record, leaving the file on disk."""
    from cellview.db.templates import delete_template
    from cellview.utils.ui import ui

    if delete_template(conn, name):
        ui.success(f"Removed template '{name}' from the database.")
    else:
        ui.warning(f"Template '{name}' not found in the database.")


def handle_template_show(
    conn: duckdb.DuckDBPyConnection, *, name: str
) -> None:
    """Show the stored details for one template."""
    from cellview.db.templates import get_template_record
    from cellview.explore._template_registry import sync_filesystem_to_db
    from cellview.utils.ui import ui

    sync_filesystem_to_db(conn)
    rec = get_template_record(conn, name)
    if rec is None:
        ui.error(f"Template '{name}' not found.")
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


def main() -> None:
    """Entry point for the ``cellview`` console script."""
    from cellview.cli import cli
    from omero_screen.config import configure_logging

    # Configure logging once, at the entry point (file @ INFO, console off —
    # CellView's user output goes through its Rich UI). Tune with
    # $OMERO_SCREEN_LOG_LEVEL / $OMERO_SCREEN_LOG_FILE.
    configure_logging()

    try:
        cli.main(standalone_mode=False)
    except CellViewError as e:
        e.display()
        sys.exit(1)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
