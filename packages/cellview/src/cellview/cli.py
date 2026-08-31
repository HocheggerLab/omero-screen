"""Click command-line interface for CellView.

The exported :data:`cli` group is the single source of truth for the command
surface: Great Docs renders it as CLI reference and ``CliRunner`` drives it in
tests. Command callbacks stay thin — they parse, then delegate to handlers in
:mod:`cellview.main`, which are imported lazily so that ``--help`` and
documentation discovery do not pull in DuckDB, pandas or the OMERO stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

    from cellview.db.db import CellViewDB


# Shared help text, kept in one place so the three import routes stay in step.
NUCLEUS_HELP = (
    "Name of the nucleus (DNA-segmentation) channel as it appears in the "
    "input data, e.g. 'DAPI', 'Hoechst', 'H2B_RFP'. "
    "Plate/screen routes default to the plate's channel annotation. "
    "CSV route prompts interactively when omitted. Use this flag to "
    "override the default or to run non-interactively."
)
PROJECT_HELP = (
    "Existing project ID to import into. Skips the interactive project "
    "prompt — useful when importing several plates in one go."
)
EXPERIMENT_HELP = (
    "Existing experiment ID to import into. Skips the interactive "
    "experiment prompt. Implies its parent project, so --project is "
    "optional; when both are given they must agree."
)


class Context:
    """Carries the ``--db`` choice and opens the database on first use.

    ``explore`` runs without a database, so connecting eagerly in the group
    callback would create or migrate a database file for a command that never
    touches it. The connection is opened on demand and closed by Click when
    the command finishes.
    """

    def __init__(self, db_path: Path | None) -> None:
        """Record the database path without opening anything yet."""
        self.db_path = db_path
        self._db: CellViewDB | None = None
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def db(self) -> CellViewDB:
        """The CellView database handle, created on first access."""
        if self._db is None:
            from cellview.db.db import CellViewDB

            self._db = CellViewDB(self.db_path)
        return self._db

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """An open DuckDB connection, created on first access."""
        if self._conn is None:
            self._conn = self.db.connect()
        return self._conn

    def close(self) -> None:
        """Close the connection if one was ever opened."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _add_target_options(func: Any) -> Any:
    """Attach the shared --project / --experiment options to a command."""
    func = click.option(
        "--experiment",
        type=int,
        default=None,
        metavar="EXPERIMENT_ID",
        help=EXPERIMENT_HELP,
    )(func)
    func = click.option(
        "--project",
        type=int,
        default=None,
        metavar="PROJECT_ID",
        help=PROJECT_HELP,
    )(func)
    return func


def _nucleus_option(func: Any) -> Any:
    """Attach the shared --nucleus-channel option to a command."""
    return click.option(
        "--nucleus-channel",
        type=str,
        default=None,
        metavar="CH",
        help=NUCLEUS_HELP,
    )(func)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=False,
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to the DuckDB database file. "
        "Defaults to ~/.cellview/cellview.duckdb"
    ),
)
@click.version_option(package_name="cellview")
@click.pass_context
def cli(ctx: click.Context, db: Path | None) -> None:
    """CellView: manage and explore single-cell measurement data.

    Import measurements produced by the OMERO-Screen pipeline into a local
    DuckDB database, inspect projects, experiments and plates, and launch
    notebooks for interactive analysis.
    """
    ctx.obj = Context(db_path=db)
    ctx.call_on_close(ctx.obj.close)
    # argparse printed help and exited 0 for a bare `cellview`; Click 8.2
    # would raise a usage error (exit 2) instead. Preserve the old contract.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------


@cli.command()
@click.pass_obj
def projects(obj: Context) -> None:
    """List all projects with experiment counts."""
    from cellview.db.display import display_projects

    display_projects(obj.conn)


@cli.command()
@click.argument("project_id", metavar="ID", type=int)
@click.pass_obj
def project(obj: Context, project_id: int) -> None:
    """Show experiments and plates for project ID."""
    from cellview.db.display import display_single_project

    display_single_project(obj.conn, project_id)


@cli.command()
@click.argument("experiment_id", metavar="ID", type=int)
@click.pass_obj
def experiment(obj: Context, experiment_id: int) -> None:
    """Show plates, channels and variables for experiment ID."""
    from cellview.db.display import display_experiment

    display_experiment(obj.conn, experiment_id)


@cli.command()
@click.argument("plate_id", metavar="ID", type=int)
@click.pass_obj
def plate(obj: Context, plate_id: int) -> None:
    """Show summary, conditions and measurements for plate ID."""
    from cellview.db.display import display_plate_summary

    display_plate_summary(plate_id, obj.conn)


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


@cli.group("import", invoke_without_command=True, no_args_is_help=False)
@click.pass_context
def import_group(ctx: click.Context) -> None:
    """Import data from a CSV file, an OMERO plate, or a screen."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@import_group.command("csv")
@click.argument("path", type=click.Path(path_type=Path))
@_nucleus_option
@_add_target_options
@click.pass_obj
def import_csv(
    obj: Context,
    path: Path,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import measurements from the CSV file at PATH."""
    from cellview.main import handle_import_csv

    handle_import_csv(
        obj,
        path=path,
        nucleus_channel=nucleus_channel,
        project=project,
        experiment=experiment,
    )


@import_group.command("plate")
@click.argument("ids", type=int, nargs=-1, required=True, metavar="IDS...")
@click.option(
    "--interactive",
    is_flag=True,
    help="Force interactive project/experiment selection.",
)
@_nucleus_option
@_add_target_options
@click.pass_obj
def import_plate(
    obj: Context,
    ids: tuple[int, ...],
    interactive: bool,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import one or more plates by ID.

    Several plates given at once must belong to the same screen; this is
    checked before anything is written.
    """
    from cellview.main import handle_import_plate

    handle_import_plate(
        obj,
        ids=list(ids),
        interactive=interactive,
        nucleus_channel=nucleus_channel,
        project=project,
        experiment=experiment,
    )


@import_group.command("screen")
@click.argument("screen_id", metavar="ID", type=int)
@click.option(
    "--interactive",
    is_flag=True,
    help="Force interactive project/experiment selection.",
)
@_nucleus_option
@_add_target_options
@click.pass_obj
def import_screen(
    obj: Context,
    screen_id: int,
    interactive: bool,
    nucleus_channel: str | None,
    project: int | None,
    experiment: int | None,
) -> None:
    """Import every plate belonging to screen ID."""
    from cellview.main import handle_import_screen

    handle_import_screen(
        obj,
        screen_id=screen_id,
        interactive=interactive,
        nucleus_channel=nucleus_channel,
        project=project,
        experiment=experiment,
    )


# --------------------------------------------------------------------------
# Edit
# --------------------------------------------------------------------------


@cli.group("edit", invoke_without_command=True, no_args_is_help=False)
@click.pass_context
def edit_group(ctx: click.Context) -> None:
    """Edit project or experiment metadata."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@edit_group.command("project")
@click.argument("project_id", metavar="ID", type=int)
@click.pass_obj
def edit_project_command(obj: Context, project_id: int) -> None:
    """Edit a project's name and description."""
    from cellview.db.edit import edit_project

    edit_project(project_id, obj.conn)


@edit_group.command("experiment")
@click.argument("experiment_id", metavar="ID", type=int)
@click.pass_obj
def edit_experiment_command(obj: Context, experiment_id: int) -> None:
    """Edit an experiment's name and description."""
    from cellview.db.edit import edit_experiment

    edit_experiment(experiment_id, obj.conn)


# --------------------------------------------------------------------------
# Export, delete, clean
# --------------------------------------------------------------------------


@cli.command()
@click.argument("plate_id", metavar="ID", type=int)
@click.pass_obj
def export(obj: Context, plate_id: int) -> None:
    """Export the measurements for plate ID."""
    from cellview.exporters.db_to_pandas import export_pandas_df

    df, variable_names = export_pandas_df(plate_id, obj.conn)
    click.echo(
        f"Exported plate {plate_id}: {len(df)} rows, "
        f"variables: {variable_names}"
    )


@cli.group("delete", invoke_without_command=True, no_args_is_help=False)
@click.pass_context
def delete_group(ctx: click.Context) -> None:
    """Delete data from the database."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@delete_group.command("plate")
@click.argument("ids", type=int, nargs=-1, required=True, metavar="ID...")
@click.pass_obj
def delete_plate(obj: Context, ids: tuple[int, ...]) -> None:
    """Delete one or more plates and all their associated data.

    Plates are deleted in the order given, then orphaned records are cleaned
    up in a single pass.
    """
    from cellview.db.clean_up import clean_up_db, del_measurements_by_plate_id

    for plate_id in ids:
        del_measurements_by_plate_id(obj.db, obj.conn, plate_id)
    # One pass at the end: cleanup is global (it walks the whole
    # project->measurement chain) and prints a results table, so per-plate
    # calls would repeat work and noise.
    clean_up_db(obj.db, obj.conn)


@cli.command()
@click.pass_obj
def clean(obj: Context) -> None:
    """Clean up orphaned records in the database."""
    from cellview.db.clean_up import clean_up_db

    clean_up_db(obj.db, obj.conn)


# --------------------------------------------------------------------------
# Explore - the only command that needs no database connection
# --------------------------------------------------------------------------


@cli.command()
@click.argument("plate_ids", nargs=-1, metavar="[PLATE_IDS]...")
@click.option(
    "--experiment",
    type=str,
    default=None,
    metavar="EXPERIMENT",
    help="Explore all plates from an experiment (name or ID).",
)
@click.option(
    "--template",
    type=str,
    default="cellcycle",
    show_default=True,
    metavar="NAME",
    help="Template notebook to use.",
)
@click.option(
    "--fresh",
    is_flag=True,
    help="Regenerate the notebook even if it already exists.",
)
@click.option(
    "--no-napari",
    "no_napari",
    is_flag=True,
    help="Skip launching napari.",
)
@click.option(
    "--code",
    is_flag=True,
    help="Open the notebook folder in VS Code instead of JupyterLab.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help=(
        "Print a JSON context snapshot (schema, conditions, stats, "
        "notebooks) to stdout and exit. Used by the agentic skill."
    ),
)
def explore(
    plate_ids: tuple[str, ...],
    experiment: str | None,
    template: str,
    fresh: bool,
    no_napari: bool,
    code: bool,
    json_output: bool,
) -> None:
    """Launch a Jupyter notebook for interactive data exploration.

    PLATE_IDS are plate IDs, or a notebook name such as plates_3602_3603
    from which the IDs are extracted.
    """
    from cellview.main import handle_explore

    handle_explore(
        plate_ids=list(plate_ids),
        experiment=experiment,
        template=template,
        fresh=fresh,
        no_napari=no_napari,
        code=code,
        json_output=json_output,
    )


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


@cli.group("template", invoke_without_command=True)
@click.pass_context
def template_group(ctx: click.Context) -> None:
    """Manage analysis notebook templates.

    With no subcommand this lists the registered templates, matching the
    behaviour of 'cellview template list'.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(template_list)


@template_group.command("list")
@click.pass_obj
def template_list(obj: Context) -> None:
    """List all registered templates."""
    from cellview.main import handle_template_list

    handle_template_list(obj.conn)


@template_group.command("add")
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "--name",
    type=str,
    default=None,
    help="Override the template name (default: filename stem).",
)
@click.option(
    "--description",
    type=str,
    default=None,
    help="Short description shown in listings.",
)
@click.pass_obj
def template_add(
    obj: Context, path: Path, name: str | None, description: str | None
) -> None:
    """Register the template file at PATH in the database."""
    from cellview.main import handle_template_add

    handle_template_add(
        obj.conn, path=path, name=name, description=description
    )


@template_group.command("remove")
@click.argument("name", type=str)
@click.pass_obj
def template_remove(obj: Context, name: str) -> None:
    """Remove template NAME from the database.

    The template file itself is left on disk.
    """
    from cellview.main import handle_template_remove

    handle_template_remove(obj.conn, name=name)


@template_group.command("show")
@click.argument("name", type=str)
@click.pass_obj
def template_show(obj: Context, name: str) -> None:
    """Show details for template NAME."""
    from cellview.main import handle_template_show

    handle_template_show(obj.conn, name=name)


@template_group.command("sync")
@click.pass_obj
def template_sync(obj: Context) -> None:
    """Scan the filesystem and register all discovered templates."""
    from cellview.main import handle_template_sync

    handle_template_sync(obj.conn)
