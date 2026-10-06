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


@cli.command("repair-tracks")
@click.argument("plate_id", metavar="ID", type=int)
@click.option(
    "--well",
    "wells",
    multiple=True,
    help="Restrict to this well (repeatable). Default: every well.",
)
@click.option(
    "--marker",
    default="Geminin",
    show_default=True,
    help="Channel whose nuclear signal is degraded at anaphase (PIP-FUCCI: geminin).",
)
@click.option(
    "--dry-run", is_flag=True, help="Report what would change; write nothing."
)
@click.option(
    "--events",
    "events_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write every repair decision (rule, frame, position) to this CSV.",
)
@click.pass_obj
def repair_tracks(
    obj: Context,
    plate_id: int,
    wells: tuple[str, ...],
    marker: str,
    dry_run: bool,
    events_path: Path | None,
) -> None:
    """Repair the tracked lineages of plate ID from segmentation flicker.

    Reads the tracker's immutable track_id_raw / parent_track_id_raw and
    writes the curated track_id / parent_track_id, so re-running gives the
    same result. Only plates with a mitotic marker (geminin) are repaired.
    """
    from cellview.tracks.plate import MarkerNotFoundError, repair_plate

    try:
        summaries = repair_plate(
            obj.conn,
            plate_id,
            marker,
            list(wells) or None,
            dry_run,
            events_path,
        )
    except (MarkerNotFoundError, ValueError) as err:
        raise click.ClickException(str(err)) from err
    verb = "Would repair" if dry_run else "Repaired"
    for s in summaries:
        click.echo(
            f"{verb} {s.well}: tracks {s.tracks_before} -> {s.tracks_after}, "
            f"divisions {s.divisions_before} -> {s.divisions_after}"
        )


def _parse_anchor(
    _ctx: Any, _param: Any, value: str | None
) -> list[int] | None:
    """Parse an anchor written ``FRAME:LABEL`` (e.g. ``72:524``)."""
    if value is None:
        return None
    try:
        frame, label = value.split(":")
        return [int(frame), int(label)]
    except ValueError as err:
        raise click.BadParameter("use FRAME:LABEL, e.g. 72:524") from err


@cli.group("curate", invoke_without_command=True, no_args_is_help=False)
@click.pass_context
def curate_group(ctx: click.Context) -> None:
    """Curate tracked lineages through a replayable edit log.

    Every correction is appended to a JSON-lines log and the curated lineage
    is rebuilt by replaying it on the automatic repair. Cells are named by an
    anchor FRAME:LABEL, the raw mask label of the cell in that frame.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


_LOG_ARG = click.argument(
    "log", type=click.Path(dir_okay=False, path_type=Path)
)
_PLATE_OPT = click.option(
    "--plate", "plate_id", type=int, required=True, help="OMERO plate id."
)
_MARKER_OPT = click.option(
    "--marker",
    default="Geminin",
    show_default=True,
    help="Mitotic marker channel for the repair.",
)


@curate_group.command("add")
@_LOG_ARG
@click.argument(
    "op",
    type=click.Choice(
        [
            "link",
            "unlink",
            "set_parent",
            "clear_parent",
            "swap",
            "event",
            "set_outcome",
            "exclude",
            "note",
        ]
    ),
)
@_PLATE_OPT
@click.option("--well", required=True, help="Well the edit applies to.")
@click.option(
    "--cell",
    callback=_parse_anchor,
    required=True,
    help="Anchor FRAME:LABEL of the cell.",
)
@click.option("--frame", type=int, help="Frame (link, unlink, swap, event).")
@click.option(
    "--label", type=int, help="Raw label the cell continues as (link)."
)
@click.option(
    "--parent",
    callback=_parse_anchor,
    help="Anchor of the parent (set_parent).",
)
@click.option(
    "--other", callback=_parse_anchor, help="Anchor of the other cell (swap)."
)
@click.option(
    "--kind",
    type=click.Choice(["mitosis", "death", "slippage"]),
    help="Event kind.",
)
@click.option("--outcome", help="Outcome (set_outcome).")
@click.option("--text", help="Note text, or exclusion reason.")
@click.option("--reason", default="", help="Why the edit is made.")
@click.option(
    "--author",
    type=click.Choice(["human", "agent"]),
    default="human",
    show_default=True,
)
@click.option(
    "--confirmed-by",
    type=click.Choice(["human"]),
    help="Who approved an agent's edit.",
)
@_MARKER_OPT
@click.pass_obj
def curate_add(
    obj: Context,
    log: Path,
    op: str,
    plate_id: int,
    well: str,
    cell: list[int],
    frame: int | None,
    label: int | None,
    parent: list[int] | None,
    other: list[int] | None,
    kind: str | None,
    outcome: str | None,
    text: str | None,
    reason: str,
    author: str,
    confirmed_by: str | None,
    marker: str,
) -> None:
    """Validate an edit against the plate and append it to LOG."""
    from cellview.tracks.edit import EditError, EditLog
    from cellview.tracks.plate import MarkerNotFoundError, well_bases

    args: dict[str, Any] = {"cell": cell}
    for key, value in (
        ("frame", frame),
        ("label", label),
        ("parent", parent),
        ("other", other),
        ("kind", kind),
        ("outcome", outcome),
        ("text" if op == "note" else "reason", text),
    ):
        if value is not None:
            args[key] = value
    try:
        base = well_bases(obj.conn, plate_id, marker, [well])[well][1]
        edit = EditLog(log).append(
            op,
            well,
            args,
            author=author,
            confirmed_by=confirmed_by,
            reason=reason,
            base=base,
        )
    except (EditError, MarkerNotFoundError, KeyError) as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"{edit.id} {edit.op} {well} {edit.args}")


@curate_group.command("undo")
@_LOG_ARG
@_PLATE_OPT
@click.option("--well", required=True, help="Well whose last edit to revert.")
@_MARKER_OPT
@click.pass_obj
def curate_undo(
    obj: Context, log: Path, plate_id: int, well: str, marker: str
) -> None:
    """Revert the last live edit for a well (appends a revert entry)."""
    from cellview.tracks.edit import EditError, EditLog
    from cellview.tracks.plate import well_bases

    try:
        base = well_bases(obj.conn, plate_id, marker, [well])[well][1]
        edit = EditLog(log).undo(well, base=base)
    except EditError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"{edit.id} reverts {edit.args['target']}")


@curate_group.command("replay")
@_LOG_ARG
@_PLATE_OPT
@click.option(
    "--well",
    "wells",
    multiple=True,
    help="Restrict to this well (repeatable).",
)
@click.option(
    "--write",
    is_flag=True,
    help="Write curated track_id / parent_track_id to CellView.",
)
@click.option(
    "--annotations",
    "annotations_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write per-track annotations (events, outcome, exclusion, notes) to this JSON file.",
)
@_MARKER_OPT
@click.pass_obj
def curate_replay(
    obj: Context,
    log: Path,
    plate_id: int,
    wells: tuple[str, ...],
    write: bool,
    annotations_path: Path | None,
    marker: str,
) -> None:
    """Rebuild curated lineages: automatic repair plus every edit in LOG."""
    import json

    from cellview.tracks.edit import EditError, EditLog
    from cellview.tracks.plate import MarkerNotFoundError, curate_plate

    try:
        curated = curate_plate(
            obj.conn,
            plate_id,
            EditLog(log),
            marker,
            list(wells) or None,
            write,
        )
    except (EditError, MarkerNotFoundError) as err:
        raise click.ClickException(str(err)) from err
    for well, cur in curated.items():
        n_tracks = len(set(cur.tracks.values()))
        n_div = len({p for p in cur.parents.values() if p})
        click.echo(
            f"{'Wrote' if write else 'Replayed'} {well}: {n_tracks} tracks, "
            f"{n_div} divisions, {len(cur.annotations)} annotated"
        )
    if annotations_path is not None:
        payload = {
            well: {str(tid): notes for tid, notes in cur.annotations.items()}
            for well, cur in curated.items()
        }
        annotations_path.write_text(json.dumps(payload, indent=2))


@curate_group.command("show")
@_LOG_ARG
@click.option("--well", help="Only this well.")
def curate_show(log: Path, well: str | None) -> None:
    """List the entries of LOG, marking reverted ones."""
    from cellview.tracks.edit import EditLog

    entries = EditLog(log).entries()
    reverted = {e.args.get("target") for e in entries if e.op == "revert"}
    for e in entries:
        if well and e.well != well:
            continue
        flag = " (reverted)" if e.id in reverted else ""
        who = e.author + (f"/{e.confirmed_by}" if e.confirmed_by else "")
        click.echo(
            f"{e.id} {e.time} {who:12} {e.well} {e.op} {e.args}{flag}  {e.reason}"
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
