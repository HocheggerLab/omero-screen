"""Click command-line interface for the Omero Screen training-data database.

This module provides ``omero-train``:

- Migrating existing training data to the database
- Listing and managing classifiers
- Exporting training data
- Viewing statistics

The exported :data:`cli` group is what Great Docs renders as CLI reference and
what ``CliRunner`` drives in tests, so it must stay importable without pulling
in napari, Qt or Torch. Database and pandas imports therefore live inside the
command callbacks, not at module scope.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .database import TrainingDB

console = Console()


class VariadicOptionCommand(click.Command):
    """Preserve argparse-style ``--option VALUE [VALUE ...]`` options.

    argparse's ``nargs="+"`` accepts several values after a single flag;
    Click's ``multiple=True`` expects the flag to be repeated. Existing shell
    scripts use the argparse spelling, so rewrite it into the Click form
    before parsing and accept both.

    A near-identical copy lives in ``cellclass.cli``. The duplication is
    deliberate: ``omero_utils`` — the obvious shared home — calls
    ``set_env_vars()`` at import time, which would drag OMERO configuration
    into every ``--help`` invocation and break the lightweight-import rule.
    """

    def __init__(
        self,
        *args: Any,
        variadic_options: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialise a command with options that consume values until the next flag."""
        self._variadic_options = frozenset(variadic_options)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Expand variadic values into Click's repeatable-option representation."""
        expanded: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token not in self._variadic_options:
                expanded.append(token)
                index += 1
                continue

            index += 1
            if index == len(args) or args[index].startswith("-"):
                expanded.append(token)
                continue
            while index < len(args) and not args[index].startswith("-"):
                expanded.extend((token, args[index]))
                index += 1
        return super().parse_args(ctx, expanded)


def resolve_classifier(db: TrainingDB, identifier: str) -> dict[str, Any]:
    """Resolve a classifier by numeric ID, falling back to its name.

    Args:
        db: An open :class:`TrainingDB` handle.
        identifier: Either a numeric classifier ID or a classifier name.

    Returns:
        The classifier record.

    Raises:
        SystemExit: If no classifier matches, after printing an error.
    """
    if identifier.isdigit():
        clf = db.get_classifier_by_id(int(identifier))
        if clf:
            return clf

    clf = db.get_classifier(identifier)
    if clf:
        return clf

    console.print(f"[red]Classifier '{identifier}' not found.[/red]")
    sys.exit(1)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(package_name="omero-screen-napari")
def cli() -> None:
    """Manage the Omero Screen training-data database.

    Migrate existing '.npy' training-data files into the database, list
    available classifiers, inspect their statistics, and export annotations
    for analysis.

    Classifiers can be referred to by either their ID (e.g. 1) or their
    name (e.g. 'Experiment_1').
    """


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Simulate the migration without writing to the database.",
)
@click.option(
    "--path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Base directory containing classifier folders "
        "(default: ~/omeroscreen_trainingdata)."
    ),
)
def migrate(dry_run: bool, path: Path | None) -> None:
    """Migrate existing .npy files into the database.

    Scans the given directory for training-data folders and imports them.
    """
    handle_migrate(path=path, dry_run=dry_run)


@cli.command("list")
def list_command() -> None:
    """List all classifiers.

    Shows a table of every classifier with its ID, name and summary
    statistics.
    """
    handle_list()


@cli.command()
@click.argument("classifier")
def stats(classifier: str) -> None:
    """Show statistics for CLASSIFIER.

    CLASSIFIER is an ID or a name, e.g. 1 or 'MyClassifier'. Prints
    general statistics, the class distribution, and a per-image breakdown.
    """
    handle_stats(classifier=classifier)


@cli.command()
@click.argument("classifier")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "json", "parquet"]),
    default="csv",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: {name}_export.{format}).",
)
@click.option("--plate", type=int, default=None, help="Filter by plate ID.")
@click.option("--well", default=None, help="Filter by well.")
def export(
    classifier: str,
    output_format: str,
    output: Path | None,
    plate: int | None,
    well: str | None,
) -> None:
    """Export training data for CLASSIFIER to a file.

    CLASSIFIER is an ID or a name. Annotations are written as CSV, JSON or
    Parquet, optionally filtered by plate and well.
    """
    handle_export(
        classifier=classifier,
        output_format=output_format,
        output=output,
        plate=plate,
        well=well,
    )


@cli.command(cls=VariadicOptionCommand, variadic_options=("--plate",))
@click.argument("identifiers", nargs=-1, required=True)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--plate",
    type=int,
    multiple=True,
    help=(
        "Delete only sessions for the given plate ID(s). Accepts one or more "
        "values. If omitted, the entire classifier is deleted."
    ),
)
def delete(
    identifiers: tuple[str, ...], yes: bool, plate: tuple[int, ...]
) -> None:
    """Delete one or more classifiers.

    IDENTIFIERS are IDs or names, e.g. 1 2 'MyClassifier'. Deletes the
    classifier along with its classes, sessions and annotations, and removes
    the associated .npy files and folder from disk.
    """
    handle_delete(
        identifiers=list(identifiers), yes=yes, plate=list(plate) or None
    )


def handle_migrate(path: Path | None, dry_run: bool) -> None:
    """Run the training-data migration and report the outcome."""
    from .migrator import migrate_all_classifiers

    console.print(Panel("Starting Migration", style="bold blue"))

    migrate_all_classifiers(base_dir=path, dry_run=dry_run)

    if dry_run:
        console.print(
            "[yellow]This was a DRY RUN. No changes were made.[/yellow]"
        )


def handle_list() -> None:
    """Print a table of every classifier with its summary statistics."""
    from .database import TrainingDB

    db = TrainingDB()
    classifiers = db.list_classifiers()

    if not classifiers:
        console.print("[yellow]No classifiers found.[/yellow]")
        return

    table = Table(title="Training Data Classifiers")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Created At", style="dim")
    table.add_column("Classes", style="yellow")
    table.add_column("Sessions", justify="right")
    table.add_column("Annotations", justify="right")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching stats...", total=None)

        for clf in classifiers:
            name = clf["name"]
            try:
                n_sessions = db.get_session_count(name)
                n_annotations = db.get_total_annotations(name)
            except Exception:
                n_sessions = 0
                n_annotations = 0

            classes_str = clf.get("class_labels") or ""

            table.add_row(
                str(clf["id"]),
                name,
                clf["description"] or "",
                str(clf["created_at"]),
                classes_str,
                str(n_sessions),
                str(n_annotations),
            )

    console.print(table)


def handle_stats(classifier: str) -> None:
    """Print general statistics, class distribution and per-image breakdown."""
    from .database import TrainingDB

    db = TrainingDB()

    record = resolve_classifier(db, classifier)
    name = record["name"]

    console.print(Panel(f"Statistics for: [bold]{name}[/bold]", style="blue"))

    n_sessions = db.get_session_count(name)
    n_annotations = db.get_total_annotations(name)
    classes = db.get_classes(name)

    grid = Table.grid(padding=1)
    grid.add_row("ID:", str(record["id"]))
    grid.add_row("Total Sessions:", str(n_sessions))
    grid.add_row("Total Annotations:", str(n_annotations))
    grid.add_row("Defined Classes:", ", ".join(classes))
    console.print(grid)
    console.print()

    dist = db.get_class_distribution(name)

    table = Table(title="Class Distribution")
    table.add_column("Class Label", style="green")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for label, count in dist.items():
        percentage = (count / n_annotations * 100) if n_annotations > 0 else 0
        table.add_row(label, str(count), f"{percentage:.1f}%")

    console.print(table)
    console.print()

    stats_rows = db.get_image_stats(name)
    if not stats_rows:
        console.print("[yellow]No per-image data found.[/yellow]")
        return

    table = Table(title=f"Detailed Statistics: {name}")
    table.add_column("Plate ID", justify="right", style="cyan")
    table.add_column("Well", style="green")
    table.add_column("Image ID", justify="right")
    table.add_column("Timepoint", justify="right")
    table.add_column("Total Cells", justify="right", style="bold")
    table.add_column("Class Breakdown")

    for s in stats_rows:
        breakdown = ", ".join(
            [f"{k}: {v}" for k, v in s["class_distribution"].items()]
        )
        table.add_row(
            str(s["plate_id"]),
            str(s["well"]),
            str(s["image_id"]),
            str(s["timepoint"]),
            str(s["total_cells"]),
            breakdown,
        )

    console.print(table)


def handle_delete(
    identifiers: list[str], yes: bool, plate: list[int] | None
) -> None:
    """Delete classifiers, or only their sessions for the given plates."""
    from .database import TrainingDB

    db_instance = TrainingDB()

    targets = []
    for ident in identifiers:
        try:
            clf = resolve_classifier(db_instance, ident)
            targets.append(clf)
        except SystemExit:
            # resolve_classifier printed the error already
            return

    console.print(
        Panel(f"Delete Operation: {len(targets)} classifier(s)", style="red")
    )

    total_sessions = 0
    total_annotations = 0
    deletion_plan: list[dict[str, Any]] = []

    for clf in targets:
        name = clf["name"]

        if plate:
            all_sessions = db_instance.list_sessions(name)
            target_plates = set(plate)
            sessions_to_delete = [
                s for s in all_sessions if s["plate_id"] in target_plates
            ]

            if not sessions_to_delete:
                console.print(
                    f"[yellow]Skipping '{name}': No sessions found for plates {plate}.[/yellow]"
                )
                continue

            n_sess = len(sessions_to_delete)
            deletion_plan.append(
                {
                    "type": "partial",
                    "classifier": clf,
                    "sessions": sessions_to_delete,
                    "desc": f"Sessions for Plate(s) {plate}",
                }
            )
            total_sessions += n_sess

        else:
            n_sess = db_instance.get_session_count(name)
            n_ann = db_instance.get_total_annotations(name)

            deletion_plan.append(
                {
                    "type": "full",
                    "classifier": clf,
                    "n_sessions": n_sess,
                    "n_annotations": n_ann,
                    "desc": "Entire Classifier",
                }
            )
            total_sessions += n_sess
            total_annotations += n_ann

    if not deletion_plan:
        console.print("[yellow]Nothing to delete.[/yellow]")
        return

    for plan in deletion_plan:
        clf_name = plan["classifier"]["name"]
        if plan["type"] == "full":
            console.print(
                f"[bold]{clf_name}[/bold]: {plan['desc']} ({plan['n_sessions']} sessions, {plan['n_annotations']} annotations)"
            )
        else:
            console.print(
                f"[bold]{clf_name}[/bold]: {plan['desc']} ({len(plan['sessions'])} sessions)"
            )

    if not yes:
        confirm = console.input(
            "\n[bold red]Are you sure you want to proceed with deletion? [y/N]: [/bold red]"
        )
        if confirm.lower() not in ["y", "yes"]:
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    training_data_dir = Path.home() / "omeroscreen_trainingdata"

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Deleting...", total=len(deletion_plan))

            for plan in deletion_plan:
                clf_name = plan["classifier"]["name"]
                classifier_dir = training_data_dir / clf_name

                if plan["type"] == "full":
                    all_sessions = db_instance.list_sessions(clf_name)
                    for session in all_sessions:
                        npy_path = Path(session["file_path"])
                        if npy_path.exists():
                            npy_path.unlink()

                    if db_instance.delete_classifier(clf_name):
                        console.print(
                            f"[green]Deleted classifier '{clf_name}' from database.[/green]"
                        )
                    else:
                        console.print(
                            f"[red]Failed to delete classifier '{clf_name}' from database.[/red]"
                        )

                    if classifier_dir.exists():
                        shutil.rmtree(classifier_dir)
                        console.print(
                            f"[green]Removed folder: {classifier_dir}[/green]"
                        )

                else:
                    count = 0
                    for session in plan["sessions"]:
                        npy_path = Path(session["file_path"])
                        if npy_path.exists():
                            npy_path.unlink()
                        if db_instance.delete_session(session["id"]):
                            count += 1
                    console.print(
                        f"[green]Deleted {count} sessions (+ NPY files) from '{clf_name}'.[/green]"
                    )

                    remaining = db_instance.get_session_count(clf_name)
                    if remaining == 0:
                        console.print(
                            f"[yellow]No sessions remain for '{clf_name}'.[/yellow]"
                        )
                        confirm_full = console.input(
                            f"[bold red]Delete entire classifier '{clf_name}' and its folder? [y/N]: [/bold red]"
                        )
                        if confirm_full.lower() in ["y", "yes"]:
                            db_instance.delete_classifier(clf_name)
                            if classifier_dir.exists():
                                shutil.rmtree(classifier_dir)
                            console.print(
                                f"[green]Removed classifier '{clf_name}' and folder {classifier_dir}.[/green]"
                            )

                progress.advance(task)

    except Exception as e:
        console.print(f"\n[red]Error during deletion: {e}[/red]")


def handle_export(
    classifier: str,
    output_format: str,
    output: Path | None,
    plate: int | None,
    well: str | None,
) -> None:
    """Write a classifier's annotations to a CSV, JSON or Parquet file."""
    import pandas as pd

    from .database import TrainingDB

    db = TrainingDB()

    record = resolve_classifier(db, classifier)
    name = record["name"]

    console.print(f"Fetching data for [bold]{name}[/bold]...")

    try:
        data = db.get_annotations_by_classifier(
            classifier_name=name, plate_id=plate, well=well
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if not data:
        console.print("[yellow]No data found to export.[/yellow]")
        return

    df = pd.DataFrame(data)

    output_path = output or Path(f"{name}_export.{output_format}")

    console.print(f"Exporting {len(df)} rows to {output_path}...")

    try:
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif output_format == "parquet":
            df.to_parquet(output_path, index=False)

        console.print(f"[green]Successfully exported to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Entry point for the ``omero-train`` console script.

    Preserves the pre-Click contract: invoking with no subcommand prints help
    and exits 1, a user interrupt exits 130, and an unexpected error exits 1.
    """
    try:
        cli.main(standalone_mode=False)
    except click.NoSuchOption as e:
        console.print(f"[red]{e.format_message()}[/red]")
        sys.exit(2)
    except click.UsageError as e:
        # No subcommand given: argparse printed help and exited 1.
        if e.ctx is not None and not sys.argv[1:]:
            click.echo(e.ctx.get_help())
            sys.exit(1)
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Abort:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
