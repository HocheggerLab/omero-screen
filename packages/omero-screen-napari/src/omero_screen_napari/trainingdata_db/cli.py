"""CLI tool for managing Omero Screen Training Data.

This module provides a command-line interface for:
- Migrating existing training data to the database
- Listing and managing classifiers
- Exporting training data
- Viewing statistics
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .database import TrainingDB
from .migrator import migrate_all_classifiers

console = Console()


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="""
Manage Omero Screen Training Data Database.

This CLI allows you to:
  - Migrate existing .npy training data files into the SQLite database.
  - List available classifiers and see summary statistics.
  - view detailed statistics for specific classifiers/images.
  - Export training data to CSV, JSON, or Parquet for analysis.

You can refer to classifiers by either their ID (e.g., 1) or Name (e.g., 'Experiment_1').
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute"
    )

    # Command: migrate
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate existing .npy files to database",
        description="Scan the specified directory for training data folders and import them into the database.",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without writing to DB",
    )
    migrate_parser.add_argument(
        "--path",
        type=Path,
        help="Base directory containing classifier folders (default: ~/omeroscreen_trainingdata)",
    )

    # Command: list
    subparsers.add_parser(
        "list",
        help="List all classifiers",
        description="Show a table of all available classifiers with their IDs, names, and summary stats.",
    )

    # Command: stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show statistics for a classifier",
        description="Show general statistics and class distribution for a specific classifier.",
    )
    stats_parser.add_argument(
        "classifier",
        help="ID or Name of the classifier (e.g., 1 or 'MyClassifier')",
    )

    # Command: export
    export_parser = subparsers.add_parser(
        "export",
        help="Export training data to file",
        description="Export annotations to a file (CSV, JSON, Parquet) with optional filtering.",
    )
    export_parser.add_argument(
        "classifier",
        help="ID or Name of the classifier (e.g., 1 or 'MyClassifier')",
    )
    export_parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: {name}_export.{format})",
    )
    export_parser.add_argument("--plate", type=int, help="Filter by Plate ID")
    export_parser.add_argument("--well", help="Filter by Well")

    # Command: delete
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a classifier",
        description="Delete a classifier and all its associated data (classes, sessions, annotations).",
    )
    delete_parser.add_argument(
        "identifiers",
        nargs="+",
        help="IDs or Names of the classifiers to delete (e.g., 1 2 'MyClassifier')",
    )
    delete_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    delete_parser.add_argument(
        "--plate",
        type=int,
        nargs="+",
        help="Delete only sessions for specific Plate ID(s). If not specified, deletes entire classifier.",
    )

    return parser


def resolve_classifier(db: TrainingDB, identifier: str) -> dict[str, Any]:
    """Resolve classifier by ID (if int) or name."""
    # Try as ID first if it looks like an integer
    if identifier.isdigit():
        clf = db.get_classifier_by_id(int(identifier))
        if clf:
            return clf

    # Try as name
    clf = db.get_classifier(identifier)
    if clf:
        return clf

    console.print(f"[red]Classifier '{identifier}' not found.[/red]")
    sys.exit(1)


def handle_migrate(args: argparse.Namespace) -> None:
    """Handle migrate command."""
    base_dir = args.path
    console.print(Panel("Starting Migration", style="bold blue"))

    # The migrator prints its own logs, but we can wrap it or just call it.
    # Since migrator uses logger, we rely on that or the summary it prints.
    migrate_all_classifiers(base_dir=base_dir, dry_run=args.dry_run)

    if args.dry_run:
        console.print(
            "[yellow]This was a DRY RUN. No changes were made.[/yellow]"
        )


def handle_list(args: argparse.Namespace) -> None:
    """Handle list command."""
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


def handle_stats(args: argparse.Namespace) -> None:
    """Handle stats command."""
    db = TrainingDB()
    identifier = args.classifier

    classifier = resolve_classifier(db, identifier)
    name = classifier["name"]

    console.print(Panel(f"Statistics for: [bold]{name}[/bold]", style="blue"))

    # General Stats
    n_sessions = db.get_session_count(name)
    n_annotations = db.get_total_annotations(name)
    classes = db.get_classes(name)

    grid = Table.grid(padding=1)
    grid.add_row("ID:", str(classifier["id"]))
    grid.add_row("Total Sessions:", str(n_sessions))
    grid.add_row("Total Annotations:", str(n_annotations))
    grid.add_row("Defined Classes:", ", ".join(classes))
    console.print(grid)
    console.print()

    # Class Distribution
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

    # Detailed Stats
    stats = db.get_image_stats(name)
    if not stats:
        console.print("[yellow]No per-image data found.[/yellow]")
        return

    table = Table(title=f"Detailed Statistics: {name}")
    table.add_column("Plate ID", justify="right", style="cyan")
    table.add_column("Well", style="green")
    table.add_column("Image ID", justify="right")
    table.add_column("Timepoint", justify="right")
    table.add_column("Total Cells", justify="right", style="bold")
    table.add_column("Class Breakdown")

    for s in stats:
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


def handle_delete(args: argparse.Namespace) -> None:
    """Handle delete command."""

    identifiers = args.identifiers
    db_instance = TrainingDB()

    # Resolve all classifiers first
    targets = []
    for ident in identifiers:
        try:
            # We use a slight variant of resolve logic here to avoid sys.exit inside loop if possible,
            # but resolve_classifier currently exits. Let's just use it and fail fast for now,
            # or better, catch the exit? resolve_classifier exits on failure.
            # Ideally we refactor resolve, but for now let's assume valid inputs or fail on first invalid.
            clf = resolve_classifier(db_instance, ident)
            targets.append(clf)
        except SystemExit:
            # resolve_classifier printed the error already
            return

    # Collect stats for all targets
    console.print(
        Panel(f"Delete Operation: {len(targets)} classifier(s)", style="red")
    )

    total_sessions = 0
    total_annotations = 0
    deletion_plan: list[dict[str, Any]] = []

    for clf in targets:
        name = clf["name"]

        if args.plate:
            # Partial delete logic per classifier
            all_sessions = db_instance.list_sessions(name)
            target_plates = set(args.plate)
            sessions_to_delete = [
                s for s in all_sessions if s["plate_id"] in target_plates
            ]

            if not sessions_to_delete:
                console.print(
                    f"[yellow]Skipping '{name}': No sessions found for plates {args.plate}.[/yellow]"
                )
                continue

            n_sess = len(sessions_to_delete)
            # Count annotations for these sessions if possible, or just estimate
            # Since we don't have a cheap way to count annotations for list of sessions without query
            # We'll just list session count.
            deletion_plan.append(
                {
                    "type": "partial",
                    "classifier": clf,
                    "sessions": sessions_to_delete,
                    "desc": f"Sessions for Plate(s) {args.plate}",
                }
            )
            total_sessions += n_sess

        else:
            # Full delete
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

    # Summary
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

    if not args.yes:
        confirm = console.input(
            "\n[bold red]Are you sure you want to proceed with deletion? [y/N]: [/bold red]"
        )
        if confirm.lower() not in ["y", "yes"]:
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    # Execute
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
                    # Delete NPY files for all sessions first
                    all_sessions = db_instance.list_sessions(clf_name)
                    for session in all_sessions:
                        npy_path = Path(session["file_path"])
                        if npy_path.exists():
                            npy_path.unlink()

                    # Delete classifier from DB
                    if db_instance.delete_classifier(clf_name):
                        console.print(
                            f"[green]Deleted classifier '{clf_name}' from database.[/green]"
                        )
                    else:
                        console.print(
                            f"[red]Failed to delete classifier '{clf_name}' from database.[/red]"
                        )

                    # Remove classifier folder on disk
                    if classifier_dir.exists():
                        shutil.rmtree(classifier_dir)
                        console.print(
                            f"[green]Removed folder: {classifier_dir}[/green]"
                        )

                else:
                    # Partial: delete selected sessions + their NPY files
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

                    # If no sessions remain, clean up the whole classifier
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


def handle_export(args: argparse.Namespace) -> None:
    """Handle export command."""
    db = TrainingDB()
    identifier = args.classifier

    classifier = resolve_classifier(db, identifier)
    name = classifier["name"]

    console.print(f"Fetching data for [bold]{name}[/bold]...")

    # Fetch data
    try:
        data = db.get_annotations_by_classifier(
            classifier_name=name, plate_id=args.plate, well=args.well
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if not data:
        console.print("[yellow]No data found to export.[/yellow]")
        return

    df = pd.DataFrame(data)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"{name}_export.{args.format}")

    # Export
    console.print(f"Exporting {len(df)} rows to {output_path}...")

    try:
        if args.format == "csv":
            df.to_csv(output_path, index=False)
        elif args.format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif args.format == "parquet":
            df.to_parquet(output_path, index=False)

        console.print(f"[green]Successfully exported to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "migrate":
            handle_migrate(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "stats":
            handle_stats(args)
        elif args.command == "export":
            handle_export(args)
        elif args.command == "delete":
            handle_delete(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
