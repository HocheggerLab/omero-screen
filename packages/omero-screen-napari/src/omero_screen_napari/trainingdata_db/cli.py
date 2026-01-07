"""CLI tool for managing Omero Screen Training Data.

This module provides a command-line interface for:
- Migrating existing training data to the database
- Listing and managing classifiers
- Exporting training data
- Viewing statistics
"""

import argparse
import sys
from pathlib import Path

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
        description="Manage Omero Screen Training Data Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute"
    )

    # Command: migrate
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate existing .npy files to database"
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
    subparsers.add_parser("list", help="List all classifiers")

    # Command: stats
    stats_parser = subparsers.add_parser(
        "stats", help="Show statistics for a classifier"
    )
    stats_parser.add_argument("classifier", help="Name of the classifier")

    # Command: export
    export_parser = subparsers.add_parser(
        "export", help="Export training data to file"
    )
    export_parser.add_argument("classifier", help="Name of the classifier")
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
        help="Output file path (default: {classifier}_export.{format})",
    )
    export_parser.add_argument("--plate", type=int, help="Filter by Plate ID")
    export_parser.add_argument("--well", help="Filter by Well")

    return parser


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

            table.add_row(
                str(clf["id"]),
                name,
                clf["description"] or "",
                str(clf["created_at"]),
                str(n_sessions),
                str(n_annotations),
            )

    console.print(table)


def handle_stats(args: argparse.Namespace) -> None:
    """Handle stats command."""
    db = TrainingDB()
    name = args.classifier

    classifier = db.get_classifier(name)
    if not classifier:
        console.print(f"[red]Classifier '{name}' not found.[/red]")
        sys.exit(1)

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


def handle_export(args: argparse.Namespace) -> None:
    """Handle export command."""
    db = TrainingDB()
    name = args.classifier

    if not db.get_classifier(name):
        console.print(f"[red]Classifier '{name}' not found.[/red]")
        sys.exit(1)

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
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
