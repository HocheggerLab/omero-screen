"""Command-line argument parser for CellView.

Provides subcommand-based CLI using argparse subparsers.
"""

import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """Create and return the command line argument parser.

    Returns:
        The configured argument parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="cellview",
        description="CellView: manage and explore single-cell measurement data.",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to the DuckDB database file. "
        "Defaults to ~/.cellview/cellview.duckdb",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- Display commands ---
    subparsers.add_parser(
        "projects",
        help="List all projects with experiment counts.",
    )

    p_project = subparsers.add_parser(
        "project",
        help="Show experiments and plates for a project.",
    )
    p_project.add_argument("id", type=int, help="Project ID.")

    p_experiment = subparsers.add_parser(
        "experiment",
        help="Show plates, channels, and variables for an experiment.",
    )
    p_experiment.add_argument("id", type=int, help="Experiment ID.")

    p_plate = subparsers.add_parser(
        "plate",
        help="Show summary, conditions, and measurements for a plate.",
    )
    p_plate.add_argument("id", type=int, help="Plate ID.")

    # --- Import commands ---
    p_import = subparsers.add_parser(
        "import",
        help="Import data from CSV, OMERO plate, or screen.",
    )
    import_sub = p_import.add_subparsers(dest="import_command")

    p_import_csv = import_sub.add_parser(
        "csv",
        help="Import from a CSV file.",
    )
    p_import_csv.add_argument("path", type=Path, help="Path to the CSV file.")

    p_import_plate = import_sub.add_parser(
        "plate",
        help="Import one or more plates by ID.",
    )
    p_import_plate.add_argument(
        "ids", type=int, nargs="+", help="Plate ID(s) to import."
    )
    p_import_plate.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive project/experiment selection.",
    )

    p_import_screen = import_sub.add_parser(
        "screen",
        help="Import all plates from a screen.",
    )
    p_import_screen.add_argument("id", type=int, help="Screen ID.")
    p_import_screen.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive project/experiment selection.",
    )

    # --- Edit commands ---
    p_edit = subparsers.add_parser(
        "edit",
        help="Edit project or experiment metadata.",
    )
    edit_sub = p_edit.add_subparsers(dest="edit_command")

    p_edit_project = edit_sub.add_parser(
        "project",
        help="Edit a project's name and description.",
    )
    p_edit_project.add_argument("id", type=int, help="Project ID.")

    p_edit_experiment = edit_sub.add_parser(
        "experiment",
        help="Edit an experiment's name and description.",
    )
    p_edit_experiment.add_argument("id", type=int, help="Experiment ID.")

    # --- Export ---
    p_export = subparsers.add_parser(
        "export",
        help="Export plate data.",
    )
    p_export.add_argument("id", type=int, help="Plate ID to export.")

    # --- Delete ---
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete data.",
    )
    delete_sub = p_delete.add_subparsers(dest="delete_command")

    p_delete_plate = delete_sub.add_parser(
        "plate",
        help="Delete a plate and all its associated data.",
    )
    p_delete_plate.add_argument("id", type=int, help="Plate ID.")

    # --- Clean ---
    subparsers.add_parser(
        "clean",
        help="Clean up orphaned records in the database.",
    )

    # --- Explore ---
    p_explore = subparsers.add_parser(
        "explore",
        help="Launch a Jupyter notebook for interactive data exploration.",
    )
    p_explore.add_argument(
        "plate_ids",
        type=str,
        nargs="*",
        help="Plate ID(s) or notebook name (e.g. plates_3602_3603).",
    )
    p_explore.add_argument(
        "--experiment",
        type=str,
        metavar="EXPERIMENT",
        help="Explore all plates from an experiment (name or ID).",
    )
    p_explore.add_argument(
        "--template",
        type=str,
        default="cellcycle",
        metavar="NAME",
        help="Template notebook to use (default: cellcycle).",
    )
    p_explore.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate the notebook even if it already exists.",
    )
    p_explore.add_argument(
        "--no-napari",
        action="store_true",
        help="Skip launching napari.",
    )
    p_explore.add_argument(
        "--code",
        action="store_true",
        help="Open the notebook folder in VS Code instead of JupyterLab.",
    )
    p_explore.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print a JSON context snapshot (schema, conditions, stats, notebooks) "
            "to stdout and exit. Used by the agentic skill."
        ),
    )

    # --- Template management ---
    p_template = subparsers.add_parser(
        "template",
        help="Manage analysis notebook templates.",
    )
    template_sub = p_template.add_subparsers(dest="template_command")

    template_sub.add_parser(
        "list",
        help="List all registered templates.",
    )

    p_template_add = template_sub.add_parser(
        "add",
        help="Register a template file in the DB.",
    )
    p_template_add.add_argument(
        "path", type=Path, help="Path to the template file."
    )
    p_template_add.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the template name (default: filename stem).",
    )
    p_template_add.add_argument(
        "--description",
        type=str,
        default=None,
        help="Short description shown in listings.",
    )

    p_template_remove = template_sub.add_parser(
        "remove",
        help="Remove a template record from the DB (does not delete the file).",
    )
    p_template_remove.add_argument(
        "name", type=str, help="Template name to remove."
    )

    p_template_show = template_sub.add_parser(
        "show",
        help="Show details for a single template.",
    )
    p_template_show.add_argument("name", type=str, help="Template name.")

    template_sub.add_parser(
        "sync",
        help="Scan filesystem and register all discovered templates into the DB.",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        An object containing the parsed arguments.
    """
    return get_parser().parse_args()
