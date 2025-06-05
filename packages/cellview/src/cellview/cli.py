"""Module for parsing command-line arguments.

This module provides a function to parse command-line arguments using the argparse library.
It supports various options for database connection, CSV file import, plate ID, project listing,

"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    This function parses command-line arguments using the argparse library.
    It supports various options for database connection, CSV file import, plate ID, project listing,
    and cleanup.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional path to the DuckDB database file. Defaults to ~/cellview_data/cellview.duckdb",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        help="To trigger import, supply the path to the CSV file.",
    )

    parser.add_argument(
        "--plate-id",
        type=int,
        help="The ID of the plate to import.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up the database before importing.",
    )

    parser.add_argument(
        "--plate",
        type=int,
        help="Display information about a specific plate.",
    )

    parser.add_argument(
        "--projects",
        action="store_true",
        help="Display name, description, and number of experiments for all projects.",
    )

    parser.add_argument(
        "--project",
        type=int,
        help="Display all experiments and their associated plate_ids for a single project.",
    )

    parser.add_argument(
        "--experiment",
        type=int,
        help="Display all plates and their associated measurements for a single experiment.",
    )

    parser.add_argument(
        "--edit-project",
        type=int,
        help="Select a project by its ID and edit its name and description.",
    )

    parser.add_argument(
        "--edit-experiment",
        type=int,
        help="Select an experiment by its ID and edit its name and description.",
    )

    parser.add_argument(
        "--delete-plate",
        type=int,
        help="Delete a plate by its ID. This will also delete all associated data.",
    )

    parser.add_argument(
        "--export-plate",
        type=int,
        help="Export a plate by its ID.",
    )

    return parser.parse_args()
