import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        help="Display information about all projects.",
    )

    return parser.parse_args()
