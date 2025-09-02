"""Module for loading data from a plate or a CSV file into the database.

This module provides a function to load data from a plate or a CSV file into the database.
"""

import argparse

import duckdb
import pandas as pd

from cellview.db.db import CellViewDB
from cellview.exporters.db_to_pandas import export_pandas_df
from cellview.importers.import_functions import import_data
from cellview.utils.state import CellViewState, create_cellview_state


def cellview_load_data_with_injection(
    *plate_ids: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data from plates using dependency injection (preferred approach).

    This function loads data from one or more plates using dependency injection
    instead of the singleton pattern, providing better testability and thread safety.

    Args:
        *plate_ids: The plate IDs to load data from.

    Returns:
        A tuple containing a pandas DataFrame and a list of variable names.
    """
    db = CellViewDB()
    df_list = []
    for plate_id in plate_ids:
        conn = db.connect()
        if not _check_plate_exists(plate_id, conn):
            args = argparse.Namespace(plate_id=plate_id, csv=None)
            # Use dependency injection to create state
            state = create_cellview_state(args)
            import_data(db, state, conn=conn)
        df, variable_names = export_pandas_df(plate_id, conn)
        df_list.append(df)
    conn.close()
    return pd.concat(df_list), variable_names


def cellview_load_data(*plate_ids: int) -> tuple[pd.DataFrame, list[str]]:
    """Load data from a plate or a CSV file into the database (dependency injection version).

    This function loads data from a plate or a CSV file into the database using
    dependency injection for better testability and thread safety.

    Args:
        *plate_ids: The plate IDs to load data from.

    Returns:
        A tuple containing a pandas DataFrame and a list of variable names.
    """
    return cellview_load_data_with_injection(*plate_ids)


def cellview_load_data_legacy(
    *plate_ids: int,
) -> tuple[pd.DataFrame, list[str]]:
    # sourcery skip: hoist-statement-from-if, inline-immediately-returned-variable
    """Load data from a plate or a CSV file into the database (legacy singleton version).

    This function loads data from a plate or a CSV file into the database using
    the singleton pattern. Kept for backward compatibility.

    Args:
        *plate_ids: The plate IDs to load data from.

    Returns:
        A tuple containing a pandas DataFrame and a list of variable names.
    """
    db = CellViewDB()
    df_list = []
    for plate_id in plate_ids:
        conn = db.connect()
        if not _check_plate_exists(plate_id, conn):
            args = argparse.Namespace(plate_id=plate_id, csv=None)
            state = CellViewState.get_instance(args)
            import_data(db, state, conn=conn)
        df, variable_names = export_pandas_df(plate_id, conn)
        df_list.append(df)
    conn.close()
    return pd.concat(df_list), variable_names


def _check_plate_exists(
    plate_id: int, conn: duckdb.DuckDBPyConnection
) -> bool:
    """Check if a plate exists in the database."""
    return bool(
        result := conn.execute(  # noqa: F841
            """
        SELECT r.plate_id, p.project_name, e.experiment_name
        FROM repeats r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        JOIN projects p ON e.project_id = p.project_id
        WHERE r.plate_id = ?
        """,
            [plate_id],
        ).fetchone()
    )
