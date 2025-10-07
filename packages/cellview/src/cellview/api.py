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
    experiment: str | int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data from plates using dependency injection (preferred approach).

    This function loads data from one or more plates using dependency injection
    instead of the singleton pattern, providing better testability and thread safety.

    Args:
        *plate_ids: The plate IDs to load data from. Ignored if experiment is provided.
        experiment: Optional experiment identifier. Can be either:
                   - An integer (experiment_id)
                   - A string (experiment_name)
                   If provided, loads all plates from this experiment.

    Returns:
        A tuple containing a pandas DataFrame and a list of variable names.
    """
    db = CellViewDB()
    conn = db.connect()

    # Determine which plates to load
    if experiment is not None:
        # Load all plates from the specified experiment
        if isinstance(experiment, int):
            target_plate_ids = _get_plates_from_experiment_id(experiment, conn)
        else:
            target_plate_ids = _get_plates_from_experiment_name(
                experiment, conn
            )

        if not target_plate_ids:
            conn.close()
            raise ValueError(f"No plates found for experiment '{experiment}'")
    else:
        # Use provided plate IDs
        target_plate_ids = list(plate_ids)
        if not target_plate_ids:
            conn.close()
            raise ValueError("Either plate_ids or experiment must be provided")

    df_list = []
    for plate_id in target_plate_ids:
        if not _check_plate_exists(plate_id, conn):
            args = argparse.Namespace(plate_id=plate_id, csv=None)
            # Use dependency injection to create state
            state = create_cellview_state(args)
            import_data(db, state, conn=conn)
        df, variable_names = export_pandas_df(plate_id, conn)
        df_list.append(df)

    conn.close()
    return pd.concat(df_list), variable_names


def cellview_load_data(
    *plate_ids: int, experiment: str | int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Load data from a plate or a CSV file into the database (dependency injection version).

    This function loads data from a plate or a CSV file into the database using
    dependency injection for better testability and thread safety.

    Args:
        *plate_ids: The plate IDs to load data from. Ignored if experiment is provided.
        experiment: Optional experiment identifier. Can be either:
                   - An integer (experiment_id)
                   - A string (experiment_name)
                   If provided, loads all plates from this experiment.

    Returns:
        A tuple containing a pandas DataFrame and a list of variable names.

    Examples:
        # Load specific plates
        df, vars = cellview_load_data(123, 456)

        # Load all plates from experiment by ID
        df, vars = cellview_load_data(experiment=6)

        # Load all plates from experiment by name
        df, vars = cellview_load_data(experiment="palb_washout_recovery")
    """
    return cellview_load_data_with_injection(*plate_ids, experiment=experiment)


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


def _get_plates_from_experiment_id(
    experiment_id: int, conn: duckdb.DuckDBPyConnection
) -> list[int]:
    """Get all plate IDs from a given experiment ID."""
    result = conn.execute(
        """
        SELECT DISTINCT r.plate_id
        FROM repeats r
        WHERE r.experiment_id = ?
        ORDER BY r.plate_id
        """,
        [experiment_id],
    ).fetchall()
    return [row[0] for row in result]


def _get_plates_from_experiment_name(
    experiment_name: str, conn: duckdb.DuckDBPyConnection
) -> list[int]:
    """Get all plate IDs from a given experiment name."""
    result = conn.execute(
        """
        SELECT DISTINCT r.plate_id
        FROM repeats r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.experiment_name = ?
        ORDER BY r.plate_id
        """,
        [experiment_name],
    ).fetchall()
    return [row[0] for row in result]


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
