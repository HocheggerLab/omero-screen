"""Module for cleaning up the database.

This module provides functions for cleaning up the database by removing orphaned records
and fixing schema issues (e.g., removing problematic columns with numeric suffixes).
"""

import contextlib

import duckdb
from rich.table import Table

from cellview.db.db import CellViewDB
from cellview.utils.ui import CellViewUI, Colors
from omero_screen.config import get_logger

logger = get_logger(__name__)


ui = CellViewUI()


def clean_up_db(db: CellViewDB, conn: duckdb.DuckDBPyConnection) -> None:
    """Clean up the database by removing orphaned records and fixing schema issues.

    The cleanup process works recursively from top to bottom:
    1. Clean schema (remove problematic columns with numeric suffixes)
    2. Check projects -> experiments
    3. Check experiments -> repeats
    4. Check repeats -> conditions
    5. Check conditions -> measurements
    6. Repeat until no more orphaned records are found

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Raises:
        Exception: If an error occurs during the cleanup process
    """
    # Create a table to display cleanup results
    table = Table(
        title="Database Cleanup Results", title_style=Colors.TITLE.value
    )
    table.add_column("Operation", style=Colors.SECONDARY.value)
    table.add_column("Records Affected", style=Colors.ACCENT.value)

    # Start a transaction to ensure atomicity
    # conn.begin() # Removed: Auto-commit mode for best-effort cleanup
    try:
        total_cleaned = 0
        iteration = 1

        ui.header(
            "Database Cleanup", "Analyzing and removing orphaned records"
        )

        # Clean schema first (remove problematic suffixed columns)
        ui.progress("Checking schema for problematic columns")
        schema_cols_cleaned = clean_schema_columns(db, conn)
        if schema_cols_cleaned > 0:
            ui.success(
                f"Cleaned {schema_cols_cleaned} problematic schema columns"
            )

        while True:
            iteration_cleaned = 0

            # 1. Clean measurements (Bottom-up: refer to nothing useful)
            ui.progress("Checking measurements")
            m_count = del_orphaned_measurements(db, conn)
            iteration_cleaned += m_count

            # 2. Clean condition variables
            ui.progress("Checking condition variables")
            cv_count = del_orphaned_condition_variables(db, conn)
            iteration_cleaned += cv_count

            # 3. Clean conditions ( aggressive: OR means if either half is missing, it's garbage )
            ui.progress("Checking conditions")
            c_count = del_orphaned_conditions(db, conn)
            iteration_cleaned += c_count

            # 4. Clean repeats
            ui.progress("Checking repeats")
            r_count = del_orphaned_repeats(db, conn)
            iteration_cleaned += r_count

            # 5. Clean experiments
            ui.progress("Checking experiments")
            e_count = del_orphaned_experiments(db, conn)
            iteration_cleaned += e_count

            # 6. Clean projects
            ui.progress("Checking projects")
            p_count = del_orphaned_projects(db, conn)
            iteration_cleaned += p_count

            total_cleaned += iteration_cleaned
            if iteration_cleaned == 0:
                break  # No more orphaned records found
            iteration += 1
            if iteration > 10:  # Safety break
                break

        # Commit the transaction if all operations succeed
        # conn.commit() # Removed: Auto-commit mode

        # Display the results
        ui.header(
            "Cleanup Summary", "Results of the database cleanup operation"
        )
        ui.console.print(table)

        if total_cleaned > 0:
            ui.success(f"Total records cleaned: {total_cleaned}")
        else:
            ui.info("No orphaned records found. Database is clean.")

    except Exception as e:
        # Rollback on any error
        # conn.rollback() # Removed: No transaction
        ui.error(f"Error during cleanup: {str(e)}")
        # If standard cleanup fails, attempt deep cleanup
        try:
            deep_clean_db(db, conn)
        except Exception as deep_e:
            ui.error(f"Deep cleanup also failed: {str(deep_e)}")
        raise e
    finally:
        # Final safety cleanup for any dangling experiments/projects without children
        # We don't start a new transaction here, just try to clear them up
        with contextlib.suppress(Exception):
            del_orphaned_projects(db, conn)


def deep_clean_db(db: CellViewDB, conn: duckdb.DuckDBPyConnection) -> None:
    """Perform an aggressive database scan and cleanup of all broken references.

    This function explicitly checks for children pointing to non-existent parents
    across all tables, performs schema cleanup, and a full bottom-up orphan cleanup.
    """
    ui.header(
        "Deep Database Cleanup", "Aggressively removing all dangling records"
    )
    # Note: We do NOT start a transaction here (conn.begin()).
    # Deep cleanup is a "best effort" operation. If one delete fails (e.g. strict FK),
    # we want to continue trying to delete other records.
    # Individual delete statements will be auto-committed.

    try:
        # 0. Clean schema first (remove problematic suffixed columns)
        schema_cols_cleaned = clean_schema_columns(db, conn)
        if schema_cols_cleaned > 0:
            ui.success(
                f"Deep cleanup removed {schema_cols_cleaned} problematic schema columns"
            )

        # 1. Clean broken references (Children pointing to missing parents)

        # Measurements -> Conditions
        m_count = conn.execute("""
            DELETE FROM measurements
            WHERE condition_id NOT IN (SELECT condition_id FROM conditions)
        """).rowcount
        if m_count > 0:
            ui.info(f"Cleaned {m_count} measurements with missing conditions")

        # Condition Variables -> Conditions
        cv_count = conn.execute("""
            DELETE FROM condition_variables
            WHERE condition_id NOT IN (SELECT condition_id FROM conditions)
        """).rowcount
        if cv_count > 0:
            ui.info(
                f"Cleaned {cv_count} condition variables with missing conditions"
            )

        # Conditions -> Repeats
        c_count = conn.execute("""
            DELETE FROM conditions
            WHERE repeat_id NOT IN (SELECT repeat_id FROM repeats)
        """).rowcount
        if c_count > 0:
            ui.info(f"Cleaned {c_count} conditions with missing repeats")

        # Repeats -> Experiments
        r_count = conn.execute("""
            DELETE FROM repeats
            WHERE experiment_id NOT IN (SELECT experiment_id FROM experiments)
        """).rowcount
        if r_count > 0:
            ui.info(f"Cleaned {r_count} repeats with missing experiments")

        # Experiments -> Projects
        e_count = conn.execute("""
            DELETE FROM experiments
            WHERE project_id NOT IN (SELECT project_id FROM projects)
        """).rowcount
        if e_count > 0:
            ui.info(f"Cleaned {e_count} experiments with missing projects")

        # 2. Perform exhaustive orphan cleanup (Parents with no children)
        # Loop multiple times to handle multi-level orphans (e.g. project -> exp -> repeat)
        total_orphans = 0
        for _ in range(3):
            iteration_cleaned = 0
            iteration_cleaned += del_orphaned_measurements(db, conn)
            iteration_cleaned += del_orphaned_condition_variables(db, conn)
            iteration_cleaned += del_orphaned_conditions(db, conn)
            iteration_cleaned += del_orphaned_repeats(db, conn)
            iteration_cleaned += del_orphaned_experiments(db, conn)
            iteration_cleaned += del_orphaned_projects(db, conn)
            total_orphans += iteration_cleaned
            if iteration_cleaned == 0:
                break

        # conn.commit() # Removed: Auto-commit mode
        if total_orphans > 0:
            ui.success(
                f"Deep cleanup removed {total_orphans} orphaned records"
            )
        ui.success("Deep database cleanup completed successfully")

    except Exception as e:
        # conn.rollback() # Removed: No transaction to rollback
        ui.error(f"Deep cleanup failed: {str(e)}")
        # We don't re-raise here to allow the application to proceed if possible,
        # or at least show what WAS cleaned.


def del_orphaned_projects(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete projects that don't have any experiments.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of projects deleted
    """
    # Find orphans first
    orphans = conn.execute("""
        SELECT project_id, project_name
        FROM projects
        WHERE project_id NOT IN (
            SELECT DISTINCT project_id
            FROM experiments
            WHERE project_id IS NOT NULL
        )
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for p_id, p_name in orphans:
        try:
            # Triple check dependencies before delete for diagnosis
            refs = conn.execute(
                "SELECT experiment_id FROM experiments WHERE project_id = ?",
                [p_id],
            ).fetchall()

            if refs:
                ui.warning(
                    f"Project {p_name} (ID: {p_id}) was marked as orphan, but "
                    f"DEBUG check found remaining experiments: {refs}"
                )
                continue

            conn.execute("DELETE FROM projects WHERE project_id = ?", [p_id])
            count += 1
            ui.info(f"Deleted orphaned project: {p_name} (ID: {p_id})")
        except Exception as e:
            ui.warning(
                f"Could not delete project {p_name} (ID: {p_id}). It might be referenced by a table not explicitly checked: {str(e)}"
            )

    return count


def del_orphaned_experiments(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete experiments that don't have any repeats or valid projects.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of experiments deleted
    """
    # Find orphans
    orphans = conn.execute("""
        SELECT experiment_id, experiment_name
        FROM experiments
        WHERE experiment_id NOT IN (
            SELECT DISTINCT experiment_id
            FROM repeats
            WHERE experiment_id IS NOT NULL
        )
        OR project_id NOT IN (SELECT project_id FROM projects)
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for e_id, e_name in orphans:
        try:
            conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?", [e_id]
            )
            count += 1
            ui.info(f"Deleted orphaned experiment: {e_name} (ID: {e_id})")
        except Exception as e:
            ui.warning(
                f"Could not delete experiment {e_name} (ID: {e_id}): {str(e)}"
            )
    return count


def del_orphaned_repeats(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete repeats that don't have any conditions or valid experiments.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of repeats deleted
    """
    # Find orphans
    orphans = conn.execute("""
        SELECT repeat_id, experiment_id, plate_id
        FROM repeats
        WHERE repeat_id NOT IN (
            SELECT DISTINCT repeat_id
            FROM conditions
            WHERE repeat_id IS NOT NULL
        )
        OR experiment_id NOT IN (SELECT experiment_id FROM experiments)
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for r_id, _, plate_id in orphans:
        try:
            conn.execute("DELETE FROM repeats WHERE repeat_id = ?", [r_id])
            count += 1
            ui.info(f"Deleted orphaned repeat: ID {r_id} (Plate {plate_id})")
        except Exception as e:
            ui.warning(f"Could not delete repeat ID {r_id}: {str(e)}")
    return count


def del_orphaned_conditions(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete conditions that don't have any measurements or valid repeats.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of conditions deleted
    """
    # First, delete conditions that point to missing repeats (Broken references)
    broken = conn.execute("""
        SELECT condition_id, well
        FROM conditions
        WHERE repeat_id NOT IN (SELECT repeat_id FROM repeats)
    """).fetchall()

    for c_id, well in broken:
        try:
            conn.execute(
                "DELETE FROM conditions WHERE condition_id = ?", [c_id]
            )
            ui.info(f"Cleaned broken condition {well} (ID: {c_id})")
        except Exception:
            pass

    # Then find conditions missing data (No measurements OR no condition variables)
    orphans = conn.execute("""
        SELECT condition_id, well
        FROM conditions
        WHERE condition_id NOT IN (
            SELECT DISTINCT condition_id
            FROM measurements
            WHERE condition_id IS NOT NULL
        )
        OR condition_id NOT IN (
            SELECT DISTINCT condition_id
            FROM condition_variables
            WHERE condition_id IS NOT NULL
        )
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for c_id, well in orphans:
        try:
            # 1. Clean child records first (Redundant but safe)
            conn.execute(
                "DELETE FROM condition_variables WHERE condition_id = ?",
                [c_id],
            )
            conn.execute(
                "DELETE FROM measurements WHERE condition_id = ?", [c_id]
            )

            # 2. Delete the condition
            conn.execute(
                "DELETE FROM conditions WHERE condition_id = ?", [c_id]
            )
            count += 1
            ui.info(f"Deleted orphaned condition: {well} (ID: {c_id})")
        except Exception as e:
            ui.warning(
                f"Could not delete condition {well} (ID: {c_id}): {str(e)}"
            )
    return count


def del_orphaned_measurements(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete measurements that don't have a valid condition_id reference.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of measurements deleted
    """
    orphans = conn.execute("""
        SELECT measurement_id, condition_id
        FROM measurements
        WHERE condition_id NOT IN (SELECT condition_id FROM conditions)
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for m_id, _ in orphans:
        try:
            conn.execute(
                "DELETE FROM measurements WHERE measurement_id = ?", [m_id]
            )
            count += 1
        except Exception:
            pass

    if count > 0:
        ui.info(f"Deleted {count} orphaned measurements")
    return count


def del_orphaned_condition_variables(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete condition variables that don't have a valid condition_id reference.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of condition variables deleted
    """
    orphans = conn.execute("""
        SELECT variable_id, condition_id
        FROM condition_variables
        WHERE condition_id NOT IN (
            SELECT DISTINCT condition_id
            FROM conditions
            WHERE condition_id IS NOT NULL
        )
    """).fetchall()

    if not orphans:
        return 0

    count = 0
    for v_id, _ in orphans:
        try:
            conn.execute(
                "DELETE FROM condition_variables WHERE variable_id = ?", [v_id]
            )
            count += 1
        except Exception:
            pass

    if count > 0:
        ui.info(f"Deleted {count} orphaned condition variables")
    return count


def find_repeats_without_measurements(conn: duckdb.DuckDBPyConnection) -> None:
    """Find and display repeats that have conditions but no measurements.

    Args:
        conn: The database connection
    """
    if result := conn.execute(
        """
        SELECT r.repeat_id, r.plate_id, r.experiment_id, r.date,
               COUNT(DISTINCT c.condition_id) as condition_count,
               COUNT(DISTINCT m.measurement_id) as measurement_count
        FROM repeats r
        LEFT JOIN conditions c ON r.repeat_id = c.repeat_id
        LEFT JOIN measurements m ON c.condition_id = m.condition_id
        GROUP BY r.repeat_id, r.plate_id, r.experiment_id, r.date
        HAVING condition_count > 0 AND measurement_count = 0
    """
    ).fetchall():
        ui.warning("Found repeats with conditions but no measurements")

        table = Table(
            title="Problematic Repeats", title_style=Colors.TITLE.value
        )
        table.add_column("Repeat ID", style=Colors.PRIMARY.value)
        table.add_column("Plate ID", style=Colors.PRIMARY.value)
        table.add_column("Experiment ID", style=Colors.PRIMARY.value)
        table.add_column("Date", style=Colors.PRIMARY.value)
        table.add_column("Condition Count", style=Colors.PRIMARY.value)
        table.add_column("Measurement Count", style=Colors.PRIMARY.value)

        for row in result:
            table.add_row(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
            )
        ui.console.print(table)
    else:
        ui.info("No repeats found with conditions but no measurements")


def del_conditions(db: CellViewDB, conn: duckdb.DuckDBPyConnection) -> None:
    """Delete all conditions from the database.

    Args:
        db: The CellViewDB instance
        conn: The database connection
    """
    conn.execute("DELETE FROM conditions")
    ui.warning("All conditions deleted from database")


def del_measurements_by_plate_id(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection, plate_id: int
) -> int:
    """Delete all measurements associated with a specific plate_id.

    Args:
        db: The CellViewDB instance
        conn: The database connection
        plate_id: The ID of the plate whose measurements should be deleted

    Returns:
        int: The number of measurements deleted

    """
    # First count the measurements to be deleted
    result = conn.execute(
        """
        SELECT COUNT(*)
        FROM measurements m
        JOIN conditions c ON m.condition_id = c.condition_id
        JOIN repeats r ON c.repeat_id = r.repeat_id
        WHERE r.plate_id = ?
    """,
        [plate_id],
    ).fetchone()

    count = result[0] if result else 0

    if count > 0:
        # Delete the measurements
        conn.execute(
            """
            DELETE FROM measurements
            WHERE condition_id IN (
                SELECT c.condition_id
                FROM conditions c
                JOIN repeats r ON c.repeat_id = r.repeat_id
                WHERE r.plate_id = ?
            )
        """,
            [plate_id],
        )
        # We don't commit here, let the caller handle it or let it auto-commit
        # clean_up_db(db, conn) # Removed recursive call to avoid transaction issues
        ui.header(
            f"Plate {plate_id} Successfully Deleted",
        )
        ui.success(
            f"Deleted {count} measurements as well as all associated data for plate {plate_id}"
        )
    else:
        ui.info(f"No measurements found for plate {plate_id}")

    return count


def clean_schema_columns(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Clean up problematic schema columns with numeric suffixes.

    This function removes columns from the measurements table that have
    numeric suffixes (e.g., .0, .1, .2, .3). These columns are typically
    created when pandas DataFrames with duplicate column names are imported,
    and they cause import failures because new data doesn't have values for
    these spurious columns.

    Args:
        db: The CellViewDB instance
        conn: The database connection

    Returns:
        The number of columns dropped

    Raises:
        Exception: If unable to clean schema columns
    """
    try:
        # Get all columns from measurements table
        result = conn.execute("SELECT * FROM measurements LIMIT 0").description
        all_columns = [col[0] for col in result]

        # Find columns with numeric suffixes (.0, .1, .2, .3, etc.)
        import re

        suffix_pattern = re.compile(r"\.\d+$")
        suffixed_columns = [
            col for col in all_columns if suffix_pattern.search(col)
        ]

        if not suffixed_columns:
            logger.debug("No problematic schema columns found")
            return 0

        # Drop each suffixed column
        dropped_count = 0
        failed_columns = []

        for col in suffixed_columns:
            try:
                # Use double quotes for column names with special characters
                conn.execute(f'ALTER TABLE measurements DROP COLUMN "{col}"')
                logger.info(f"Dropped problematic column: {col}")
                dropped_count += 1
            except Exception as e:
                logger.warning(f"Failed to drop column {col}: {e}")
                failed_columns.append(col)

        # Report results
        if dropped_count > 0:
            ui.info(
                f"Removed {dropped_count} problematic schema columns from measurements table"
            )

        if failed_columns:
            ui.warning(
                f"Failed to drop {len(failed_columns)} columns: {', '.join(failed_columns[:5])}"
            )

        return dropped_count

    except Exception as e:
        logger.error(f"Error during schema cleanup: {e}")
        ui.warning(f"Schema cleanup encountered an error: {e}")
        return 0
