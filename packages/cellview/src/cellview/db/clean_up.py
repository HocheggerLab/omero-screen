import duckdb
from rich.table import Table

from cellview.db.db import CellViewDB
from cellview.utils.ui import CellViewUI, Colors

ui = CellViewUI()


def clean_up_db(db: CellViewDB, conn: duckdb.DuckDBPyConnection) -> None:
    """Clean up the database by removing orphaned records.

    The cleanup process works recursively from top to bottom:
    1. Check projects -> experiments
    2. Check experiments -> repeats
    3. Check repeats -> conditions
    4. Check conditions -> measurements
    5. Repeat until no more orphaned records are found
    """

    # Create a table to display cleanup results
    table = Table(
        title="Database Cleanup Results", title_style=Colors.TITLE.value
    )
    table.add_column("Operation", style=Colors.SECONDARY.value)
    table.add_column("Records Affected", style=Colors.ACCENT.value)

    # Start a transaction to ensure atomicity
    conn.begin()
    try:
        total_cleaned = 0
        iteration = 1

        ui.header(
            "Database Cleanup", "Analyzing and removing orphaned records"
        )

        while True:
            iteration_cleaned = 0

            # Check and clean projects
            ui.progress("Checking projects")
            p_count = del_orphaned_projects(db, conn)
            if p_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Projects",
                    str(p_count),
                )
                iteration_cleaned += p_count
                continue  # Start over if we found orphaned projects

            # Check and clean experiments
            ui.progress("Checking experiments")
            e_count = del_orphaned_experiments(db, conn)
            if e_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Experiments",
                    str(e_count),
                )
                iteration_cleaned += e_count
                continue  # Start over if we found orphaned experiments

            # Check and clean repeats
            ui.progress("Checking repeats")
            r_count = del_orphaned_repeats(db, conn)
            if r_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Repeats",
                    str(r_count),
                )
                iteration_cleaned += r_count
                continue  # Start over if we found orphaned repeats

            # Check and clean measurements first (bottom-up)
            ui.progress("Checking measurements")
            m_count = del_orphaned_measurements(db, conn)
            if m_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Measurements",
                    str(m_count),
                )
                iteration_cleaned += m_count
                continue  # Start over if we found orphaned measurements

            # Check and clean condition variables
            ui.progress("Checking condition variables")
            cv_count = del_orphaned_condition_variables(db, conn)
            if cv_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Condition Variables",
                    str(cv_count),
                )
                iteration_cleaned += cv_count
                continue  # Start over if we found orphaned condition variables

            # Check and clean conditions
            ui.progress("Checking conditions")
            c_count = del_orphaned_conditions(db, conn)
            if c_count > 0:
                table.add_row(
                    f"Iteration {iteration}: Orphaned Conditions",
                    str(c_count),
                )
                iteration_cleaned += c_count
                continue  # Start over if we found orphaned conditions

            total_cleaned += iteration_cleaned
            if iteration_cleaned == 0:
                break  # No more orphaned records found
            iteration += 1

        # Commit the transaction if all operations succeed
        conn.commit()

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
        conn.rollback()
        ui.error(f"Error during cleanup: {str(e)}")
        raise e


def del_orphaned_projects(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete projects that don't have any experiments."""
    result = conn.execute("""
        SELECT COUNT(*)
        FROM projects p
        WHERE NOT EXISTS (
            SELECT 1
            FROM experiments e
            WHERE e.project_id = p.project_id
        )
    """).fetchone()
    count = result[0] if result else 0

    if count > 0:
        conn.execute("""
            DELETE FROM projects p
            WHERE NOT EXISTS (
                SELECT 1
                FROM experiments e
                WHERE e.project_id = p.project_id
            )
        """)
    return count


def del_orphaned_experiments(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete experiments that don't have any repeats."""
    result = conn.execute("""
        SELECT COUNT(*)
        FROM experiments e
        WHERE NOT EXISTS (
            SELECT 1
            FROM repeats r
            WHERE r.experiment_id = e.experiment_id
        )
    """).fetchone()
    count = result[0] if result else 0

    if count > 0:
        conn.execute("""
            DELETE FROM experiments e
            WHERE NOT EXISTS (
                SELECT 1
                FROM repeats r
                WHERE r.experiment_id = e.experiment_id
            )
        """)
    return count


def del_orphaned_repeats(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete repeats that don't have any conditions."""
    result = conn.execute("""
        SELECT COUNT(*)
        FROM repeats r
        WHERE NOT EXISTS (
            SELECT 1
            FROM conditions c
            WHERE c.repeat_id = r.repeat_id
        )
    """).fetchone()
    count = result[0] if result else 0

    if count > 0:
        conn.execute("""
            DELETE FROM repeats r
            WHERE NOT EXISTS (
                SELECT 1
                FROM conditions c
                WHERE c.repeat_id = r.repeat_id
            )
        """)
    return count


def del_orphaned_conditions(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete conditions that don't have any measurements or condition variables."""
    # Find conditions without measurements OR without condition variables
    result = conn.execute("""
        SELECT c.condition_id, c.repeat_id, c.well, c.cell_line,
               r.plate_id, r.experiment_id, e.project_id
        FROM conditions c
        JOIN repeats r ON c.repeat_id = r.repeat_id
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM measurements m
            WHERE m.condition_id = c.condition_id
        )
        OR NOT EXISTS (
            SELECT 1
            FROM condition_variables cv
            WHERE cv.condition_id = c.condition_id
        )
    """).fetchall()

    if not result:
        return 0

    # Group results by plate_id
    plates: dict[int, list[tuple[int, int, str, str, int, int, int]]] = {}
    for row in result:
        plate_id = row[4]
        if plate_id not in plates:
            plates[plate_id] = []
        plates[plate_id].append(row)

    total_deleted = 0
    # Process each plate
    for plate_id, plate_conditions in plates.items():
        ui.header(
            f"Plate {plate_id} Cleanup",
            "Removing orphaned conditions and related data",
        )

        # Create table for this plate's conditions
        table = Table(
            title=f"Records to be deleted from plate {plate_id}",
            title_style=Colors.TITLE.value,
        )
        table.add_column("Condition ID", style=Colors.PRIMARY.value)
        table.add_column("Well", style=Colors.SECONDARY.value)
        table.add_column("Cell Line", style=Colors.INFO.value)

        for condition_id, _, well, cell_line, _, _, _ in plate_conditions:
            table.add_row(str(condition_id), str(well), str(cell_line))

            # Count and delete condition variables
            cv_result = conn.execute(
                """
                SELECT COUNT(*)
                FROM condition_variables
                WHERE condition_id = ?
            """,
                [condition_id],
            ).fetchone()
            cv_count = cv_result[0] if cv_result else 0
            if cv_count > 0:
                conn.execute(
                    """
                    DELETE FROM condition_variables
                    WHERE condition_id = ?
                """,
                    [condition_id],
                )
                conn.commit()  # Commit after deleting condition variables
                total_deleted += cv_count

            # Count and delete measurements
            m_result = conn.execute(
                """
                SELECT COUNT(*)
                FROM measurements
                WHERE condition_id = ?
            """,
                [condition_id],
            ).fetchone()
            m_count = m_result[0] if m_result else 0
            if m_count > 0:
                conn.execute(
                    """
                    DELETE FROM measurements
                    WHERE condition_id = ?
                """,
                    [condition_id],
                )
                conn.commit()  # Commit after deleting measurements
                total_deleted += m_count

            # Delete the condition itself
            conn.execute(
                """
                DELETE FROM conditions
                WHERE condition_id = ?
            """,
                [condition_id],
            )
            conn.commit()  # Commit after deleting condition
            total_deleted += 1

        ui.console.print(table)
        ui.success(f"Deleted {total_deleted} records from plate {plate_id}")

    return total_deleted


def del_orphaned_measurements(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete measurements that don't have a valid condition_id reference."""
    result = conn.execute("""
        SELECT COUNT(*)
        FROM measurements m
        WHERE NOT EXISTS (
            SELECT 1
            FROM conditions c
            WHERE c.condition_id = m.condition_id
        )
    """).fetchone()
    count = result[0] if result else 0

    if count > 0:
        conn.execute("""
            DELETE FROM measurements m
            WHERE NOT EXISTS (
                SELECT 1
                FROM conditions c
                WHERE c.condition_id = m.condition_id
            )
        """)
    return count


def del_orphaned_condition_variables(
    db: CellViewDB, conn: duckdb.DuckDBPyConnection
) -> int:
    """Delete condition variables that don't have a valid condition_id reference."""
    result = conn.execute("""
        SELECT COUNT(*)
        FROM condition_variables cv
        WHERE NOT EXISTS (
            SELECT 1
            FROM conditions c
            WHERE c.condition_id = cv.condition_id
        )
    """).fetchone()
    count = result[0] if result else 0

    if count > 0:
        conn.execute("""
            DELETE FROM condition_variables cv
            WHERE NOT EXISTS (
                SELECT 1
                FROM conditions c
                WHERE c.condition_id = cv.condition_id
            )
        """)
    return count


def find_repeats_without_measurements(conn: duckdb.DuckDBPyConnection) -> None:
    """Find and display repeats that have conditions but no measurements."""
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
            DELETE FROM measurements m
            USING conditions c, repeats r
            WHERE m.condition_id = c.condition_id
            AND c.repeat_id = r.repeat_id
            AND r.plate_id = ?
        """,
            [plate_id],
        )
        conn.commit()
        clean_up_db(db, conn)
        ui.header(
            f"Plate {plate_id} Successfully Deleted",
        )
        ui.success(
            f"Deleted {count} measurements as well as all associated data for plate {plate_id}"
        )
    else:
        ui.info(f"No measurements found for plate {plate_id}")

    return count
