"""Module for displaying data from the CellView database.

This module provides functions for displaying data from the CellView database.
"""

from collections import defaultdict
from typing import Any

import duckdb

from cellview.utils.ui import Colors, display_table, ui


def plate_project_query(
    plate_id: int, conn: duckdb.DuckDBPyConnection
) -> None:
    """Display the project information for a given plate.

    Args:
        plate_id: The ID of the plate
        conn: The database connection
    """
    query = """
  SELECT
    r.plate_id,
    p.project_id,
    p.project_name,
    e.experiment_id,
    e.experiment_name,
    r.date,
    r.lab_member,
    r.channel_0,
    r.channel_1,
    r.channel_2,
    r.channel_3,
    r.classifier,
    (
        SELECT COUNT(*)
        FROM repeats r2
        WHERE r2.experiment_id = e.experiment_id
    ) AS total_repeats,
  FROM repeats r
  JOIN experiments e ON r.experiment_id = e.experiment_id
  JOIN projects p ON e.project_id = p.project_id
  WHERE r.plate_id = ?
    """

    result = conn.execute(query, [plate_id]).fetchone()
    if not result:
        ui.error(f"No data found for plate ID {plate_id}")
        return
    display_table(
        conn,
        title="🧪 Plate Summary",
        rows=[result],
        style_columns=[0],  # <-- highlight first column
        highlight_style=Colors.SECONDARY.value,
    )


def conditions_query(plate_id: int, con: duckdb.DuckDBPyConnection) -> None:
    """Display the conditions for a given plate.

    Args:
        plate_id: The ID of the plate
        con: The database connection
    """
    query = """
    SELECT
        c.well,
        c.well_id,
        c.cell_line,
        c.antibody,
        c.antibody_1,
        c.antibody_2,
        c.antibody_3,
        cv.variable_name,
        cv.variable_value
    FROM repeats r
    JOIN conditions c ON r.repeat_id = c.repeat_id
    LEFT JOIN condition_variables cv ON c.condition_id = cv.condition_id
    WHERE r.plate_id = ?
    ORDER BY c.well, cv.variable_name
    """

    rows = con.execute(query, [plate_id]).fetchall()
    assert con.description is not None

    # Group by well
    wells: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "well_id": None,
            "cell_line": None,
            "antibodies": [],
            "variables": {},
        }
    )

    for row in rows:
        (
            well,
            well_id,
            cell_line,
            antibody,
            antibody_1,
            antibody_2,
            antibody_3,
            var_name,
            var_value,
        ) = row

        well_data = wells[well]
        well_data["well_id"] = well_id
        well_data["cell_line"] = cell_line

        # Add non-None antibodies to the list
        new_antibodies: list[str] = list(
            filter(None, [antibody, antibody_1, antibody_2, antibody_3])
        )
        well_data["antibodies"] = new_antibodies

        # Only add variables if variable name exists and ensure variables is a dict
        if var_name is not None:
            if "variables" not in well_data or well_data["variables"] is None:
                well_data["variables"] = {}
            well_data["variables"][var_name] = var_value

    # Prepare final row list
    final_rows = []
    for well, info in wells.items():
        # Ensure variables is a dict before calling items()
        variables = info.get("variables", {}) or {}
        variable_str = ", ".join(f"{k}={v}" for k, v in variables.items())

        # Ensure antibodies is a list before using it
        antibodies = info.get("antibodies", []) or []
        antibody_str = ", ".join(antibodies) if antibodies else "-"
        final_rows.append(
            (
                well,
                info["well_id"],
                info["cell_line"],
                antibody_str,
                variable_str,
            )
        )

    columns = ["Well", "Well ID", "Cell Line", "Antibodies", "Variables"]

    display_table(
        con,
        title="🧫 Plate Conditions",
        rows=final_rows,
        columns=columns,
        style_columns=[0],
        highlight_style=Colors.SECONDARY.value,
    )


def measurements_query(
    plate_id: int, con: duckdb.DuckDBPyConnection, limit: int = 5
) -> None:
    """Display the measurements for a given plate.

    Args:
        plate_id: The ID of the plate
        con: The database connection
        limit: The number of measurements to display
    """
    query = """
    SELECT m.*
    FROM repeats r
    JOIN conditions c ON r.repeat_id = c.repeat_id
    JOIN measurements m ON c.condition_id = m.condition_id
    WHERE r.plate_id = ?
    ORDER BY m.measurement_id
    LIMIT ?
    """

    rows = con.execute(query, [plate_id, limit]).fetchall()
    assert con.description is not None, "Query did not return any description"
    column_names = [desc[0] for desc in con.description]

    if not rows:
        ui.warning(f"No measurements found for plate ID {plate_id}")
        return

    # Extract measurement IDs for headers
    measurement_ids = [
        str(row[column_names.index("measurement_id")]) for row in rows
    ]

    # Pivot the data:
    # Dictionary with key = measurement column, value = list of values
    pivot = {}
    for col_index, col_name in enumerate(column_names):
        if col_name in ("measurement_id", "condition_id"):  # Skip ID columns
            continue
        pivot[col_name] = [str(row[col_index]) for row in rows]

    # Now build the Rich table manually
    from rich.table import Table

    table = Table(show_lines=True)
    table.add_column("Measurement", style=Colors.TITLE.value)
    for mid in measurement_ids:
        table.add_column(f"ID {mid}", style=Colors.INFO.value)

    for field, values in pivot.items():
        table.add_row(field, *values)

    # Prepare data for display_table

    columns = ["Measurement"] + [f"ID {mid}" for mid in measurement_ids]
    rows = [(field, *values) for field, values in pivot.items()]

    display_table(
        con=con,
        title="📈 Measurement Overview",
        rows=rows,
        columns=columns,
        style_columns=[0],  # Highlight first column
    )


def display_plate_summary(
    plate_id: int, con: duckdb.DuckDBPyConnection
) -> None:
    """Display the summary of a given plate.

    Args:
        plate_id: The ID of the plate
        con: The database connection
    """
    plate_project_query(plate_id, con)
    conditions_query(plate_id, con)
    measurements_query(plate_id, con)


def display_projects(con: duckdb.DuckDBPyConnection) -> None:
    """Display the projects in the database.

    Args:
        con: The database connection
    """
    query = """
    SELECT
        p.project_name,
        p.project_id,
        e.experiment_name,
        e.experiment_id,
        COUNT(r.repeat_id) AS repeat_count,
        LIST(DISTINCT r.plate_id) AS plate_ids
    FROM projects p
    JOIN experiments e ON p.project_id = e.project_id
    LEFT JOIN repeats r ON e.experiment_id = r.experiment_id
    GROUP BY p.project_id, p.project_name, e.experiment_id, e.experiment_name
    ORDER BY p.project_id, e.experiment_id;
    """
    rows = con.execute(query).fetchall()
    display_table(
        con,
        title="Summary of Projects and Experiments",
        rows=rows,
        style_columns=[0],  # <-- highlight first column
        highlight_style=Colors.SECONDARY.value,
    )
