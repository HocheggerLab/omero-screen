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

    result_proxy = conn.execute(query, [plate_id])
    assert result_proxy.description is not None
    column_names = [desc[0] for desc in result_proxy.description]
    result = result_proxy.fetchone()
    if not result:
        ui.error(f"No data found for plate ID {plate_id}")
        return
    rows = list(zip(column_names, result, strict=False))

    display_table(
        conn,
        title="🧪 Plate Summary",
        rows=rows,
        columns=["Field", "Value"],
        style_columns=[0],  # highlight first column
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

    pivot = {
        col_name: [str(row[col_index]) for row in rows]
        for col_index, col_name in enumerate(column_names)
        if col_name not in ("measurement_id", "condition_id")
    }
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


def display_projects(
    con: duckdb.DuckDBPyConnection, project_id: int | None = None
) -> None:
    """Display the projects in the database.

    Args:
        con: The database connection
        project_id: Optional project ID to filter by
    """
    if project_id is not None:
        query = """
        SELECT
            p.project_name,
            p.project_id,
            p.description,
            COUNT(e.experiment_id) AS experiment_count
        FROM projects p
        LEFT JOIN experiments e ON p.project_id = e.project_id
        WHERE p.project_id = ?
        GROUP BY p.project_id, p.project_name, p.description
        ORDER BY p.project_id;
        """
        rows = con.execute(query, [project_id]).fetchall()
    else:
        query = """
        SELECT
            p.project_name,
            p.project_id,
            p.description,
            COUNT(e.experiment_id) AS experiment_count
        FROM projects p
        LEFT JOIN experiments e ON p.project_id = e.project_id
        GROUP BY p.project_id, p.project_name, p.description
        ORDER BY p.project_id;
        """
        rows = con.execute(query).fetchall()
    display_table(
        con,
        title="Summary of Projects",
        rows=rows,
        columns=[
            "Project Name",
            "Project ID",
            "Description",
            "Experiment Count",
        ],
        style_columns=[0],  # highlight first column
        highlight_style=Colors.SECONDARY.value,
    )


def display_single_project(
    con: duckdb.DuckDBPyConnection, project_id: int
) -> None:
    """Display all experiments and their associated plate_ids for a single project, grouped by experiment.

    Args:
        con: The database connection
        project_id: The ID of the project to display
    """
    query = """
    SELECT
        e.experiment_id,
        e.experiment_name,
        e.description,
        r.plate_id
    FROM experiments e
    LEFT JOIN repeats r ON e.experiment_id = r.experiment_id
    WHERE e.project_id = ?
    ORDER BY e.experiment_id, r.plate_id
    """
    rows = con.execute(query, [project_id]).fetchall()
    if not rows:
        ui.warning(f"No experiments found for project ID {project_id}")
        return

    # Group by experiment
    exp_dict = {}
    for exp_id, exp_name, desc, plate_id in rows:
        if exp_id not in exp_dict:
            exp_dict[exp_id] = {
                "experiment_name": exp_name,
                "description": desc,
                "plate_ids": [],
            }
        if plate_id is not None:
            exp_dict[exp_id]["plate_ids"].append(str(plate_id))

    display_rows = []
    for exp_id, info in exp_dict.items():
        plate_ids_str = (
            ", ".join(info["plate_ids"]) if info["plate_ids"] else "-"
        )
        display_rows.append(
            (
                exp_id,
                info["experiment_name"],
                info["description"],
                plate_ids_str,
            )
        )

    columns = ["Experiment ID", "Experiment Name", "Description", "Plate IDs"]
    display_table(
        con,
        title=f"Experiments and Plates for Project {project_id}",
        rows=display_rows,
        columns=columns,
        style_columns=[0],  # highlight experiment id
        highlight_style=Colors.SECONDARY.value,
    )


def display_experiment(
    con: duckdb.DuckDBPyConnection, experiment_id: int
) -> None:
    """Display a summary for the experiment, then channels and variable information for each plate in the experiment."""
    # Show experiment summary table (like in project, but for one experiment)
    summary_query = """
    SELECT
        e.experiment_id,
        e.experiment_name,
        e.description,
        GROUP_CONCAT(DISTINCT r.plate_id ORDER BY r.plate_id) AS plate_ids
    FROM experiments e
    LEFT JOIN repeats r ON e.experiment_id = r.experiment_id
    WHERE e.experiment_id = ?
    GROUP BY e.experiment_id, e.experiment_name, e.description
    """
    summary_rows = con.execute(summary_query, [experiment_id]).fetchall()
    summary_columns = [
        "Experiment ID",
        "Experiment Name",
        "Description",
        "Plate IDs",
    ]
    display_table(
        con,
        title=f"Summary for Experiment {experiment_id}",
        rows=summary_rows,
        columns=summary_columns,
        style_columns=[0],
        highlight_style=Colors.SECONDARY.value,
    )

    # Get all column names from repeats table except repeat_id, experiment_id, plate_id
    repeats_columns_query = """
        PRAGMA table_info(repeats);
    """
    columns_info = con.execute(repeats_columns_query).fetchall()
    repeats_columns = [
        col[1]
        for col in columns_info
        if col[1] not in ("repeat_id", "experiment_id", "plate_id")
    ]

    # Query for all relevant info
    repeats_cols_str = ", ".join(f"r.{col}" for col in repeats_columns)
    query = f"""
    SELECT
        r.plate_id,
        {repeats_cols_str},
        cv.variable_name,
        c.cell_line
    FROM repeats r
    LEFT JOIN conditions c ON r.repeat_id = c.repeat_id
    LEFT JOIN condition_variables cv ON c.condition_id = cv.condition_id
    WHERE r.experiment_id = ?
    ORDER BY r.plate_id, cv.variable_name
    """
    rows = con.execute(query, [experiment_id]).fetchall()
    if not rows:
        ui.warning(f"No plates found for experiment ID {experiment_id}")
        return

    # Group by plate_id and collect repeats info, variable names, and cell lines
    plate_dict = {}
    for row in rows:
        plate_id = row[0]
        repeats_info = row[1 : 1 + len(repeats_columns)]
        var_name = row[1 + len(repeats_columns)]
        cell_line = row[1 + len(repeats_columns) + 1]
        if plate_id not in plate_dict:
            plate_dict[plate_id] = {
                "repeats_info": repeats_info,
                "variable_names": set(),
                "cell_lines": set(),
            }
        if var_name is not None:
            plate_dict[plate_id]["variable_names"].add(var_name)
        if cell_line is not None:
            plate_dict[plate_id]["cell_lines"].add(cell_line)

    display_rows = []
    # Find the actual channel columns
    channel_columns = [
        col for col in repeats_columns if col.startswith("channel_")
    ]
    classifier_col = next(
        (col for col in repeats_columns if col == "classifier"), None
    )
    for plate_id, info in plate_dict.items():
        channels = [
            info["repeats_info"][repeats_columns.index(col)]
            for col in channel_columns
            if col in repeats_columns
        ]
        channels_str = ", ".join(
            [str(ch) for ch in channels if ch is not None]
        )
        classifier = (
            info["repeats_info"][repeats_columns.index(classifier_col)]
            if classifier_col
            else None
        )
        classifier_str = str(classifier) if classifier else "-"
        cell_line_str = (
            ", ".join(sorted(info["cell_lines"]))
            if info["cell_lines"]
            else "-"
        )
        variables_str = (
            ", ".join(sorted(info["variable_names"]))
            if info["variable_names"]
            else "-"
        )
        display_rows.append(
            (
                plate_id,
                channels_str,
                classifier_str,
                cell_line_str,
                variables_str,
            )
        )

    columns = ["Plate ID", "Channels", "Classifier", "Cell Line", "Variables"]
    display_table(
        con,
        title=f"Channels and Variables for Experiment {experiment_id}",
        rows=display_rows,
        columns=columns,
        style_columns=[0],  # highlight plate id
        highlight_style=Colors.SECONDARY.value,
    )
