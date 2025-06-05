"""This module is used to edit projects and experiments in the database."""

import duckdb

from cellview.db.display import display_projects


def edit_project(project_id: int, conn: duckdb.DuckDBPyConnection) -> None:
    """Edit a project in the database.

    Args:
        project_id: The ID of the project to edit
        conn: The database connection
    """
    # Fetch current project info
    row = conn.execute(
        "SELECT description FROM projects WHERE project_id = ?",
        [project_id],
    ).fetchone()
    if not row:
        print(f"No project found with ID {project_id}")
        return
    current_name, current_desc = row
    print(f"Editing project ID {project_id}:")
    # Prompt for new values
    # new_name = input(f"Project name [{current_name}]: ") or current_name
    new_desc = input(f"Description [{current_desc or ''}]: ") or current_desc
    # Only update if changed
    if new_desc != current_desc:
        conn.execute(
            "UPDATE projects SET description = ? WHERE project_id = ?",
            [new_desc, project_id],
        )
        print("Project updated.")
    else:
        print("No changes made.")
    # Show updated project
    display_projects(conn, project_id)


def edit_experiment(
    experiment_id: int, conn: duckdb.DuckDBPyConnection
) -> None:
    """Edit an experiment in the database.

    Args:
        experiment_id: The ID of the experiment to edit
        conn: The database connection
    """
    # Fetch current experiment info
    row = conn.execute(
        "SELECT experiment_name, description FROM experiments WHERE experiment_id = ?",
        [experiment_id],
    ).fetchone()
    if not row:
        print(f"No experiment found with ID {experiment_id}")
        return
    current_name, current_desc = row
    print(f"Editing experiment ID {experiment_id}:")
    new_name = input(f"Experiment name [{current_name}]: ") or current_name
    new_desc = input(f"Description [{current_desc or ''}]: ") or current_desc
    # Only update if changed
    if new_name != current_name or new_desc != current_desc:
        conn.execute(
            "UPDATE experiments SET experiment_name = ?, description = ? WHERE experiment_id = ?",
            [new_name, new_desc, experiment_id],
        )
        print("Experiment updated.")
    else:
        print("No changes made.")
    # Optionally, show updated experiment (could call a display function if desired)
