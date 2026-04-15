"""CLI entry point for the explore feature.

Copies a template notebook, injects plate IDs, then opens it in
JupyterLab or VS Code. Optionally launches napari alongside.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cellview.explore._registry import (
    EXPLORE_DIR,
    legacy_notebook_path_for_experiment,
    legacy_notebook_path_for_plates,
    notebook_path_for_experiment,
    notebook_path_for_plates,
)
from cellview.explore._template_registry import (
    create_notebook_from_template,
    get_template,
    list_templates,
)
from cellview.utils.ui import ui


@dataclass(frozen=True)
class ExploreTarget:
    """Resolved notebook location and plate context for an explore request."""

    notebook_path: Path
    legacy_path: Path
    plate_ids: list[int]
    label: str | int


def _resolve_experiment_plates(experiment: str | int) -> tuple[list[int], int]:
    """Resolve an experiment identifier to plate IDs.

    Args:
        experiment: Experiment name (str) or ID (int).

    Returns:
        Tuple of (plate_ids, experiment_id).

    Raises:
        SystemExit: If no plates are found for the experiment.
    """
    from cellview.db.db import CellViewDB

    db = CellViewDB()
    conn = db.connect()
    try:
        if isinstance(experiment, int):
            exp_id = experiment
            rows = conn.execute(
                "SELECT DISTINCT plate_id FROM repeats "
                "WHERE experiment_id = ? ORDER BY plate_id",
                [exp_id],
            ).fetchall()
        else:
            row = conn.execute(
                "SELECT experiment_id FROM experiments "
                "WHERE experiment_name = ?",
                [experiment],
            ).fetchone()
            if not row:
                ui.error(f"Experiment '{experiment}' not found")
                sys.exit(1)
            exp_id = row[0]
            rows = conn.execute(
                "SELECT DISTINCT plate_id FROM repeats "
                "WHERE experiment_id = ? ORDER BY plate_id",
                [exp_id],
            ).fetchall()

        plate_ids = [r[0] for r in rows]
        if not plate_ids:
            ui.error(f"No plates found for experiment {experiment}")
            sys.exit(1)
        return plate_ids, exp_id
    finally:
        conn.close()


def show_available_templates() -> None:
    """Print all available templates to the console."""
    templates = list_templates()
    if not templates:
        ui.warning(
            "No templates found. Add .ipynb files to "
            "~/.cellview/templates/ or check built-in templates."
        )
        return

    ui.header("Available templates")
    for t in templates:
        desc = f" — {t.description}" if t.description else ""
        source_tag = f"[{t.source}]"
        ui.info(f"  {t.name:20s} {source_tag:12s}{desc}")


def launch_explore(
    plate_ids: list[int] | None = None,
    experiment: str | int | None = None,
    *,
    template: str = "cellcycle",
    fresh: bool = False,
    no_napari: bool = False,
    list_templates_flag: bool = False,
    code: bool = False,
) -> None:
    """Copy a template notebook, inject plate IDs, and open it.

    Args:
        plate_ids: List of plate IDs to explore.
        experiment: Experiment name or ID. Mutually exclusive with plate_ids.
        template: Template name (without .ipynb). Defaults to "cellcycle".
        fresh: If True, regenerate the notebook even if it already exists.
        no_napari: If True, skip launching napari.
        list_templates_flag: If True, list available templates and exit.
        code: If True, force VS Code and open the parent folder.
    """
    if list_templates_flag:
        show_available_templates()
        return

    target = _resolve_target(plate_ids=plate_ids, experiment=experiment)

    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_notebook(target.legacy_path, target.notebook_path)

    if target.notebook_path.exists() and not fresh:
        ui.info(f"Opening existing notebook: {target.notebook_path}")
    else:
        tmpl = get_template(template)
        if tmpl is None:
            ui.error(
                f"Template '{template}' not found. "
                "Use --list-templates to see available templates."
            )
            sys.exit(1)

        create_notebook_from_template(
            tmpl.path,
            target.notebook_path,
            target.plate_ids,
        )
        action = "Regenerated" if fresh else "Created"
        ui.success(
            f"{action} notebook from '{template}' template: "
            f"{target.notebook_path}"
        )

    if not no_napari:
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import napari; napari.Viewer(); napari.run()",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ui.info("Napari viewer launched")

    editor = (
        "vscode"
        if code
        else os.environ.get("CELLVIEW_EDITOR", "jupyter").lower()
    )

    if editor == "vscode":
        subprocess.Popen(["code", str(target.notebook_path.parent)])
        ui.info(
            "Opened explore directory in VS Code — select the venv kernel "
            "if prompted"
        )
    else:
        subprocess.Popen(["jupyter", "lab", str(target.notebook_path)])
        ui.info("JupyterLab starting...")


def _resolve_target(
    plate_ids: list[int] | None,
    experiment: str | int | None,
) -> ExploreTarget:
    """Resolve notebook, migration, and folder metadata for explore."""
    if experiment is not None:
        resolved_plates, exp_id = _resolve_experiment_plates(experiment)
        folder_name = _folder_name_for_experiment(exp_id)
        return ExploreTarget(
            notebook_path=notebook_path_for_experiment(
                exp_id,
                folder_name=folder_name,
            ),
            legacy_path=legacy_notebook_path_for_experiment(exp_id),
            plate_ids=resolved_plates,
            label=experiment,
        )

    if plate_ids:
        folder_name = _folder_name_for_plates(plate_ids)
        return ExploreTarget(
            notebook_path=notebook_path_for_plates(
                plate_ids,
                folder_name=folder_name,
            ),
            legacy_path=legacy_notebook_path_for_plates(plate_ids),
            plate_ids=plate_ids,
            label=", ".join(str(pid) for pid in plate_ids),
        )

    ui.error("Provide plate IDs or --explore-experiment")
    sys.exit(1)


def _folder_name_for_experiment(experiment_id: int) -> str | None:
    """Return a readable folder name for an experiment notebook."""
    from cellview.db.db import CellViewDB

    db = CellViewDB()
    conn = db.connect()
    try:
        row = conn.execute(
            """
            SELECT p.project_name, e.experiment_name
            FROM experiments AS e
            LEFT JOIN projects AS p ON p.project_id = e.project_id
            WHERE e.experiment_id = ?
            """,
            [experiment_id],
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    project_name_raw, experiment_name_raw = row
    project_name = (
        str(project_name_raw) if project_name_raw is not None else None
    )
    experiment_name = (
        str(experiment_name_raw) if experiment_name_raw is not None else None
    )
    if project_name and experiment_name:
        return f"{project_name}/{experiment_name}"
    return experiment_name or project_name


def _folder_name_for_plates(plate_ids: list[int]) -> str | None:
    """Return a shared experiment/project folder when the plates imply one."""
    from cellview.db.db import CellViewDB

    placeholders = ", ".join("?" for _ in plate_ids)
    query = f"""
        SELECT DISTINCT p.project_name, e.experiment_name
        FROM repeats AS r
        LEFT JOIN experiments AS e ON e.experiment_id = r.experiment_id
        LEFT JOIN projects AS p ON p.project_id = e.project_id
        WHERE r.plate_id IN ({placeholders})
    """

    db = CellViewDB()
    conn = db.connect()
    try:
        rows = conn.execute(query, plate_ids).fetchall()
    finally:
        conn.close()

    unique_pairs = {
        (
            str(project_name) if project_name is not None else None,
            str(experiment_name) if experiment_name is not None else None,
        )
        for project_name, experiment_name in rows
    }
    if len(unique_pairs) == 1:
        project_name, experiment_name = next(iter(unique_pairs))
        if project_name and experiment_name:
            return f"{project_name}/{experiment_name}"
        return experiment_name or project_name

    unique_projects = {
        project_name
        for project_name, _ in unique_pairs
        if project_name is not None
    }
    if len(unique_projects) == 1:
        return next(iter(unique_projects))

    return None


def _migrate_legacy_notebook(legacy_path: Path, target_path: Path) -> None:
    """Move a flat-layout notebook into its new folder when needed."""
    if (
        legacy_path == target_path
        or not legacy_path.exists()
        or target_path.exists()
    ):
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.replace(target_path)
    ui.info(f"Moved notebook into folder structure: {target_path}")
