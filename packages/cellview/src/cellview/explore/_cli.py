"""CLI entry point for the explore feature.

Copies a template notebook, injects plate IDs, then opens it in
JupyterLab or VS Code.  Optionally launches napari alongside.
"""

import os
import subprocess
import sys

from cellview.explore._registry import (
    EXPLORE_DIR,
    notebook_path_for_experiment,
    notebook_path_for_plates,
)
from cellview.explore._template_registry import (
    create_notebook_from_template,
    get_template,
    list_templates,
)
from cellview.utils.ui import ui


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
) -> None:
    """Copy a template notebook, inject plate IDs, and open it.

    Args:
        plate_ids: List of plate IDs to explore.
        experiment: Experiment name or ID. Mutually exclusive with plate_ids.
        template: Template name (without .ipynb). Defaults to "cellcycle".
        fresh: If True, regenerate the notebook even if it already exists.
        no_napari: If True, skip launching napari.
        list_templates_flag: If True, list available templates and exit.
    """
    if list_templates_flag:
        show_available_templates()
        return

    # Resolve what we're exploring
    if experiment is not None:
        resolved_plates, exp_id = _resolve_experiment_plates(experiment)
        notebook_path = notebook_path_for_experiment(exp_id)
        plate_ids = resolved_plates
    elif plate_ids:
        notebook_path = notebook_path_for_plates(plate_ids)
    else:
        ui.error("Provide plate IDs or --explore-experiment")
        sys.exit(1)

    # Reuse existing notebook or create from template
    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)

    if notebook_path.exists() and not fresh:
        ui.info(f"Opening existing notebook: {notebook_path}")
    else:
        tmpl = get_template(template)
        if tmpl is None:
            ui.error(
                f"Template '{template}' not found. "
                "Use --list-templates to see available templates."
            )
            sys.exit(1)

        create_notebook_from_template(tmpl.path, notebook_path, plate_ids)
        action = "Regenerated" if fresh else "Created"
        ui.success(
            f"{action} notebook from '{template}' template: {notebook_path}"
        )

    # Launch napari (unless --no-napari)
    if not no_napari:
        subprocess.Popen(
            [sys.executable, "-c", "import napari; napari.run()"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ui.info("Napari viewer launched")

    # Open notebook in editor
    editor = os.environ.get("CELLVIEW_EDITOR", "jupyter").lower()

    if editor == "vscode":
        subprocess.Popen(["code", str(notebook_path)])
        ui.info("Opened in VS Code — select the venv kernel if prompted")
    else:
        subprocess.Popen(["jupyter", "lab", str(notebook_path)])
        ui.info("JupyterLab starting...")
