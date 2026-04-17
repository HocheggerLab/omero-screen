"""CLI entry point for the explore feature.

Copies a template notebook, injects plate IDs, then opens it in
JupyterLab or VS Code. Optionally launches napari alongside.
"""

import json
import os
import shutil
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
    BUILTIN_TEMPLATE_DIR,
    create_notebook_from_template,
    get_template,
)
from cellview.utils.ui import ui


@dataclass(frozen=True)
class ExploreTarget:
    """Resolved notebook location and plate context for an explore request."""

    notebook_path: Path
    legacy_path: Path
    plate_ids: list[int]
    label: str | int
    fmt: str = "jupyter"  # "jupyter" or "marimo"


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


def launch_explore(
    plate_ids: list[int] | None = None,
    experiment: str | int | None = None,
    *,
    template: str = "cellcycle",
    fresh: bool = False,
    no_napari: bool = False,
    code: bool = False,
) -> None:
    """Copy a template notebook, inject plate IDs, and open it.

    Automatically detects the template format:
    - ``.ipynb`` templates open in JupyterLab (or VS Code).
    - ``.py`` templates open with ``marimo edit`` (or VS Code).

    The ``CELLVIEW_EDITOR`` environment variable can override the default
    editor (``"jupyter"`` or ``"marimo"``).  Setting ``code=True`` opens the
    entire ``~/.cellview/explore`` library in VS Code with the new notebook
    focused, so you can browse all past analyses alongside the new one.

    Args:
        plate_ids: List of plate IDs to explore.
        experiment: Experiment name or ID. Mutually exclusive with plate_ids.
        template: Template name (without extension). Defaults to "cellcycle".
        fresh: If True, regenerate the notebook even if it already exists.
        no_napari: If True, skip launching napari.
        code: If True, open the explore library in VS Code.
    """
    # Resolve the template first so we know its format before building the path
    tmpl = get_template(template)
    if tmpl is None:
        ui.error(
            f"Template '{template}' not found. "
            "Use --list-templates to see available templates."
        )
        sys.exit(1)

    target = _resolve_target(
        plate_ids=plate_ids, experiment=experiment, fmt=tmpl.fmt
    )

    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_claude_md(EXPLORE_DIR)
    _migrate_legacy_notebook(target.legacy_path, target.notebook_path)

    if target.notebook_path.exists() and not fresh:
        ui.info(f"Opening existing notebook: {target.notebook_path}")
    else:
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

    _open_editor(target, code=code)


def _resolve_target(
    plate_ids: list[int] | None,
    experiment: str | int | None,
    fmt: str = "jupyter",
) -> ExploreTarget:
    """Resolve notebook, migration, and folder metadata for explore.

    Args:
        plate_ids: One or more plate IDs.
        experiment: Experiment name or ID.
        fmt: Template format — ``"jupyter"`` (default) or ``"marimo"``.
    """
    ext = ".py" if fmt == "marimo" else ".ipynb"

    if experiment is not None:
        resolved_plates, exp_id = _resolve_experiment_plates(experiment)
        folder_name = _folder_name_for_experiment(exp_id)
        return ExploreTarget(
            notebook_path=notebook_path_for_experiment(
                exp_id,
                folder_name=folder_name,
                ext=ext,
            ),
            legacy_path=legacy_notebook_path_for_experiment(exp_id),
            plate_ids=resolved_plates,
            label=experiment,
            fmt=fmt,
        )

    if plate_ids:
        folder_name = _folder_name_for_plates(plate_ids)
        return ExploreTarget(
            notebook_path=notebook_path_for_plates(
                plate_ids,
                folder_name=folder_name,
                ext=ext,
            ),
            legacy_path=legacy_notebook_path_for_plates(plate_ids),
            plate_ids=plate_ids,
            label=", ".join(str(pid) for pid in plate_ids),
            fmt=fmt,
        )

    ui.error("Provide plate IDs or --explore-experiment")
    sys.exit(1)


def _open_editor(target: ExploreTarget, *, code: bool) -> None:
    """Launch the appropriate editor for the resolved target.

    Decision logic:
    - ``code=True`` or ``CELLVIEW_EDITOR=vscode``
        → Opens the entire ``~/.cellview/explore`` library as the VS Code
          workspace, then navigates directly to the notebook file via
          ``--goto``.  This lets you browse all past analyses in the sidebar.
    - format ``"marimo"`` → ``marimo edit <file>``
    - default → ``jupyter lab <file>``

    Args:
        target: Resolved explore target carrying path and format.
        code: If True, force VS Code.
    """
    editor_env = os.environ.get("CELLVIEW_EDITOR", "").lower()

    if code or editor_env == "vscode":
        subprocess.Popen(
            ["code", str(EXPLORE_DIR), "--goto", str(target.notebook_path)]
        )
        ui.info(
            f"Opened explore library in VS Code: {EXPLORE_DIR}\n"
            f"  → focused on {target.notebook_path.name}"
        )
        return

    if target.fmt == "marimo":
        subprocess.Popen(["marimo", "edit", str(target.notebook_path)])
        ui.info("Marimo editor starting...")
        return

    # Default: JupyterLab
    subprocess.Popen(["jupyter", "lab", str(target.notebook_path)])
    ui.info("JupyterLab starting...")


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


def _ensure_claude_md(explore_dir: Path) -> None:
    """Copy the bundled CLAUDE.md into explore_dir if not already present."""
    dest = explore_dir / "CLAUDE.md"
    if dest.exists():
        return
    source = BUILTIN_TEMPLATE_DIR / "CLAUDE.md"
    if source.exists():
        shutil.copy2(source, dest)


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


def explore_json_command(
    plate_ids: list[int] | None = None,
    experiment: str | int | None = None,
) -> None:
    """Print a JSON context snapshot for a plate or experiment to stdout.

    This is the primary entry point for the agentic workflow: the agent calls
    this before writing any pandas or plotting code so it knows the exact
    column schema, available conditions, cell counts, and linked notebooks.

    Args:
        plate_ids: One or more plate IDs to describe.
        experiment: Experiment name (str) or ID (int).
    """
    from cellview.db.db import CellViewDB
    from cellview.explore._explore_json import explore_json

    db = CellViewDB()
    conn = db.connect()
    try:
        snapshot = explore_json(
            conn, plate_ids=plate_ids, experiment=experiment
        )
        print(json.dumps(snapshot, indent=2, default=str))
    finally:
        conn.close()
