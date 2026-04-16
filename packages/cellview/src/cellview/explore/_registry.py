"""Registry for tracking explore notebooks on the filesystem.

Supports both the legacy flat ``~/.cellview/explore`` layout and the newer
folder-based layout that groups notebooks by experiment/project or by plate
selection.
"""

import re
from pathlib import Path

EXPLORE_DIR = Path.home() / ".cellview" / "explore"


def notebooks_for_plate(plate_id: int) -> list[str]:
    """Return short display names of notebooks that reference a plate.

    Args:
        plate_id: The plate ID to search for.

    Returns:
        List of short names like ``"plate_12345"`` or ``"plates_12345_12378"``.
    """
    if not EXPLORE_DIR.exists():
        return []

    results: set[str] = set()
    for nb in _iter_explore_notebooks():
        stem = nb.stem
        if not stem.startswith("explore_plate"):
            continue

        suffix = stem.removeprefix("explore_")
        parts = suffix.split("_")
        ids = [int(part) for part in parts if part.isdigit()]
        if plate_id in ids:
            results.add(suffix)

    return sorted(results)


def experiment_notebook_exists(experiment_id: int) -> bool:
    """Check whether an experiment-level explore notebook exists.

    Args:
        experiment_id: The experiment ID to check.

    Returns:
        True if the notebook file exists.
    """
    canonical = legacy_notebook_path_for_experiment(experiment_id)
    if canonical.exists():
        return True

    pattern = f"explore_exp_{experiment_id}.ipynb"
    return any(nb.name == pattern for nb in _iter_explore_notebooks())


def notebook_path_for_plates(
    plate_ids: list[int],
    *,
    folder_name: str | None = None,
    ext: str = ".ipynb",
) -> Path:
    """Return the canonical notebook path for a set of plate IDs.

    Single plate  -> ``explore/plates/12345/explore_plate_12345.ipynb``
    Multiple      -> ``explore/plates/12345_12378/explore_plates_12345_12378.ipynb``

    Args:
        plate_ids: Sorted list of plate IDs.
        folder_name: Optional folder override for experiment/project grouping.
        ext: File extension, either ``".ipynb"`` (default) or ``".py"``.

    Returns:
        Path to the notebook file (may not exist yet).
    """
    sorted_ids = sorted(plate_ids)
    if len(sorted_ids) == 1:
        name = f"explore_plate_{sorted_ids[0]}"
        default_folder = EXPLORE_DIR / "plates" / str(sorted_ids[0])
    else:
        ids_str = "_".join(str(pid) for pid in sorted_ids)
        name = f"explore_plates_{ids_str}"
        default_folder = EXPLORE_DIR / "plates" / ids_str

    folder = (
        _folder_path_from_name(folder_name) if folder_name else default_folder
    )
    return folder / f"{name}{ext}"


def notebook_path_for_experiment(
    experiment_id: int,
    *,
    folder_name: str | None = None,
    ext: str = ".ipynb",
) -> Path:
    """Return the canonical notebook path for an experiment.

    Args:
        experiment_id: The experiment ID.
        folder_name: Optional readable experiment/project folder name.
        ext: File extension, either ``".ipynb"`` (default) or ``".py"``.

    Returns:
        Path to the notebook file (may not exist yet).
    """
    if folder_name:
        return (
            _folder_path_from_name(folder_name)
            / f"explore_exp_{experiment_id}{ext}"
        )
    base = legacy_notebook_path_for_experiment(experiment_id)
    return base.with_suffix(ext)


def legacy_notebook_path_for_plates(plate_ids: list[int]) -> Path:
    """Return the legacy flat explore path for a plate notebook."""
    sorted_ids = sorted(plate_ids)
    if len(sorted_ids) == 1:
        name = f"explore_plate_{sorted_ids[0]}"
    else:
        ids_str = "_".join(str(pid) for pid in sorted_ids)
        name = f"explore_plates_{ids_str}"
    return EXPLORE_DIR / f"{name}.ipynb"


def legacy_notebook_path_for_experiment(experiment_id: int) -> Path:
    """Return the legacy flat explore path for an experiment notebook."""
    return EXPLORE_DIR / f"explore_exp_{experiment_id}.ipynb"


def sanitize_folder_name(name: str) -> str:
    """Convert a single project or experiment name into a stable folder name."""
    sanitized = re.sub(r"\s+", "_", name.strip())
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._")
    return sanitized or "explore"


def _folder_path_from_name(folder_name: str) -> Path:
    """Build an explore subdirectory path from a slash-delimited folder name."""
    parts = [
        sanitize_folder_name(part)
        for part in folder_name.split("/")
        if part.strip()
    ]
    if not parts:
        return EXPLORE_DIR / "explore"

    folder = EXPLORE_DIR
    for part in parts:
        folder /= part
    return folder


def _iter_explore_notebooks() -> list[Path]:
    """Return every explore notebook in both flat and nested layouts."""
    return (
        list(EXPLORE_DIR.rglob("explore_*.ipynb"))
        if EXPLORE_DIR.exists()
        else []
    )
