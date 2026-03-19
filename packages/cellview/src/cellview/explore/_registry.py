"""Registry for tracking explore notebooks on the filesystem.

Scans ~/.cellview/explore/ for notebook files and matches them to plate
and experiment IDs based on deterministic naming conventions.
"""

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
    results: list[str] = []
    for nb in EXPLORE_DIR.glob("explore_plate*.ipynb"):
        stem = nb.stem  # e.g. "explore_plates_12345_12378"
        # extract all integers after the prefix
        suffix = stem.replace("explore_", "")
        parts = suffix.split("_")
        ids = [int(p) for p in parts if p.isdigit()]
        if plate_id in ids:
            results.append(suffix)
    return sorted(results)


def experiment_notebook_exists(experiment_id: int) -> bool:
    """Check whether an experiment-level explore notebook exists.

    Args:
        experiment_id: The experiment ID to check.

    Returns:
        True if the notebook file exists.
    """
    return (EXPLORE_DIR / f"explore_exp_{experiment_id}.ipynb").exists()


def notebook_path_for_plates(plate_ids: list[int]) -> Path:
    """Return the canonical notebook path for a set of plate IDs.

    Single plate  -> ``explore_plate_12345.ipynb``
    Multiple      -> ``explore_plates_12345_12378_12390.ipynb``

    Args:
        plate_ids: Sorted list of plate IDs.

    Returns:
        Path to the notebook file (may not exist yet).
    """
    sorted_ids = sorted(plate_ids)
    if len(sorted_ids) == 1:
        name = f"explore_plate_{sorted_ids[0]}"
    else:
        ids_str = "_".join(str(pid) for pid in sorted_ids)
        name = f"explore_plates_{ids_str}"
    return EXPLORE_DIR / f"{name}.ipynb"


def notebook_path_for_experiment(experiment_id: int) -> Path:
    """Return the canonical notebook path for an experiment.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Path to the notebook file (may not exist yet).
    """
    return EXPLORE_DIR / f"explore_exp_{experiment_id}.ipynb"
