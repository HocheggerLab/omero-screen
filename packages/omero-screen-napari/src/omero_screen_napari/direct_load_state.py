"""Remembered form state for the direct-load dialog.

The dialog is typically used in bursts: load a well, label the crops, then
come back to load more of the same population — often changing a single
field (the next image, a different class, more crops). Re-entering six
fields every time is friction, so the last successful load is written
next to the classifier's own data and restored when the dialog reopens.

State is stored per classifier, separately from ``metadata.json`` — that
file describes how the classifier was *set up* (crop size, channels, the
gallery geometry that seeds the crop-count default) and must not be
rewritten by routine UI use.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

STATE_FILENAME = "direct_load_state.json"


@dataclass
class DirectLoadState:
    """Last-used inputs of the direct-load dialog.

    Attributes:
        plate_id: OMERO plate ID.
        well: Well position (e.g. ``"A1"``).
        image_input: Image selection string (``"All"``, ``"0, 1"``, ``"3-5"``).
        timepoint: Timepoint index.
        cellcycle: Cell-cycle phase filter, ``"All"`` for no filter.
        classifier_column: CellView classifier column, empty for no filter.
        classifier_class: Class value within that column, empty for all.
        n_crops: Requested crop count; 0 means "use the metadata default".
    """

    plate_id: int = 1
    well: str = "A1"
    image_input: str = "All"
    timepoint: int = 0
    cellcycle: str = "All"
    classifier_column: str = ""
    classifier_class: str = ""
    n_crops: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectLoadState":
        """Build from a stored dict, ignoring unknown or malformed fields.

        Restoring form state must never be able to break the dialog, so
        anything unparseable falls back to that field's default rather
        than raising.

        Args:
            data: Previously stored state (possibly from an older version).

        Returns:
            A state object with every field either restored or defaulted.
        """
        if not isinstance(data, dict):
            return cls()

        defaults = cls()

        def _int(key: str, fallback: int) -> int:
            try:
                return int(data[key])
            except (KeyError, TypeError, ValueError):
                return fallback

        def _str(key: str, fallback: str) -> str:
            value = data.get(key, fallback)
            return value if isinstance(value, str) else fallback

        return cls(
            plate_id=_int("plate_id", defaults.plate_id),
            well=_str("well", defaults.well),
            image_input=_str("image_input", defaults.image_input),
            timepoint=_int("timepoint", defaults.timepoint),
            cellcycle=_str("cellcycle", defaults.cellcycle),
            classifier_column=_str(
                "classifier_column", defaults.classifier_column
            ),
            classifier_class=_str(
                "classifier_class", defaults.classifier_class
            ),
            n_crops=_int("n_crops", defaults.n_crops),
        )


def state_path(classifier_name: str, base_dir: Path | None = None) -> Path:
    """Path of the state file for one classifier.

    Args:
        classifier_name: Classifier name.
        base_dir: Training-data root; defaults to
            ``~/omeroscreen_trainingdata``.

    Returns:
        Path to the classifier's ``direct_load_state.json``.
    """
    root = base_dir or (Path.home() / "omeroscreen_trainingdata")
    return root / classifier_name / STATE_FILENAME


def load_state(
    classifier_name: str, base_dir: Path | None = None
) -> DirectLoadState:
    """Read the remembered dialog state for a classifier.

    Args:
        classifier_name: Classifier name.
        base_dir: Training-data root; defaults to
            ``~/omeroscreen_trainingdata``.

    Returns:
        The stored state, or a default state when nothing is stored or the
        file is unreadable.
    """
    path = state_path(classifier_name, base_dir)
    if not path.exists():
        return DirectLoadState()

    try:
        with path.open() as f:
            return DirectLoadState.from_dict(json.load(f))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read direct-load state from {path}: {e}")
        return DirectLoadState()


def save_state(
    classifier_name: str,
    state: DirectLoadState,
    base_dir: Path | None = None,
) -> bool:
    """Persist the dialog state for a classifier.

    Args:
        classifier_name: Classifier name.
        state: State to store.
        base_dir: Training-data root; defaults to
            ``~/omeroscreen_trainingdata``.

    Returns:
        True when the state was written. Failures are logged and reported
        as False rather than raised — losing remembered form state must
        never fail a load that otherwise succeeded.
    """
    path = state_path(classifier_name, base_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not save direct-load state to {path}: {e}")
        return False
