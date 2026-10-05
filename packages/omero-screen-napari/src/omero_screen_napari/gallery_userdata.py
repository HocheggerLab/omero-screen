from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class UserData:
    well: str = field(default_factory=str)
    segmentation: str = field(default_factory=str)
    reload: bool = field(default_factory=bool)
    crop_size: int = field(default_factory=int)
    cellcycle: str = field(default_factory=str)
    classifier_filter: str = field(default_factory=str)
    # Exact ``classifier_*`` column for ``classifier_filter``. Empty keeps the
    # old behaviour: the first classifier column containing the value.
    classifier_column: str = field(default_factory=str)
    timepoint: int = field(default_factory=int)
    columns: int = field(default_factory=int)
    rows: int = field(default_factory=int)
    contour: bool = field(default_factory=bool)
    no_background: bool = True
    # Figure panels are usually placed with their own caption, so the
    # verbose well/metadata/settings title is optional.
    show_title: bool = True
    channels: list[str] = field(default_factory=list)
    # Seed for the crop sampling, so a gallery can be reproduced. ``None``
    # samples from the global ``random`` state (the interactive default).
    seed: int | None = None

    def populate_from_dict(self, data: dict[str, Any]) -> None:
        """Replace all fields from ``data``, resetting anything not present.

        All three call sites (gallery dialog, classifier-metadata load,
        session-metadata load) supply complete dicts produced via
        ``dataclasses.asdict``, so reset-before-populate gives them a
        fresh snapshot. Without the reset, a previous load's
        ``cellcycle`` / ``channel_data`` / etc. would linger when a key
        was missing — a recurring cross-contamination bug pattern.
        """
        self.reset()
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} to {value}")
            else:
                logger.error(
                    f"Error: {key} is not a valid attribute of UserData"
                )

    def reset(self) -> None:
        """Reset to default values."""
        self.__init__()  # type: ignore[misc]
