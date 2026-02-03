from dataclasses import dataclass, field
from typing import Any

from omero_screen.config import get_logger

logger = get_logger(__name__)


@dataclass
class UserData:
    well: str = field(default_factory=str)
    segmentation: str = field(default_factory=str)
    reload: bool = field(default_factory=bool)
    crop_size: int = field(default_factory=int)
    cellcycle: str = field(default_factory=str)
    timepoint: int = field(default_factory=int)
    columns: int = field(default_factory=int)
    rows: int = field(default_factory=int)
    contour: bool = field(default_factory=bool)
    no_background: bool = True
    channels: list[str] = field(default_factory=list)

    def populate_from_dict(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug("Updated %s to %s", key, value)
            else:
                logger.error(
                    "Error: %s is not a valid attribute of UserData", key
                )

    def reset(self) -> None:
        """Reset to default values."""
        self.__init__()  # type: ignore[misc]
