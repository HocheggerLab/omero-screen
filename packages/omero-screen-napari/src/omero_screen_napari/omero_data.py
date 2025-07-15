import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from omero.gateway import _DatasetWrapper, _PlateWrapper, _WellWrapper
from omero_screen.config import get_logger

# Initialize logger with the module's name
logger = get_logger(__name__)


def get_project_id() -> int:
    """_summary_
    Fetch Omero PROJECT_ID from the environment, converting it to int if necessary
    Returns:
        int: project id to find flat field masks and segmentation directory in Omero
    """
    default_project_id = 0
    return int(os.getenv("PROJECT_ID", default_project_id))


def get_data_path() -> Path:
    """_summary_
    Fetch data_path from the environment
    Returns:
        Path: path to folder that saves the csv data to avoid reloading from the server.
    """
    default_data_path = "default_data_path"
    return Path.home() / Path(os.getenv("DATA_PATH", default_data_path))


@dataclass
class OmeroData:
    """
    Dataclass to store all the data related to the omero project and plate.
    """

    # User Input
    well_pos_list: list[str] = field(default_factory=list)
    image_input: str = field(default_factory=str)
    image_index: list[int] = field(default_factory=list)
    # Screen data
    project_id: int = field(default_factory=get_project_id)
    screen_dataset: _DatasetWrapper = field(
        default_factory=_DatasetWrapper
    )  # dataset with flatfield masks and segementations
    plate_id: int = field(default_factory=int)
    plate_name: str = field(default_factory=str)
    plate: _PlateWrapper = field(default_factory=_PlateWrapper)
    plate_data: pl.LazyFrame = field(default_factory=pl.LazyFrame)
    data_path: Path = field(default_factory=get_data_path)
    csv_path: Path = field(default_factory=Path)
    flatfield_masks: NDArray[Any] = field(
        default_factory=lambda: np.empty((0,))
    )
    pixel_size: Optional[tuple[float, float]] = field(default=None)
    channel_data: dict[str, Any] = field(default_factory=dict)
    intensities: dict[int, Any] = field(default_factory=dict)

    # Well data

    well_list: list[_WellWrapper] = field(default_factory=list)
    well_id_list: list[int] = field(default_factory=list)
    well_metadata_list: list[dict[str, Any]] = field(default_factory=list)
    well_ifdata: pl.DataFrame = field(default_factory=pl.DataFrame)
    well_image_index: list[int] = field(default_factory=list)

    # Image data
    images: NDArray[Any] = field(default_factory=lambda: np.empty((0,)))
    image_ids: list[int] = field(default_factory=list)
    labels: NDArray[Any] = field(default_factory=lambda: np.empty((0,)))

    # Stitched images
    stitched_images: NDArray[Any] = field(
        default_factory=lambda: np.empty((0,))
    )

    # gallery data
    cropped_images: list[NDArray[Any]] = field(default_factory=list)
    cropped_labels: list[NDArray[Any]] = field(default_factory=list)
    selected_images: list[NDArray[Any]] = field(default_factory=list)
    selected_labels: list[NDArray[Any]] = field(default_factory=list)
    selected_crops: list[NDArray[Any]] = field(default_factory=list)
    selected_classes: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.well_pos_list = []
        self.image_input = ""
        self.image_index = []
        self.project_id = get_project_id()
        self.screen_dataset = _DatasetWrapper()
        self.plate_id = 0
        self.plate_name = ""
        self.plate = _PlateWrapper()
        self.plate_data = pl.LazyFrame()
        self.data_path = get_data_path()
        self.csv_path = Path()
        self.flatfield_masks = np.empty((0,))
        self.pixel_size = None
        self.channel_data = {}
        self.intensities = {}
        self.well_list = []
        self.well_id_list = []
        self.well_metadata_list = []
        self.well_ifdata = pl.DataFrame()
        self.images = np.empty((0,))
        self.image_ids = []
        self.labels = np.empty((0,))
        self.stitched_images = np.empty((0,))
        self.cropped_images = []
        self.cropped_labels = []
        self.selected_images = []
        self.selected_labels = []
        self.selected_crops = []
        self.selected_classes = []

    def reset_well_and_image_data(self) -> None:
        """
        Resets the well and image data to their default states.
        """
        self.well_list = []
        self.well_id_list = []
        self.well_metadata_list = []
        self.well_ifdata = pl.DataFrame()
        self.well_image_index = []
        self.images = np.empty((0,))
        self.image_ids = []
        self.labels = np.empty((0,))
