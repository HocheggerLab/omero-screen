import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from omero.gateway import (
    BlitzGateway,
    _DatasetWrapper,
    _PlateWrapper,
    _WellWrapper,
)
from omero_screen.constants import OmeroScreenNS
from omero_utils.map_anns import parse_annotations


def get_dataset_id(conn: BlitzGateway, plate_id: int) -> int:
    """Fetch Omero dataset id for the plate.
    Args:
        conn: Omero connection
        plate_id: Plate ID
    Returns:
        int: Dataset ID (or 0 if not found)
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        return 0
    anns = parse_annotations(plate, ns=OmeroScreenNS.DATASET)
    return int(anns.get("Dataset", 0)) if anns else 0


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
    crop_start: tuple[int, ...] = field(default_factory=tuple)
    crop_length: tuple[int, ...] = field(default_factory=tuple)

    # Well data

    well_list: list[_WellWrapper] = field(default_factory=list)
    well_id_list: list[int] = field(default_factory=list)
    well_metadata_list: list[dict[str, Any]] = field(default_factory=list)
    well_ifdata: pl.DataFrame = field(default_factory=pl.DataFrame)
    well_image_index: list[int] = field(default_factory=list)

    # Image data
    images: NDArray[Any] = field(default_factory=lambda: np.empty((0,)))
    image_ids: list[int] = field(default_factory=list)
    image_positions: list[tuple[float, float] | None] = field(
        default_factory=list
    )
    labels: NDArray[Any] = field(default_factory=lambda: np.empty((0,)))

    # Stitched images
    stitched_images: NDArray[Any] = field(
        default_factory=lambda: np.empty((0,))
    )

    # gallery data
    cropped_images: list[NDArray[Any]] = field(default_factory=list)
    cropped_labels: list[NDArray[Any]] = field(default_factory=list)
    cropped_cell_meta: list[dict[str, Any]] = field(default_factory=list)
    selected_images: list[NDArray[Any]] = field(default_factory=list)
    selected_labels: list[NDArray[Any]] = field(default_factory=list)
    selected_crops: list[NDArray[Any]] = field(default_factory=list)
    selected_classes: list[str] = field(default_factory=list)
    selected_cell_meta: list[dict[str, Any]] = field(default_factory=list)
    session_file_path: Optional[Path] = field(default=None)

    def reset(self) -> None:
        self.well_pos_list = []
        self.image_input = ""
        self.image_index = []
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
        self.crop_start = ()
        self.crop_length = ()
        self.well_list = []
        self.well_id_list = []
        self.well_metadata_list = []
        self.well_ifdata = pl.DataFrame()
        self.images = np.empty((0,))
        self.image_ids = []
        self.image_positions = []
        self.labels = np.empty((0,))
        self.stitched_images = np.empty((0,))
        self.cropped_images = []
        self.cropped_labels = []
        self.cropped_cell_meta = []
        self.selected_images = []
        self.selected_labels = []
        self.selected_crops = []
        self.selected_classes = []
        self.selected_cell_meta = []
        self.session_file_path = None

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
        self.image_positions = []
        self.labels = np.empty((0,))


class OmeroConnection:
    """Provide a connection to OMERO.

    Connection credentials are obtained from the environment.
    """

    def __init__(self) -> None:
        self.username = os.getenv("USERNAME")
        self.password = os.getenv("PASSWORD")
        self.host = os.getenv("HOST")
        self.conn: BlitzGateway | None = None

    def get_conn(self) -> BlitzGateway:
        """Get the current connection to OMERO, creating one if required.

        The connection is stored for reuse.

        Returns:
            OMERO connection

        Raises:
            RuntimeError: if a required connection cannot be opened.
        """
        conn = self.conn
        if conn is None:
            conn = self.create_conn()
            self.conn = conn
        return conn

    def close(self, hard: bool = True) -> None:
        """Close the current connection.

        Args:
            hard: If true perform a hard close.
        """
        conn = self.conn
        if conn is not None:
            self.conn = None
            conn.close(hard=hard)

    def create_conn(self) -> BlitzGateway:
        """Create a new connection to OMERO.

        Returns:
            OMERO connection

        Raises:
            RuntimeError: if a connection cannot be opened.
        """
        conn = BlitzGateway(self.username, self.password, host=self.host)
        conn.connect()
        if not conn.isConnected():
            raise RuntimeError(
                f"Failed to establish connection to OMERO server at {self.host} as {self.username}"
            )
        conn.c.enableKeepAlive(60)
        return conn
