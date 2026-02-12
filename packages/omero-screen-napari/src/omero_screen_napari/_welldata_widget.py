"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""

import contextlib
import os
from typing import Any, Optional

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils import progress as napari_progress
from napari.viewer import Viewer
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from vispy.color import Colormap

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.plate_cache import (
    cache_plate as cache_plate_full,
)
from omero_screen_napari.plate_cache import (
    is_plate_cached,
    load_from_cache,
)
from omero_screen_napari.welldata_api import (
    parse_omero_data,
    stitch_images,
    stitch_labels,
)

# Logging
logger = get_logger(__name__)


class MetadataWidget(QWidget):  # type: ignore
    """
    A custom QWidget that displays metadata in a QLabel. The metadata is displayed as key-value pairs.

    Inherits from:
    QWidget: Base class for all user interface objects in PyQt5.

    Attributes:
    layout (QVBoxLayout): Layout for the widget.
    label (QLabel): Label where the metadata is displayed.

    Args:
    metadata (dict): The metadata to be displayed. It should be a dictionary where the keys are the metadata
    fields and the values are the metadata values.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__()
        self._layout = QVBoxLayout()
        self.label = QLabel()
        label_text = "".join(
            f"{key}: {value}\n" for key, value in metadata.items()
        )
        self.label.setText(
            label_text.rstrip()
        )  # Remove the last newline character

        self._layout.addWidget(self.label)
        self.setLayout(self._layout)


# Mock event object with the current_step attribute
class MockEvent:
    def __init__(self, source: Any) -> None:
        self.source = source


# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[MetadataWidget] = None

# Combine Welldata and Stiched data widgets


def well_widget_combined() -> Container:  # type: ignore
    """
    This function combines the well and stitched data widgets into a single widget.
    """
    # Call the magic factories to get the widget instances
    welldata_widget_instance = welldata_widget()
    stitched_data_widget_instance = stitched_data_widget()
    return Container(
        widgets=[
            welldata_widget_instance,
            stitched_data_widget_instance,
        ]
    )


# Widget to call Omero and load well images


@magic_factory(call_button="Enter")
def welldata_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos_list: str = "Well Position",
    images: str = "All",
    time: str = "All",
    cache: bool = False,
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    plate_num = int(plate_id)

    if cache:
        start_cache_worker(plate_num)

    try:
        # Fast path: load from cache without OMERO connection
        if is_plate_cached(plate_num):
            logger.info("Loading plate %d from cache (fast path)", plate_num)
            load_from_cache(
                omero_data, plate_num, well_pos_list, images, time=time
            )
        else:
            parse_omero_data(
                omero_data, plate_id, well_pos_list, images, time=time
            )
        clear_viewer_layers(viewer)
        add_image_to_viewer(viewer)
        set_color_maps(viewer)
        add_label_layers(viewer)

        def slider_position_change(event: Any) -> None:
            current_position = event.source.current_step[0]
            handle_metadata_widget(viewer, current_position)

        viewer.dims.events.current_step.connect(slider_position_change)
        # _initial_position = viewer.dims.current_step[0]
        mock_event = MockEvent(viewer.dims)
        slider_position_change(mock_event)
    except Exception as e:
        logger.error(f"Error in welldata_widget: {e}")
        # MessageBox is already shown in parse_omero_data if it failed there
        # But we catch other potential errors here
        if "ValueError" not in str(
            type(e)
        ):  # Avoid double message for the common ValueError
            from qtpy.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"An unexpected error occurred: {e}")
            msg.setWindowTitle("Widget Error")
            msg.exec_()


_active_cache_worker: Any = None
_active_cache_plate_id: int | None = None


def start_cache_worker(plate_id: int) -> None:
    """Start background caching for a plate using plate_cache.

    Downloads all metadata, flatfield-corrected images, and labels
    so that subsequent well navigation is fully offline.  Shows a
    progress bar in the napari activity dock.

    Skips if a worker is already running for the same plate or
    if the plate is already fully cached.
    """
    global _active_cache_worker, _active_cache_plate_id

    # Already fully cached — skip
    if is_plate_cached(plate_id):
        logger.info(
            "Plate %d already cached — skipping background download",
            plate_id,
        )
        return

    # Already caching this plate — skip
    if (
        _active_cache_worker is not None
        and _active_cache_worker.is_running
        and _active_cache_plate_id == plate_id
    ):
        logger.info(
            "Cache worker already running for plate %d — skipping",
            plate_id,
        )
        return

    # Different plate or finished — cancel old worker if still running
    if _active_cache_worker is not None and _active_cache_worker.is_running:
        logger.info(
            "Cancelling cache worker for plate %d",
            _active_cache_plate_id,
        )
        _active_cache_worker.quit()

    # Create a connection manually (worker is async, can't use decorator)
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    conn = BlitzGateway(username, password, host=host)
    conn.connect()
    if not conn.isConnected():
        raise RuntimeError(
            f"Failed to establish connection to OMERO server at {host} as {username}"
        )

    # Napari progress bar — created lazily on the first progress signal
    # when we know the total count.
    pbr: list[Any] = []  # mutable container so closures can share it
    _prev_done = 0

    def on_finished() -> None:
        global _active_cache_worker
        conn.close(hard=True)
        _active_cache_worker = None
        if pbr:
            pbr[0].set_description(f"Plate {plate_id} cached")
            pbr[0].close()
        logger.info("Cache worker finished for plate %d", plate_id)

    def on_error(exc: BaseException) -> None:
        global _active_cache_worker
        conn.close(hard=True)
        _active_cache_worker = None
        if pbr:
            pbr[0].set_description(f"Cache error: {exc}")
            pbr[0].close()
        logger.error("Cache worker error for plate %d: %s", plate_id, exc)

    def on_progress(prog: tuple[int, int]) -> None:
        nonlocal _prev_done
        done, total = prog
        if total <= 0:
            return
        # Create the bar on first real signal so total is correct from the start
        if not pbr:
            pbr.append(
                napari_progress(total=total, desc=f"Caching plate {plate_id}")
            )
        delta = done - _prev_done
        if delta > 0:
            pbr[0].update(delta)
        _prev_done = done

    worker = create_worker(cache_plate_full, plate_id, conn, max_workers=3)
    worker.yielded.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.errored.connect(on_error)
    worker.start()

    _active_cache_worker = worker
    _active_cache_plate_id = plate_id
    logger.info("Started cache worker for plate %d", plate_id)


def clear_viewer_layers(viewer: Viewer) -> None:
    while len(viewer.layers) > 0:
        viewer.layers.pop(0)


def add_image_to_viewer(viewer: Viewer) -> None:
    num_channels = omero_data.images.shape[-1]
    logger.debug(
        f"The images shape is {omero_data.images.shape} ({omero_data.images.dtype})"
    )
    channel_names: dict[int, str] = {
        int(value): key for key, value in omero_data.channel_data.items()
    }
    for i in range(num_channels):
        image_data = omero_data.images[..., i]
        layer = viewer.add_image(image_data, scale=omero_data.pixel_size)
        assert isinstance(layer, Image), (
            "Expected layer to be an instance of Image"
        )
        layer.contrast_limits_range = (0, 65535)
        specific_intensities = omero_data.intensities[i]
        layer.contrast_limits = specific_intensities
        layer.blending = "additive"
        layer.events.contrast_limits.connect(on_contrast_change)
        layer.name = channel_names[i]

    # Configure the scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"


def on_contrast_change(event: Any) -> None:
    """
    Event handler for changes in contrast limits.

    Parameters:
    - event: The event object containing information about the change.
    """
    # Access the layer through the event's source attribute
    layer = event.source
    channel_number = int(omero_data.channel_data[layer.name])
    omero_data.intensities[channel_number] = tuple(layer.contrast_limits)


def handle_metadata_widget(viewer: Viewer, slider_position: int) -> None:
    global metadata_widget

    # Calculate which well's metadata to use based on the slider position
    images_per_well = len(omero_data.image_index)
    if images_per_well == 0:
        return
    well_index = slider_position // images_per_well
    if not omero_data.well_metadata_list:
        return
    well_index = min(well_index, len(omero_data.well_metadata_list) - 1)

    if metadata_widget is not None:
        with contextlib.suppress(LookupError):
            viewer.window.remove_dock_widget(metadata_widget)  # type: ignore
    well_metadata = omero_data.well_metadata_list[well_index]
    metadata_widget = MetadataWidget(well_metadata)
    viewer.window.add_dock_widget(metadata_widget)


def set_color_maps(viewer: Viewer) -> None:
    channel_names = [layer.name for layer in viewer.layers]
    color_maps: list[str | Colormap] = _generate_color_map(channel_names)
    for i, c in enumerate(color_maps):
        viewer.layers[i].colormap = c


def add_label_layers(
    viewer: Viewer, labels: Optional[np.ndarray[Any, np.dtype[Any]]] = None
) -> None:
    scale = omero_data.pixel_size
    if labels is None:
        labels = omero_data.labels
    if labels is None or labels.size == 0:
        return
    logger.debug(f"The labels shape is {labels.shape} ({labels.dtype})")
    if labels.shape[-1] == 1:
        viewer.add_labels(
            np.squeeze(labels).astype(int),
            name="Nuclei Masks",
            scale=scale,
        )
    elif labels.shape[-1] == 2:
        channel_1_masks = labels[..., 0].astype(int)
        channel_2_masks = labels[..., 1].astype(int)
        viewer.add_labels(channel_1_masks, name="Nuclei Masks", scale=scale)
        viewer.add_labels(channel_2_masks, name="Cell Masks", scale=scale)
    else:
        raise ValueError("Invalid segmentation label shape")


def _generate_color_map(channel_names: list[str]) -> list[str | Colormap]:
    """
    Generate a list of color maps for the channels
    :param channel_names: channel names
    :return: color maps
    """
    # Napari supports vispy or matplotlib colormap names

    # Determine the number of channels
    num_channels = len(channel_names)

    if num_channels == 1:
        return ["gray"]

    # Default channel color assignments
    special_channels = {"DAPI": "blue", "Tub": "green", "EdU": "gray"}

    # Other color assignments. This list is used in reverse order amd repeated as required.
    # Supports using a Colormap. This requires the RBG value of the final color.
    remaining_colors: list[str | Colormap] = [
        "gray",
        Colormap(["black", "#ff2f92"]),  # strawberry
        Colormap(["black", "#8efa00"]),  # lime
        Colormap(["black", "#009193"]),  # teal
        Colormap(["black", "#00fdff"]),  # turquoise
        Colormap(["black", "#aa7942"]),  # brown
        Colormap(["black", "#ffc0cb"]),  # pink
        "bop orange",
        "bop blue",
        "bop purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "red",
    ]
    # Do not run out of colours
    remaining_colors.extend(
        remaining_colors * (num_channels // len(remaining_colors))
    )
    logger.debug(f"Remaining colors: {remaining_colors}")

    return [
        special_channels[ch]
        if ch in special_channels
        else remaining_colors.pop()
        for ch in channel_names
    ]


@magic_factory(call_button="Enter")
def stitched_data_widget(
    viewer: Viewer,
    rotation: float = 0.15,
    overlap_x: int = 7,
    overlap_y: int = 7,
    edge: int = 7,
    mode: str = "reflect",
) -> None:
    clear_viewer_layers(viewer)
    stitched_images = stitch_images(
        omero_data,
        rotation=rotation,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        edge=edge,
        mode=mode,
    )
    logger.debug(
        f"Stitched shape {stitched_images.shape} ({stitched_images.dtype})"
    )
    names = ["Stitched Image"] * len(omero_data.channel_data)
    for k, v in omero_data.channel_data.items():
        names[int(v)] = k
    viewer.add_image(
        stitched_images,
        contrast_limits=list(omero_data.intensities[0]),
        gamma=1,
        channel_axis=-1,
        scale=omero_data.pixel_size,
        name=names,
    )
    set_color_maps(viewer)
    if len(omero_data.labels):
        stitched_labels = stitch_labels(
            omero_data,
            rotation=rotation,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
        )
        add_label_layers(viewer, labels=stitched_labels)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
    viewer.scale_bar.color = "white"
