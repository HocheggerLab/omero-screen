"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""

import os
from typing import Any, Optional

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.viewer import Viewer
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from vispy.color import Colormap

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.omero_image import cache_plate_images
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
    if cache:
        cache_plate(int(plate_id))

    try:
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


def cache_plate(plate_id: int, conn: BlitzGateway | None = None) -> None:
    # Note: Cannot use omero_connect as the worker is asynchronous
    # and the connection is cleaned up after this function exits.
    # Have to create a connection manually and then cleanup when
    # the worker terminates.
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    conn = BlitzGateway(username, password, host=host)
    conn.connect()
    if not conn.isConnected():
        raise RuntimeError(
            f"Failed to establish connection to OMERO server at {host} as {username}"
        )

    # Create a method to close the connection on return.
    def close_conn(nbytes: int) -> None:
        conn.close(hard=True)

    worker = create_worker(cache_plate_images, conn, plate_id)
    worker.returned.connect(close_conn)
    worker.start()


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
