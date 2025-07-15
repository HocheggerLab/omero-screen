"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""


# Logging

from typing import Any, Optional

import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari.viewer import Viewer
from omero_screen.config import get_logger
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import parse_omero_data

# Initialize logger with the module's name
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


# Widget to call Omero and load well images


@magic_factory(call_button="Enter")
def welldata_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos_list: str = "Well Position",
    images: str = "All",
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    print(
        f"Starting welldata_widget with plate_id={plate_id}, well_pos_list={well_pos_list}, images={images}"
    )
    parse_omero_data(omero_data, plate_id, well_pos_list, images)
    print(
        f"After parse_omero_data: omero_data.images.shape={omero_data.images.shape}"
    )
    clear_viewer_layers(viewer)
    add_image_to_viewer(viewer)
    set_color_maps(viewer)
    add_label_layers(viewer)

    def slider_position_change(event: Any) -> None:
        current_position = event.source.current_step[0]
        handle_metadata_widget(viewer, current_position)

    viewer.dims.events.current_step.connect(slider_position_change)
    _initial_position = viewer.dims.current_step[0]
    mock_event = MockEvent(viewer.dims)
    slider_position_change(mock_event)


def clear_viewer_layers(viewer: Viewer) -> None:
    while len(viewer.layers) > 0:
        viewer.layers.pop(0)


def add_image_to_viewer(viewer: Viewer) -> None:
    num_channels = omero_data.images.shape[-1]
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
    well_index = slider_position // images_per_well
    well_index = min(well_index, len(omero_data.well_metadata_list) - 1)

    if metadata_widget is not None:
        viewer.window.remove_dock_widget(metadata_widget)  # type: ignore
    well_metadata = omero_data.well_metadata_list[well_index]
    metadata_widget = MetadataWidget(well_metadata)
    viewer.window.add_dock_widget(metadata_widget)


def set_color_maps(viewer: Viewer) -> None:
    channel_names: dict[str, int] = {
        key: int(value) for key, value in omero_data.channel_data.items()
    }
    color_maps: dict[str, str] = _generate_color_map(channel_names)
    for name, index in channel_names.items():
        layer = viewer.layers[index]
        layer.name = name
        layer.colormap = color_maps[name]


def add_label_layers(viewer: Viewer) -> None:
    scale = omero_data.pixel_size
    print(f"The labels shape loaded in line 159 is {omero_data.labels.shape}")
    if omero_data.labels.shape[-1] == 1:
        viewer.add_labels(
            np.squeeze(omero_data.labels).astype(int),
            name="Nuclei Masks",
            scale=scale,
        )
    elif omero_data.labels.shape[-1] == 2:
        channel_1_masks = omero_data.labels[..., 0].astype(int)
        channel_2_masks = omero_data.labels[..., 1].astype(int)
        viewer.add_labels(channel_1_masks, name="Nuclei Masks", scale=scale)
        viewer.add_labels(channel_2_masks, name="Cell Masks", scale=scale)
    else:
        raise ValueError("Invalid segmentation label shape")


def _generate_color_map(channel_names: dict[str, int]) -> dict[str, str]:
    """
    Generate a color map dictionary for the channels
    :param channel_names: dictionary of channel names and indices
    :return: a dictionary with channel names as keys and color names as values
    """
    # Initialize the color map dictionary
    color_map_dict = {}

    # Determine the number of channels
    num_channels = len(channel_names)

    # If there is only one channel, set it to gray
    if num_channels == 1:
        single_key = list(channel_names.keys())[0]
        color_map_dict[single_key] = "gray"
        return color_map_dict

    color_map_dict = {"DAPI": "blue"}

    # Default color assignments for Tub and EdU
    special_channels = {"Tub": "green", "EdU": "gray"}

    # Determine remaining color options based on presence of Tub and EdU
    remaining_colors = ["red"]
    if "Tub" not in channel_names:
        remaining_colors.append("green")
    if "EdU" not in channel_names:
        remaining_colors.append("gray")

    # Assign colors to Tub and EdU if present
    for channel in special_channels:
        if channel in channel_names:
            color_map_dict[channel] = special_channels[channel]

    # Assign remaining colors to any other channels
    remaining_channels = [
        channel for channel in channel_names if channel not in color_map_dict
    ]
    for channel, color in zip(
        remaining_channels, remaining_colors, strict=False
    ):
        color_map_dict[channel] = color

    return color_map_dict
