"""This module handles the widget to call Omero and load all images from a single well
from multiple aligned plates into napari.
The plugin can be run from napari as Aligned Plate Widget under Plugins.
"""

import re

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image
from napari.viewer import Viewer
from omero_screen.config import get_logger

from omero_screen_napari._welldata_widget import (
    add_label_layers,
    clear_viewer_layers,
    set_color_maps,
)
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import (
    get_plate_alignments,
    parse_omero_data,
)

logger = get_logger(__name__)


def aligned_plate_widget_gui() -> Container:  # type: ignore[type-arg]
    """This function combines the widgets into a single widget."""
    # Call the magic factories to get the widget instances
    aligned_plate_widget_instance = aligned_plate_widget()
    return Container(
        widgets=[
            aligned_plate_widget_instance,
        ]
    )


# Widget to call Omero and load well images
@magic_factory(call_button="Enter")
def aligned_plate_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos: str = "Well Position",
    image: int = 0,
    sample_alignments: bool = False,
) -> None:
    """This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.

    For aligned plates, the primary plate's agg_data.csv contains all channel data
    from all aligned plates, so we only need to import cellview data once.
    """
    # Single well only
    if not re.match("^[A-Z]+[0-9]+$", well_pos):
        raise ValueError("Invalid well position: " + well_pos)

    # Get alignment for the plate
    alignments = get_plate_alignments(
        int(plate_id), sample_alignments=sample_alignments
    )
    plates = alignments["plate"].unique()
    logger.info("Loaded alignments for plates: %s", plates)

    all_channels: set[str] = set()

    # Load primary plate with cellview data (includes all channel data from aligned plates)
    parse_omero_data(omero_data, plate_id, well_pos, str(image))
    clear_viewer_layers(viewer)
    _add_image_to_viewer(viewer, all_channels)
    labels = omero_data.labels

    all_channels = set(omero_data.channel_data.keys())

    # Load aligned plates (images only, skip cellview import since data already loaded)
    for plate_other in plates:
        # Get the images (not the labels, and skip cellview since already imported)
        parse_omero_data(
            omero_data,
            str(plate_other),
            well_pos,
            str(image),
            options=["skip_cellview"],
        )
        # Translate
        mask = (alignments["well"] == well_pos) & (
            alignments["plate"] == plate_other
        )
        if sample_alignments:
            mask = mask & (alignments["image_id"] == omero_data.image_ids[0])
        df = alignments[mask]
        if df.empty:
            raise Exception(
                f"Plate {plate_other} is missing alignment for well: {well_pos}"
            )
        # Translation maps plate_id to plate_other so negate
        trans = (-df.iloc[0]["x"], -df.iloc[0]["y"])
        logger.info("Plate %d %s translation %s", plate_other, well_pos, trans)

        # Filter channels already added to the viewer (e.g duplicate alignment channel)
        _add_image_to_viewer(viewer, all_channels, trans)

    set_color_maps(viewer)

    add_label_layers(viewer, labels)


def _add_image_to_viewer(
    viewer: Viewer,
    all_channels: set[str],
    trans: tuple[float, float] | None = None,
) -> None:
    num_channels = omero_data.images.shape[-1]
    logger.debug(
        "The images shape is %s (%s)",
        omero_data.images.shape,
        omero_data.images.dtype,
    )
    channel_names: dict[int, str] = {
        int(value): key for key, value in omero_data.channel_data.items()
    }
    # Create translation
    translate = None
    if trans and omero_data.pixel_size is not None:
        # A 1-D array of factors to shift each axis by. X is the last axis.
        # Translation is broadcast to 0 in leading dimensions.
        # Scaling is applied before translation so we must scale the translation.
        translate = [
            trans[1] * omero_data.pixel_size[1],
            trans[0] * omero_data.pixel_size[0],
        ]

    for i in range(num_channels):
        if channel_names[i] in all_channels:
            continue
        all_channels.add(channel_names[i])
        image_data = omero_data.images[..., i]
        layer = viewer.add_image(
            image_data, scale=omero_data.pixel_size, translate=translate
        )
        assert isinstance(layer, Image), (
            "Expected layer to be an instance of Image"
        )
        layer.contrast_limits_range = (0, 65535)
        layer.contrast_limits = (np.min(image_data), np.max(image_data))
        layer.blending = "additive"
        layer.name = channel_names[i]

    # Configure the scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
