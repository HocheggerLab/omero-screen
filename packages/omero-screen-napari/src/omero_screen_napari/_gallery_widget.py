import logging

from magicgui import magic_factory
from magicgui.widgets import Container
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_api import run_gallery_parser, show_gallery
from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data_singleton import omero_data

logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)


def gallery_gui_widget() -> Container:  # type: ignore
    # Call the magic factories to get the widget instances
    gallery_widget_instance = gallery_widget()
    reset_widget_instance = reset_widget()
    analysis_widget_instance = run_analysis_widget()
    return Container(
        widgets=[
            gallery_widget_instance,
            reset_widget_instance,
            analysis_widget_instance,
        ]
    )


@magic_factory(
    call_button="Enter",
)
def reset_widget() -> None:
    omero_data.cropped_images = []
    omero_data.cropped_labels = []


@magic_factory(
    call_button="Enter",
)
def run_analysis_widget(wells: str, galleries: int) -> None:
    well_list = wells.split(", ")
    run_gallery_parser(omero_data, userdata, well_list, galleries)


@magic_factory(
    call_button="Enter",
    segmentation={"choices": ["nucleus", "cell"]},
    crop_size={"choices": [20, 30, 50, 100, 200]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def gallery_widget(
    # viewer: "napari.viewer.Viewer",
    well: str = "",  # Magicgui doesn't support dynamic defaults easily, handled in body
    *,  # allow non default arguments after this
    segmentation: str,
    crop_size: int,
    cellcycle: str,
    timepoint: int = 0,
    columns: int = 4,
    rows: int = 4,
    reload: bool = True,
    contour: bool = True,
    blue_channel: str = "DAPI",
    green_channel: str = "Tub",
    red_channel: str = "EdU",
) -> None:
    channels = [blue_channel, green_channel, red_channel]  # to match rgb order
    channels = [channel for channel in channels if channel != ""]
    if not well and omero_data.well_pos_list:
        well = omero_data.well_pos_list[0]
        logger.info("Using default well: %s", well)

    if not well:
        logger.warning(
            "No well selected. Please enter a valid well identifier."
        )
        return

    user_data_dict = {
        "well": well,
        "segmentation": segmentation,
        "reload": reload,
        "crop_size": crop_size,
        "cellcycle": cellcycle,
        "timepoint": timepoint,
        "columns": columns,
        "rows": rows,
        "contour": contour,
        "channels": channels,
    }
    try:
        # UserData.set_omero_data_channel_keys(omero_data.channel_data.keys())
        userdata.populate_from_dict(user_data_dict)
        show_gallery(omero_data, userdata)
    except ValueError as e:
        logger.error("Gallery Error: %s", e)
        QMessageBox.critical(None, "Gallery Error", str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected Error: %s", e)
        QMessageBox.critical(None, "Unexpected Error", str(e))
