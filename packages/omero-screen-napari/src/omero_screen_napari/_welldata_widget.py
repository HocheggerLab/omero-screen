"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""

import contextlib
import os
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari.qt.threading import create_worker
from napari.utils import progress as napari_progress
from napari.viewer import Viewer
from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from qtpy.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vispy.color import Colormap

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.plate_cache import (
    cache_plate as cache_plate_full,
)
from omero_screen_napari.plate_cache import (
    get_all_cached_plates,
    is_plate_cached,
    is_plate_fully_cached,
    load_from_cache,
)
from omero_screen_napari.position_stitching import (
    has_valid_positions,
    stitch_from_positions,
    stitch_labels_from_positions,
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


class CachedPlatesTable(QWidget):  # type: ignore[misc]
    """Compact table showing plates available in the local cache.

    Double-click a row to populate the plate_id field in welldata_widget.

    Args:
        on_plate_selected: Callback receiving the plate_id as string.
        on_resume_cache: Callback receiving the plate_id as int to resume caching.
    """

    def __init__(
        self,
        on_plate_selected: Callable[[str], None] | None = None,
        on_resume_cache: Callable[[int], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_plate_selected = on_plate_selected
        self._on_resume_cache = on_resume_cache

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>Cached Plates</b>"))
        self._resume_btn = QPushButton("Resume")
        self._resume_btn.setFixedWidth(60)
        self._resume_btn.setToolTip("Resume caching for the selected plate")
        self._resume_btn.clicked.connect(self._on_resume_clicked)
        self._resume_btn.setEnabled(False)
        header_row.addWidget(self._resume_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(60)
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Plate ID", "Name", "Status"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMaximumHeight(120)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        self.refresh()

    def refresh(self) -> None:
        """Re-scan the cache and repopulate the table."""
        plates = get_all_cached_plates()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(plates))
        for row, (plate_id, plate_name) in enumerate(plates):
            self.table.setItem(row, 0, QTableWidgetItem(str(plate_id)))
            self.table.setItem(row, 1, QTableWidgetItem(plate_name))
            fully = is_plate_fully_cached(plate_id)
            status_item = QTableWidgetItem("Cached" if fully else "Partial")
            self.table.setItem(row, 2, status_item)
        self.table.setSortingEnabled(True)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        """Enable Resume button only when a partial plate is selected."""
        row = self.table.currentRow()
        if row < 0:
            self._resume_btn.setEnabled(False)
            return
        status_item = self.table.item(row, 2)
        self._resume_btn.setEnabled(
            status_item is not None and status_item.text() == "Partial"
        )

    def _on_resume_clicked(self) -> None:
        """Resume caching for the selected partial plate."""
        row = self.table.currentRow()
        if row < 0 or self._on_resume_cache is None:
            return
        item = self.table.item(row, 0)
        if item:
            self._on_resume_cache(int(item.text()))

    def _on_double_click(self, row: int, _column: int) -> None:
        """Populate the plate_id field on double-click."""
        if self._on_plate_selected is None:
            return
        item = self.table.item(row, 0)
        if item:
            self._on_plate_selected(item.text())


# Mock event object with the current_step attribute
class MockEvent:
    def __init__(self, source: Any) -> None:
        self.source = source


# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[MetadataWidget] = None

# Reference to the stitch widget so auto-stitch can read its parameters
_stitch_widget_ref: Any = None

# Combine Welldata and Stiched data widgets


_cached_plates_table_ref: CachedPlatesTable | None = None


def well_widget_combined() -> QWidget:  # type: ignore
    """Combine the well and stitched data widgets with a Plate Info button."""
    global _stitch_widget_ref, _cached_plates_table_ref
    from omero_screen_napari._plate_info_dialog import PlateInfoDialog

    welldata_instance = welldata_widget()
    stitched_instance = stitched_data_widget()
    _stitch_widget_ref = stitched_instance

    # Insert "Plate Info" button into the welldata form, right below plate_id
    # Layout order: [0] Enter button, [1] plate_id field, [2] insert here
    plate_info_btn = QPushButton("Plate Info")
    native_layout = welldata_instance.native.layout()
    native_layout.insertWidget(2, plate_info_btn)

    # Cached plates table — double-click populates plate_id field,
    # Resume button restarts caching for partially cached plates
    cached_table = CachedPlatesTable(
        on_plate_selected=lambda pid: setattr(
            welldata_instance.plate_id, "value", pid
        ),
        on_resume_cache=start_cache_worker,
    )
    _cached_plates_table_ref = cached_table

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(cached_table)
    layout.addWidget(welldata_instance.native)
    layout.addWidget(stitched_instance.native)

    plate_info_btn.clicked.connect(
        lambda: _open_plate_info(welldata_instance, PlateInfoDialog, widget)
    )
    return widget


def _open_plate_info(
    welldata_instance: Any,
    dialog_cls: type,
    parent: QWidget,
) -> None:
    """Open the Plate Info dialog for the current plate ID."""
    plate_id_str = welldata_instance.plate_id.value
    try:
        plate_id = int(plate_id_str)
    except (ValueError, TypeError):
        QMessageBox.warning(
            parent, "Invalid Plate ID", "Enter a valid plate ID first."
        )
        return

    def load_callback(well_pos: str) -> None:
        welldata_instance.well_pos_list.value = well_pos
        welldata_instance()

    dialog = dialog_cls(
        plate_id, on_load_callback=load_callback, parent=parent
    )
    dialog.exec_()


# Defaults matching the stitched_data_widget signature (Operetta calibration)
_STITCH_DEFAULTS: dict[str, Any] = {
    "rotation": 0.15,
    "precise_rotation": False,
    "overlap_x": 7,
    "overlap_y": 7,
    "edge": 7,
    "mode": "reflect",
}


def _get_stitch_params() -> dict[str, Any]:
    """Read current stitch parameters from the sibling stitch widget.

    Falls back to Operetta defaults when the widget is not available.
    """
    w = _stitch_widget_ref
    if w is not None:
        try:
            precise = w.precise_rotation.value
            return {
                "rotation": w.rotation.value if precise else 0.0,
                "overlap_x": w.overlap_x.value,
                "overlap_y": w.overlap_y.value,
                "edge": w.edge.value,
                "mode": w.mode.value,
            }
        except AttributeError:
            pass
    defaults = dict(_STITCH_DEFAULTS)
    if not defaults["precise_rotation"]:
        defaults["rotation"] = 0.0
    return defaults


# Widget to call Omero and load well images


def _is_already_loaded(
    omero_data: Any, plate_id: int, well_pos_list: str, images: str
) -> bool:
    """Check whether omero_data already holds the requested plate/wells/images."""
    if omero_data.plate_id != plate_id:
        return False
    if omero_data.images.size == 0:
        return False
    requested_wells = [w.strip() for w in well_pos_list.split(",")]
    if omero_data.well_pos_list != requested_wells:
        return False
    return bool(omero_data.image_input == images)


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
            if _is_already_loaded(
                omero_data, plate_num, well_pos_list, images
            ):
                logger.info(
                    "Plate %d already in memory, skipping reload", plate_num
                )
            else:
                logger.info(
                    "Loading plate %d from cache (fast path)", plate_num
                )
                load_from_cache(
                    omero_data, plate_num, well_pos_list, images, time=time
                )

            n_wells = len(omero_data.well_id_list)
            n_per_well = len(omero_data.image_index)

            # Check first well's positions to decide if stitching is possible
            first_well_pos = omero_data.image_positions[:n_per_well]
            if n_per_well > 0 and has_valid_positions(first_well_pos):
                logger.info(
                    "Auto-stitching %d well(s) from stage positions", n_wells
                )
                sp = _get_stitch_params()
                stitched_imgs: list[np.ndarray[Any, np.dtype[Any]]] = []
                stitched_lbls: list[np.ndarray[Any, np.dtype[Any]]] = []

                for w in range(n_wells):
                    start = w * n_per_well
                    end = start + n_per_well
                    well_images = omero_data.images[start:end]
                    well_positions = omero_data.image_positions[start:end]

                    stitched_imgs.append(
                        stitch_from_positions(
                            well_images,
                            well_positions,  # type: ignore[arg-type]
                            omero_data.pixel_size,  # type: ignore[arg-type]
                            rotation=sp["rotation"],
                            edge=sp["edge"],
                            mode=sp["mode"],
                            fallback_overlap=(
                                sp["overlap_x"],
                                sp["overlap_y"],
                            ),
                        )
                    )

                    if omero_data.labels.size > 0:
                        well_labels = omero_data.labels[start:end]
                        stitched_lbls.append(
                            stitch_labels_from_positions(
                                well_labels,
                                well_positions,  # type: ignore[arg-type]
                                omero_data.pixel_size,  # type: ignore[arg-type]
                                rotation=sp["rotation"],
                                fallback_overlap=(
                                    sp["overlap_x"],
                                    sp["overlap_y"],
                                ),
                            )
                        )

                if n_wells == 1:
                    result_img = stitched_imgs[0]
                    result_lbl = stitched_lbls[0] if stitched_lbls else None
                else:
                    result_img = np.stack(stitched_imgs)
                    result_lbl = (
                        np.stack(stitched_lbls) if stitched_lbls else None
                    )

                clear_viewer_layers(viewer)
                _display_stitched(viewer, result_img, result_lbl)

                # For multi-well, each slider position = one well
                iw_override = 1 if n_wells > 1 else None

                def slider_position_change(event: Any) -> None:
                    pos = event.source.current_step[0]
                    handle_metadata_widget(
                        viewer, pos, images_per_well_override=iw_override
                    )

                viewer.dims.events.current_step.connect(slider_position_change)
                mock_event = MockEvent(viewer.dims)
                slider_position_change(mock_event)
            else:
                clear_viewer_layers(viewer)
                add_image_to_viewer(viewer)
                set_color_maps(viewer)
                add_label_layers(viewer)

                def slider_position_change(event: Any) -> None:
                    pos = event.source.current_step[0]
                    handle_metadata_widget(viewer, pos)

                viewer.dims.events.current_step.connect(slider_position_change)
                mock_event = MockEvent(viewer.dims)
                slider_position_change(mock_event)
        else:
            if _is_already_loaded(
                omero_data, plate_num, well_pos_list, images
            ):
                logger.info(
                    "Plate %d already in memory, skipping reload", plate_num
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

    # Already fully cached (metadata + all images) — skip
    if is_plate_fully_cached(plate_id):
        logger.info(
            "Plate %d fully cached — skipping background download",
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
        if _cached_plates_table_ref is not None:
            _cached_plates_table_ref.refresh()
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


def handle_metadata_widget(
    viewer: Viewer,
    slider_position: int,
    images_per_well_override: int | None = None,
) -> None:
    global metadata_widget

    # Calculate which well's metadata to use based on the slider position
    images_per_well = images_per_well_override or len(omero_data.image_index)
    if images_per_well == 0:
        return
    well_index = slider_position // images_per_well
    if not omero_data.well_metadata_list:
        return
    well_index = min(well_index, len(omero_data.well_metadata_list) - 1)

    if metadata_widget is not None:
        with contextlib.suppress(LookupError):
            viewer.window.remove_dock_widget(metadata_widget)  # type: ignore

    # Include well position in the displayed metadata
    well_metadata: dict[str, Any] = {}
    if well_index < len(omero_data.well_pos_list):
        well_metadata["Well"] = omero_data.well_pos_list[well_index]
    well_metadata.update(omero_data.well_metadata_list[well_index])

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


def _display_stitched(
    viewer: Viewer,
    stitched_images: np.ndarray[Any, np.dtype[Any]],
    stitched_labels: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
) -> None:
    """Display a stitched image in the viewer with channel colouring.

    Handles both single-well (Y, X, C) and multi-well (N, Y, X, C) shapes.
    For multi-well, napari creates a slider for the well dimension.
    """
    num_channels = stitched_images.shape[-1]
    names = ["Stitched Image"] * num_channels
    for k, v in omero_data.channel_data.items():
        idx = int(v)
        if idx < num_channels:
            names[idx] = k

    # Per-channel contrast limits from stored intensities
    per_channel_limits = [
        list(omero_data.intensities.get(i, (0, 65535)))
        for i in range(num_channels)
    ]

    viewer.add_image(
        stitched_images,
        contrast_limits=per_channel_limits,
        gamma=1,
        channel_axis=-1,
        scale=omero_data.pixel_size,
        name=names,
    )
    # Set slider range to full 16-bit so user can adjust beyond auto limits
    for layer in viewer.layers:
        if isinstance(layer, Image):
            layer.contrast_limits_range = (0, 65535)

    set_color_maps(viewer)
    if stitched_labels is not None:
        # Single-well labels are (Y, X, C) — need batch dim for add_label_layers
        # Multi-well labels are (N, Y, X, C) — already have it
        if stitched_labels.ndim == 3:
            stitched_labels = stitched_labels[np.newaxis, ...]
        add_label_layers(viewer, labels=stitched_labels)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"


@magic_factory(call_button="Enter")
def stitched_data_widget(
    viewer: Viewer,
    rotation: float = 0.15,
    precise_rotation: bool = False,
    overlap_x: int = 7,
    overlap_y: int = 7,
    edge: int = 7,
    mode: str = "reflect",
) -> None:
    effective_rotation = rotation if precise_rotation else 0.0
    clear_viewer_layers(viewer)
    stitched_images = stitch_images(
        omero_data,
        rotation=effective_rotation,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        edge=edge,
        mode=mode,
    )
    logger.debug(
        f"Stitched shape {stitched_images.shape} ({stitched_images.dtype})"
    )
    num_channels = stitched_images.shape[-1]
    names = ["Stitched Image"] * num_channels
    for k, v in omero_data.channel_data.items():
        idx = int(v)
        if idx < num_channels:
            names[idx] = k

    per_channel_limits = [
        list(omero_data.intensities.get(i, (0, 65535)))
        for i in range(num_channels)
    ]

    viewer.add_image(
        stitched_images,
        contrast_limits=per_channel_limits,
        gamma=1,
        channel_axis=-1,
        scale=omero_data.pixel_size,
        name=names,
    )
    for layer in viewer.layers:
        if isinstance(layer, Image):
            layer.contrast_limits_range = (0, 65535)
    set_color_maps(viewer)
    if len(omero_data.labels):
        stitched_labels = stitch_labels(
            omero_data,
            rotation=effective_rotation,
            overlap_x=overlap_x,
            overlap_y=overlap_y,
        )
        add_label_layers(viewer, labels=stitched_labels)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
    viewer.scale_bar.color = "white"
