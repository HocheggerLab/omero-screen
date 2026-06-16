"""
This module handles the widget to call Omero and load flatfield corrected well images
as well as segmentation masks (if avaliable) into napari.
The plugin can be run from napari as Welldata Widget under Plugins.
"""

import contextlib
import os
import re
import threading
from collections.abc import Callable, Generator
from typing import Any, Optional

import numpy as np
import polars as pl
from magicgui import magic_factory
from napari.layers import Image
from napari.qt.threading import GeneratorWorker, create_worker
from napari.utils import notifications
from napari.utils import progress as napari_progress
from napari.viewer import Viewer
from omero_screen.config import get_logger
from omero_utils.stitching import (
    has_valid_positions,
    recompose_split_labels,
    stitch_from_positions,
    stitch_labels_from_positions,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
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
from superqt.utils import ensure_main_thread
from vispy.color import Colormap

from omero_screen_napari.omero_data import OmeroConnection
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.omero_image import cache_size_limit, cache_volume
from omero_screen_napari.plate_cache import (
    cache_plate,
    delete_plate_from_cache,
    get_all_cached_plates,
    get_cached_plate_metadata,
    get_well_cache_status,
    is_plate_cached,
    is_plate_fully_cached,
    load_from_cache,
)
from omero_screen_napari.zarr_cache import (
    ZarrCacheTooSmall,
    build_plate_zarr,
    is_stitched_plate,
    load_plate_to_viewer,
    plate_zarr_path,
    resolve_target_wells,
)

# Logging
logger = get_logger(__name__)


class MetadataWidget(QWidget):  # type: ignore
    """Displays well metadata and optional classifier class counts."""

    def __init__(
        self,
        metadata: dict[str, Any],
        classifier_data: dict[str, list[tuple[str, int]]] | None = None,
    ) -> None:
        super().__init__()
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        self.label = QLabel()
        label_text = "".join(
            f"{key}: {value}\n" for key, value in metadata.items()
        )
        self.label.setText(label_text.rstrip())
        self._layout.addWidget(self.label)

        if classifier_data:
            self._layout.addWidget(QLabel("<b>Classifiers</b>"))
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(["Classifier", "Class", "Cells"])
            table.setEditTriggers(QTableWidget.NoEditTriggers)  # type: ignore[attr-defined]
            table.setSelectionMode(QTableWidget.NoSelection)  # type: ignore[attr-defined]
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # type: ignore[attr-defined, union-attr]
            table.verticalHeader().setVisible(False)  # type: ignore[union-attr]
            table.setSortingEnabled(False)
            for classifier, entries in classifier_data.items():
                for class_val, count in entries:
                    row = table.rowCount()
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(classifier))
                    table.setItem(row, 1, QTableWidgetItem(str(class_val)))
                    table.setItem(row, 2, QTableWidgetItem(str(count)))
            row_height = 25
            header_height = 26
            table.setFixedHeight(header_height + row_height * table.rowCount())
            self._layout.addWidget(table)

        self._layout.addStretch()
        self.setLayout(self._layout)


class CachedPlatesSelector(QWidget):  # type: ignore[misc]
    """Compact dropdown showing plates from the cache.

    Includes cached plates (with images in the local cache) and removed
    plates (previously cached, now evicted).  Users can select
    a plate to load, cache a plate, or forget it entirely.

    Args:
        on_plate_selected: Callback receiving the plate_id as string.
    """

    def __init__(
        self,
        on_plate_selected: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_plate_selected = on_plate_selected

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Row 1: cache size label
        self._cache_size_label = QLabel()
        layout.addWidget(self._cache_size_label)

        # Row 2: combo + buttons
        combo_row = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._combo.setMinimumWidth(200)
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        self._combo.activated.connect(self._on_activated)
        combo_row.addWidget(self._combo, stretch=1)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedWidth(60)
        self._delete_btn.setToolTip(
            "Delete cached data, or forget a removed plate"
        )
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._delete_btn.setEnabled(False)
        combo_row.addWidget(self._delete_btn)

        self._cache_btn = QPushButton("Cache")
        self._cache_btn.setFixedWidth(60)
        self._cache_btn.setToolTip("Download plate data to local cache")
        self._cache_btn.clicked.connect(self._on_cache_clicked)
        self._cache_btn.setEnabled(False)
        combo_row.addWidget(self._cache_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(60)
        self._cancel_btn.clicked.connect(self._on_stop_clicked)
        self._cancel_btn.setEnabled(False)
        combo_row.addWidget(self._cancel_btn)
        layout.addLayout(combo_row)

        # Row 3: detail label
        self._detail_label = QLabel()
        layout.addWidget(self._detail_label)

        self.refresh()
        # Trigger selection of the current plate ID
        self._on_activated(0)

    def refresh(self) -> None:
        """Rebuild the combo from plate cache."""
        # Update cache size label
        volume_gb = cache_volume() / 2**30
        limit_gb = cache_size_limit() / 2**30
        self._cache_size_label.setText(
            f"Cache: {volume_gb:.1f} / {limit_gb:.1f} GB"
        )

        # Block signals to avoid spurious callbacks during repopulation
        self._combo.blockSignals(True)
        prev_plate_id = self._selected_plate_id()
        self._combo.clear()
        for plate_id, name, cache_date in get_all_cached_plates():
            tag = _cache_status_tag(plate_id)
            tag_str = f"  [{tag}]" if tag else ""
            display_text = f"{plate_id} - {name} [{cache_date}]{tag_str}"
            self._combo.addItem(display_text, userData=plate_id)

        # Restore previous selection if still present
        self._select_plate(prev_plate_id)

        self._combo.blockSignals(False)
        self._update_detail()

    def _selected_plate_id(self) -> int | None:
        """Return the plate_id of the currently selected combo item."""
        idx = self._combo.currentIndex()
        if idx < 0:
            return None
        return self._combo.itemData(idx, Qt.ItemDataRole.UserRole)  # type: ignore[no-any-return]

    def _select_plate(self, plate_id: int | None) -> None:
        """Select the specified plate."""
        if plate_id is None:
            return
        for i in range(self._combo.count()):
            if self._combo.itemData(i, Qt.ItemDataRole.UserRole) == plate_id:
                self._combo.setCurrentIndex(i)
                break

    def _on_index_changed(self, _index: int) -> None:
        """Update detail label and button states when selection changes."""
        self._update_detail()

    def _on_activated(self, _index: int) -> None:
        """Populate the plate_id field when user activates an item."""
        plate_id = self._selected_plate_id()
        if plate_id is not None and self._on_plate_selected is not None:
            self._on_plate_selected(str(plate_id))

    def _update_detail(self) -> None:
        """Refresh the detail label and button states for current selection."""
        plate_id = self._selected_plate_id()
        if plate_id is None:
            self._detail_label.setText("")
            self._cache_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return

        self._delete_btn.setEnabled(True)

        if is_plate_cached(plate_id):
            # Well completeness summary
            download = True
            well_status = get_well_cache_status(plate_id)
            if well_status:
                complete = sum(1 for v in well_status.values() if v)
                total = len(well_status)
                download = complete < total
                well_info = f"{complete}/{total}"
            else:
                well_info = ""
            self._cache_btn.setEnabled(download)
        else:
            well_info = "NA"
            self._cache_btn.setEnabled(True)

        # Get cache date from metadata
        meta = get_cached_plate_metadata(plate_id) or {}
        last_cached = meta.get("cache_date", "unknown")
        estimated_bytes = meta.get("estimated_bytes", 0)

        if estimated_bytes:
            byte_info = f" | Download size ~ {estimated_bytes / 2**30:.2f} GB"
        else:
            byte_info = ""

        self._detail_label.setText(
            f"Wells: {well_info}{byte_info} | Cached: {last_cached}"
        )

    def _on_delete_clicked(self) -> None:
        """Delete cached data."""
        plate_id = self._selected_plate_id()
        if plate_id is None:
            return

        reply = QMessageBox.question(
            self,
            "Delete Cached Plate",
            f"Delete all cached data for plate {plate_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            delete_plate_from_cache(plate_id)
            # Also evict the zarr-cache copy (separate store, separate
            # eviction path). Without this, orphan zarr directories
            # accumulate on disk indefinitely.
            try:
                from omero_screen_napari.zarr_cache.eviction import (
                    evict_plate as evict_zarr_plate,
                )

                evict_zarr_plate(plate_id)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.warning(
                    "Failed to evict zarr cache for plate %d: %s",
                    plate_id,
                    exc,
                )
            self.refresh()

    def _on_cache_clicked(self) -> None:
        """Start or resume caching for the selected plate.

        Dispatches to the zarr writer for plates that were processed in
        stitched mode (per the segmentation dataset), otherwise to the
        existing per-field diskcache worker. The decision is made by
        checking OMERO once via a short-lived connection.
        """
        plate_id = self._selected_plate_id()
        if plate_id is None:
            return

        try:
            if _plate_is_stitched(plate_id):
                start_zarr_build_worker(plate_id, self)
                self.enable_cancel_button()
                return
        except Exception as e:  # noqa: BLE001 — surface the failure, then fall back
            logger.warning(
                "Could not determine stitched-mode for plate %d "
                "(falling back to diskcache): %s",
                plate_id,
                e,
            )

        start_cache_worker(plate_id)
        self.enable_cancel_button()

    def enable_cancel_button(self) -> None:
        """Enable the cancel button to stop the active download."""
        # Note: Ideally we would like a callback event when the
        # caching starts to enable the cancel button and
        # when it completes to disable the cancel button.
        # Presently we just poll the active download.
        self._cache_timer: QTimer | None = QTimer(self)
        self._cache_timer.timeout.connect(self._poll_cache_status)
        self._cache_timer.start(1000)

    def _poll_cache_status(self) -> None:
        """Check for newly cached wells and update the table."""
        if get_active_download() != 0:
            self._cancel_btn.setEnabled(True)
        else:
            self._cancel_btn.setEnabled(False)
            t = self._cache_timer
            self._cache_timer = None
            if t is not None:
                t.stop()

    def _on_stop_clicked(self) -> None:
        """Stop caching data."""
        self._cancel_btn.setEnabled(False)
        _stop_active_download()


# Mock event object with the current_step attribute
class MockEvent:
    def __init__(self, source: Any) -> None:
        self.source = source


class BackgroundGeneratorWorker(GeneratorWorker):  # type: ignore[misc]
    """Worker that runs a generator in the background."""

    stop_flag: threading.Event | None = None

    def quit(self) -> None:
        super().quit()
        # Override the quit method to set the stop flag.
        # Using print statements in the base GeneratorWorker it can be seen
        # that when napari is closed the quit method is called and the
        # _abort_requested flag is detected in the work() loop method.
        # However when the signal aborted.emit() method is called
        # the connected method is not invoked. Using a custom worker
        # to perform actions on quit does work.
        if self.stop_flag is not None:
            self.stop_flag.set()


# Global variable to keep track of the existing metadata widget
metadata_widget: Optional[MetadataWidget] = None

# Reference to the stitch widget so auto-stitch can read its parameters
_stitch_widget_ref: Any = None

# Combine Welldata and Stiched data widgets


_cached_plates_selector_ref: CachedPlatesSelector | None = None


def well_widget_combined() -> QWidget:  # type: ignore
    """Combine the well and stitched data widgets with a Plate Info button."""
    global _stitch_widget_ref, _cached_plates_selector_ref
    from omero_screen_napari._plate_info_dialog import PlateInfoDialog

    welldata_instance = welldata_widget()
    stitched_instance = stitched_data_widget()
    _stitch_widget_ref = stitched_instance

    # Insert "Plate Info" button into the welldata form, right below plate_id
    # Layout order: [0] Enter button, [1] plate_id field, [2] insert here
    plate_info_btn = QPushButton("Plate Info")
    native_layout = welldata_instance.native.layout()
    native_layout.insertWidget(2, plate_info_btn)

    # Cached plates selector — selecting a plate populates plate_id field
    cached_selector = CachedPlatesSelector(
        on_plate_selected=lambda pid: setattr(
            welldata_instance.plate_id, "value", pid
        ),
    )
    _cached_plates_selector_ref = cached_selector

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(cached_selector)
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

    def build_callback(plate_id: int) -> None:
        if _cached_plates_selector_ref is not None:
            # If the plate is not currently selected then
            # refresh the plate list (for unknown plates) and select it
            if plate_id != _cached_plates_selector_ref._selected_plate_id():
                _cached_plates_selector_ref.refresh()
                _cached_plates_selector_ref._select_plate(plate_id)
            else:
                # Ensure the detail is correct
                _cached_plates_selector_ref._update_detail()

    def load_callback(well_pos: str) -> None:
        welldata_instance.well_pos_list.value = well_pos
        welldata_instance()

    def cache_callback(plate_id: int, wells: list[str] | None = None) -> None:
        # Stitched-mode plates use the OME-Zarr cache (lazy chunk loads,
        # bounded memory). Per-field plates use the legacy diskcache.
        # Mirrors the dispatch in CachedPlatesSelector._on_cache_clicked.
        # ``wells`` (ticked in the plate-info dialog) restricts the zarr
        # build to a subset — key for huge live-cell plates.
        try:
            if _plate_is_stitched(plate_id):
                start_zarr_build_worker(plate_id, parent, wells=wells)
                if _cached_plates_selector_ref is not None:
                    _cached_plates_selector_ref.enable_cancel_button()
                return
        except Exception as exc:  # noqa: BLE001 — fall back, never block
            logger.warning(
                "Could not determine stitched-mode for plate %d "
                "(falling back to diskcache): %s",
                plate_id,
                exc,
            )
        start_cache_worker(plate_id)
        if _cached_plates_selector_ref is not None:
            _cached_plates_selector_ref.enable_cancel_button()

    dialog = dialog_cls(
        plate_id,
        on_build_callback=build_callback,
        on_load_callback=load_callback,
        on_cache_callback=cache_callback,
        parent=parent,
    )
    dialog.exec_()


# Defaults matching the stitched_data_widget signature (Operetta calibration)
_STITCH_DEFAULTS: dict[str, Any] = {
    "stitch": True,
    "overlap_x": 7,
    "overlap_y": 7,
    "translate_x": -3,
    "translate_y": 3,
    "edge": 7,
}


def _get_stitch_params() -> dict[str, Any]:
    """Read current stitch parameters from the sibling stitch widget.

    Falls back to Operetta defaults when the widget is not available.
    """
    w = _stitch_widget_ref
    if w is not None:
        try:
            return {
                "stitch": w.stitch.value,
                "overlap_x": w.overlap_x.value,
                "overlap_y": w.overlap_y.value,
                "translate_x": w.translate_x.value,
                "translate_y": w.translate_y.value,
                "edge": w.edge.value,
            }
        except AttributeError:
            pass
    return _STITCH_DEFAULTS.copy()


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
    if omero_data.image_input != images:
        return False
    # If plate_data is empty the gallery widget will fail; reload to populate it.
    return len(omero_data.plate_data.collect_schema()) != 0


@magic_factory(call_button="Enter")
def welldata_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos_list: str = "Well Position",
    images: str = "All",
    time: str = "All",
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    plate_num = int(plate_id)

    try:
        # Zarr cache overrides the per-field diskcache when present.
        # Data is already stitched on disk so we hand zarr arrays to
        # napari directly — no background worker, lazy chunk loads.
        if plate_zarr_path(plate_num).exists():
            logger.info(
                "Loading plate %d from zarr cache (wells=%s)",
                plate_num,
                well_pos_list,
            )
            loaded = load_plate_to_viewer(
                viewer, plate_num, well_pos_input=well_pos_list
            )
            if not loaded:
                notifications.show_warning(
                    f"No wells matched '{well_pos_list}' in the zarr "
                    f"cache for plate {plate_num}."
                )
            elif well_pos_list.strip().lower() == "all":
                notifications.show_info(
                    f"Loaded {len(loaded)} cached well(s) from zarr: "
                    f"{', '.join(loaded)}. To add more, rebuild via the "
                    f"Cache button."
                )
            return

        # Safety net: a stitched-mode plate loaded via the per-field path
        # has to re-stitch in Python at view time, which blows up RAM
        # (9 fields × N timepoints × C channels × Y × X in float). Warn
        # the user and steer them to "Cache Plate" → zarr build first.
        try:
            if _plate_is_stitched(plate_num):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Stitched plate — build zarr cache first")
                msg.setText(
                    f"Plate {plate_num} was processed in stitched mode but "
                    f"has no OME-Zarr cache yet.\n\n"
                    f"Loading via the per-field path will re-stitch in "
                    f"memory and can exhaust RAM for time-lapse data.\n\n"
                    f"Open the Plate Info dialog and click 'Cache Plate' "
                    f"to build the zarr cache once, then loading is "
                    f"fast and bounded-memory."
                )
                msg.exec_()
                return
        except Exception as exc:  # noqa: BLE001 — safety net only
            logger.debug(
                "Stitched-mode probe failed for plate %d (ignored): %s",
                plate_num,
                exc,
            )

        if _is_already_loaded(omero_data, plate_num, well_pos_list, images):
            logger.info(
                "Plate %d already in memory, skipping reload", plate_num
            )
            _display_plate(viewer)
        else:
            # Improve GUI responsiveness by loading in a background
            # worker. Display the results via call backs.
            start_data_worker(
                viewer, omero_data, plate_num, well_pos_list, images, time=time
            )

    except Exception as e:
        logger.error(f"Error in welldata_widget: {e}")
        # MessageBox is already shown in parse_omero_data if it failed there
        # But we catch other potential errors here
        if "ValueError" not in str(
            type(e)
        ):  # Avoid double message for the common ValueError
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"An unexpected error occurred: {e}")
            msg.setWindowTitle("Widget Error")
            msg.exec_()


def _display_plate(viewer: Viewer) -> None:
    n_wells = len(omero_data.well_id_list)
    n_per_well = len(omero_data.image_index)
    iw_override = None

    # Check all wells can be stitched
    sp = _get_stitch_params()
    stitch = sp.get("stitch") and n_per_well > 0
    if stitch:
        for i in range(n_wells):
            j = i * n_per_well
            if not has_valid_positions(
                omero_data.image_positions[j : j + n_per_well]
            ):
                logger.warning(
                    "Unable to stitch well %s from stage positions",
                    omero_data.well_pos_list[i],
                )
                stitch = False
                break

    if stitch:
        logger.info("Auto-stitching %d well(s) from stage positions", n_wells)
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
                    edge=sp["edge"],
                    overlap_x=sp["overlap_x"],
                    overlap_y=sp["overlap_y"],
                    translate_x=sp["translate_x"],
                    translate_y=sp["translate_y"],
                )
            )

            if omero_data.labels.size > 0:
                well_labels = omero_data.labels[start:end]
                if omero_data.label_stitched_mode:
                    # Phase-1 stitched-canvas masks: globally-unique IDs,
                    # lossless reassembly via non-zero copy. Tile size is
                    # the spatial extent of each per-field mask.
                    tile_h = int(well_labels.shape[-3])
                    tile_w = int(well_labels.shape[-2])
                    stitched_lbls.append(
                        recompose_split_labels(
                            well_labels,
                            well_positions,  # type: ignore[arg-type]
                            tile_h=tile_h,
                            tile_w=tile_w,
                            overlap_x=sp["overlap_x"],
                            overlap_y=sp["overlap_y"],
                            translate_x=sp["translate_x"],
                            translate_y=sp["translate_y"],
                        )
                    )
                else:
                    # Legacy per-field-segmentation masks: independent label
                    # spaces, merged via merge_labels overlap fusion.
                    stitched_lbls.append(
                        stitch_labels_from_positions(
                            well_labels,
                            well_positions,  # type: ignore[arg-type]
                            overlap_x=sp["overlap_x"],
                            overlap_y=sp["overlap_y"],
                            translate_x=sp["translate_x"],
                            translate_y=sp["translate_y"],
                        )
                    )

        if n_wells == 1:
            result_img = stitched_imgs[0]
            result_lbl = stitched_lbls[0] if stitched_lbls else None
        else:
            result_img = np.stack(stitched_imgs)
            result_lbl = np.stack(stitched_lbls) if stitched_lbls else None

        clear_viewer_layers(viewer)
        _display_stitched(viewer, result_img, result_lbl)
        # For multi-well, each slider position = one well
        iw_override = 1 if n_wells > 1 else None
    else:
        logger.info("Displaying %d well(s)", n_wells)
        clear_viewer_layers(viewer)
        add_image_to_viewer(viewer)
        set_color_maps(viewer)
        add_label_layers(viewer)

    def slider_position_change(event: Any) -> None:
        pos = event.source.current_step[0]
        handle_metadata_widget(
            viewer, pos, images_per_well_override=iw_override
        )

    viewer.dims.events.current_step.connect(slider_position_change)
    mock_event = MockEvent(viewer.dims)
    slider_position_change(mock_event)


_active_lock = threading.Lock()
_active_cache_plate_id = 0
_active_cache_stop_flag = threading.Event()


# --------------------------------------------------------------------------- #
# Backend dispatch                                                            #
# --------------------------------------------------------------------------- #


def _cache_status_tag(plate_id: int) -> str:
    """Return a short tag describing which backend will serve a load.

    Returns ``"zarr"`` when a zarr cache exists for the plate — the zarr
    path overrides the diskcache on load, so we report the *active*
    backend rather than every store that happens to exist. Falls back
    to ``"disk"`` when only the legacy per-field cache is present.
    """
    if plate_zarr_path(plate_id).exists():
        return "zarr"
    if is_plate_cached(plate_id):
        return "disk"
    return ""


def _plate_is_stitched(plate_id: int) -> bool:
    """One-shot OMERO check: does this plate carry stitched-mode masks?

    Opens and closes its own short-lived connection so the call is safe
    to make from the Qt thread before deciding which worker to spawn.
    """
    omero_conn = OmeroConnection()
    try:
        conn = omero_conn.get_conn()
        return is_stitched_plate(conn, plate_id)
    finally:
        omero_conn.close(hard=False)


def start_zarr_build_worker(
    plate_id: int, parent: QWidget, wells: list[str] | None = None
) -> None:
    """Start a background zarr build for ``plate_id``.

    Mirrors the structure of :func:`start_cache_worker` so the rest of
    the widget (cancel button, completion refresh) keeps working. The
    worker yields well IDs as each one is written, which the progress
    bar uses for incremental updates.

    ``wells`` restricts the build to a subset of well positions (e.g. the
    wells ticked in the plate-info dialog); ``None`` builds the whole plate.
    """
    global _active_lock, _active_cache_plate_id, _active_cache_stop_flag

    with _active_lock:
        if not _active_cache_stop_flag.is_set():
            if _active_cache_plate_id == plate_id:
                logger.info(
                    "Zarr build already running for plate %d — skipping",
                    plate_id,
                )
                return
            if _active_cache_plate_id != 0:
                logger.info(
                    "Cancelling cache worker for plate %d",
                    _active_cache_plate_id,
                )
                _active_cache_stop_flag.set()

        omero_conn = OmeroConnection()
        stop_flag = threading.Event()
        pbr: list[Any] = []
        completed: list[str] = []

        # Resolve the work *synchronously* on the main thread so we can
        # build the determinate progress bar before spawning the worker.
        # ``napari_progress`` creates a QObject and must run on the GUI
        # thread (otherwise Qt prints ``setParent: Cannot set parent,
        # new parent is in a different thread`` and the bar never
        # appears). One small HQL query — typically <1s.
        try:
            main_conn = omero_conn.get_conn()
            target_wells = resolve_target_wells(plate_id, main_conn)
            if wells is not None:
                # Restrict to the user's ticked wells, but keep the
                # resolver's empty-well / resumability filtering.
                requested = set(wells)
                target_wells = [w for w in target_wells if w in requested]
                logger.info(
                    "Zarr build for plate %d restricted to %d selected "
                    "well(s): %s",
                    plate_id,
                    len(target_wells),
                    sorted(target_wells),
                )
        except Exception as exc:  # noqa: BLE001 — surface and bail
            logger.error(
                "Failed to plan zarr build for plate %d: %s", plate_id, exc
            )
            notifications.show_error(f"Zarr build planning failed: {exc}")
            omero_conn.close(hard=False)
            return

        total_wells = len(target_wells)
        # napari's QProgressBar only accepts *integer* values, but we want
        # smooth sub-well animation. Give each well an integer sub-resolution:
        # the bar runs 0..total_wells*STEPS and every update is an int.
        steps = 1000
        first_well = target_wells[0] if target_wells else None
        initial_desc = (
            f"Zarr plate {plate_id}: building {first_well} (0/{total_wells})"
            if first_well is not None
            else f"Zarr plate {plate_id}: nothing to build"
        )
        pbr.append(
            napari_progress(total=total_wells * steps, desc=initial_desc)
        )

        if total_wells == 0:
            # Everything already cached / no non-empty wells. Close the
            # bar cleanly and skip spawning the worker.
            pbr[0].close()
            omero_conn.close(hard=False)
            notifications.show_info(
                f"Plate {plate_id}: zarr cache already complete"
            )
            return

        # Sub-well progress. The builder streams each well's image + label
        # arrays and reports a 0–1 fraction via ``step_cb``; we translate that
        # into fractional advances of the (integer-well) bar so long live-cell
        # wells animate instead of sitting at 0 until they jump a whole well.
        # base/pos are integer step counts (well_index * steps + sub-progress).
        sub = {"base": 0, "pos": 0, "well": 0}

        @ensure_main_thread  # type: ignore[misc]
        def on_step(well_id: str, frac: float) -> None:
            """Fractional bar update (integer steps); marshalled to the GUI."""
            if not pbr:
                return
            # Cap below the next well's tick until on_yield confirms the well
            # is on disk. All quantities are integer steps.
            target = sub["base"] + int(min(frac, 0.999) * steps)
            delta = target - sub["pos"]
            if delta > 0:
                pbr[0].update(delta)
                sub["pos"] = target
            pbr[0].set_description(
                f"Zarr plate {plate_id}: building {well_id} "
                f"({sub['well']}/{total_wells}, {int(frac * 100)}%)"
            )

        def _generator() -> Generator[str, None, str | None]:
            """Wrap the builder so we can close the OMERO connection on exit."""
            try:
                conn = omero_conn.get_conn()
                # Passing ``omero_conn`` enables parallel per-field
                # downloads inside the builder (one BlitzGateway per
                # worker thread). Matches the diskcache's ``max_workers=3``.
                # Pre-resolved ``wells`` keeps the builder's internal
                # filter consistent with what we sized the bar for.
                for well_id in build_plate_zarr(
                    plate_id,
                    conn,
                    wells=target_wells,
                    # @ensure_main_thread wraps on_step's return as a Future;
                    # the builder calls it fire-and-forget, so the runtime
                    # contract (str, float) -> None still holds.
                    step_cb=on_step,  # type: ignore[arg-type]
                    omero_conn=omero_conn,
                    # 3 OMERO connections in flight. Empirically the
                    # sweet spot for the Sussex OMERO link — higher
                    # values don't help because the bottleneck is
                    # client-side bandwidth, not server concurrency.
                    max_workers=3,
                ):
                    if stop_flag.is_set():
                        return "Cancelled by user"
                    yield well_id
            except ZarrCacheTooSmall as e:
                return f"Cache too small: {e}"
            finally:
                omero_conn.close(hard=False)
            return None

        worker = create_worker(_generator, _worker_class=GeneratorWorker)
        worker.stop_flag = stop_flag

        def on_yield(well_id: str) -> None:
            """Runs on the main thread (Qt marshals via QueuedConnection)."""
            completed.append(well_id)
            done = len(completed)
            # Snap the bar up to this well's integer step boundary, then reset
            # the sub-well baseline for the next well. All integer steps.
            snap = done * steps - sub["pos"]
            if snap > 0:
                pbr[0].update(snap)
            sub["base"] = done * steps
            sub["pos"] = done * steps
            sub["well"] = done
            # Preview the next well so the bar reads as "now building Y"
            # rather than only the last completed well.
            if done < total_wells:
                next_well = target_wells[done]
                pbr[0].set_description(
                    f"Zarr plate {plate_id}: wrote {well_id} "
                    f"({done}/{total_wells}), building {next_well}"
                )
            else:
                pbr[0].set_description(
                    f"Zarr plate {plate_id}: wrote {well_id} "
                    f"({done}/{total_wells})"
                )

        def on_finished() -> None:
            logger.info(
                "Zarr build finished for plate %d (%d wells)",
                plate_id,
                len(completed),
            )
            stop_flag.set()
            if pbr:
                pbr[0].set_description(
                    f"Plate {plate_id}: {len(completed)} wells [zarr]"
                )
                pbr[0].close()
            if _cached_plates_selector_ref is not None:
                _cached_plates_selector_ref.refresh()

        def on_error(exc: BaseException) -> None:
            logger.error("Zarr build error for plate %d: %s", plate_id, exc)
            stop_flag.set()
            if pbr:
                pbr[0].set_description(f"Zarr build error: {exc}")
                pbr[0].close()
            notifications.show_error(f"Zarr build failed: {exc}")

        def on_returned(msg: str | None) -> None:
            if msg:
                notifications.show_warning(msg)

        worker.yielded.connect(on_yield)
        worker.finished.connect(on_finished)
        worker.errored.connect(on_error)
        worker.returned.connect(on_returned)
        worker.start()

        _active_cache_plate_id = plate_id
        _active_cache_stop_flag = stop_flag

    logger.info("Started zarr build worker for plate %d", plate_id)


def start_cache_worker(plate_id: int) -> None:
    """Start background caching for a plate using plate_cache.

    Downloads all metadata, flatfield-corrected images, and labels
    so that subsequent well navigation is fully offline.  Shows a
    progress bar in the napari activity dock.

    Skips if a worker is already running for the same plate or
    if the plate is already fully cached.
    """
    # Global reference to the active download objects.
    # Only allowed to update when holding the lock.
    global _active_lock, _active_cache_plate_id, _active_cache_stop_flag

    # Already fully cached (metadata + all images) — skip
    if is_plate_fully_cached(plate_id):
        logger.info(
            "Plate %d fully cached — skipping background download",
            plate_id,
        )
        return

    # The active download lock prevents duplicate calls to start a cache worker.
    # Only one worker should be active and this method can check the existing worker
    # and stop it if required.
    with _active_lock:
        # Check active download
        if not _active_cache_stop_flag.is_set():
            # A download is running, or never started when plate_id == 0.
            # Same plate ID -> already downloading this plate.
            if _active_cache_plate_id == plate_id:
                logger.info(
                    "Cache worker already running for plate %d — skipping",
                    plate_id,
                )
                return
            # Non-zero plate ID -> already downloading another plate.
            if _active_cache_plate_id != 0:
                logger.info(
                    "Cancelling cache worker for plate %d",
                    _active_cache_plate_id,
                )
                _active_cache_stop_flag.set()

        # From here we are committed to starting a new download.
        # Create connection details.
        connection = OmeroConnection()
        # This flag can be used to stop a running download. It is set when
        # the download ends to signal the download is no longer active.
        stop_flag = threading.Event()

        # Napari progress bar — created lazily on the first progress signal
        # when we know the total count.
        pbr: list[Any] = []  # mutable container so closures can share it
        _prev_done = 0

        max_workers = int(os.getenv("OMERO_SCREEN_IMAGE_CACHE_WORKERS", "3"))
        worker = create_worker(
            cache_plate,
            plate_id,
            connection,
            stop_flag,
            max_workers=max_workers,
            _worker_class=BackgroundGeneratorWorker,
        )
        worker.stop_flag = stop_flag

        def on_finished() -> Any:
            logger.info("Cache worker finished for plate %d", plate_id)
            # Signal the download has ended. No lock required.
            stop_flag.set()
            if pbr:
                pbr[0].set_description(f"Plate {plate_id} cached")
                pbr[0].close()
            if _cached_plates_selector_ref is not None:
                _cached_plates_selector_ref.refresh()
            return None

        def on_error(exc: BaseException) -> None:
            logger.error("Cache worker error for plate %d: %s", plate_id, exc)
            # Stop the download. No lock required.
            stop_flag.set()
            if pbr:
                pbr[0].set_description(f"Cache error: {exc}")
                pbr[0].close()

        def on_progress(prog: tuple[int, int]) -> None:
            nonlocal _prev_done
            done, total = prog
            if total <= 0:
                return
            # Create the bar on first real signal so total is correct from the start
            if not pbr:
                pbr.append(
                    napari_progress(
                        total=total, desc=f"Caching plate {plate_id}"
                    )
                )
            delta = done - _prev_done
            if delta > 0:
                pbr[0].update(delta)
            _prev_done = done

        def on_aborted() -> None:
            logger.warning("Cache worker aborted for plate %d", plate_id)
            # Stop the download. No lock required.
            stop_flag.set()

        def on_returned(msg: str | None) -> None:
            if msg:
                notifications.show_warning(msg)

        worker.yielded.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.errored.connect(on_error)
        worker.aborted.connect(on_aborted)
        worker.returned.connect(on_returned)
        worker.start()

        # The download has started. Store objects to allow it to be cancelled.
        _active_cache_plate_id = plate_id
        _active_cache_stop_flag = stop_flag
        # End-of with _active_lock

    logger.info("Started cache worker for plate %d", plate_id)


def get_active_download() -> int:
    """Get the plate ID of the active download, or zero if no download is active."""
    with _active_lock:
        return (
            0 if _active_cache_stop_flag.is_set() else _active_cache_plate_id
        )


def _stop_active_download() -> None:
    """Stop the active download."""
    with _active_lock:
        if (
            _active_cache_plate_id != 0
            and not _active_cache_stop_flag.is_set()
        ):
            logger.info(
                "Cancelling cache worker for plate %d",
                _active_cache_plate_id,
            )
            _active_cache_stop_flag.set()


_data_lock = threading.Lock()
_data_description = ""
_data_stop_flag = threading.Event()


def start_data_worker(
    viewer: Viewer,
    omero_data: Any,
    plate_id: int,
    well_pos_input: str,
    image_input: str,
    time: str = "All",
    cache_images: bool = True,
) -> None:
    """Start background loading of plate data.

    Downloads plate data. Shows a progress bar in the napari activity dock.

    Skips if a worker is already running.
    """
    # Global reference to the active data objects.
    # Only allowed to update when holding the lock.
    global _data_lock, _data_description, _data_stop_flag

    description = f"{plate_id}:{well_pos_input}:{image_input}:{time}"

    # The data lock prevents duplicate calls to start a data worker.
    # Only one worker should be active and this method can check the existing worker
    # and stop it if required.
    with _data_lock:
        # Check active download
        if not _data_stop_flag.is_set():
            # A download is running, or never started when description is empty.
            if _data_description == description:
                logger.info(
                    "Data worker already running for plate %d — skipping",
                    plate_id,
                )
                return
            if _data_description != "":
                old_plate = _data_description[: _data_description.index(":")]
                logger.info(
                    "Cancelling data worker for plate %s",
                    old_plate,
                )
                _data_stop_flag.set()

        # From here we are committed to starting a new download.
        # Create connection details.
        connection = OmeroConnection()
        # This flag can be used to stop a running download. It is set when
        # the download ends to signal the download is no longer active.
        stop_flag = threading.Event()

        # Napari progress bar — created lazily on the first progress signal
        # when we know the total count.
        pbr: list[Any] = []  # mutable container so closures can share it
        _prev_done = 0
        _total = 0

        worker = create_worker(
            load_from_cache,
            connection,
            omero_data,
            plate_id,
            well_pos_input,
            image_input,
            stop_flag,
            time=time,
            cache_images=cache_images,
            _worker_class=BackgroundGeneratorWorker,
        )
        worker.stop_flag = stop_flag

        def on_finished() -> Any:
            logger.info("Data worker finished for plate %d", plate_id)
            if pbr:
                pbr[0].set_description(f"Plate {plate_id} loaded")
                pbr[0].close()
                # Wells may have been downloaded so update the detail label
                if _cached_plates_selector_ref is not None:
                    _cached_plates_selector_ref._update_detail()
                # Display if done, otherwise assume load was cancelled
                if _prev_done == _total:
                    _display_plate(viewer)

        def on_error(exc: BaseException) -> None:
            logger.error("Data worker error for plate %d: %s", plate_id, exc)
            if pbr:
                pbr[0].set_description(f"Data error: {exc}")
                pbr[0].close()

        def on_progress(prog: tuple[int, int]) -> None:
            nonlocal _prev_done, _total
            done, total = prog
            if total <= 0:
                return
            # Create the bar on first real signal so total is correct from the start
            if not pbr:
                _total = total
                pbr.append(
                    napari_progress(
                        total=total, desc=f"Loading plate {plate_id}"
                    )
                )
            delta = done - _prev_done
            if delta > 0:
                pbr[0].update(delta)
            _prev_done = done

        def on_aborted() -> None:
            logger.warning("Data worker aborted for plate %d", plate_id)
            if pbr:
                pbr[0].set_description(f"Plate {plate_id} cancelled")
                pbr[0].close()

        worker.yielded.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.errored.connect(on_error)
        # Note: This does not seem to be called when napari exits and aborts threads
        # in the global pool. So we also check abort_requested in on_progress.
        worker.aborted.connect(on_aborted)
        worker.start()

        # The download has started. Store objects to allow it to be cancelled.
        _data_description = description
        _data_stop_flag = stop_flag
        # End-of with _active_lock

    logger.info("Started data worker for plate %d", plate_id)


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


def _extract_classifier_data(
    well_pos: str | None,
) -> dict[str, list[tuple[str, int]]] | None:
    """Return per-class cell counts for each classifier column in the current well."""
    if well_pos is None:
        return None
    try:
        schema = omero_data.plate_data.collect_schema()
        all_cols = schema.names()
        classifier_cols = [c for c in all_cols if c.startswith("classifier_")]
        if not classifier_cols:
            return None
        well_df = omero_data.plate_data.filter(
            pl.col("well") == well_pos
        ).collect()
        result: dict[str, list[tuple[str, int]]] = {}
        for col in classifier_cols:
            counts = (
                well_df.select(pl.col(col).drop_nulls())
                .get_column(col)
                .value_counts()
                .sort("count", descending=True)
            )
            label = col.removeprefix("classifier_")
            result[label] = [(str(r[0]), r[1]) for r in counts.iter_rows()]
        return result if any(result.values()) else None
    except Exception:
        logger.warning(
            "Could not extract classifier data for well %s", well_pos
        )
        return None


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
    well_pos = (
        omero_data.well_pos_list[well_index]
        if well_index < len(omero_data.well_pos_list)
        else None
    )
    if well_pos:
        well_metadata["Well"] = well_pos
    well_metadata.update(omero_data.well_metadata_list[well_index])

    classifier_data = _extract_classifier_data(well_pos)
    metadata_widget = MetadataWidget(well_metadata, classifier_data)
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


_NUCLEUS_ALIASES = frozenset({"dapi", "hoechst", "dna", "h2b_rfp"})
_CELL_ALIASES = frozenset({"tub", "gapdh"})


def _role_based_color_map(channel_names: list[str]) -> dict[str, str]:
    """Build name→colour map by resolving nuclei (blue) and cell (green) roles.

    Mirrors the resolution rules in :func:`omero_screen.metadata_parser.resolve_channel_roles`
    so napari display stays consistent with pipeline segmentation choices.
    """
    colors: dict[str, str] = {}
    for name in channel_names:
        lowered = name.lower()
        tokens = {tok for tok in re.split(r"[_-]", lowered) if tok}
        if lowered.endswith("_nucleus"):
            colors[name] = "blue"
        elif lowered.endswith("_cell"):
            colors[name] = "green"
        elif lowered in _NUCLEUS_ALIASES and "blue" not in colors.values():
            colors[name] = "blue"
        elif tokens & _CELL_ALIASES and "green" not in colors.values():
            colors[name] = "green"
    return colors


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

    # Default channel color assignments. Colours are assigned by *role* not by
    # literal name so plates using non-standard nuclei/cell stains (e.g. Hoechst,
    # CellMask, H2B_RFP) still render with consistent blue/green channels.
    special_channels = _role_based_color_map(channel_names)
    special_channels.setdefault("EdU", "gray")

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
    # Set slider range to full 16-bit and wire contrast-change events so that
    # gallery_api._scale_full_image picks up user adjustments.
    for layer in viewer.layers:
        if isinstance(layer, Image):
            layer.contrast_limits_range = (0, 65535)
            layer.events.contrast_limits.connect(on_contrast_change)

    set_color_maps(viewer)
    if stitched_labels is not None:
        # Single-well labels are (Y, X, C) — need batch dim for add_label_layers
        # Multi-well labels are (N, Y, X, C) — already have it
        if stitched_labels.ndim == 3:
            stitched_labels = stitched_labels[np.newaxis, ...]
        add_label_layers(viewer, labels=stitched_labels)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
    # viewer.reset_view()


@magic_factory(
    call_button="Enter",
    translate_x={"step": 1, "min": -20, "max": 20},
    translate_y={"step": 1, "min": -20, "max": 20},
)
def stitched_data_widget(
    viewer: Viewer,
    stitch: bool = True,
    overlap_x: int = 7,
    overlap_y: int = 7,
    translate_x: int = -3,
    translate_y: int = 3,
    edge: int = 7,
) -> None:
    _display_plate(viewer)
