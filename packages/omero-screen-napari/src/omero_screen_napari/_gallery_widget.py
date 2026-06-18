from loguru import logger
from magicgui import magic_factory
from magicgui.widgets import Container
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari._setup_training_widget import (
    ImageNavigator,
    MetaDataSaver,
)
from omero_screen_napari.gallery_api import show_gallery
from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data_singleton import omero_data

_RED_HINTS = ("rfp", "mcherry", "txred", "cy3", "h2b_rfp", "rfp_h2b")
_GREEN_HINTS = ("gfp", "yfp", "fitc", "tub_gfp", "edu")
_BLUE_HINTS = ("dapi", "hoechst", "cfp")


def _auto_pick_rgb_channels(
    available: list[str],
) -> tuple[str, str, str]:
    """Pick (red, green, blue) defaults from the loaded plate's channels.

    Returns an empty string for a slot when no available channel name
    matches the heuristic for that colour. The user can still override
    the picks in the widget before clicking Enter.
    """

    def _first_match(hints: tuple[str, ...]) -> str:
        for ch in available:
            low = ch.lower()
            if any(h in low for h in hints):
                return ch
        return ""

    red = _first_match(_RED_HINTS)
    green = _first_match(_GREEN_HINTS)
    blue = _first_match(_BLUE_HINTS)
    return red, green, blue


def _commit_viewer_contrast_to_intensities() -> None:
    """Pull live contrast_limits from the napari viewer's Image layers.

    The gallery centres + scales each crop using
    ``omero_data.intensities[channel_index]``. By default these are set
    at well-load time (from canvas percentiles or CellView), so the
    gallery is "frozen" — adjusting the viewer's contrast slider has no
    effect. Reading the current ``contrast_limits`` here, just before
    ``show_gallery`` runs, lets the user iterate contrast in the viewer
    and see it reflected in the next gallery rebuild.

    Layer matching: napari layer ``name`` is the channel name (set in
    ``zarr_cache.display._add_image_layers``); we map that back to the
    channel index via ``omero_data.channel_data``.
    """
    try:
        from napari.viewer import current_viewer

        viewer = current_viewer()
    except Exception:  # noqa: BLE001
        return
    if viewer is None:
        return

    channel_data = getattr(omero_data, "channel_data", None) or {}
    if not channel_data:
        return

    name_to_index: dict[str, int] = {}
    for name, value in channel_data.items():
        try:
            name_to_index[str(name)] = int(float(value))
        except (TypeError, ValueError):
            continue

    new_intensities = dict(getattr(omero_data, "intensities", {}) or {})
    updated_any = False
    for layer in viewer.layers:
        if layer.__class__.__name__ != "Image":
            continue
        ch_idx = name_to_index.get(layer.name)
        if ch_idx is None:
            continue
        cl = getattr(layer, "contrast_limits", None)
        if cl is None:
            continue
        try:
            lo, hi = float(cl[0]), float(cl[1])
        except (TypeError, ValueError, IndexError):
            continue
        new_intensities[ch_idx] = (int(lo), int(hi))
        updated_any = True

    if updated_any:
        omero_data.intensities = new_intensities
        logger.info(f"Gallery contrast synced from viewer: {new_intensities}")


def gallery_gui_widget() -> Container:  # type: ignore
    from omero_screen_napari._logging import init_plugin_logging

    init_plugin_logging()
    gallery_widget_instance = gallery_widget()
    reset_widget_instance = reset_widget()
    container = Container(  # type: ignore[type-var]
        widgets=[gallery_widget_instance, reset_widget_instance]
    )
    setup = ClassifierSetupWidget()
    container.native.layout().addWidget(setup.widget)
    container._classifier_setup = setup  # prevent GC: signals reference self via bound methods  # type: ignore[attr-defined]
    return container


class ClassifierSetupWidget:
    """Simple Qt widget for creating a new classifier from the gallery."""

    def __init__(self) -> None:
        self.image_navigator = ImageNavigator(None)
        self.meta_data_saver: MetaDataSaver | None = None

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Section label
        title = QLabel("<b>Create Classifier</b>")
        layout.addWidget(title)

        # Class name row
        class_row = QWidget()
        class_layout = QHBoxLayout(class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText("Class name…")
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_class)
        self._class_input.returnPressed.connect(self._add_class)
        class_layout.addWidget(self._class_input)
        class_layout.addWidget(add_btn)
        layout.addWidget(class_row)

        # Class list
        self._class_list = QListWidget()
        self._class_list.setMaximumHeight(90)
        self._refresh_class_list()
        layout.addWidget(self._class_list)

        # Cell cycle row
        cc_row = QWidget()
        cc_layout = QHBoxLayout(cc_row)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.addWidget(QLabel("Cell cycle:"))
        self._cellcycle_combo = QComboBox()
        for phase in ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]:
            self._cellcycle_combo.addItem(phase)
        cc_layout.addWidget(self._cellcycle_combo)
        layout.addWidget(cc_row)

        # Classifier name row
        name_row = QWidget()
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.addWidget(QLabel("Classifier name:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. mitosis_rpe")
        name_layout.addWidget(self._name_input)
        layout.addWidget(name_row)

        # Save button
        save_btn = QPushButton("Save classifier")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    def _add_class(self) -> None:
        name = self._class_input.text().strip()
        if name:
            self.image_navigator.add_class(name)
            self._class_input.clear()
            self._refresh_class_list()

    def _refresh_class_list(self) -> None:
        self._class_list.clear()
        for cls in self.image_navigator.class_options:
            self._class_list.addItem(cls)

    def _save(self) -> None:
        classifier_name = self._name_input.text().strip()
        if not classifier_name:
            QMessageBox.warning(
                self.widget, "Missing name", "Please enter a classifier name."
            )
            return

        # Persist the chosen cell cycle phase into userdata before saving so
        # it is captured in metadata.json and restored when sessions are loaded.
        userdata.cellcycle = self._cellcycle_combo.currentText()

        self.meta_data_saver = MetaDataSaver(
            classifier_name, omero_data, userdata, self.image_navigator
        )
        self.meta_data_saver.update_classifier_name(classifier_name)
        self.meta_data_saver.save_data()

        # Reset for next classifier
        self.image_navigator.reset_class_options()
        self._refresh_class_list()
        self._name_input.clear()


@magic_factory(
    call_button="Enter",
)
def reset_widget() -> None:
    omero_data.cropped_images = []
    omero_data.cropped_labels = []


@magic_factory(
    call_button="Enter",
    segmentation={"choices": ["nucleus", "cell"]},
    crop_size={"choices": [20, 30, 50, 100, 200]},
    cellcycle={"choices": ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]},
)
def gallery_widget(
    well: str = "",
    *,
    segmentation: str,
    crop_size: int,
    cellcycle: str,
    classifier_filter: str = "",
    timepoint: int = 0,
    columns: int = 4,
    rows: int = 4,
    reload: bool = True,
    contour: bool = True,
    no_background: bool = True,
    red_channel: str = "",
    green_channel: str = "",
    blue_channel: str = "",
) -> None:
    # Auto-pick RGB defaults from the loaded plate's channels when the
    # user leaves a slot blank, or when the literal value isn't in the
    # available channel list (covers stale hard-coded defaults like
    # "DAPI" / "Tub" / "EdU" on plates that use different names).
    available = list((omero_data.channel_data or {}).keys())
    auto_red, auto_green, auto_blue = _auto_pick_rgb_channels(available)

    def _resolve(user_value: str, auto_value: str) -> str:
        if user_value and user_value in available:
            return user_value
        return auto_value

    red_channel = _resolve(red_channel, auto_red)
    green_channel = _resolve(green_channel, auto_green)
    blue_channel = _resolve(blue_channel, auto_blue)

    # Order matters: ``fill_missing_channels`` maps the list as
    # [R=ch0, G=ch1, B=ch2]. With two channels the dispatcher packs
    # them into R and G and leaves blue empty — the user prefers this
    # to a slot-preserving G+B mapping for fluorescence imaging.
    channels = [red_channel, green_channel, blue_channel]
    channels = [channel for channel in channels if channel != ""]
    if not well and omero_data.well_pos_list:
        well = omero_data.well_pos_list[0]
        logger.info(f"Using default well: {well}")

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
        "classifier_filter": classifier_filter,
        "timepoint": timepoint,
        "columns": columns,
        "rows": rows,
        "contour": contour,
        "no_background": no_background,
        "channels": channels,
    }
    try:
        userdata.populate_from_dict(user_data_dict)
        _commit_viewer_contrast_to_intensities()
        show_gallery(omero_data, userdata)
    except ValueError as e:
        logger.exception(f"Gallery Error: {e}")
        QMessageBox.critical(None, "Gallery Error", str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Unexpected Error: {e}")
        QMessageBox.critical(None, "Unexpected Error", str(e))
