from magicgui import magic_factory
from magicgui.widgets import Container
from omero_screen.config import get_logger
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

logger = get_logger(__name__)


def gallery_gui_widget() -> Container:  # type: ignore
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
    blue_channel: str = "DAPI",
    green_channel: str = "Tub",
    red_channel: str = "EdU",
) -> None:
    channels = [blue_channel, green_channel, red_channel]
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
        show_gallery(omero_data, userdata)
    except ValueError as e:
        logger.exception("Gallery Error: %s", e)
        QMessageBox.critical(None, "Gallery Error", str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected Error: %s", e)
        QMessageBox.critical(None, "Unexpected Error", str(e))
