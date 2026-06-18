"""Direct load dialog for selecting plate/well/image to annotate.

This module provides the DirectLoadDialog widget that allows users to
input plate/well/image IDs and load crops directly from OMERO without
requiring welldata_widget pre-loading.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.direct_omero_loader import load_crops_from_omero

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData
    from omero_screen_napari.trainingdata_db.database import TrainingDB


class DirectLoadDialog(QDialog):  # type: ignore[misc]
    """Dialog for loading data directly from OMERO for annotation.

    Allows user to input:
    - Plate ID
    - Well ID (e.g., "A1")
    - Image ID
    - Timepoint (optional)

    Then fetches crops directly from OMERO and loads into viewer.
    """

    def __init__(
        self,
        classifier_name: str,
        db: "TrainingDB",
        omero_data: "OmeroData",
        user_data: "UserData",
        on_load_callback: "Callable[[], None] | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the direct load dialog.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance
            omero_data: OmeroData singleton
            user_data: UserData singleton
            on_load_callback: Callback when data is loaded successfully
            parent: Parent widget
        """
        super().__init__(parent)
        self.classifier_name = classifier_name
        self.db = db
        self.omero_data = omero_data
        self.user_data = user_data
        self.on_load_callback = on_load_callback

        self.setWindowTitle(f"Load Data for Annotation - {classifier_name}")
        self.setMinimumWidth(500)

        # Create layout
        main_layout = QVBoxLayout()

        # Input form
        form_layout = QFormLayout()

        # Plate ID
        self.plate_input = QSpinBox()
        self.plate_input.setRange(1, 999999)
        self.plate_input.setValue(1)
        form_layout.addRow("Plate ID:", self.plate_input)

        # Well ID
        self.well_input = QComboBox()
        self.well_input.setEditable(True)
        # Populate with common well positions
        self._populate_well_options()
        form_layout.addRow("Well:", self.well_input)

        # Image selection (same format as welldata widget)
        self.image_input = QLineEdit()
        self.image_input.setText("All")
        self.image_input.setToolTip(
            "Image selection: 'All', '0', '0, 1, 2', or '3-5'"
        )
        form_layout.addRow("Images:", self.image_input)

        # Timepoint
        self.timepoint_input = QSpinBox()
        self.timepoint_input.setRange(0, 999)
        self.timepoint_input.setValue(0)
        form_layout.addRow("Timepoint:", self.timepoint_input)

        # Cell cycle phase filter
        self.cellcycle_input = QComboBox()
        self.cellcycle_input.addItems(
            ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]
        )
        form_layout.addRow("Cell Cycle Phase:", self.cellcycle_input)

        # Classifier filter (populated after Validate)
        self._classifier_col_combo = QComboBox()
        self._classifier_col_combo.addItem("— validate first —")
        self._classifier_col_combo.setEnabled(False)
        self._classifier_col_combo.currentTextChanged.connect(
            self._on_classifier_col_changed
        )
        form_layout.addRow("Classifier:", self._classifier_col_combo)

        self._classifier_class_combo = QComboBox()
        self._classifier_class_combo.addItem("All")
        self._classifier_class_combo.setEnabled(False)
        form_layout.addRow("Class:", self._classifier_class_combo)

        self._plate_id_for_classes: int | None = None

        main_layout.addLayout(form_layout)

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("Enter plate/well/image IDs above")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Metadata section
        metadata_group = QGroupBox("Classifier Metadata")
        metadata_layout = QVBoxLayout()
        self.metadata_label = QLabel(self._get_classifier_metadata_text())
        self.metadata_label.setWordWrap(True)
        metadata_layout.addWidget(self.metadata_label)
        metadata_group.setLayout(metadata_layout)
        main_layout.addWidget(metadata_group)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Validate button
        validate_button = QPushButton("Validate")
        validate_button.clicked.connect(self._on_validate)
        button_layout.addWidget(validate_button)

        # Load button
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self._on_load)
        button_layout.addWidget(self.load_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _populate_well_options(self) -> None:
        """Populate well dropdown with common well positions."""
        # Generate well positions for 96-well plate (A-H, 1-12)
        wells = []
        for row in "ABCDEFGH":
            for col in range(1, 13):
                wells.append(f"{row}{col}")
        self.well_input.addItems(wells)
        self.well_input.setCurrentText("A1")

    def _get_classifier_metadata_text(self) -> str:
        """Get classifier metadata for display.

        Returns:
            Formatted metadata text
        """
        try:
            import json

            metadata_path = (
                Path.home()
                / "omeroscreen_trainingdata"
                / self.classifier_name
                / "metadata.json"
            )

            if not metadata_path.exists():
                return "Metadata file not found"

            with metadata_path.open() as f:
                metadata = json.load(f)

            user_data = metadata.get("user_data", {})
            crop_size = user_data.get("crop_size", "N/A")
            channels = user_data.get("channels", [])
            segmentation = user_data.get("segmentation", "N/A")
            cellcycle = user_data.get("cellcycle", "N/A")

            text = f"""
• Crop size: {crop_size}px
• Channels: {", ".join(str(ch) for ch in channels)}
• Segmentation: {segmentation}
• Cell cycle: {cellcycle}
            """.strip()

            return text

        except Exception as e:
            logger.warning(f"Could not load classifier metadata: {e}")
            return f"Error loading metadata: {e}"

    def _on_validate(self) -> None:
        """Validate the input data."""
        plate_id = self.plate_input.value()
        well_id = self.well_input.currentText()
        image_input = self.image_input.text().strip()

        # Basic validation
        if not well_id:
            QMessageBox.warning(
                self, "Validation Error", "Please enter a well ID"
            )
            return

        if not image_input:
            QMessageBox.warning(
                self, "Validation Error", "Please enter an image selection"
            )
            return

        # Update preview
        preview_text = f"""
<b>Plate:</b> {plate_id}<br>
<b>Well:</b> {well_id}<br>
<b>Images:</b> {image_input}<br>
<b>Timepoint:</b> {self.timepoint_input.value()}<br>
<br>
<i>Click "Load Data" to fetch crops from OMERO</i>
        """.strip()

        self.preview_label.setText(preview_text)
        self._populate_classifier_options(plate_id)

    def _populate_classifier_options(self, plate_id: int) -> None:
        """Query CellView for classifier columns available for this plate.

        Previously this called ``cellview_load_data(plate_id)`` and pulled
        the entire plate (~695k rows × 287 cols, ~3 s) just to read column
        names + per-column unique values. We now do a ``PRAGMA
        table_info`` against ``measurements`` for the column list, and
        defer the per-column ``SELECT DISTINCT`` to
        ``_on_classifier_col_changed`` (only fires when the user actually
        picks a classifier).
        """
        self._plate_id_for_classes = plate_id
        try:
            from cellview.db.db import CellViewDB

            db = CellViewDB()
            conn = db.connect()
            try:
                cols = conn.execute(
                    "PRAGMA table_info(measurements)"
                ).fetchall()
                classifier_cols = sorted(
                    row[1] for row in cols if row[1].startswith("classifier_")
                )
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not list classifier columns for plate {plate_id}: {e}"
            )
            classifier_cols = []

        self._classifier_col_combo.blockSignals(True)
        self._classifier_col_combo.clear()
        if classifier_cols:
            self._classifier_col_combo.addItem("None")
            for col in classifier_cols:
                self._classifier_col_combo.addItem(col)
            self._classifier_col_combo.setEnabled(True)
        else:
            self._classifier_col_combo.addItem(
                "(no classifiers in this plate)"
            )
            self._classifier_col_combo.setEnabled(False)
        self._classifier_col_combo.blockSignals(False)

        self._classifier_class_combo.clear()
        self._classifier_class_combo.addItem("All")
        self._classifier_class_combo.setEnabled(False)

    def _on_classifier_col_changed(self, col_name: str) -> None:
        """Populate the class dropdown when a classifier column is selected.

        Issues a ``SELECT DISTINCT col FROM measurements ... WHERE
        plate_id = ?`` against DuckDB rather than pulling the full plate.
        The plate predicate keeps it sub-100ms even on large plates.
        """
        self._classifier_class_combo.clear()
        self._classifier_class_combo.addItem("All")

        if not col_name or col_name == "None":
            self._classifier_class_combo.setEnabled(False)
            return

        plate_id = getattr(self, "_plate_id_for_classes", None)
        if plate_id is None:
            self._classifier_class_combo.setEnabled(False)
            return

        try:
            from cellview.db.db import CellViewDB

            db = CellViewDB()
            conn = db.connect()
            try:
                # Column name is taken from PRAGMA table_info output, so
                # it's safe to interpolate into the query string; DuckDB
                # parameters can't bind identifiers.
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT m."{col_name}"
                    FROM measurements m
                    JOIN conditions c ON m.condition_id = c.condition_id
                    JOIN repeats r ON c.repeat_id = r.repeat_id
                    WHERE r.plate_id = ? AND m."{col_name}" IS NOT NULL
                    ORDER BY m."{col_name}"
                    """,
                    [plate_id],
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not list classes for {col_name} on plate {plate_id}: {e}"
            )
            self._classifier_class_combo.setEnabled(False)
            return

        for row in rows:
            self._classifier_class_combo.addItem(str(row[0]))
        self._classifier_class_combo.setEnabled(True)

    def _selected_classifier_column(self) -> str:
        col = str(self._classifier_col_combo.currentText())
        if col in (
            "None",
            "— validate first —",
            "(no classifiers in this plate)",
            "",
        ):
            return ""
        return col

    def _selected_classifier_class(self) -> str:
        cls = str(self._classifier_class_combo.currentText())
        return "" if cls == "All" else cls

    def _on_load(self) -> None:
        """Load data from OMERO."""
        plate_id = self.plate_input.value()
        well_id = self.well_input.currentText()
        image_input = self.image_input.text().strip()

        if not well_id:
            QMessageBox.warning(
                self, "Validation Error", "Please enter a well ID"
            )
            return

        if not image_input:
            QMessageBox.warning(
                self, "Validation Error", "Please enter an image selection"
            )
            return

        # Show loading message
        self.load_button.setEnabled(False)
        self.load_button.setText("Loading...")

        try:
            # Call direct loader
            success, message = load_crops_from_omero(
                plate_id=plate_id,
                well_id=well_id,
                image_input=image_input,
                classifier_name=self.classifier_name,
                omero_data=self.omero_data,
                user_data=self.user_data,
                cellcycle=self.cellcycle_input.currentText(),
                classifier_column=self._selected_classifier_column(),
                classifier_class=self._selected_classifier_class(),
                timepoint=self.timepoint_input.value(),
            )

            if success:
                logger.info(f"Successfully loaded data: {message}")
                QMessageBox.information(self, "Success", message)
                self.accept()
                # Trigger callback
                if self.on_load_callback:
                    self.on_load_callback()
            else:
                logger.error(f"Failed to load data: {message}")
                QMessageBox.critical(self, "Load Failed", message)

        except Exception as e:
            logger.exception(f"Unexpected error loading data: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Unexpected error: {e!s}",
            )

        finally:
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Data")
