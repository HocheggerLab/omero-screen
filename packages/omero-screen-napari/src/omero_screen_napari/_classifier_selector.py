"""Classifier selector widget with database integration.

This module provides a reusable classifier selection component that displays
available classifiers from the training database and shows detailed statistics.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from magicgui.widgets import Container, Label
from omero_screen.config import get_logger
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.trainingdata_db.database import TrainingDB

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class DetailedStatsDialog(QDialog):  # type: ignore[misc]
    """Dialog showing detailed per-image statistics in a table."""

    def __init__(
        self,
        classifier_name: str,
        db: TrainingDB,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance for querying
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Detailed Statistics - {classifier_name}")
        self.setMinimumSize(800, 400)

        # Create layout
        layout = QVBoxLayout()

        # Add table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            [
                "Plate ID",
                "Well",
                "Image ID",
                "Timepoint",
                "Total Cells",
                "Class Distribution",
            ]
        )

        # Make table read-only and enable sorting
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)

        # Load data
        self._load_data(classifier_name, db)

        # Resize columns to content
        header = self.table.horizontalHeader()
        if header:
            for i in range(5):  # First 5 columns
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(
                5, QHeaderView.Stretch
            )  # Last column stretches

        layout.addWidget(self.table)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def _load_data(self, classifier_name: str, db: TrainingDB) -> None:
        """Load per-image statistics from database.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance
        """
        try:
            stats = db.get_image_stats(classifier_name)

            if not stats:
                self.table.setRowCount(1)
                self.table.setItem(0, 0, QTableWidgetItem("No data available"))
                return

            # Populate table
            self.table.setRowCount(len(stats))
            for row_idx, stat in enumerate(stats):
                # Plate ID
                self.table.setItem(
                    row_idx, 0, QTableWidgetItem(str(stat["plate_id"]))
                )
                # Well
                self.table.setItem(row_idx, 1, QTableWidgetItem(stat["well"]))
                # Image ID
                self.table.setItem(
                    row_idx, 2, QTableWidgetItem(str(stat["image_id"]))
                )
                # Timepoint
                self.table.setItem(
                    row_idx, 3, QTableWidgetItem(str(stat["timepoint"]))
                )
                # Total Cells
                self.table.setItem(
                    row_idx, 4, QTableWidgetItem(str(stat["total_cells"]))
                )
                # Class Distribution
                dist = stat["class_distribution"]
                dist_str = ", ".join(
                    [f"{label}: {count}" for label, count in dist.items()]
                )
                self.table.setItem(row_idx, 5, QTableWidgetItem(dist_str))

        except Exception as e:
            logger.exception(f"Failed to load detailed statistics: {e}")
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e!s}"))


class ClassifierInfoPanel:
    """Info panel showing classifier statistics.

    Displays information matching the `omero-train stats` CLI output format:
    - General stats (ID, sessions, annotations, classes)
    - Class distribution with percentages
    - Button to view detailed per-image statistics in popup
    """

    def __init__(self) -> None:
        """Initialize the info panel."""
        # Use magicgui Label widget for simplicity
        self.info_label = Label(value="Select a classifier to view details")
        self.info_label.visible = False  # Hidden by default

        # Button for detailed stats (Qt widget, not magicgui)
        self.detail_button = QPushButton("View Detailed Statistics")
        self.detail_button.clicked.connect(self._show_detailed_stats)
        self.detail_button.setVisible(False)  # Hidden by default

        # Store current classifier and db for detail view
        self._current_classifier: str | None = None
        self._current_db: TrainingDB | None = None

    def update_info(self, classifier_name: str, db: TrainingDB) -> None:
        """Update panel with statistics for the selected classifier.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance for querying
        """
        # Store for detailed stats view
        self._current_classifier = classifier_name
        self._current_db = db

        try:
            # Get classifier info
            classifier = db.get_classifier(classifier_name)
            if not classifier:
                self.info_label.value = (
                    f"Classifier '{classifier_name}' not found in database"
                )
                self.info_label.visible = True
                self.detail_button.setVisible(False)
                return

            # Query stats
            n_sessions = db.get_session_count(classifier_name)
            n_annotations = db.get_total_annotations(classifier_name)
            classes = db.get_classes(classifier_name)
            dist = db.get_class_distribution(classifier_name)

            # Format stats text matching CLI output
            info_text = self._format_stats(
                classifier["id"],
                n_sessions,
                n_annotations,
                classes,
                dist,
            )

            self.info_label.value = info_text
            self.info_label.visible = True  # Show when data is loaded

            # Show detail button only if there are sessions/annotations
            self.detail_button.setVisible(n_sessions > 0)

        except Exception as e:
            logger.error(f"Failed to load classifier stats: {e}")
            self.info_label.value = f"Error loading statistics: {e!s}"
            self.info_label.visible = True
            self.detail_button.setVisible(False)

    def _format_stats(
        self,
        classifier_id: int,
        n_sessions: int,
        n_annotations: int,
        classes: list[str],
        distribution: dict[str, int],
    ) -> str:
        """Format statistics text matching omero-train stats output.

        Args:
            classifier_id: Classifier ID
            n_sessions: Total number of sessions
            n_annotations: Total number of annotations
            classes: List of defined class labels
            distribution: Dict of class label -> count

        Returns:
            Formatted text for display
        """
        # General Stats Section
        classes_str = ", ".join(classes) if classes else "None"
        text = "Dataset Information:\n\n"
        text += f"ID: {classifier_id}\n"
        text += f"Total Sessions: {n_sessions}\n"
        text += f"Total Annotations: {n_annotations}\n"
        text += f"Defined Classes: {classes_str}\n"

        # Class Distribution Section
        if distribution and n_annotations > 0:
            text += "\nClass Distribution:\n"
            for label, count in distribution.items():
                percentage = count / n_annotations * 100
                text += f"  {label:12s} {count:6d}  ({percentage:5.1f}%)\n"
        elif n_annotations == 0:
            text += "\nNo annotations yet"

        return text

    def clear(self) -> None:
        """Clear the info panel and hide it."""
        self.info_label.value = "Select a classifier to view details"
        self.info_label.visible = False
        self.detail_button.setVisible(False)
        self._current_classifier = None
        self._current_db = None

    def _show_detailed_stats(self) -> None:
        """Show detailed statistics dialog."""
        if self._current_classifier and self._current_db:
            try:
                dialog = DetailedStatsDialog(
                    self._current_classifier,
                    self._current_db,
                    parent=self.detail_button,
                )
                dialog.exec_()
            except Exception as e:
                logger.exception(f"Failed to show detailed statistics: {e}")


class ClassifierSelector:
    """Classifier selection widget with database integration.

    Provides a dropdown to select from available classifiers and displays
    detailed statistics in an expandable info panel.
    """

    def __init__(
        self,
        db: TrainingDB | None = None,
        auto_fill_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the classifier selector.

        Args:
            db: TrainingDB instance. If None, creates a new one.
            auto_fill_callback: Optional callback to trigger when classifier
                is selected. Receives the classifier name as argument.
        """
        self.db = db if db is not None else TrainingDB()
        self.auto_fill_callback = auto_fill_callback
        self._initializing = True  # Flag to prevent auto-trigger on init

        # Create Qt widget container for label and combobox
        self.selector_widget = QWidget()
        selector_layout = QVBoxLayout()
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(2)

        self.label = QLabel("Select Classifier:")
        self.combobox = QComboBox()
        self.combobox.addItem("-- Select --")

        selector_layout.addWidget(self.label)
        selector_layout.addWidget(self.combobox)
        self.selector_widget.setLayout(selector_layout)

        # Create info panel
        self.info_panel = ClassifierInfoPanel()

        # Initial load (before connecting signal to avoid trigger)
        self.refresh_classifiers()

        # Connect selection handler AFTER initial load
        self.combobox.currentTextChanged.connect(self._on_selection_changed)

        self._initializing = False

    def refresh_classifiers(self) -> None:
        """Query database and update dropdown choices."""
        try:
            logger.info(
                f"Loading classifiers from database: {self.db.db_path}"
            )
            classifiers = self.db.list_classifiers()
            logger.info(
                f"Found {len(classifiers) if classifiers else 0} classifiers"
            )

            # Store current value to preserve selection if refreshing
            current_text = self.combobox.currentText()

            # Clear and repopulate
            self.combobox.clear()

            if not classifiers:
                logger.warning("No classifiers found in database")
                self.combobox.addItem("-- Select --")
                self.combobox.addItem("(No classifiers available)")
                self.combobox.setCurrentIndex(0)
                self.info_panel.clear()
                return

            # Add placeholder and classifier names
            self.combobox.addItem("-- Select --")
            for clf in classifiers:
                self.combobox.addItem(clf["name"])

            # Restore previous selection if it still exists
            if current_text and current_text != "-- Select --":
                index = self.combobox.findText(current_text)
                if index >= 0:
                    self.combobox.setCurrentIndex(index)
                    logger.info(f"Restored selection: {current_text}")
                else:
                    self.combobox.setCurrentIndex(0)
            else:
                self.combobox.setCurrentIndex(0)

            logger.info(f"Loaded {len(classifiers)} classifiers successfully")

        except Exception as e:
            logger.exception(f"Failed to load classifiers from database: {e}")
            self.combobox.clear()
            self.combobox.addItem("-- Select --")
            self.combobox.addItem("(Error loading classifiers)")
            self.combobox.setCurrentIndex(0)
            self.info_panel.clear()

    def _on_selection_changed(self, text: str) -> None:
        """Handle classifier selection change.

        Args:
            text: Selected classifier name
        """
        # Ignore changes during initialization
        if self._initializing:
            return

        # Clear info panel if placeholder or no valid selection
        if not text or text == "-- Select --" or text.startswith("("):
            self.info_panel.clear()
            return

        # Update info panel with selected classifier
        self.info_panel.update_info(text, self.db)

        # Trigger auto-fill callback if provided
        if self.auto_fill_callback:
            try:
                self.auto_fill_callback(text)
            except Exception as e:
                logger.warning(f"Auto-fill callback failed: {e}")

    def get_widgets(self) -> list[Label]:
        """Get list of widgets to add to parent container.

        Returns:
            List containing selector widget and info panel label
        """
        # Return the combined Qt widget and the magicgui info label
        # Note: magicgui Containers don't handle raw Qt widgets well
        # So we just return the magicgui info_label which works
        return [
            self.info_panel.info_label,
        ]

    def get_selector_widget(self) -> QWidget:
        """Get the Qt widget containing the selector.

        Returns:
            QWidget containing label and combobox
        """
        return self.selector_widget

    def create_widget(self) -> Container:  # type: ignore
        """Create magicgui Container with dropdown and info panel.

        Returns:
            Container with combobox and info panel widgets
        """
        return Container(widgets=self.get_widgets())
