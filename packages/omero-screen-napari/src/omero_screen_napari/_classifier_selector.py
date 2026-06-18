"""Classifier selector widget with database integration.

This module provides a reusable classifier selection component that displays
available classifiers from the training database and shows detailed statistics.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from magicgui.widgets import Container, Label
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.trainingdata_db.database import TrainingDB


class ClassifierInfoPanel:
    """Info panel showing classifier statistics.

    Displays information matching the `omero-train stats` CLI output format:
    - General stats (ID, sessions, annotations, classes)
    - Class distribution with percentages
    - Button to open unified session manager
    """

    def __init__(
        self,
        on_session_loaded_callback: Callable[[], None] | None = None,
        on_direct_load_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the info panel.

        Args:
            on_session_loaded_callback: Optional callback when session is loaded
            on_direct_load_callback: Optional callback when new data is loaded via OMERO
        """
        self.info_label = Label(value="Select a classifier to view details")
        self.info_label.visible = False

        # Button for managing sessions (Qt widget, not magicgui)
        self.manage_button = QPushButton("Manage Sessions")
        self.manage_button.clicked.connect(self._show_session_manager)
        self.manage_button.setVisible(False)

        # Store current classifier and db for session manager
        self._current_classifier: str | None = None
        self._current_db: TrainingDB | None = None
        self._omero_data: Any = None
        self._user_data: Any = None

        # Store callbacks
        self._on_session_loaded_callback = on_session_loaded_callback
        self._on_direct_load_callback = on_direct_load_callback

    def update_info(
        self,
        classifier_name: str,
        db: TrainingDB,
        omero_data: Any = None,
        user_data: Any = None,
    ) -> None:
        """Update panel with statistics for the selected classifier.

        Args:
            classifier_name: Name of the classifier
            db: TrainingDB instance for querying
            omero_data: OmeroData instance (for session manager)
            user_data: UserData instance (for session manager)
        """
        self._current_classifier = classifier_name
        self._current_db = db
        self._omero_data = omero_data
        self._user_data = user_data

        try:
            classifier = db.get_classifier(classifier_name)
            if not classifier:
                self.info_label.value = (
                    f"Classifier '{classifier_name}' not found in database"
                )
                self.info_label.visible = True
                self.manage_button.setVisible(False)
                return

            # Query stats
            n_sessions = db.get_session_count(classifier_name)
            n_annotations = db.get_total_annotations(classifier_name)
            classes = db.get_classes(classifier_name)
            dist = db.get_class_distribution(classifier_name)

            info_text = self._format_stats(
                classifier["id"],
                n_sessions,
                n_annotations,
                classes,
                dist,
            )

            self.info_label.value = info_text
            self.info_label.visible = True

            # Show manage button if classifier has metadata or sessions
            has_metadata = (
                Path.home()
                / "omeroscreen_trainingdata"
                / classifier_name
                / "metadata.json"
            ).exists()
            has_sessions = n_sessions > 0
            self.manage_button.setVisible(has_metadata or has_sessions)

        except Exception as e:
            logger.error(f"Failed to load classifier stats: {e}")
            self.info_label.value = f"Error loading statistics: {e!s}"
            self.info_label.visible = True
            self.manage_button.setVisible(False)

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
        classes_str = ", ".join(classes) if classes else "None"
        text = "Dataset Information:\n\n"
        text += f"ID: {classifier_id}\n"
        text += f"Total Sessions: {n_sessions}\n"
        text += f"Total Annotations: {n_annotations}\n"
        text += f"Defined Classes: {classes_str}\n"

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
        self.manage_button.setVisible(False)
        self._current_classifier = None
        self._current_db = None
        self._omero_data = None
        self._user_data = None

    def _show_session_manager(self) -> None:
        """Show unified session manager dialog."""
        if self._current_classifier and self._current_db:
            try:
                from omero_screen_napari._session_manager_widget import (
                    AnnotationSessionManager,
                )

                od = (
                    self._omero_data
                    if self._omero_data is not None
                    else omero_data
                )
                ud = (
                    self._user_data
                    if self._user_data is not None
                    else userdata
                )

                dialog = AnnotationSessionManager(
                    classifier_name=self._current_classifier,
                    db=self._current_db,
                    omero_data=od,
                    user_data=ud,
                    on_session_loaded_callback=self._on_session_loaded_callback,
                    on_direct_load_callback=self._on_direct_load_callback,
                    parent=self.manage_button,
                )
                dialog.exec_()
            except Exception as e:
                logger.exception(f"Failed to show session manager: {e}")


class ClassifierSelector:
    """Classifier selection widget with database integration.

    Provides a dropdown to select from available classifiers and displays
    detailed statistics in an expandable info panel.
    """

    def __init__(
        self,
        db: TrainingDB | None = None,
        auto_fill_callback: Callable[[str], None] | None = None,
        on_session_loaded_callback: Callable[[], None] | None = None,
        on_direct_load_callback: Callable[[], None] | None = None,
        omero_data: Any = None,
        user_data: Any = None,
    ) -> None:
        """Initialize the classifier selector.

        Args:
            db: TrainingDB instance. If None, creates a new one.
            auto_fill_callback: Optional callback to trigger when classifier
                is selected. Receives the classifier name as argument.
            on_session_loaded_callback: Optional callback to trigger when a
                session is loaded from the session manager.
            on_direct_load_callback: Optional callback to trigger when new
                data is loaded via direct OMERO loading.
            omero_data: OmeroData instance for session manager.
            user_data: UserData instance for session manager.
        """
        self.db = db if db is not None else TrainingDB()
        self.auto_fill_callback = auto_fill_callback
        self.omero_data = omero_data
        self.user_data = user_data
        self._initializing = True

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
        self.info_panel = ClassifierInfoPanel(
            on_session_loaded_callback=on_session_loaded_callback,
            on_direct_load_callback=on_direct_load_callback,
        )

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

            current_text = self.combobox.currentText()

            self.combobox.clear()

            if not classifiers:
                logger.warning("No classifiers found in database")
                self.combobox.addItem("-- Select --")
                self.combobox.addItem("(No classifiers available)")
                self.combobox.setCurrentIndex(0)
                self.info_panel.clear()
                return

            self.combobox.addItem("-- Select --")
            for clf in classifiers:
                self.combobox.addItem(clf["name"])

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
        if self._initializing:
            return

        if not text or text == "-- Select --" or text.startswith("("):
            self.info_panel.clear()
            return

        self.info_panel.update_info(
            text, self.db, omero_data=self.omero_data, user_data=self.user_data
        )

        if self.auto_fill_callback:
            try:
                self.auto_fill_callback(text)
            except Exception as e:
                logger.warning(f"Auto-fill callback failed: {e}")

    def get_widgets(self) -> list[Label]:
        """Get list of widgets to add to parent container.

        Returns:
            List containing info panel label
        """
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
