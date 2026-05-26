import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, RadioButtons
from omero_screen.config import get_logger
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari._classifier_selector import ClassifierSelector
from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.trainingdata_db.database import TrainingDB

if TYPE_CHECKING:
    from napari.viewer import Viewer

    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData

logger = get_logger(__name__)


def training_widget(
    class_name: str | None = None,
    user_data: "UserData | None" = userdata,
) -> Container:  # type: ignore
    widget = TrainingWidget(class_name, user_data, omero_data)
    return widget.container


class ImageNavigator:
    def __init__(
        self, class_options: list[str] | None, omero_data_inst: "OmeroData"
    ) -> None:
        self.omero_data = omero_data_inst
        self.current_index = 0
        self.first_load = True  # Flag to check if it is the first image load
        self.saved_contrast_limits: Any = (
            None  # Variable to save user settings
        )
        self.class_options = (
            [
                "unassigned",
            ]
            if class_options is None
            else class_options
        )
        self.class_choice = RadioButtons(
            label="Select class:",
            choices=self.class_options,
        )
        self.class_choice.value = "unassigned"
        self.class_choice.changed.connect(self.assign_class)

    def next_image(self) -> None:
        if self.omero_data.selected_images:
            self.current_index = (self.current_index + 1) % len(
                self.omero_data.selected_images
            )
            self.update_image()

    def previous_image(self) -> None:
        if self.omero_data.selected_images:
            self.current_index = (self.current_index - 1) % len(
                self.omero_data.selected_images
            )
            self.update_image()

    def update_image(self) -> None:
        viewer = napari.current_viewer()
        current_choices = self.class_choice.choices
        self.class_choice.changed.disconnect(self.assign_class)

        self._save_current_settings(viewer)
        self._clear_existing_layers(viewer)

        if self.omero_data.selected_images:
            image = self.omero_data.selected_images[self.current_index]
            logger.info(f"Selected image shape: {image.shape}")

            self._add_image_to_viewer(viewer, image)
            self._verify_layer_added(viewer)
            self._apply_saved_contrast_limits(viewer)

            self.first_load = False  # Update the flag after the first load
        else:
            logger.warning("No selected images to load.")

        self._restore_class_choices(current_choices)
        self._refresh_viewer(viewer)

    def _save_current_settings(self, viewer: "Viewer") -> None:
        if not self.first_load and viewer.layers:
            self.saved_contrast_limits = viewer.layers[0].contrast_limits
            logger.info(
                f"Saving contrast limits: {self.saved_contrast_limits}"
            )

    def _clear_existing_layers(self, viewer: "Viewer") -> None:
        viewer.layers.clear()
        logger.info("Viewer layers cleared.")

    def _add_image_to_viewer(
        self, viewer: "Viewer", image: "np.ndarray[Any, Any]"
    ) -> None:
        try:
            if image.shape[-1] == 1:
                self._add_grayscale_image(viewer, image)
            else:
                self._add_rgb_image(viewer, image)
        except Exception as e:
            logger.error(f"Error adding image to viewer: {e}")

    def _add_grayscale_image(
        self, viewer: "Viewer", image: "np.ndarray[Any, Any]"
    ) -> None:
        grayscale_image = np.mean(image, axis=-1)
        logger.debug(f"Grayscale image shape: {grayscale_image.shape}")
        inverted_image = 1.0 - grayscale_image  # Invert the image
        viewer.add_image(
            inverted_image,
            name=f"Cropped Image {self.current_index}",
            colormap="gray",
        )
        logger.debug("Inverted grayscale image added to viewer.")

    def _add_rgb_image(
        self, viewer: "Viewer", image: "np.ndarray[Any, Any]"
    ) -> None:
        viewer.add_image(
            image, name=f"Cropped Image {self.current_index}", rgb=True
        )
        logger.debug("RGB image added to viewer.")

    def _verify_layer_added(self, viewer: "Viewer") -> None:
        if not viewer.layers:
            logger.error(
                "No layers present in the viewer after adding the image."
            )
            return

    def _apply_saved_contrast_limits(self, viewer: "Viewer") -> None:
        if not self.first_load and self.saved_contrast_limits:
            min_intensity, max_intensity = (
                viewer.layers[0].data.min(),
                viewer.layers[0].data.max(),
            )
            if (
                self.saved_contrast_limits[0] >= min_intensity
                and self.saved_contrast_limits[1] <= max_intensity
            ):
                logger.debug(
                    f"Applying contrast limits: {self.saved_contrast_limits}"
                )
                viewer.layers[0].contrast_limits = self.saved_contrast_limits
            else:
                logger.warning(
                    f"Contrast limits {self.saved_contrast_limits} are out of range for the new image intensity values."
                )
        else:
            logger.debug("No contrast limits to apply or first image load.")

    def _restore_class_choices(self, current_choices: Any) -> None:
        self.class_choice.choices = current_choices
        self.class_choice.changed.connect(self.assign_class)
        self.update_class_choice()
        logger.debug("Class choices restored and signal reconnected.")

    def _refresh_viewer(self, viewer: "Viewer") -> None:
        viewer.update_console({"layers": viewer.layers})
        logger.debug("Viewer updated successfully.")

    def reset_for_new_dataset(self) -> None:
        """Reset display state so napari auto-scales the new dataset."""
        self.first_load = True
        self.saved_contrast_limits = None

    def assign_class(self, class_name: str) -> None:
        if self.omero_data.selected_classes:
            self.omero_data.selected_classes[self.current_index] = class_name

    def assign_all_to_class(self, class_name: str) -> None:
        if self.omero_data.selected_classes:
            n = len(self.omero_data.selected_classes)
            self.omero_data.selected_classes[:] = [class_name] * n
            self.update_class_choice()
            logger.info(f"Assigned all {n} cells to class '{class_name}'")

    def update_class_choice(self) -> None:
        if self.omero_data.selected_classes:
            current_class = self.omero_data.selected_classes[
                self.current_index
            ]
        else:
            current_class = "unassigned"
        self.class_choice.value = (
            current_class
            if current_class in self.class_choice.choices
            else "unassigned"
        )


class TrainingWidget:
    def __init__(
        self,
        class_name: str | None,
        user_data: "UserData | None",
        omero_data_inst: "OmeroData",
        class_options: list[str] | None = None,
    ) -> None:
        self.image_navigator = ImageNavigator(class_options, omero_data_inst)
        self.user_data = user_data
        self.omero_data = omero_data_inst
        self.class_name = class_name

        self.training_data_saver: TrainingDataSaver | None = None
        self.setup_key_bindings(napari.current_viewer())

        # Initialize database and classifier selector
        self.db = TrainingDB()
        self.classifier_selector = ClassifierSelector(
            db=self.db,
            auto_fill_callback=None,
            on_session_loaded_callback=self._on_session_loaded,
            on_direct_load_callback=self._on_direct_load,
            omero_data=omero_data_inst,
            user_data=user_data,
        )

        self.next_image_widget = magicgui(call_button="Next Image")(
            self.next_image
        )
        self.previous_image_widget = magicgui(call_button="Previous Image")(
            self.previous_image
        )
        self.save_training_data_widget = magicgui(
            call_button="Save training data",
        )(self.save_training_data)

        self._assign_all_widget: QWidget = (
            QWidget()
        )  # placeholder; built in create_container
        self.container = self.create_container()

    def _build_assign_all_widget(self) -> QWidget:
        """Build a widget with one 'Assign all' button per non-unassigned class."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        label = QLabel("Assign all cells to:")
        layout.addWidget(label)
        for class_name in self.image_navigator.class_options:
            if class_name == "unassigned":
                continue
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton(class_name)
            btn.setToolTip(f"Set every cell in this dataset to '{class_name}'")
            btn.clicked.connect(
                lambda checked, cn=class_name: self._confirm_assign_all(cn)
            )
            row_layout.addWidget(btn)
            layout.addWidget(row)
        return widget

    def _confirm_assign_all(self, class_name: str) -> None:
        n = len(self.image_navigator.omero_data.selected_classes or [])
        reply = QMessageBox.question(
            None,
            "Assign all cells",
            f"Assign all {n} cells to '{class_name}'?\nThis will overwrite existing assignments.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.image_navigator.assign_all_to_class(class_name)

    def update_class_options(self, class_options: list[str]) -> None:
        self.image_navigator.class_options = class_options
        self.image_navigator.class_choice.choices = class_options
        # Rebuild the assign-all buttons for the new class list
        new_widget = self._build_assign_all_widget()
        layout = self.container.native.layout()
        layout.replaceWidget(self._assign_all_widget, new_widget)
        self._assign_all_widget.deleteLater()
        self._assign_all_widget = new_widget
        self.image_navigator.update_class_choice()

    def next_image(self) -> None:
        if not self.omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.next_image()

    def previous_image(self) -> None:
        if not self.omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.previous_image()

    def save_training_data(self) -> None:
        # TrainingDataSaver reads omero_data.session_file_path live on
        # every save (see TrainingDataSaver._resolve_paths), so we don't
        # need to re-create the saver after a session/direct load mutates
        # that field — the next save picks up the new path automatically.
        if self.training_data_saver:
            self.training_data_saver.save_data_wrapper()
        else:
            print("Training data saver not initialized.")

    def setup_key_bindings(self, viewer: "Viewer") -> None:
        @viewer.bind_key("w", overwrite=True)
        def trigger_next_image(event: Any = None) -> None:
            self.next_image()

        @viewer.bind_key("q", overwrite=True)
        def trigger_previous_image(event: Any = None) -> None:
            self.previous_image()

    def _on_session_loaded(self) -> None:
        """Callback triggered when a session is loaded from the browser.

        This updates the viewer to display the first image from the loaded session.
        """
        if not self.omero_data.selected_images:
            logger.warning("No images loaded from session")
            return

        logger.info(
            f"Session loaded with {len(self.omero_data.selected_images)} images"
        )

        # Extract unique class labels from the loaded data
        if self.omero_data.selected_classes:
            unique_classes = list(set(self.omero_data.selected_classes))
            logger.info(f"Found unique classes in session: {unique_classes}")

            # Also try to load class options from metadata
            classifier_name = self.classifier_selector.combobox.currentText()
            if classifier_name and classifier_name != "-- Select --":
                try:
                    metadata_path = (
                        Path.home()
                        / "omeroscreen_trainingdata"
                        / classifier_name
                        / "metadata.json"
                    )
                    if metadata_path.exists():
                        with metadata_path.open() as f:
                            metadata = json.load(f)
                        class_options = metadata.get(
                            "class_options", unique_classes
                        )
                        logger.info(
                            f"Loaded class options from metadata: {class_options}"
                        )
                        self.update_class_options(class_options)
                    else:
                        # Use unique classes from the data if no metadata
                        logger.info(
                            "No metadata found, using unique classes from data"
                        )
                        self.update_class_options(unique_classes)
                except Exception as e:
                    logger.warning(
                        f"Could not load class options, using unique classes: {e}"
                    )
                    self.update_class_options(unique_classes)

        # (Re-)create TrainingDataSaver, reusing the loaded session's file path
        # so that saving updates the existing NPY rather than creating a new one.
        classifier_name = self.classifier_selector.combobox.currentText()
        if (
            classifier_name
            and classifier_name != "-- Select --"
            and not classifier_name.startswith("(")
            and self.user_data
        ):
            self.class_name = classifier_name
            self.training_data_saver = TrainingDataSaver(
                classifier_name,
                self.omero_data,
                self.user_data,
                self.image_navigator,
            )
            logger.info(
                f"TrainingDataSaver (re)initialized for classifier {classifier_name}"
            )

        self.image_navigator.current_index = 0
        self.image_navigator.reset_for_new_dataset()
        self.image_navigator.update_image()
        logger.info(
            f"Displaying {len(self.omero_data.selected_images)} "
            "images from loaded session"
        )

    def _on_direct_load(self) -> None:
        """Callback triggered when data is loaded directly from OMERO.

        This updates the viewer to display the first image from the loaded data.
        """
        if not self.omero_data.selected_images:
            logger.warning("No images loaded via direct OMERO loading")
            return

        logger.info(
            f"Direct load callback triggered with {len(self.omero_data.selected_images)} images"
        )

        # Get classifier name from the classifier selector
        classifier_name = self.classifier_selector.combobox.currentText()
        if (
            classifier_name
            and classifier_name != "-- Select --"
            and not classifier_name.startswith("(")
        ):
            self.class_name = classifier_name
            logger.info(f"Using classifier: {classifier_name}")

            # Load class options from metadata
            try:
                metadata_path = (
                    Path.home()
                    / "omeroscreen_trainingdata"
                    / classifier_name
                    / "metadata.json"
                )
                if metadata_path.exists():
                    with metadata_path.open() as f:
                        metadata = json.load(f)
                    class_options = metadata.get(
                        "class_options", ["unassigned"]
                    )
                    logger.info(f"Loaded class options: {class_options}")
                    self.update_class_options(class_options)

                    # Always (re-)create TrainingDataSaver so file paths
                    # reflect the newly-loaded data, not a previous session.
                    # Clear session_file_path so a new indexed file is created.
                    if self.class_name and self.user_data:
                        self.omero_data.session_file_path = None
                        self.training_data_saver = TrainingDataSaver(
                            self.class_name,
                            self.omero_data,
                            self.user_data,
                            self.image_navigator,
                        )
                        logger.info(
                            f"TrainingDataSaver (re)initialized for classifier {self.class_name}"
                        )
                else:
                    logger.warning(f"Metadata file not found: {metadata_path}")
            except Exception as e:
                logger.exception(f"Could not load class options: {e}")
        else:
            logger.warning(
                "No classifier selected - cannot load class options"
            )

        # Update viewer
        self.image_navigator.current_index = 0
        self.image_navigator.reset_for_new_dataset()
        self.image_navigator.update_image()
        logger.info(
            f"Displaying {len(self.omero_data.selected_images)} "
            "images loaded directly from OMERO"
        )

    def create_container(self) -> Container:  # type: ignore
        # Create container with magicgui widgets
        widgets = [
            self.previous_image_widget,
            self.next_image_widget,
            self.image_navigator.class_choice,
            self.save_training_data_widget,
        ]
        container = Container(widgets=widgets)

        # Insert Qt selector widget at the top of the native layout
        # Access the native Qt layout and insert our selector widget
        layout = container.native.layout()
        layout.insertWidget(0, self.classifier_selector.get_selector_widget())
        # Insert info panel label after selector
        layout.insertWidget(
            1, self.classifier_selector.info_panel.info_label.native
        )
        # Insert manage button after info panel
        layout.insertWidget(
            2, self.classifier_selector.info_panel.manage_button
        )

        # Build and insert the assign-all buttons widget after class_choice
        self._assign_all_widget = self._build_assign_all_widget()
        # class_choice is at index 5 (0=selector, 1=info, 2=manage, 3=prev, 4=next, 5=class_choice)
        # append after the last inserted widget; find class_choice position and insert after it
        class_choice_native = self.image_navigator.class_choice.native
        for i in range(layout.count()):
            if (
                layout.itemAt(i)
                and layout.itemAt(i).widget() is class_choice_native
            ):
                layout.insertWidget(i + 1, self._assign_all_widget)
                break
        else:
            layout.addWidget(self._assign_all_widget)

        return container


class TrainingDataSaver:
    """Saves training data + metadata for the current annotation session.

    Path and metadata derivation happens lazily at save time via properties
    rather than being cached in ``__init__``. Earlier versions cached
    ``file_path`` once at construction; after a direct-load or
    session-load mutated ``omero_data.session_file_path``, the cached
    path went stale and writes landed in the wrong file. The previous
    workaround was to re-create the saver before every save — gone now
    that ``file_path``/``file_name``/``meta_data_path``/``metadata`` are
    derived live from the singletons.
    """

    def __init__(
        self,
        classifier_name: str,
        omero_data_inst: "OmeroData",
        user_data: "UserData",
        image_navigator: ImageNavigator,
    ) -> None:
        self.classifier_name = classifier_name
        self.omero_data = omero_data_inst
        self.user_data = user_data
        self.image_navigator = image_navigator
        self.home_dir = Path.home() / "omeroscreen_trainingdata"
        self.classifier_dir = self.home_dir / self.classifier_name
        # training_dict is rebuilt on every save (see `save_both` /
        # `_save_training_data`); the initial value is just so callers
        # that introspect it before the first save get something sane.
        self.training_dict = self._create_training_dict()

    @property
    def file_name(self) -> str:
        return self._resolve_paths()[0]

    @property
    def file_path(self) -> Path:
        return self._resolve_paths()[1]

    @property
    def meta_data_path(self) -> Path:
        return self._resolve_paths()[2]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._create_metadata_dict()

    def save_data_wrapper(self) -> None:
        try:
            self._validate_classifier_name(self.classifier_name)
            if not self.classifier_dir.exists():
                self._create_and_save()
                return

            file_check = self._check_directory_contents()
            self._handle_saving_logic(file_check)
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            _show_error_message(str(e))

    def _create_and_save(self) -> None:
        self._create_directory(self.classifier_dir)
        self.save_both()

    def _check_directory_contents(self) -> str:
        logger.info(f"Directory {self.classifier_dir} already exists.")
        contents = list(self.classifier_dir.iterdir())
        return self.check_files(contents)

    def _handle_saving_logic(self, file_check: str) -> None:
        if file_check in ["empty", "neither"]:
            self.save_both()
        elif file_check == "metadata":
            self._handle_metadata_present()
        elif file_check == "training_data":
            self._handle_training_data_present()
        elif file_check == "both":
            self._handle_both_present()

    def _handle_metadata_present(self) -> None:
        if self.compare_metadata():
            self._save_training_data()
        elif _show_confirmation_dialog(
            f"Metadata has changed. Do you want to overwrite them and save {self.file_name}?"
        ):
            self.save_both()

    def _handle_training_data_present(self) -> None:
        if _show_confirmation_dialog(
            f"The file {self.file_name} already exists without metadata. Do you want to overwrite the file and save the metadata?"
        ):
            self.save_both()

    def _handle_both_present(self) -> None:
        if self.compare_metadata():
            if _show_confirmation_dialog(
                f"Do you want to overwrite {self.file_name}?"
            ):
                self._save_training_data()
        elif _show_confirmation_dialog(
            f"{self.file_name} and metadata have changed. Do you want to overwrite both?"
        ):
            self.save_both()

    def save_both(self) -> None:
        self.training_dict = self._create_training_dict()
        np.save(self.file_path, self.training_dict)  # type: ignore
        self._save_metadata(self.meta_data_path, self.metadata)
        self._save_to_database()

        logger.info(f"File and metadata saved to: {self.classifier_dir}")
        _show_success_message(
            f"Data for {self.file_name} and metadata successfully saved."
        )

    def _save_training_data(self) -> None:
        self.training_dict = self._create_training_dict()
        np.save(self.file_path, self.training_dict)  # type: ignore

        # Save to DB
        self._save_to_database()

        logger.info(
            f"File saved to: {self.file_path}, metadata already present"
        )
        _show_success_message(
            f"Data for image {self.file_name} successfully saved, with metadata present."
        )

    def _resolve_paths(self) -> tuple[str, Path, Path]:
        """Compute ``(file_name, file_path, meta_data_path)`` live.

        Reads ``omero_data.session_file_path`` on every call so that any
        load that updates the singleton (session reload, direct load,
        setup widget's initial save) is reflected immediately. If a
        session path is set, that file is reused in place. Otherwise the
        path is derived from current plate/well/image/timepoint, with
        ``_2``/``_3`` suffixes appended to avoid clobbering existing
        files.
        """
        meta_data_path = self.classifier_dir / "metadata.json"
        existing = self.omero_data.session_file_path
        if existing is not None:
            return existing.name, existing, meta_data_path
        plate = self.omero_data.plate_id
        well = self.omero_data.well_pos_list[0]
        image = self.omero_data.image_input
        time_point = self.user_data.timepoint
        base = f"{plate}_{well}_{image}_{time_point}"
        file_path = self.classifier_dir / f"{base}.npy"
        if file_path.exists():
            index = 2
            while (self.classifier_dir / f"{base}_{index}.npy").exists():
                index += 1
            file_path = self.classifier_dir / f"{base}_{index}.npy"
        return file_path.name, file_path, meta_data_path

    def check_files(self, contents: list[Path]) -> str:
        has_metadata = any(file.name == "metadata.json" for file in contents)
        has_self_file = any(file.name == self.file_name for file in contents)

        if not contents:
            return "empty"
        if has_metadata:
            return "both" if has_self_file else "metadata"
        return "training_data" if has_self_file else "neither"

    def compare_metadata(self) -> bool:
        with self.meta_data_path.open("r") as json_file:
            existing_metadata = json.load(json_file)
        return existing_metadata == self.metadata  # type: ignore

    def _validate_classifier_name(self, text_input: str) -> None:
        if not text_input.strip():
            logger.error("No classifier name provided.")
            raise ValueError(
                "Failed to create directory: no classifier name provided."
            )

    def _create_directory(self, directory: Path) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise ValueError(
                f"Failed to create directory {directory}: {e}"
            ) from e

    def _save_to_database(self) -> None:
        """Save training data to SQLite database."""
        try:
            db = TrainingDB()

            # Store image_input string in metadata for session manager display
            self.metadata["image_input"] = self.omero_data.image_input

            # Resolve Image ID
            try:
                # Try to get single image ID from image_input
                image_id = int(self.omero_data.image_input)
            except (ValueError, TypeError):
                # Fallback: use first ID from the list if available
                if self.omero_data.image_ids:
                    image_id = self.omero_data.image_ids[0]
                else:
                    logger.warning(
                        "Could not resolve Image ID for database. Defaulting to 0."
                    )
                    image_id = 0

            # Look up by file path so that indexed files (_2.npy, _3.npy …)
            # always create a fresh session rather than updating the first one.
            existing_session = db.get_session_by_file_path(str(self.file_path))

            if existing_session:
                session_id = existing_session["id"]
                db.update_session(
                    session_id,
                    file_path=str(self.file_path),
                    metadata=self.metadata,
                )
            else:
                session_id = db.create_session(
                    classifier_name=self.classifier_name,
                    plate_id=self.omero_data.plate_id,
                    well=self.omero_data.well_pos_list[0],
                    image_id=image_id,
                    timepoint=self.user_data.timepoint,
                    file_path=str(self.file_path),
                    metadata=self.metadata,
                )

            # Prepare annotations – use sequential crop index for
            # cell_index to guarantee uniqueness; np.max(label_mask)
            # can collide when neighbouring cells bleed into crops.
            annotations = [
                (idx, class_label)
                for idx, class_label in enumerate(
                    self.omero_data.selected_classes
                )
            ]

            # Replace annotations
            db.delete_annotations(session_id)
            if annotations:
                db.add_annotations(
                    session_id,
                    annotations,
                    cell_meta=self.omero_data.selected_cell_meta or None,
                )

            logger.info(
                f"Saved {len(annotations)} annotations to database session {session_id}."
            )

        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            _show_error_message(f"Database Save Error: {e}")

    def _create_training_dict(self) -> dict[str, Any]:
        logger.info(
            f"Creating training data dictionary with {len(self.omero_data.selected_classes)} entries."
        )
        crops = self.omero_data.selected_crops
        if crops and np.issubdtype(crops[0].dtype, np.integer):
            iinfo = np.iinfo(crops[0].dtype)
            crops = [c.astype(np.float32) / iinfo.max for c in crops]
        return {
            "data": (
                crops,
                self.omero_data.selected_labels,
            ),
            "target": self.omero_data.selected_classes,
        }

    def _create_metadata_dict(self) -> dict[str, Any]:
        user_data_dict = asdict(self.user_data)
        user_data_dict.pop("well", None)
        metadata: dict[str, Any] = {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
        }
        # Save channel_data so saved sessions can resolve channel names
        if self.omero_data.channel_data:
            metadata["channel_data"] = dict(self.omero_data.channel_data)
        return metadata

    def _save_metadata(
        self, meta_data_path: Path, metadata: "dict[str, Any]"
    ) -> None:
        with meta_data_path.open("w") as json_file:
            json.dump(metadata, json_file)


def _show_error_message(message: str) -> None:
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(message)
    msg_box.setWindowTitle("Error")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()


def _show_success_message(message: str) -> None:
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setText(message)
    msg_box.setWindowTitle("Success")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()


def _show_confirmation_dialog(message: str) -> bool:
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(message)
    msg_box.setWindowTitle("Warning")
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    reply = msg_box.exec_()
    return bool(reply == QMessageBox.Yes)
