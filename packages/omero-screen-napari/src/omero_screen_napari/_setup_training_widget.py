import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger
from magicgui import magicgui
from magicgui.widgets import Container, Label
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_userdata_singleton import (
    userdata as global_user_data,
)
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.trainingdata_db.database import TrainingDB

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData


def setup_training_widget(
    class_options: list[str] | None = None,
    class_name: str | None = None,
    user_data: "UserData | None" = global_user_data,
) -> Container:  # type: ignore
    widget = SetupTrainingWidget(
        class_options, class_name, user_data, omero_data
    )
    return widget.container


class ImageNavigator:
    def __init__(self, class_options: list[str] | None) -> None:
        self.class_options = (
            ["unassigned"] if class_options is None else class_options
        )
        self.class_labels = Container(
            widgets=[
                Label(value=class_name) for class_name in self.class_options
            ]
        )

    def add_class(self, class_name: str) -> None:
        if class_name and class_name not in self.class_options:
            self.class_options.append(class_name)
            self.refresh_class_labels()
            logger.debug(f"Class {class_name} added to choices.")

    def reset_class_options(self) -> None:
        self.class_options = ["unassigned"]
        self.refresh_class_labels()
        logger.debug("Class choices reset to default.")

    def refresh_class_labels(self) -> None:
        self.class_labels.clear()
        for class_name in self.class_options:
            self.class_labels.append(Label(value=class_name))


class SetupTrainingWidget:
    def __init__(
        self,
        class_options: list[str] | None,
        class_name: str | None,
        user_data: "UserData | None",
        omero_data_inst: "OmeroData",
    ) -> None:
        self.image_navigator = ImageNavigator(class_options)
        self.user_data = user_data
        self.omero_data = omero_data_inst
        self.class_name = class_name or "Classifier Name"

        self.db = TrainingDB()

        self.meta_data_saver = MetaDataSaver(
            self.class_name,
            self.omero_data,
            self.user_data,
            self.image_navigator,
        )

        _CELLCYCLE_CHOICES = ["All", "G1", "S", "G2/M", "G2", "M", "Polyploid"]

        self.add_class_widget = magicgui(
            call_button="Enter", text_input={"label": "Class name"}
        )(self.add_class)
        self.reset_class_options_widget = magicgui(
            call_button="Reset class options"
        )(self.reset_class_options)
        self.cellcycle_widget = magicgui(
            cellcycle={"choices": _CELLCYCLE_CHOICES, "label": "Cell cycle"},
        )(self._set_cellcycle)
        self.save_meta_data_widget = magicgui(
            call_button="Save metadata",
            text_input={"label": "filename", "value": self.class_name},
        )(self.save_meta_data)

        self.container = self.create_container()

    def add_class(self, text_input: str) -> None:
        if text_input:
            self.image_navigator.add_class(text_input)
            self.add_class_widget.text_input.value = ""

    def reset_class_options(self) -> None:
        self.image_navigator.reset_class_options()

    def _set_cellcycle(self, cellcycle: str = "All") -> None:
        if self.user_data is not None:
            self.user_data.cellcycle = cellcycle

    def save_meta_data(self, text_input: str) -> None:
        # Sync cellcycle selection into user_data before saving
        if self.user_data is not None:
            self.user_data.cellcycle = self.cellcycle_widget.cellcycle.value
        if new_classifier_name := text_input.strip():
            self.meta_data_saver.update_classifier_name(new_classifier_name)
        self.meta_data_saver.save_data()

    def create_container(self) -> Container:  # type: ignore
        widgets = [
            self.add_class_widget,
            self.reset_class_options_widget,
            self.image_navigator.class_labels,
            self.cellcycle_widget,
            self.save_meta_data_widget,
        ]
        return Container(widgets=widgets)


class MetaDataSaver:
    def __init__(
        self,
        classifier_name: str,
        omero_data_inst: "OmeroData",
        user_data: "UserData | None",
        image_navigator: ImageNavigator,
    ) -> None:
        self.classifier_name = classifier_name
        self.omero_data = omero_data_inst
        self.user_data = user_data
        self.image_navigator = image_navigator
        self.home_dir = Path.home() / "omeroscreen_trainingdata"
        self.classifier_dir = self.home_dir / self.classifier_name
        self.meta_data_path = self._set_paths()
        self.metadata = self._create_metadata_dict()
        self._update_paths_and_metadata()
        self.on_save_callback: Any = None  # Callback to trigger after save

    def save_data(self) -> None:
        # Rebuild metadata from current user_data state so that settings
        # changed after widget creation (contour, no_background, etc.) are captured.
        self.metadata = self._create_metadata_dict()
        try:
            self._validate_classifier_name(self.classifier_name)

            # Database integration
            from omero_screen_napari.trainingdata_db.database import TrainingDB

            db = TrainingDB()
            exists_in_db = db.get_classifier(self.classifier_name)

            if not self.classifier_dir.exists():
                if exists_in_db:
                    raise ValueError(
                        f"Classifier '{self.classifier_name}' already exists in the database. "
                        "Please choose a different name."
                    )

                # Register new classifier in DB
                db.create_classifier(self.classifier_name)
                if self.image_navigator.class_options:
                    db.add_classes(
                        self.classifier_name,
                        self.image_navigator.class_options,
                    )
                self._create_and_save()
                self._save_initial_session(db)

                # Trigger callback after successful save
                if self.on_save_callback:
                    try:
                        self.on_save_callback()
                    except Exception as e:
                        logger.warning(f"Save callback failed: {e}")
                return

            # Directory exists
            if not exists_in_db:
                # Add existing local classifier to DB if missing
                logger.info(
                    f"Adding existing local classifier '{self.classifier_name}' to database."
                )
                db.create_classifier(self.classifier_name)
                # We should also populate classes if possible, but reading from metadata file might be needed
                # For now, let's just create the classifier.
                # If we have class_options loaded in self (which we should if we loaded metadata), add them
                if self.image_navigator.class_options:
                    try:
                        db.add_classes(
                            self.classifier_name,
                            self.image_navigator.class_options,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not populate classes for existing classifier: {e}"
                        )

            file_check = self._check_directory_contents()
            self._handle_saving_logic(file_check)
            self._save_initial_session(db)

            # Trigger callback after successful save
            if self.on_save_callback:
                try:
                    self.on_save_callback()
                except Exception as e:
                    logger.warning(f"Save callback failed: {e}")

        except Exception as e:
            logger.exception("Error saving data")
            self._show_error_message(str(e))

    def update_classifier_name(self, new_classifier_name: str) -> None:
        self.classifier_name = new_classifier_name
        self._update_paths_and_metadata()

    def _update_paths_and_metadata(self) -> None:
        self.classifier_dir = self.home_dir / self.classifier_name
        self.meta_data_path = self._set_paths()
        self.metadata = self._create_metadata_dict()

    def _create_and_save(self) -> None:
        self._create_directory(self.classifier_dir)
        self._save_metadata()

    def _save_initial_session(self, db: "TrainingDB") -> None:
        """Save the current in-memory gallery data as the initial session.

        This ensures that when a new classifier is created, the crop data
        already loaded in the gallery widget is persisted as an NPY file
        and registered in the database so it appears in session management.

        The gallery widget runs with classifier=False, so selected_crops is
        not populated. We re-parse with classifier=True using the gallery's
        rows/columns so the session contains the same number of crops the
        user chose for the gallery display.
        """
        # Allow zarr-loaded plates through: in zarr mode we deliberately
        # leave ``omero_data.images`` empty and rely on the gallery's
        # zarr fast path to fetch crops. The presence of ``image_ids``
        # (populated for both backends) is the right "do we have data?"
        # signal.
        has_in_memory = self.omero_data.images.size > 0
        has_zarr_loadable = bool(self.omero_data.image_ids)
        if not has_in_memory and not has_zarr_loadable:
            logger.info(
                "No images in memory and no zarr-loadable IDs — "
                "skipping initial session save."
            )
            return
        if not self.user_data:
            logger.info("No user_data — skipping initial session save.")
            return

        # Re-parse crops with classifier=True to populate selected_crops.
        # Keep the gallery's rows/columns so the session size matches the
        # gallery display (e.g. 4×4 = 16 crops, not all 211).
        from omero_screen_napari.gallery_api import (
            RandomImageParser,
            parse_crops_into_omero_data,
        )

        # Exclude cells already annotated for this classifier+well
        well = (
            self.omero_data.well_pos_list[0]
            if self.omero_data.well_pos_list
            else "unknown"
        )
        try:
            excluded = db.get_used_centroids(
                self.classifier_name, self.omero_data.plate_id, well
            )
        except Exception as exc:
            logger.warning(f"Could not query used centroids: {exc}")
            excluded = set()

        parse_crops_into_omero_data(
            self.omero_data, self.user_data, excluded_centroids=excluded
        )
        selector = RandomImageParser(
            self.omero_data, self.user_data, classifier=True
        )
        selector.parse_random_images()

        if not self.omero_data.selected_crops:
            logger.info("No crops generated — skipping initial session save.")
            return

        # Populate selected_classes with "unassigned" for all crops
        self.omero_data.selected_classes = [
            "unassigned" for _ in self.omero_data.selected_crops
        ]

        # Resolve image ID for DB record
        try:
            image_id = int(self.omero_data.image_input)
        except (ValueError, TypeError):
            if self.omero_data.image_ids:
                image_id = self.omero_data.image_ids[0]
            else:
                image_id = 0

        well = (
            self.omero_data.well_pos_list[0]
            if self.omero_data.well_pos_list
            else "unknown"
        )
        timepoint = self.user_data.timepoint

        # Build file path using image_input (not image_id) to match
        # TrainingWidget._set_paths() convention. Append an incrementing
        # index when the base name already exists so repeated runs on the
        # same well never overwrite a previous session.
        base = f"{self.omero_data.plate_id}_{well}_{self.omero_data.image_input}_{timepoint}"
        file_path = self.classifier_dir / f"{base}.npy"
        if file_path.exists():
            index = 2
            while (self.classifier_dir / f"{base}_{index}.npy").exists():
                index += 1
            file_path = self.classifier_dir / f"{base}_{index}.npy"

        # Normalize crops to float32 so loading via session_utils produces
        # consistent [0, 1] data regardless of original OMERO dtype (uint16 etc.)
        crops = self.omero_data.selected_crops
        if crops and np.issubdtype(crops[0].dtype, np.integer):
            iinfo = np.iinfo(crops[0].dtype)
            crops = [c.astype(np.float32) / iinfo.max for c in crops]

        # Save NPY file (same format as TrainingDataSaver._create_training_dict)
        training_dict: dict[str, Any] = {
            "data": (
                crops,
                self.omero_data.selected_labels,
            ),
            "target": self.omero_data.selected_classes,
        }
        np.save(file_path, training_dict, allow_pickle=True)  # type: ignore[arg-type,call-overload]
        logger.info(f"Initial session NPY saved to {file_path}")

        # Tell the training widget which file to use for subsequent saves.
        # Without this the training widget's stale saver would write to the
        # previous session's NPY, overwriting it.
        self.omero_data.session_file_path = file_path

        # Store image_input string in metadata for session manager display
        self.metadata["image_input"] = self.omero_data.image_input

        # Create DB session
        try:
            session_id = db.create_session(
                classifier_name=self.classifier_name,
                plate_id=self.omero_data.plate_id,
                well=well,
                image_id=image_id,
                timepoint=timepoint,
                file_path=str(file_path),
                metadata=self.metadata,
            )

            # Save annotations (all "unassigned")
            # Use sequential crop index for cell_index to guarantee
            # uniqueness; np.max(label_mask) can collide when
            # neighbouring cells bleed into multiple crops.
            annotations: list[tuple[int, str]] = [
                (idx, class_label)
                for idx, class_label in enumerate(
                    self.omero_data.selected_classes
                )
            ]

            if annotations:
                db.add_annotations(
                    session_id,
                    annotations,
                    cell_meta=self.omero_data.selected_cell_meta or None,
                )

            logger.info(
                f"Initial session {session_id} created with "
                f"{len(annotations)} annotations."
            )
        except Exception as e:
            logger.error(f"Failed to save initial session to database: {e}")

    def _check_directory_contents(self) -> Literal["metadata", "no_data"]:
        logger.info(f"Directory {self.classifier_dir} already exists.")
        contents = list(self.classifier_dir.iterdir())
        return self._check_files(contents)

    def _handle_saving_logic(self, file_check: str) -> None:
        if file_check == "no_data":
            self._save_metadata()
        elif file_check == "metadata":
            if self._compare_metadata():
                if self._show_confirmation_dialog(
                    "Old metadata already present. Do you want to overwrite it "
                    "anyway? Nothing will change"
                ):
                    self._save_metadata()
            elif self._show_confirmation_dialog(
                "Old metadata present but different. Do you want to change them?"
            ):
                self._save_metadata()
        else:
            logger.error(f"Problem with file check in {self.classifier_dir}")
            raise ValueError(
                f"Problem with file check in {self.classifier_dir}"
            )

    def _save_metadata(self) -> None:
        with self.meta_data_path.open("w") as json_file:
            json.dump(self.metadata, json_file)
        logger.info(f"Training Metadata saved to: {self.classifier_dir}")
        self._show_success_message(
            f"Metadata successfully saved to : {self.classifier_dir}."
        )

    def _set_paths(self) -> Path:
        return self.classifier_dir / "metadata.json"

    def _check_files(
        self, contents: list[Path]
    ) -> Literal["metadata", "no_data"]:
        has_metadata = any(file.name == "metadata.json" for file in contents)
        return "metadata" if has_metadata else "no_data"

    def _compare_metadata(self) -> bool:
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

    def _create_metadata_dict(self) -> dict[str, Any]:
        if not self.user_data:
            return {
                "user_data": None,
                "class_options": self.image_navigator.class_options,
            }
        user_data_dict = asdict(self.user_data)
        user_data_dict.pop("well", None)
        # Compute n_crops from gallery grid so loaders know how many
        # crops each session should contain without loading the NPY.
        rows = user_data_dict.get("rows", 0)
        columns = user_data_dict.get("columns", 0)
        n_crops = rows * columns if rows > 0 and columns > 0 else 0

        metadata: dict[str, Any] = {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
            "n_crops": n_crops,
        }
        # Save channel_data so saved sessions can resolve channel names
        if self.omero_data.channel_data:
            metadata["channel_data"] = dict(self.omero_data.channel_data)
        return metadata

    def _show_error_message(self, message: str) -> None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _show_success_message(self, message: str) -> None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Success")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _show_confirmation_dialog(self, message: str) -> bool:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Warning")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        reply = msg_box.exec_()
        return bool(reply == QMessageBox.Yes)
