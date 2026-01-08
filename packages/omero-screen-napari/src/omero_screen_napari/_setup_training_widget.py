import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from magicgui import magicgui
from magicgui.widgets import Container, Label
from omero_screen.config import get_logger
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_userdata_singleton import (
    userdata as global_user_data,
)
from omero_screen_napari.omero_data_singleton import omero_data

if TYPE_CHECKING:
    from omero_screen_napari.gallery_userdata import UserData
    from omero_screen_napari.omero_data import OmeroData

logger = get_logger(__name__)


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
        self.meta_data_saver = MetaDataSaver(
            self.class_name,
            self.omero_data,
            self.user_data,
            self.image_navigator,
        )
        self.add_class_widget = magicgui(
            call_button="Enter", text_input={"label": "Class name"}
        )(self.add_class)
        self.reset_class_options_widget = magicgui(
            call_button="Reset class options"
        )(self.reset_class_options)
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

    def save_meta_data(self, text_input: str) -> None:
        if new_classifier_name := text_input.strip():
            self.meta_data_saver.update_classifier_name(new_classifier_name)
        self.meta_data_saver.save_data()

    def create_container(self) -> Container:  # type: ignore
        return Container(
            widgets=[
                self.add_class_widget,
                self.reset_class_options_widget,
                self.image_navigator.class_labels,
                self.save_meta_data_widget,
            ]
        )


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

    def save_data(self) -> None:
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
        return {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
        }

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
