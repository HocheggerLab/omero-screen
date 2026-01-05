
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
from omero_screen_napari._setup_training_widget import (
    ImageNavigator as SetupImageNavigator,
    MetaDataSaver,
    SetupTrainingWidget,
)
from omero_screen_napari._training_widget import (
    ImageNavigator as TrainingImageNavigator,
    TrainingDataSaver,
    TrainingWidget,
)
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_omero_data():
    mock = MagicMock(spec=OmeroData)
    mock.plate_id = "Plate1"
    mock.well_pos_list = ["A1"]
    mock.image_input = "Image1"
    mock.selected_images = [np.zeros((10, 10, 3)), np.zeros((10, 10, 3))]
    mock.selected_crops = [np.zeros((10, 10, 3)), np.zeros((10, 10, 3))]
    mock.selected_labels = [np.zeros((10, 10)), np.zeros((10, 10))]
    mock.selected_classes = ["class1", "class2"]
    return mock

@pytest.fixture
def real_user_data():
    return UserData(well="A1", timepoint=0)

@pytest.fixture
def mock_container():
    with patch("omero_screen_napari._setup_training_widget.Container") as mock:
        yield mock

@pytest.fixture
def mock_magicgui():
    with patch("omero_screen_napari._setup_training_widget.magicgui") as mock:
        def decorator(func):
            widget_mock = MagicMock()
            # We don't need side_effect=func here because we call the methods directly
            # and the widget attribute is just used to access sub-widgets (text_input)
            return widget_mock
        mock.return_value = decorator
        yield mock

@pytest.fixture
def mock_training_magicgui():
    with patch("omero_screen_napari._training_widget.magicgui") as mock:
        def decorator(func):
            widget_mock = MagicMock()
            return widget_mock
        mock.return_value = decorator
        yield mock

@pytest.fixture
def mock_training_container():
     with patch("omero_screen_napari._training_widget.Container") as mock:
        yield mock


# --- SetupTrainingWidget Tests ---

class TestSetupImageNavigator:
    def test_init_defaults(self):
        nav = SetupImageNavigator(None)
        assert nav.class_options == ["unassigned"]
        assert nav.class_labels is not None

    def test_add_class(self):
        nav = SetupImageNavigator(None)
        nav.add_class("new_class")
        assert "new_class" in nav.class_options

    def test_add_class_duplicate(self):
        nav = SetupImageNavigator(["A"])
        nav.add_class("A")
        assert nav.class_options.count("A") == 1

    def test_reset_class_options(self):
        nav = SetupImageNavigator(["A", "B"])
        nav.reset_class_options()
        assert nav.class_options == ["unassigned"]

class TestMetaDataSaver:
    def test_validate_classifier_name_valid(self, mock_omero_data, real_user_data):
        nav = MagicMock(spec=SetupImageNavigator)
        nav.class_options = ["unassigned"]
        saver = MetaDataSaver("classifier", mock_omero_data, real_user_data, nav)
        saver._validate_classifier_name("valid_name")

    def test_validate_classifier_name_invalid(self, mock_omero_data, real_user_data):
        nav = MagicMock(spec=SetupImageNavigator)
        nav.class_options = ["unassigned"]
        saver = MetaDataSaver("classifier", mock_omero_data, real_user_data, nav)
        with pytest.raises(ValueError, match="no classifier name provided"):
            saver._validate_classifier_name("")

    def test_create_metadata_dict(self, mock_omero_data, real_user_data):
        nav = MagicMock(spec=SetupImageNavigator)
        nav.class_options = ["A", "B"]

        saver = MetaDataSaver("classifier", mock_omero_data, real_user_data, nav)
        metadata = saver._create_metadata_dict()

        assert "well" not in metadata["user_data"]
        assert metadata["class_options"] == ["A", "B"]

    @patch("omero_screen_napari._setup_training_widget.Path.home")
    @patch("json.dump")
    def test_save_metadata(self, mock_json_dump, mock_home, mock_omero_data, real_user_data):
        mock_home.return_value = Path("/tmp")
        nav = MagicMock(spec=SetupImageNavigator)
        nav.class_options = ["unassigned"]
        saver = MetaDataSaver("test_class", mock_omero_data, real_user_data, nav)
        saver._show_success_message = MagicMock()

        # Mock meta_data_path so we don't need actual filesystem
        saver.meta_data_path = MagicMock()

        saver._save_metadata()

        saver.meta_data_path.open.assert_called_with("w")
        mock_json_dump.assert_called()

class TestSetupTrainingWidgetClass:
    def test_add_class_integration(self, mock_omero_data, real_user_data, mock_magicgui, mock_container):
        widget = SetupTrainingWidget(None, "test", real_user_data, mock_omero_data)

        # Test the add_class method logic
        widget.add_class("NewClass")

        assert "NewClass" in widget.image_navigator.class_options
        # Verify value was reset. self.add_class_widget is the mock returned by mock_magicgui decorator
        # In the code: self.add_class_widget.text_input.value = ""
        # Accessing .value on the mock just returns another mock/value, setting it sets the attribute on the mock.
        assert widget.add_class_widget.text_input.value == ""

# --- TrainingWidget Tests ---

class TestTrainingImageNavigator:
    def test_next_image_cycle(self, mock_omero_data):
        mock_omero_data.selected_images = [1, 2, 3] # 3 images
        nav = TrainingImageNavigator(None, mock_omero_data)
        nav.update_image = MagicMock()

        assert nav.current_index == 0
        nav.next_image()
        assert nav.current_index == 1
        nav.next_image()
        assert nav.current_index == 2
        nav.next_image()
        assert nav.current_index == 0

    def test_previous_image_cycle(self, mock_omero_data):
        mock_omero_data.selected_images = [1, 2, 3]
        nav = TrainingImageNavigator(None, mock_omero_data)
        nav.update_image = MagicMock()

        assert nav.current_index == 0
        nav.previous_image()
        assert nav.current_index == 2
        nav.previous_image()
        assert nav.current_index == 1

    def test_assign_class(self, mock_omero_data):
        mock_omero_data.selected_classes = ["unassigned", "unassigned"]
        nav = TrainingImageNavigator(None, mock_omero_data)

        nav.current_index = 0
        nav.assign_class("ClassA")
        assert mock_omero_data.selected_classes[0] == "ClassA"

        nav.current_index = 1
        nav.assign_class("ClassB")
        assert mock_omero_data.selected_classes[1] == "ClassB"

class TestTrainingDataSaver:
    def test_create_training_dict(self, mock_omero_data, real_user_data):
        nav = MagicMock()
        saver = TrainingDataSaver("test", mock_omero_data, real_user_data, nav)

        data_dict = saver._create_training_dict()
        assert "data" in data_dict
        assert "target" in data_dict
        assert data_dict["target"] == ["class1", "class2"]

    @patch("omero_screen_napari._training_widget.Path.home")
    @patch("numpy.save")
    @patch("omero_screen_napari._training_widget.TrainingDataSaver._save_metadata")
    def test_save_training_data_call(self, mock_save_meta, mock_np_save, mock_home, mock_omero_data, real_user_data):
        mock_home.return_value = Path("/tmp")
        nav = MagicMock()
        saver = TrainingDataSaver("test", mock_omero_data, real_user_data, nav)

        with patch("omero_screen_napari._training_widget._show_success_message"):
            saver._save_training_data()

        mock_np_save.assert_called()
        mock_save_meta.assert_not_called()


class TestTrainingWidgetClass:
    @patch("omero_screen_napari._training_widget.napari.current_viewer")
    def test_load_image_no_name(self, mock_viewer, mock_omero_data, real_user_data, mock_training_magicgui, mock_training_container):
        widget = TrainingWidget(None, real_user_data, mock_omero_data)

        with patch("omero_screen_napari._training_widget._show_error_message") as mock_err:
            widget.load_image("")
            mock_err.assert_called()

    @patch("omero_screen_napari._training_widget.napari.current_viewer")
    @patch("omero_screen_napari._training_widget.TrainingWidget._set_paths")
    def test_load_image_file_not_found(self, mock_set_paths, mock_viewer, mock_omero_data, real_user_data, mock_training_magicgui, mock_training_container):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_set_paths.return_value = ("file.npy", mock_path, mock_path)

        widget = TrainingWidget(None, real_user_data, mock_omero_data)

        with patch("omero_screen_napari._training_widget._show_error_message") as mock_err:
            with patch.object(widget, '_parse_metadata', side_effect=FileNotFoundError("Mock")):
                 widget.load_image("test_class")
                 mock_err.assert_called_with("Metadata file not found: " + str(mock_path))
