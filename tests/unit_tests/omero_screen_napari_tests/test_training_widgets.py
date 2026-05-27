
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Mock ClassifierSelector in sys.modules BEFORE importing widget modules
sys.modules['omero_screen_napari._classifier_selector'] = MagicMock()

# Import modules first (magicgui needs to import normally)
from omero_screen_napari._setup_training_widget import \
    ImageNavigator as SetupImageNavigator
from omero_screen_napari._setup_training_widget import MetaDataSaver
from omero_screen_napari._training_widget import \
    ImageNavigator as TrainingImageNavigator
from omero_screen_napari._training_widget import TrainingDataSaver
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData


# Create mock widget classes for headless testing
class MockContainer:
    def __init__(self, widgets=None):
        self.widgets = widgets or []

    def clear(self):
        self.widgets.clear()

    def append(self, widget):
        self.widgets.append(widget)

class MockLabel:
    def __init__(self, value=""):
        self.value = value
        self.visible = False

class MockRadioButtons:
    def __init__(self, label="", choices=None):
        self.label = label
        self.choices = choices or []
        self.value = choices[0] if choices else None
        self.changed = MagicMock()

    def connect(self, callback):
        self.changed.connect(callback)

# NOTE: SetupTrainingWidget and TrainingWidget tests have been removed because they
# require Qt widgets (ClassifierSelector) which need QApplication to run.

# --- Mocks and Fixtures ---

# Fixture to patch widget classes for headless testing
# This prevents Qt initialization in headless CI environments
@pytest.fixture(scope="module", autouse=True)
def mock_widgets():
    """Patch magicgui widgets to prevent Qt initialization in headless environments."""
    import omero_screen_napari._setup_training_widget as setup_module
    import omero_screen_napari._training_widget as training_module

    # Use patch.object to properly handle the patching
    patches = [
        patch.object(setup_module, 'Container', MockContainer),
        patch.object(setup_module, 'Label', MockLabel),
        patch.object(training_module, 'Container', MockContainer),
        patch.object(training_module, 'RadioButtons', MockRadioButtons),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()

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
    mock.channel_data = {}
    # TrainingDataSaver reads this live; default to None so the
    # non-session code path runs. Tests that want the session-path
    # behaviour override this to a Path.
    mock.session_file_path = None
    return mock

@pytest.fixture
def real_user_data():
    return UserData(well="A1", timepoint=0)

# Removed fixtures for SetupTrainingWidget and TrainingWidget tests
# (mock_container, mock_magicgui, mock_training_magicgui, mock_training_container)
# These are no longer needed since we removed the tests that require Qt widgets


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
    # REMOVED: test_add_class_integration - requires Qt widgets (ClassifierSelector)
    # The SetupTrainingWidget now creates ClassifierSelector in __init__ which needs QApplication
    pass

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

    def test_file_path_reflects_session_file_path_changes_post_init(
        self, mock_omero_data, real_user_data
    ):
        # Regression: TrainingDataSaver used to cache file_path in __init__,
        # so a session/direct load that mutated omero_data.session_file_path
        # after construction wouldn't be picked up — saves would land in
        # the wrong file. The properties now read live on every access.
        nav = MagicMock()
        saver = TrainingDataSaver("test", mock_omero_data, real_user_data, nav)
        first_path = saver.file_path  # derived from plate/well/image/timepoint

        new_path = Path("/tmp/some_other_session.npy")
        mock_omero_data.session_file_path = new_path
        assert saver.file_path == new_path
        assert saver.file_path != first_path
        assert saver.file_name == new_path.name

    def test_file_path_stable_across_accesses_after_file_created(
        self, mock_omero_data, real_user_data, tmp_path
    ):
        # Regression: file_path was a property that re-ran the `.exists()`
        # dedup on every access. save_both writes the NPY via self.file_path,
        # then _save_to_database reads self.file_path again — once the file
        # exists, the second read bumped to a `_2` name and the DB recorded a
        # path that doesn't exist on disk. The resolution is now memoised, so
        # it stays stable across accesses even after the file appears.
        mock_omero_data.session_file_path = None
        mock_omero_data.plate_id = 4053
        mock_omero_data.well_pos_list = ["D1"]
        mock_omero_data.image_input = "All"
        real_user_data.timepoint = 5
        nav = MagicMock()
        saver = TrainingDataSaver("test", mock_omero_data, real_user_data, nav)
        saver.classifier_dir = tmp_path

        first = saver.file_path
        assert first.name == "4053_D1_All_5.npy"
        # Simulate np.save creating the file mid-save.
        first.write_bytes(b"")
        # Subsequent accesses must NOT bump to _5_2.npy.
        assert saver.file_path == first
        assert saver.file_name == "4053_D1_All_5.npy"

    @patch("omero_screen_napari._training_widget.Path.home")
    @patch("numpy.save")
    @patch("omero_screen_napari._training_widget.TrainingDataSaver._save_metadata")
    @patch("omero_screen_napari._training_widget.TrainingDataSaver._save_to_database")
    def test_save_training_data_call(self, mock_save_db, mock_save_meta, mock_np_save, mock_home, mock_omero_data, real_user_data):
        mock_home.return_value = Path("/tmp")
        nav = MagicMock()
        saver = TrainingDataSaver("test", mock_omero_data, real_user_data, nav)

        with patch("omero_screen_napari._training_widget._show_success_message"):
            saver._save_training_data()

        mock_np_save.assert_called()
        mock_save_meta.assert_not_called()
        mock_save_db.assert_called_once()


class TestTrainingWidgetClass:
    # REMOVED: test_load_image_no_name - requires Qt widgets (ClassifierSelector)
    # REMOVED: test_load_image_file_not_found - requires Qt widgets (ClassifierSelector)
    # The TrainingWidget now creates ClassifierSelector in __init__ which needs QApplication
    pass
