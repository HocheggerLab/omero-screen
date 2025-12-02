import sys
print("Starting test script...", file=sys.stderr)
from unittest.mock import MagicMock

# Mock Ice and omero modules before they are imported
sys.modules["Ice"] = MagicMock()
sys.modules["omero"] = MagicMock()
sys.modules["omero.gateway"] = MagicMock()

import argparse
from unittest.mock import patch

# Now we can import cellview modules
# We need to make sure omero_connect decorator is mocked or handles the mock
# In state.py: from omero_utils.omero_connect import omero_connect
# We might need to mock omero_utils too if it imports Ice
sys.modules["omero_utils"] = MagicMock()
sys.modules["omero_utils.omero_connect"] = MagicMock()
sys.modules["omero_utils.attachments"] = MagicMock()
# We need to define omero_connect as a decorator that just calls the function
def mock_omero_connect(func):
    return func
sys.modules["omero_utils.omero_connect"].omero_connect = mock_omero_connect

from cellview.cli import parse_args
from cellview.main import main_with_dependency_injection
from cellview.utils.state import CellViewStateCore

def test_cli_screen_id_argument():
    """Test that --screen-id argument is parsed correctly."""
    with patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        db=None, csv=None, plate_id=None, screen_id=123, clean=False, plate=None,
        projects=False, project=None, experiment=None, edit_project=None,
        edit_experiment=None, delete_plate=None, export_plate=None, interactive=False
    )):
        args = parse_args()
        assert args.screen_id == 123

@patch("cellview.main.create_cellview_state")
@patch("cellview.main.import_data")
@patch("cellview.main.CellViewDB")
def test_main_screen_id_flow(mock_db_cls, mock_import_data, mock_create_state):
    """Test the flow when --screen-id is provided."""
    # Setup mocks
    mock_db = mock_db_cls.return_value
    mock_conn = mock_db.connect.return_value

    # Mock state and get_plates_from_screen
    mock_state = MagicMock(spec=CellViewStateCore)
    mock_state.get_plates_from_screen.return_value = [101, 102]
    mock_create_state.return_value = mock_state

    # Mock args
    mock_args = argparse.Namespace(
        db=None, csv=None, plate_id=None, screen_id=123, clean=False, plate=None,
        projects=False, project=None, experiment=None, edit_project=None,
        edit_experiment=None, delete_plate=None, export_plate=None, interactive=False
    )

    with patch("cellview.main.parse_args", return_value=mock_args):
        main_with_dependency_injection()

    # Verify get_plates_from_screen was called
    mock_state.get_plates_from_screen.assert_called_once_with(123)

    # Verify import_data was called twice (once for each plate)
    assert mock_import_data.call_count == 2

    # Verify create_cellview_state was called 3 times:
    # 1. Initial call with screen_id
    # 2. Call for plate 101
    # 3. Call for plate 102
    assert mock_create_state.call_count == 3

    # Verify args passed to create_cellview_state for plates
    # We can inspect the calls
    calls = mock_create_state.call_args_list

    # First call has screen_id=123
    assert calls[0][0][0].screen_id == 123

    # Second call should have plate_id=101 and screen_id=None
    args_plate_1 = calls[1][0][0]
    assert args_plate_1.plate_id == 101
    assert args_plate_1.screen_id is None

    # Third call should have plate_id=102 and screen_id=None
    args_plate_2 = calls[2][0][0]
    assert args_plate_2.plate_id == 102
    assert args_plate_2.screen_id is None

@patch("cellview.utils.state.omero_connect")
def test_get_plates_from_screen(mock_omero_connect):
    """Test get_plates_from_screen method."""
    # Setup mock connection and screen
    mock_conn = MagicMock()
    mock_screen = MagicMock()
    mock_plate1 = MagicMock()
    mock_plate1.getId.return_value = 101
    mock_plate2 = MagicMock()
    mock_plate2.getId.return_value = 102

    mock_screen.listChildren.return_value = [mock_plate1, mock_plate2]
    mock_conn.getObject.return_value = mock_screen

    # Setup omero_connect decorator to just call the function with mock_conn
    def side_effect(func):
        def wrapper(*args, **kwargs):
            return func(*args, conn=mock_conn, **kwargs)
        return wrapper
    mock_omero_connect.side_effect = side_effect

    # Create state instance
    state = CellViewStateCore(ui=MagicMock())

    # Call method
    plate_ids = state.get_plates_from_screen(123, conn=mock_conn)

    assert plate_ids == [101, 102]
    mock_conn.getObject.assert_called_with("Screen", 123)

@patch("cellview.main.create_cellview_state")
@patch("cellview.main.import_data")
@patch("cellview.main.CellViewDB")
def test_main_multiple_plate_ids_flow(mock_db_cls, mock_import_data, mock_create_state):
    """Test the flow when multiple --plate-id are provided."""
    # Setup mocks
    mock_db = mock_db_cls.return_value
    mock_conn = mock_db.connect.return_value

    # Mock state
    mock_state = MagicMock(spec=CellViewStateCore)
    mock_create_state.return_value = mock_state

    # Mock args
    mock_args = argparse.Namespace(
        db=None, csv=None, plate_id=[101, 102], screen_id=None, clean=False, plate=None,
        projects=False, project=None, experiment=None, edit_project=None,
        edit_experiment=None, delete_plate=None, export_plate=None, interactive=False
    )

    with patch("cellview.main.parse_args", return_value=mock_args):
        main_with_dependency_injection()

    # Verify validate_plates_same_screen was called
    mock_state.validate_plates_same_screen.assert_called_once_with([101, 102])

    # Verify import_data was called twice (once for each plate)
    assert mock_import_data.call_count == 2

    # Verify create_cellview_state was called 3 times:
    # 1. Initial call
    # 2. Call for plate 101
    # 3. Call for plate 102
    assert mock_create_state.call_count == 3

@patch("cellview.utils.state.omero_connect")
def test_validate_plates_same_screen(mock_omero_connect):
    """Test validate_plates_same_screen method."""
    # Setup mock connection and plates
    mock_conn = MagicMock()

    # Plate 1
    mock_plate1 = MagicMock()
    mock_screen1 = MagicMock()
    mock_screen1.getId.return_value = 999
    mock_plate1.getParent.return_value = mock_screen1

    # Plate 2
    mock_plate2 = MagicMock()
    mock_screen2 = MagicMock()
    mock_screen2.getId.return_value = 999
    mock_plate2.getParent.return_value = mock_screen2

    # Plate 3 (different screen)
    mock_plate3 = MagicMock()
    mock_screen3 = MagicMock()
    mock_screen3.getId.return_value = 888
    mock_plate3.getParent.return_value = mock_screen3

    def get_object_side_effect(obj_type, obj_id):
        if obj_type == "Plate":
            if obj_id == 101: return mock_plate1
            if obj_id == 102: return mock_plate2
            if obj_id == 103: return mock_plate3
        return None
    mock_conn.getObject.side_effect = get_object_side_effect

    # Setup omero_connect decorator
    def side_effect(func):
        def wrapper(*args, **kwargs):
            return func(*args, conn=mock_conn, **kwargs)
        return wrapper
    mock_omero_connect.side_effect = side_effect

    # Create state instance
    state = CellViewStateCore(ui=MagicMock())

    # Test valid case (same screen)
    state.validate_plates_same_screen([101, 102], conn=mock_conn)

    # Test invalid case (different screens)
    from cellview.utils.error_classes import DataError
    try:
        state.validate_plates_same_screen([101, 103], conn=mock_conn)
        assert False, "Should have raised DataError"
    except DataError:
        pass

@patch("cellview.main.CellViewState.get_instance")
@patch("cellview.main.import_data")
@patch("cellview.main.CellViewDB")
def test_legacy_main_multiple_plate_ids_flow(mock_db_cls, mock_import_data, mock_get_instance):
    """Test the legacy main flow when multiple --plate-id are provided."""
    from cellview.main import main

    # Setup mocks
    mock_db = mock_db_cls.return_value
    mock_conn = mock_db.connect.return_value

    # Mock state
    mock_state = MagicMock(spec=CellViewStateCore)
    mock_get_instance.return_value = mock_state

    # Mock args
    mock_args = argparse.Namespace(
        db=None, csv=None, plate_id=[101, 102], screen_id=None, clean=False, plate=None,
        projects=False, project=None, experiment=None, edit_project=None,
        edit_experiment=None, delete_plate=None, export_plate=None, interactive=False
    )

    with patch("cellview.main.parse_args", return_value=mock_args):
        main()

    # Verify validate_plates_same_screen was called
    mock_state.validate_plates_same_screen.assert_called_once_with([101, 102])

    # Verify import_data was called twice (once for each plate)
    assert mock_import_data.call_count == 2

    # Verify get_instance was called 3 times:
    # 1. Initial call
    # 2. Call for plate 101
    # 3. Call for plate 102
    assert mock_get_instance.call_count == 3

if __name__ == "__main__":
    test_cli_screen_id_argument()
    test_main_screen_id_flow()
    test_get_plates_from_screen()
    test_main_multiple_plate_ids_flow()
    test_validate_plates_same_screen()
    test_legacy_main_multiple_plate_ids_flow()
    print("All tests passed!")
