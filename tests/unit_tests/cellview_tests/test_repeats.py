import re
from unittest.mock import MagicMock, call, patch

import pytest
from cellview.importers.repeats import RepeatsManager
from cellview.utils.error_classes import DataError, StateError
from cellview.utils.state import CellViewState


def normalize_sql(sql: str) -> str:
    """Normalize SQL query by removing extra whitespace."""
    return re.sub(r"\s+", " ", sql.strip())


def assert_sql_equal(actual_sql: str, expected_sql: str, params=None):
    """Assert that two SQL queries are equal, ignoring whitespace."""
    assert normalize_sql(actual_sql) == normalize_sql(expected_sql)
    if params is not None:
        assert actual_sql[1] == params


@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    return mock_conn


@pytest.fixture
def mock_state():
    """Create a mock state object."""
    state = CellViewState()
    state.experiment_id = 1
    state.plate_id = 1
    state.date = "2024-03-26"
    state.lab_member = "Test User"
    state.channel_0 = "DAPI"
    state.channel_1 = "Tub"
    state.channel_2 = "p21"
    state.channel_3 = "EdU"
    return state


@pytest.fixture
def repeats_manager(mock_db, mock_state):
    """Create a RepeatsManager instance with mocked dependencies."""
    with patch(
        "cellview.utils.state.CellViewState.get_instance",
        return_value=mock_state,
    ):
        return RepeatsManager(mock_db)


def test_fetch_existing_repeats(repeats_manager, mock_db):
    """Test fetching existing repeats from the database."""
    # Setup mock data
    mock_data = [
        (
            1,
            1,
            1,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
        (
            2,
            1,
            2,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
    ]
    mock_db.execute.return_value.fetchall.return_value = mock_data

    # Test the function
    result = repeats_manager._fetch_existing_repeats()

    # Verify the results
    assert result == mock_data

    # Verify SQL query
    expected_sql = """
        SELECT repeat_id, experiment_id, plate_id, date, lab_member, channel_0, channel_1, channel_2, channel_3
        FROM repeats
        WHERE experiment_id = ?
        ORDER BY repeat_id
    """
    actual_call = mock_db.execute.call_args
    assert_sql_equal(actual_call[0][0], expected_sql)
    assert actual_call[0][1] == [1]


def test_check_plate_duplicate_no_duplicate(repeats_manager):
    """Test plate duplicate check when no duplicates exist."""
    mock_repeats = [
        (
            1,
            1,
            2,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
        (
            2,
            1,
            3,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
    ]

    # This should not raise an error
    repeats_manager._check_plate_duplicate(mock_repeats)


def test_check_plate_duplicate_with_duplicate(repeats_manager):
    """Test plate duplicate check when a duplicate exists."""
    mock_repeats = [
        (
            1,
            1,
            1,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
        (
            2,
            1,
            2,
            "2024-03-26",
            "Test User",
            "DAPI",
            "Tub",
            "p21",
            "EdU",
        ),
    ]

    # This should raise a DataError
    with pytest.raises(DataError) as exc_info:
        repeats_manager._check_plate_duplicate(mock_repeats)

    assert "Duplicate Plate ID Found!" in str(exc_info.value)


def test_create_new_repeat_success(repeats_manager, mock_db):
    """Test successful creation of a new repeat."""
    # Setup mock data
    mock_db.execute.return_value.fetchone.return_value = (1,)

    # Setup state
    state = CellViewState.get_instance()
    state.experiment_id = 1
    state.plate_id = 1
    state.date = "2024-03-26"
    state.lab_member = "Test User"
    state.channel_0 = "DAPI"
    state.channel_1 = "Tub"
    state.channel_2 = "p21"
    state.channel_3 = "EdU"

    # Test the function
    repeat_id = repeats_manager._create_new_repeat()

    # Verify the results
    assert repeat_id == 1

    # Check that both queries were called in the correct order
    assert mock_db.execute.call_count == 2

    # Verify SQL queries
    expected_insert_sql = """
        INSERT INTO repeats (experiment_id, plate_id, date, lab_member, channel_0, channel_1, channel_2, channel_3)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    calls = mock_db.execute.call_args_list
    assert_sql_equal(
        calls[0][0][0],
        expected_insert_sql,
    )
    assert calls[0][0][1] == (
        1,
        1,
        "2024-03-26",
        "Test User",
        "DAPI",
        "Tub",
        "p21",
        "EdU",
    )
    assert calls[1] == call("SELECT currval('repeat_id_seq')")


def test_create_new_repeat_no_experiment_id(repeats_manager, mock_state):
    """Test creating a new repeat when no experiment is selected."""
    mock_state.experiment_id = None

    with pytest.raises(StateError) as exc_info:
        repeats_manager.create_new_repeat()

    assert "No experiment selected" in str(exc_info.value)


def test_create_new_repeat_no_plate_id(repeats_manager, mock_state):
    """Test creating a new repeat when no plate ID is provided."""
    mock_state.plate_id = None
    mock_state.df = MagicMock()

    with pytest.raises(DataError) as exc_info:
        repeats_manager.create_new_repeat()

    assert "No plate ID provided in CSV file" in str(exc_info.value)
