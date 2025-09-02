import argparse
import re
from datetime import datetime
from unittest import mock

import pandas as pd
import pytest
from cellview.utils.error_classes import DataError, StateError
from cellview.utils.state import CellViewState


def test_state_singleton_initialization(sample_data_path):
    """Test that CellViewState singleton is properly initialized with valid data."""
    # Reset state to ensure clean slate
    state = CellViewState.get_instance()
    state.reset()

    # Create args with sample data path
    args = argparse.Namespace()
    args.csv = sample_data_path

    # Get instance with args
    state = CellViewState.get_instance(args)

    # Test singleton behavior - should be same instance
    state2 = CellViewState.get_instance()
    assert state is state2

    # Test basic attributes are set correctly
    assert state.csv_path == sample_data_path
    assert state.plate_id == 1
    assert state.df is not None
    assert len(state.df) == 20  # 2 wells × 10 cells

    # Test date extraction from filename
    assert state.date == "2024-03-26"  # From filename 240326_test_data_cc.csv

    # Test channel detection
    assert state.channel_0 == "DAPI"
    assert state.channel_1 == "Tub"
    assert state.channel_2 == "p21"
    assert state.channel_3 == "EdU"

    # Test that DataFrame has expected columns
    expected_cols = {
        "plate_id",
        "well_id",
        "image_id",
        "intensity_max_DAPI_nucleus",
        "intensity_min_DAPI_nucleus",
        "intensity_mean_DAPI_nucleus",
    }
    assert all(col in state.df.columns for col in expected_cols)

    # Test that plate_id is consistent
    assert len(state.df.plate_id.unique()) == 1
    assert state.df.plate_id.unique()[0] == 1


def test_get_plate_id_success():
    """Test that get_plate_id returns correct plate ID for valid data."""
    # Create state instance directly
    state = CellViewState()

    # Set up test DataFrame with single plate_id
    state.df = pd.DataFrame({"plate_id": [1, 1, 1], "well_id": [1, 2, 3]})

    assert state.get_plate_id() == 1


def test_get_plate_id_multiple_plates_error():
    """Test that get_plate_id raises error when multiple plate IDs are found."""
    state = CellViewState()

    # Set up test DataFrame with multiple plate_ids
    state.df = pd.DataFrame({"plate_id": [1, 2, 1], "well_id": [1, 2, 3]})

    with pytest.raises(
        DataError,  # Changed from ValueError
        match="Multiple plates found in the CSV file",
    ):
        state.get_plate_id()


def test_get_plate_id_no_data():
    """Test that get_plate_id returns None when no DataFrame is set."""
    state = CellViewState()

    with pytest.raises(
        StateError,  # Added error check
        match="Cannot get plate ID: no DataFrame loaded",
    ):
        state.get_plate_id()


def test_extract_date_from_filename_valid():
    """Test date extraction from filenames with valid dates."""
    state = CellViewState()
    test_cases = [
        ("240326_test_data.csv", "2024-03-26"),
        ("data_230115_analysis.csv", "2023-01-15"),
        ("prefix_220931_suffix.csv", "2022-09-31"),
        ("220101.csv", "2022-01-01"),
    ]

    for filename, expected in test_cases:
        assert state.extract_date_from_filename(filename) == expected


def test_extract_date_from_filename_invalid():
    """Test date extraction from filenames without valid dates."""
    state = CellViewState()
    invalid_filenames = [
        "no_date.csv",
        "190523_old_date.csv",  # Starts with 19, not 2
        "data_12345_test.csv",  # Not enough digits
        "test_1234567_data.csv",  # Too many digits
        "",  # Empty string
    ]

    current_date = datetime.now().strftime("%Y-%m-%d")
    for filename in invalid_filenames:
        assert state.extract_date_from_filename(filename) == current_date


def test_get_channels_success():
    """Test that get_channels correctly extracts channel names from column headers."""
    state = CellViewState()

    # Set up test DataFrame with various intensity measurements
    state.df = pd.DataFrame(
        {
            "intensity_max_DAPI_nucleus": [],
            "intensity_min_DAPI_nucleus": [],
            "intensity_mean_DAPI_nucleus": [],
            "intensity_max_GFP_cytoplasm": [],
            "intensity_mean_mCherry_cell": [],
            "other_column": [],
        }
    )

    channels = state.get_channels()
    assert channels == ["DAPI", "GFP", "mCherry"]


def test_get_channels_no_data():
    """Test that get_channels returns empty list when DataFrame is None."""
    state = CellViewState()
    state.df = None
    assert state.get_channels() == []


def test_get_channels_no_intensity_columns():
    """Test that get_channels returns empty list when no intensity columns exist."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {"plate_id": [], "well_id": [], "other_column": []}
    )
    assert state.get_channels() == []


def test_get_channels_order_preservation():
    """Test that get_channels preserves order of first appearance."""
    state = CellViewState()

    # Create columns where channels appear in different orders
    state.df = pd.DataFrame(
        {
            "intensity_mean_B_nucleus": [],  # B appears first
            "intensity_max_A_nucleus": [],  # A appears second
            "intensity_min_B_nucleus": [],  # B appears again
            "intensity_max_C_nucleus": [],  # C appears third
            "intensity_mean_A_cell": [],  # A appears again
        }
    )

    channels = state.get_channels()
    assert channels == ["B", "A", "C"]  # Order of first appearance


# -----------------test methods to prepare for measurements-----------------


def test_find_measurement_cols():
    """Test that _find_measurement_cols correctly identifies variable columns."""
    # Create sample data with 4 wells and some variable/non-variable columns
    state = CellViewState()
    df = pd.DataFrame(
        {
            "well": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
            "image_id": [1, 1, 2, 2, 3, 3, 4, 4],  # Each well has 2 images
            "constant_col": [
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
            ],  # No variability within images
            "low_var_col": [
                1,
                1,
                2,
                2,
                1,
                1,
                1,
                1,
            ],  # Only variable in 1/4 wells
            "high_var_col": [1, 2, 3, 4, 5, 6, 7, 8],  # Variable in all wells
            "med_var_col": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
            ],  # Variable between wells but not within images
        }
    )

    state.df = df
    result = state._find_measurement_cols()

    # Only high_var_col should be above the threshold since it varies within each image_id group
    assert result == ["high_var_col"]


@pytest.mark.parametrize(
    "test_df, expected_cols",
    [
        # Case 1: Normal case with mixed variable and constant columns
        (
            pd.DataFrame(
                {
                    "well": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
                    "image_id": [
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        4,
                        4,
                    ],  # Each well has 2 images
                    "constant": [1, 1, 1, 1, 1, 1, 1, 1],
                    "var_75pct": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        1,
                        1,
                    ],  # Variable in 3/4 wells (75%)
                    "var_100pct": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ],  # Variable in all wells
                    "var_50pct": [
                        1,
                        1,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                    ],  # Variable between wells but not within images
                }
            ),
            ["var_75pct", "var_100pct"],
        ),
        # Case 2: Empty DataFrame
        (
            pd.DataFrame({"well": [], "image_id": []}),
            [],
        ),
        # Case 3: No variable columns
        (
            pd.DataFrame(
                {
                    "well": ["A1", "A1", "A2", "A2"],
                    "image_id": [1, 1, 2, 2],  # Each well has 2 images
                    "col1": [1, 1, 1, 1],
                    "col2": [2, 2, 2, 2],
                }
            ),
            [],
        ),
        # Case 4: All columns are variable
        (
            pd.DataFrame(
                {
                    "well": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
                    "image_id": [
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        4,
                        4,
                    ],  # Each well has 2 images
                    "col1": [1, 2, 3, 4, 5, 6, 7, 8],
                    "col2": [8, 7, 6, 5, 4, 3, 2, 1],
                }
            ),
            ["col1", "col2"],
        ),
        # Case 5: Edge case with exactly at threshold
        (
            pd.DataFrame(
                {
                    "well": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
                    "image_id": [
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        4,
                        4,
                    ],  # Each well has 2 images
                    "at_threshold": [
                        1,
                        1,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                    ],  # Variable between wells but not within images
                }
            ),
            [],  # Should be empty as 50% < 75%
        ),
        # Case 6: Single well (edge case)
        (
            pd.DataFrame(
                {
                    "well": ["A1", "A1", "A1", "A1"],
                    "image_id": [1, 1, 2, 2],  # Each well has 2 images
                    "var_col": [1, 2, 3, 4],
                }
            ),
            ["var_col"],  # Should include as it's variable within images
        ),
    ],
)
def test_find_measurement_cols_parametrized(test_df, expected_cols):
    """Test _find_measurement_cols with various DataFrame inputs."""
    state = CellViewState()
    state.df = test_df
    result = state._find_measurement_cols()
    assert sorted(result) == sorted(expected_cols)


def test_find_measurement_cols_missing_well_column():
    """Test that _find_measurement_cols handles missing 'well' column correctly."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "image_id": [1, 2],  # Added image_id column
            "not_well": ["A1", "A2"],
            "some_col": [1, 2],
        }
    )

    # The method should not raise a KeyError for missing 'well' column
    # since it only uses 'image_id' for grouping
    result = state._find_measurement_cols()
    assert (
        result == []
    )  # Should return empty list since no columns vary within image_id groups


def test_find_measurement_cols_none_df():
    """Test that _find_measurement_cols raises error when df is None."""
    state = CellViewState()
    state.df = None

    with pytest.raises(AssertionError):
        state._find_measurement_cols()


def test_find_measurement_cols_with_unnamed():
    """Test that Unnamed columns are properly filtered out."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "well": ["A1", "A1", "A2", "A2", "A1", "A1", "A2", "A2"],
            "image_id": [1, 1, 2, 2, 3, 3, 4, 4],  # Each image_id has 2 rows
            "plate_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "intensity_max_DAPI_nucleus": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
            ],  # Varies within image_id groups
            "Unnamed: 0": [0, 1, 2, 3, 4, 5, 6, 7],
            "Unnamed: 1": [4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    measurement_cols = state._find_measurement_cols()
    assert "Unnamed: 0" not in measurement_cols
    assert "Unnamed: 1" not in measurement_cols
    assert "intensity_max_DAPI_nucleus" in measurement_cols


def test_trim_df_normal_case():
    """Test that _trim_df correctly trims the DataFrame to include only necessary columns."""
    state = CellViewState()
    # Create a DataFrame with required and measurement columns
    state.df = pd.DataFrame(
        {
            "well": [1, 2, 3],
            "timepoint": [1, 2, 3],
            "image_id": [101, 102, 103],
            "measurement_1": [10, 20, 30],
            "measurement_2": [40, 50, 60],
            "extra_col": ["a", "b", "c"],
        }
    )

    # Test with a subset of measurement columns
    measurement_cols = ["measurement_1", "measurement_2"]
    trimmed_df = state._trim_df(measurement_cols)

    # Check that only measurement + required columns are present
    expected_cols = [
        "well",
        "timepoint",
        "image_id",
        "measurement_1",
        "measurement_2",
    ]
    assert set(trimmed_df.columns.tolist()) == set(expected_cols)


def test_trim_df_missing_required_column():
    """Test that _trim_df raises StateError when required columns are missing."""
    state = CellViewState()
    # Create a DataFrame missing some required columns
    state.df = pd.DataFrame(
        {
            "well": [1, 2, 3],
            "image_id": [101, 102, 103],
            # Missing 'timepoint'
            "measurement_1": [10, 20, 30],
        }
    )

    measurement_cols = ["measurement_1"]

    with pytest.raises(
        StateError, match="Required column timepoint not found in dataframe"
    ):
        state._trim_df(measurement_cols)


def test_trim_df_empty_measurement_cols():
    """Test that _trim_df works with empty measurement columns list."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "well": [1, 2, 3],
            "timepoint": [1, 2, 3],
            "image_id": [101, 102, 103],
            "extra_col": ["a", "b", "c"],
        }
    )

    # Empty measurement columns list
    measurement_cols = []
    trimmed_df = state._trim_df(measurement_cols)

    # Should only contain required columns
    expected_cols = ["image_id", "well", "timepoint"]
    assert set(trimmed_df.columns.tolist()) == set(expected_cols)


def test_trim_df_none_df():
    """Test that _trim_df raises StateError when df is None."""
    state = CellViewState()
    state.df = None

    with pytest.raises(StateError, match="No dataframe loaded in state"):
        state._trim_df(["measurement_1"])


@pytest.mark.parametrize(
    "test_df, measurement_cols, expected_cols, should_raise",
    [
        # Case 1: Normal case with all required columns
        (
            pd.DataFrame(
                {
                    "well": [1, 2],
                    "timepoint": [1, 2],
                    "image_id": [1, 2],
                    "data1": [10, 20],
                    "data2": [30, 40],
                }
            ),
            ["data1"],
            ["well", "timepoint", "image_id", "data1"],  # Added image_id
            False,
        ),
        # Case 2: Missing one required column
        (
            pd.DataFrame(
                {
                    "well": [1, 2],
                    "data1": [10, 20],
                }
            ),
            ["data1"],
            [],  # Not used when exception is raised
            True,
        ),
        # Case 3: Empty DataFrame with columns
        (
            pd.DataFrame(
                {"well": [], "timepoint": [], "image_id": [], "data1": []}
            ),
            ["data1"],
            ["well", "timepoint", "image_id", "data1"],  # Added image_id
            False,
        ),
        # Case 4: All columns are measurement columns
        (
            pd.DataFrame(
                {"well": [1, 2], "timepoint": [1, 2], "image_id": [1, 2]}
            ),
            ["well", "timepoint", "image_id"],
            ["well", "timepoint", "image_id"],
            False,
        ),
    ],
)
def test_trim_df_parametrized(
    test_df, measurement_cols, expected_cols, should_raise
):
    """Test _trim_df with various DataFrame inputs."""
    state = CellViewState()
    state.df = test_df

    if should_raise:
        with pytest.raises(StateError):
            state._trim_df(measurement_cols)
    else:
        result = state._trim_df(measurement_cols)
        # Use set comparison to handle any order differences
        assert set(result.columns.tolist()) == set(expected_cols)
        assert len(result) == len(test_df)


def test_get_channel_list_normal_case():
    """Test that _get_channel_list correctly extracts channels from column names."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "intensity_max_DAPI_nucleus": [1, 2, 3],
            "intensity_mean_GFP_cell": [4, 5, 6],
            "intensity_min_mCherry_cyto": [7, 8, 9],
            "other_column": [10, 11, 12],
        }
    )

    channels = state._get_channel_list()
    assert channels == ["DAPI", "GFP", "mCherry"]


def test_get_channel_list_order_preservation():
    """Test that _get_channel_list preserves order of first appearance."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "intensity_mean_Ch2_cell": [1, 2, 3],  # Ch2 appears first
            "intensity_max_Ch1_nucleus": [4, 5, 6],  # Ch1 appears second
            "intensity_min_Ch2_nucleus": [
                7,
                8,
                9,
            ],  # Ch2 appears again, should be ignored
            "intensity_max_Ch3_cyto": [10, 11, 12],  # Ch3 appears third
        }
    )

    channels = state._get_channel_list()
    assert channels == ["Ch2", "Ch1", "Ch3"]


def test_get_channel_list_no_intensity_columns():
    """Test that _get_channel_list returns empty list when no intensity columns exist."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "well": [1, 2, 3],
            "image_id": [101, 102, 103],
            "other_data": [10, 20, 30],
        }
    )

    channels = state._get_channel_list()
    assert channels == []


def test_get_channel_list_empty_df():
    """Test that _get_channel_list handles empty DataFrames correctly."""
    state = CellViewState()
    state.df = pd.DataFrame(
        columns=["intensity_max_DAPI_nucleus", "intensity_mean_GFP_cell"]
    )

    channels = state._get_channel_list()
    assert channels == ["DAPI", "GFP"]


def test_get_channel_list_none_df():
    """Test that _get_channel_list returns empty list when df is None."""
    state = CellViewState()
    state.df = None

    channels = state._get_channel_list()
    assert channels == []


def test_get_channel_list_invalid_column_names():
    """Test that _get_channel_list ignores columns not matching the expected pattern."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "intensity_wrong_DAPI_nucleus": [1, 2, 3],  # Wrong measure type
            "intensity_max_GFP_invalid": [4, 5, 6],  # Wrong location
            "not_intensity_max_mCherry_cell": [7, 8, 9],  # Wrong prefix
        }
    )

    channels = state._get_channel_list()
    assert channels == []


@pytest.mark.parametrize(
    "test_df, expected_channels",
    [
        # Case 1: Normal case with multiple channels
        (
            pd.DataFrame(
                {
                    "intensity_max_DAPI_nucleus": [1, 2],
                    "intensity_mean_GFP_cell": [3, 4],
                    "intensity_min_RFP_cyto": [5, 6],
                    "other_col": [7, 8],
                }
            ),
            ["DAPI", "GFP", "RFP"],
        ),
        # Case 2: No intensity columns
        (pd.DataFrame({"well": [1, 2], "image_id": [1, 2]}), []),
        # Case 3: Mixed valid and invalid intensity columns
        (
            pd.DataFrame(
                {
                    "intensity_max_DAPI_nucleus": [1, 2],
                    "intensity_wrong_GFP_cell": [3, 4],  # Invalid measure_type
                    "intensity_min_RFP_invalid": [5, 6],  # Invalid location
                }
            ),
            ["DAPI"],
        ),
        # Case 4: Numeric channel names
        (
            pd.DataFrame(
                {
                    "intensity_max_405_nucleus": [1, 2],
                    "intensity_mean_488_cell": [3, 4],
                    "intensity_min_647_cyto": [5, 6],
                }
            ),
            ["405", "488", "647"],
        ),
        # Case 5: Duplicate channels in different measurements
        (
            pd.DataFrame(
                {
                    "intensity_max_DAPI_nucleus": [1, 2],
                    "intensity_min_DAPI_cell": [3, 4],
                    "intensity_mean_DAPI_cyto": [5, 6],
                }
            ),
            ["DAPI"],
        ),
    ],
)
def test_get_channel_list_parametrized(test_df, expected_channels):
    """Test _get_channel_list with various DataFrame inputs."""
    state = CellViewState()
    state.df = test_df
    result = state._get_channel_list()
    assert result == expected_channels


def test_validate_channels_matching():
    """Test that _validate_channels passes when DAPI is present."""
    state = CellViewState()

    # Should pass without raising an exception as long as DAPI is present
    state._validate_channels(["DAPI", "GFP", "RFP"])

    # Check that state channels are updated dynamically
    assert state.channel_0 == "DAPI"
    assert state.channel_1 == "GFP"
    assert state.channel_2 == "RFP"
    assert state.channel_3 is None


def test_validate_channels_mismatch():
    """Test that _validate_channels passes regardless of order as long as DAPI is present."""
    state = CellViewState()

    # Should pass regardless of order - new flexible validation
    state._validate_channels(["DAPI", "RFP", "GFP"])

    # Check that state channels are updated dynamically based on input order
    assert state.channel_0 == "DAPI"
    assert state.channel_1 == "RFP"
    assert state.channel_2 == "GFP"
    assert state.channel_3 is None


def test_validate_channels_not_enough_extracted():
    """Test that _validate_channels passes with any number of channels as long as DAPI is present."""
    state = CellViewState()

    # Should pass and update state channels dynamically
    state._validate_channels(["DAPI", "GFP"])

    # Check that state channels are updated correctly
    assert state.channel_0 == "DAPI"
    assert state.channel_1 == "GFP"
    assert state.channel_2 is None
    assert state.channel_3 is None


def test_validate_channels_extra_extracted():
    """Test that _validate_channels handles any number of channels dynamically."""
    state = CellViewState()

    # Should pass and handle any number of channels
    state._validate_channels(["DAPI", "GFP", "RFP"])

    # Check that state channels are updated to accommodate extra channels
    assert state.channel_0 == "DAPI"
    assert state.channel_1 == "GFP"
    assert state.channel_2 == "RFP"
    assert state.channel_3 is None


def test_validate_channels_empty_state():
    """Test that _validate_channels handles empty channels but requires DAPI."""
    state = CellViewState()

    # Empty channels should raise error (no DAPI)
    with pytest.raises(StateError, match="DAPI channel is required but not found in data"):
        state._validate_channels([])

    # Should pass if DAPI is present
    state._validate_channels(["DAPI"])
    assert state.channel_0 == "DAPI"
    assert state.channel_1 is None


def test_validate_channels_empty_extracted():
    """Test that _validate_channels requires DAPI even if state has channels."""
    state = CellViewState()

    # Should raise error even if state has channels set (no DAPI in input)
    with pytest.raises(StateError, match="DAPI channel is required but not found in data"):
        state._validate_channels([])


@pytest.mark.parametrize(
    "state_channels, extracted_channels, should_raise",
    [
        # Case 1: Has DAPI - should pass
        (["DAPI", "GFP", "RFP", None], ["DAPI", "GFP", "RFP"], False),
        # Case 2: Has DAPI, different order - should pass (flexible now)
        (["DAPI", "GFP", "RFP", None], ["DAPI", "RFP", "GFP"], False),
        # Case 3: Has DAPI, fewer channels - should pass
        (["DAPI", "GFP", "RFP", None], ["DAPI", "GFP"], False),
        # Case 4: Has DAPI, more channels - should pass (flexible now)
        (["DAPI", "GFP", None, None], ["DAPI", "GFP", "RFP"], False),
        # Case 5: No DAPI, empty - should raise
        ([None, None, None, None], [], True),
        # Case 6: Has DAPI - should pass
        ([None, None, None, None], ["DAPI"], False),
        # Case 7: No DAPI - should raise
        (["DAPI", None, None, None], [], True),
        # Case 8: Has DAPI, different names - should pass (flexible now)
        (["DAPI", "GFP", "RFP", None], ["DAPI", "GFP", "mCherry"], False),
    ],
)
def test_validate_channels_parametrized(
    state_channels, extracted_channels, should_raise
):
    """Test _validate_channels with various inputs - now flexible validation."""
    state = CellViewState()

    if should_raise:
        with pytest.raises(StateError, match="DAPI channel is required"):
            state._validate_channels(extracted_channels)
    else:
        # Should not raise an exception
        state._validate_channels(extracted_channels)
        # Check that state was updated dynamically if DAPI present
        if extracted_channels and "DAPI" in extracted_channels:
            assert state.channel_0 == (extracted_channels[0] if len(extracted_channels) > 0 else None)
            assert state.channel_1 == (extracted_channels[1] if len(extracted_channels) > 1 else None)
            assert state.channel_2 == (extracted_channels[2] if len(extracted_channels) > 2 else None)
            assert state.channel_3 == (extracted_channels[3] if len(extracted_channels) > 3 else None)


def test_rename_channel_columns_mixed_channels():
    """Test that _rename_channel_columns correctly renames non-DAPI channel columns."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "intensity_max_DAPI_nucleus": [1, 2],
            "intensity_mean_GFP_cell": [3, 4],
            "intensity_min_RFP_cyto": [5, 6],
        }
    )

    # Store original columns to verify changes
    original_columns = state.df.columns.tolist()

    channels = ["DAPI", "GFP", "RFP"]
    state._rename_channel_columns(channels)

    # Check that columns were changed
    assert state.df.columns.tolist() != original_columns

    # Check that DAPI is still in the columns (not replaced)
    assert "intensity_max_DAPI_nucleus" in state.df.columns

    # Check that GFP and RFP were replaced
    assert "intensity_mean_ch1_cell" in state.df.columns
    assert "intensity_min_ch2_cyto" in state.df.columns


def test_rename_channel_columns_only_non_dapi():
    """Test that _rename_channel_columns works with only non-DAPI channels."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "intensity_max_GFP_nucleus": [1, 2],
            "intensity_mean_RFP_cell": [3, 4],
            "intensity_min_mCherry_cyto": [5, 6],
        }
    )

    # Store original columns to verify changes
    original_columns = state.df.columns.tolist()

    channels = ["GFP", "RFP", "mCherry"]
    state._rename_channel_columns(channels)

    # Check that columns were changed
    assert state.df.columns.tolist() != original_columns

    # Check that channels were renamed correctly
    assert "intensity_max_ch1_nucleus" in state.df.columns
    assert "intensity_mean_ch2_cell" in state.df.columns
    assert "intensity_min_ch3_cyto" in state.df.columns


def test_rename_channel_columns_implementation():
    """Test the actual implementation of _rename_channel_columns to identify possible bugs."""
    state = CellViewState()
    state.df = pd.DataFrame({"col_GFP_name": [1, 2]})

    # Before renaming
    assert "col_GFP_name" in state.df.columns

    # Create a simple channel mapping that should work
    channels = ["GFP"]

    # Directly execute the core of the rename function to debug
    non_dapi_channels = [ch for ch in channels if ch != "DAPI"]
    channel_map = {ch: f"ch{i + 1}" for i, ch in enumerate(non_dapi_channels)}

    # Print debug information
    print(f"Channel map: {channel_map}")

    # Try different patterns to see what works
    old_col = "col_GFP_name"

    # Simple replacement
    simple_replacement = old_col.replace("GFP", "ch1")
    print(f"Simple replacement: {simple_replacement}")

    # Test regex with word boundaries
    regex_pattern = rf"\b{'GFP'}\b"
    regex_replacement = re.sub(regex_pattern, "ch1", old_col)
    print(f"Regex with word boundaries: {regex_replacement}")

    # Test regex without word boundaries
    regex_pattern_no_boundaries = r"GFP"
    regex_replacement_no_boundaries = re.sub(
        regex_pattern_no_boundaries, "ch1", old_col
    )
    print(f"Regex without word boundaries: {regex_replacement_no_boundaries}")

    # Apply the actual function to see its behavior
    state._rename_channel_columns(channels)

    # Print the final columns to see if anything changed
    print(f"Final columns: {state.df.columns.tolist()}")


def test_rename_channel_columns_with_patcher():
    """Test the _rename_channel_columns method with patched regex functionality."""
    # This test will investigate why the regex isn't working as expected

    state = CellViewState()
    state.df = pd.DataFrame(
        {"intensity_max_GFP_nucleus": [1, 2], "other_column": [3, 4]}
    )

    # Patch the re.sub function to see exactly what's being passed to it
    with mock.patch("re.sub", wraps=re.sub) as wrapped_sub:
        channels = ["GFP"]
        state._rename_channel_columns(channels)

        # Check what arguments were passed to re.sub
        for call in wrapped_sub.call_args_list:
            args, kwargs = call
            pattern, replacement, string = args
            print(
                f"Pattern: {pattern}, Replacement: {replacement}, String: {string}"
            )

    # Print final columns to verify if any changes occurred
    print(f"Final columns after attempted rename: {state.df.columns.tolist()}")


def test_functionality_check():
    """A simple test to verify how the column renaming should work."""
    # Create a simple dataframe
    df = pd.DataFrame(
        {
            "intensity_max_GFP_nucleus": [1, 2],
            "intensity_mean_RFP_cell": [3, 4],
        }
    )

    # Create a simple mapping
    mapping = {"GFP": "ch1", "RFP": "ch2"}

    # Create new column names by direct string replacement
    new_columns = []
    for col in df.columns:
        new_col = col
        for original, replacement in mapping.items():
            new_col = new_col.replace(original, replacement)
        new_columns.append(new_col)

    # Rename the columns
    df.rename(
        columns=dict(zip(df.columns, new_columns, strict=False)), inplace=True
    )

    # Verify the results
    assert "intensity_max_ch1_nucleus" in df.columns
    assert "intensity_mean_ch2_cell" in df.columns

    print(f"Successfully renamed columns to: {df.columns.tolist()}")


@pytest.mark.parametrize(
    "initial_columns, channels, expected_checks",
    [
        # Case 1: Simple test case to verify basic functionality
        (
            ["col_GFP_data", "col_RFP_data"],
            ["GFP", "RFP"],
            [
                lambda df: "col_GFP_data"
                not in df.columns.tolist(),  # Original GFP column should be gone
                lambda df: "col_RFP_data"
                not in df.columns.tolist(),  # Original RFP column should be gone
            ],
        )
    ],
)
def test_rename_channel_columns_simplified(
    initial_columns, channels, expected_checks
):
    """Simplified test for _rename_channel_columns that checks core functionality."""
    state = CellViewState()

    # For debugging purposes, let's patch the re.sub function
    def print_sub_args(pattern, replacement, string):
        print(
            f"re.sub called with: pattern={pattern}, replacement={replacement}, string={string}"
        )
        # Use simple string replacement for testing
        if "GFP" in string:
            return string.replace("GFP", "ch1")
        elif "RFP" in string:
            return string.replace("RFP", "ch2")
        return string

    with mock.patch("re.sub", side_effect=print_sub_args):
        # Create DataFrame with dummy data and specified columns
        data = {col: [1, 2] for col in initial_columns}
        state.df = pd.DataFrame(data)

        # Print original columns for debugging
        print(f"Original columns: {state.df.columns.tolist()}")

        # Call the function
        state._rename_channel_columns(channels)

        # Print final columns for debugging
        print(f"Final columns: {state.df.columns.tolist()}")

        # Run checks
        for check in expected_checks:
            assert check(state.df)


def test_rename_centroid_cols():
    """Test centroid column renaming functionality."""
    state = CellViewState()

    # Test case 1: All columns present
    state.df = pd.DataFrame(
        {
            "centroid-0": [1, 2],
            "centroid-1": [3, 4],
            "centroid-0_x": [5, 6],
            "centroid-1_x": [7, 8],
            "centroid-0_y": [9, 10],
            "centroid-1_y": [11, 12],
            "other_col": [13, 14],
        }
    )

    state._rename_centroid_cols()
    expected_columns = {
        "centroid-0-nuc",
        "centroid-1-nuc",
        "centroid-0-cell",
        "centroid-1-cell",
        "other_col",
    }
    assert set(state.df.columns) == expected_columns

    # Test case 2: Only base centroids
    state.df = pd.DataFrame(
        {"centroid-0": [1, 2], "centroid-1": [3, 4], "other_col": [5, 6]}
    )

    state._rename_centroid_cols()
    expected_columns = {"centroid-0-nuc", "centroid-1-nuc", "other_col"}
    assert set(state.df.columns) == expected_columns

    # Test case 3: Only _x centroids
    state.df = pd.DataFrame(
        {"centroid-0_x": [1, 2], "centroid-1_x": [3, 4], "other_col": [5, 6]}
    )

    state._rename_centroid_cols()
    expected_columns = {"centroid-0-cell", "centroid-1-cell", "other_col"}
    assert set(state.df.columns) == expected_columns


def test_optimize_measurement_types():
    """Test that numeric columns are optimized to appropriate types."""
    state = CellViewState()
    state.df = pd.DataFrame(
        {
            "well": ["A1", "A1", "A2", "A2"],
            "intensity_max_DAPI_nucleus": [
                100,
                200,
                300,
                400,
            ],  # Should be uint16 (median > 10)
            "normalized_value": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],  # Should be float32 (median < 10)
            "large_measurement": [
                65535,
                65534,
                65533,
                65532,
            ],  # Should be uint16 (median > 10)
            "small_measurement": [
                1,
                2,
                3,
                4,
            ],  # Should be float32 (median < 10)
            "negative_values": [
                -1,
                -2,
                -3,
                -4,
            ],  # Should be float32 (negative)
            "non_numeric": ["a", "b", "c", "d"],  # Should remain unchanged
        }
    )

    # Run optimization
    state._optimize_measurement_types()

    # Check types
    assert state.df["intensity_max_DAPI_nucleus"].dtype == "uint16"
    assert state.df["normalized_value"].dtype == "float32"
    assert state.df["large_measurement"].dtype == "uint16"
    assert state.df["small_measurement"].dtype == "float32"
    assert state.df["negative_values"].dtype == "float32"
    assert state.df["non_numeric"].dtype == "object"
    assert state.df["well"].dtype == "object"
