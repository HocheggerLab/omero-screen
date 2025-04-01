from unittest.mock import patch

import pandas as pd
import pytest
from omero_utils.omero_plate import cleanup_plate

from tests.e2e_tests.e2e_excel import run_plate, run_plate_with_correct_excel

# We'll need to add a fixture for the OMERO connection in conftest.py
# For now, assuming we have a 'conn' fixture


def test_excel_success(omero_conn):
    """Test excel handling with standard DAPI channel setup"""
    result = run_plate_with_correct_excel(omero_conn, teardown=True)

    # Add assertions based on expected results
    assert result["channel_annotations"] == {
        "DAPI": "0",
        "Tub": "1",
        "EdU": "2",
    }
    assert result["well_annotations"]["cell_line"] == "RPE-1"
    assert result["well_annotations"]["condition"] in ["Ctr", "Cdk4"]


@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        # Missing required key (cell_line)
        (
            {
                "well_data": {
                    "Well": ["C2", "C5"],
                    "condition": ["Ctr", "Cdk4"],
                },
                "channel_data": {
                    "DAPI": "0",
                    "Tub": "1",
                    "EdU": "2",
                },
            },
            "Missing required keys in well data: cell_line",
        ),
        # Non-list value in well data
        (
            {
                "well_data": {
                    "Well": ["C2", "C5"],
                    "cell_line": "RPE-1",  # String instead of list
                    "condition": ["Ctr", "Cdk4"],
                },
                "channel_data": {
                    "DAPI": "0",
                    "Tub": "1",
                    "EdU": "2",
                },
            },
            "Values must be lists for all keys. Non-list values found for: cell_line",
        ),
        # Inconsistent list lengths
        (
            {
                "well_data": {
                    "Well": ["C2", "C5"],
                    "cell_line": ["RPE-1", "RPE-1", "RPE-1"],  # Extra value
                    "condition": ["Ctr", "Cdk4"],
                },
                "channel_data": {
                    "DAPI": "0",
                    "Tub": "1",
                    "EdU": "2",
                },
            },
            "All well data lists must have the same length",
        ),
        # Missing nuclei channel
        (
            {
                "well_data": {
                    "Well": ["C2", "C5"],
                    "cell_line": ["RPE-1", "RPE-1"],
                    "condition": ["Ctr", "Cdk4"],
                },
                "channel_data": {
                    "Tub": "0",
                    "EdU": "1",
                },
            },
            "At least one nuclei channel (DAPI/Hoechst/DNA/RFP) is required",
        ),
        # Invalid channel data structure
        (
            {
                "well_data": {
                    "Well": ["C2", "C5"],
                    "cell_line": ["RPE-1", "RPE-1"],
                    "condition": ["Ctr", "Cdk4"],
                },
                "channel_data": {
                    "DAPI": 1,
                    "Tub": 2,
                    "EdU": 3,
                },
            },
            "Channel data must be a dictionary with string values",
        ),
    ],
)
def test_excel_validation_failures(omero_conn, invalid_data, expected_error):
    """Test various validation failures in Excel data handling"""
    # Create valid test data
    test_data = {
        "Sheet1": pd.DataFrame(
            {"Channels": ["DAPI", "Tub", "EdU"], "Index": [0, 1, 2]}
        ),
        "Sheet2": pd.DataFrame(
            {
                "Well": ["C2", "C5"],
                "cell_line": ["RPE-1", "RPE-1"],
                "condition": ["Ctr", "Cdk4"],
            }
        ),
    }

    # Get the plate ID before running the test
    plate_id = None
    try:
        # Run the test with teardown=False to get the plate ID
        result = run_plate(omero_conn, teardown=False, correct_df=test_data)
        plate_id = result["plate_id"]

        # Patch the _load_data_from_excel method to return our invalid data
        with patch(
            "omero_screen.metadata_parser.MetadataParser._load_data_from_excel"
        ) as mock_load:
            mock_load.return_value = (
                invalid_data["channel_data"],
                invalid_data["well_data"],
            )

            with pytest.raises(Exception) as exc_info:
                run_plate(omero_conn, teardown=False, correct_df=test_data)
            assert expected_error in str(exc_info.value)
    finally:
        plate = omero_conn.getObject("Plate", plate_id)
        cleanup_plate(omero_conn, plate)
