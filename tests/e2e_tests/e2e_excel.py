from typing import Any, Optional

import pandas as pd
from omero.gateway import BlitzGateway
from omero_utils.map_anns import parse_annotations
from omero_utils.omero_plate import cleanup_plate

from omero_screen.metadata_parser import MetadataParser
from tests.e2e_tests.e2e_setup import e2e_excel_setup, excel_file_handling


# Helper functions for test data generation
def get_channel_test_data() -> dict[str, pd.DataFrame]:
    """Return standard test data with DAPI, Tub, EdU channels"""
    return {
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


def get_nodapi_test_data() -> dict[str, pd.DataFrame]:
    """Return test data without DAPI channel"""
    return {
        "Sheet1": pd.DataFrame({"Channels": ["Tub", "EdU"], "Index": [0, 1]}),
        "Sheet2": pd.DataFrame(
            {
                "Well": ["C2", "C5"],
                "cell_line": ["RPE-1", "RPE-1"],
                "condition": ["Ctr", "Cdk4"],
            }
        ),
    }


# Core test functions
def missing_plate(conn: BlitzGateway, plate_id: int) -> dict[str, Any]:
    """Test basic metadata parsing functionality"""
    assert plate_id  # the e2erun has to pass a plate id here!

    parser = MetadataParser(conn, plate_id)
    parser._parse_metadata()

    # Return result for inspection/assertion instead of just printing
    result = {"well_data": parser.well_data}

    # Print for manual runs
    print(f"Well data: {parser.well_data}")

    return result


def run_plate(
    conn: BlitzGateway,
    teardown: bool = True,
    correct_df: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, Any]:
    """Test the excel file handling functionality

    Args:
        conn: OMERO connection
        teardown: Whether to clean up the plate after test
        correct_df: Test data to use for the Excel file

    Returns:
        dict: Test results including annotations and metadata
    """
    # Setup
    plate_id = e2e_excel_setup(conn)
    excel_file_handling(conn, plate_id, correct_df)

    # Test execution
    parser = MetadataParser(conn, plate_id)
    parser.manage_metadata()

    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate with ID {plate_id} not found")

    wells = list(plate.listChildren())
    if not wells:
        raise ValueError(f"No wells found in plate {plate_id}")

    well = wells[0]
    channel_annotations = parse_annotations(plate)
    well_annotations = parse_annotations(well)

    # Format result for both pytest assertions and manual inspection
    result = {
        "plate_id": plate_id,
        "channel_annotations": channel_annotations,
        "well_annotations": well_annotations,
    }

    # Print output for manual runs
    print(f"Plate annotations: {channel_annotations}")
    print(f"Well annotations: {well_annotations}")

    # Cleanup if requested
    if teardown:
        cleanup_plate(conn, plate)

    return result


# Specific test case functions
def run_plate_with_correct_excel(
    conn: BlitzGateway, teardown: bool = True
) -> dict[str, Any]:
    """Test with standard channel configuration including DAPI"""
    correct_df = get_channel_test_data()
    return run_plate(conn, teardown, correct_df)


def run_plate_noDAPI(
    conn: BlitzGateway, teardown: bool = True
) -> dict[str, Any]:
    """Test with alternative channel configuration without DAPI"""
    correct_df = get_nodapi_test_data()
    return run_plate(conn, teardown, correct_df)
