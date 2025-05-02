from typing import Any, Optional

import pandas as pd
from omero.gateway import BlitzGateway, PlateWrapper
from omero_utils.attachments import delete_excel_attachment
from omero_utils.map_anns import delete_map_annotations, parse_annotations

from omero_screen.metadata_parser import MetadataParser
from tests.e2e_tests.e2e_setup import excel_file_handling


# Helper functions for test data generation
def get_channel_test_data() -> dict[str, pd.DataFrame]:
    """Return standard test data with DAPI, Tub, EdU channels"""
    return {
        "Sheet1": pd.DataFrame(
            {"Channels": ["DAPI", "Tub", "EdU"], "Index": [0, 1, 2]}
        ),
        "Sheet2": pd.DataFrame(
            {
                "Well": ["A1", "B1"],
                "cell_line": ["RPE-1", "RPE-1"],
                "condition": ["Ctr", "Cdk4"],
            }
        ),
    }


def get_nodapi_test_data() -> dict[str, pd.DataFrame]:
    """Return test data without DAPI channel"""
    d = get_channel_test_data()
    d["Sheet1"] = pd.DataFrame({"Channels": ["Tub", "EdU"], "Index": [0, 1]})
    return d


def get_wrongwell_test_data() -> dict[str, pd.DataFrame]:
    """Return test data with the wrong well names"""
    d = get_channel_test_data()
    d["Sheet2"]["Well"] = ["A1", "B2"]  # B2 is incorrect
    return d


def get_multierror_test_data() -> dict[str, pd.DataFrame]:
    """Return test data without DAPI channel and wrong well names"""
    d = get_nodapi_test_data()
    d["Sheet2"]["Well"] = ["A1", "B2"]  # B1 is incorrect
    return d


# Core test functions
def missing_plate(
    conn: BlitzGateway, plate_id: int, teardown: bool = False
) -> dict[str, Any]:
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
    plate_id = 1
    excel_file_handling(conn, plate_id, correct_df)

    # Initialize plate and wells for cleanup
    plate = conn.getObject("Plate", plate_id)
    wells = list(plate.listChildren()) if plate else []

    try:
        # Validate setup
        if plate is None:
            raise ValueError(f"Plate with ID {plate_id} not found")
        if not wells:
            raise ValueError(f"No wells found in plate {plate_id}")

        # Test execution
        parser = MetadataParser(conn, plate_id)
        parser.manage_metadata()

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

        return result
    finally:
        # Cleanup if requested
        if teardown:
            clean_plate_annotations(conn, plate)


def clean_plate_annotations(conn: BlitzGateway, plate: PlateWrapper):
    """Clean the plate annotations"""
    if plate is not None:
        print("Cleaning up plate annotations")
        delete_map_annotations(conn, plate)
        delete_excel_attachment(conn, plate)

        for well in plate.listChildren():
            delete_map_annotations(conn, well)
            delete_excel_attachment(conn, well)


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


def run_plate_wrongwell(
    conn: BlitzGateway, teardown: bool = True
) -> dict[str, Any]:
    """Test with alternative channel configuration without correct well names"""
    correct_df = get_wrongwell_test_data()
    return run_plate(conn, teardown, correct_df)


def run_plate_multierror(
    conn: BlitzGateway, teardown: bool = True
) -> dict[str, Any]:
    """Test with alternative channel configuration without DAPI or correct well names"""
    correct_df = get_multierror_test_data()
    return run_plate(conn, teardown, correct_df)
