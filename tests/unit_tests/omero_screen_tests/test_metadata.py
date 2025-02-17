from io import StringIO

import pandas as pd
import pytest
from rich.console import Console

from omero_screen.metadata_parser import (
    MetadataParser,
    MetadataValidationError,
    PlateNotFoundError,
)


def test_plate_check_failure(omero_conn):
    # Create a string buffer to capture the output
    console_output = StringIO()
    test_console = Console(file=console_output, force_terminal=True)

    # Temporarily replace the default console with our test console
    parser = MetadataParser(omero_conn, 5000, console=test_console)

    with pytest.raises(PlateNotFoundError) as exc_info:
        parser._check_plate()

    # Get the captured output
    output = console_output.getvalue()

    # Test the exception message
    assert str(exc_info.value) == "A plate with id 5000 was not found!"

    # Test the rich formatting
    assert "╭─" in output  # Check for panel border
    assert "Error" in output  # Check for panel title
    assert "Plate Not Found:" in output  # Check for error type
    assert "5000" in output  # Check for plate ID in message


def test_plate_check_success(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_plate()


def test_excel_file_check_failure(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_excel_file() is None


def test_excel_file_check_success(test_plate_with_excel):
    plate_id = test_plate_with_excel.getId()
    conn = test_plate_with_excel._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_excel_file() is not None


def test_parse_excel_file(test_plate_with_excel):
    plate_id = test_plate_with_excel.getId()
    conn = test_plate_with_excel._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    if file_annotation:
        excel_file = parser._parse_excel_file(file_annotation)
    assert excel_file["Sheet1"].Channels.unique().tolist() == [
        "DAPI",
        "Tub",
        "p21",
        "EdU",
    ]
    assert excel_file["Sheet2"].condition.unique().tolist() == ["ctr", "CDK4"]


# --------------------TEST Validate Excel Data--------------------


def test_validate_excel_data_success(test_plate_with_excel):
    plate_id = test_plate_with_excel.getId()
    conn = test_plate_with_excel._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    if file_annotation:
        excel_file = parser._parse_excel_file(file_annotation)
    assert parser._validate_excel_data(excel_file)


def test_validate_excel_data_missing_sheets(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)

    # Test with empty dict
    invalid_data = {}
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_excel_data(invalid_data)
    assert "Missing required sheets" in str(exc_info.value)


def test_validate_excel_data_missing_sheet1_columns(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)
    # Missing required columns in Sheet1
    invalid_data = {
        "Sheet1": pd.DataFrame({"WrongColumn": [1, 2, 3]}),
        "Sheet2": pd.DataFrame({"cell_line": ["A", "B"]}),
    }
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_excel_data(invalid_data)
    assert "Sheet1 missing required columns" in str(exc_info.value)


def test_validate_excel_data_missing_nuclei_channel(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)

    # Data without DAPI/Hoechst/RFP channel
    invalid_data = {
        "Sheet1": pd.DataFrame({"Channels": ["GFP", "YFP"], "Index": [0, 1]}),
        "Sheet2": pd.DataFrame({"cell_line": ["A", "B"]}),
    }

    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_excel_data(invalid_data)
    assert "Sheet1 missing required columns:missing nuclei channel" in str(
        exc_info.value
    )


def test_validate_excel_data_missing_sheet2_columns(test_plate):
    plate_id = test_plate.getId()
    conn = test_plate._conn
    parser = MetadataParser(conn, plate_id)

    # Missing required column in Sheet2
    invalid_data = {
        "Sheet1": pd.DataFrame({"Channels": ["DAPI"], "Index": [1]}),
        "Sheet2": pd.DataFrame({"wrong_column": ["A", "B"]}),
    }
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_excel_data(invalid_data)
    assert "Sheet2 missing required column" in str(exc_info.value)


# --------------------excel data formatting--------------------


def test_format_channel_data(test_plate_with_excel):
    plate_id = test_plate_with_excel.getId()
    conn = test_plate_with_excel._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    if file_annotation:
        excel_file = parser._parse_excel_file(file_annotation)
    assert parser._format_channel_data(excel_file) == {
        "DAPI": 0,
        "Tub": 1,
        "p21": 2,
        "EdU": 3,
    }


def test_format_well_data(test_plate_with_excel):
    plate_id = test_plate_with_excel.getId()
    conn = test_plate_with_excel._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    if file_annotation:
        excel_file = parser._parse_excel_file(file_annotation)
    assert parser._format_well_data(excel_file) == {
        "C2": {"cell_line": "RPE-1", "condition": "ctr"},
        "C5": {"cell_line": "RPE-1", "condition": "CDK4"},
    }
