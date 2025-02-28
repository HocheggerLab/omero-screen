from io import StringIO

import pytest
from rich.console import Console

from omero_screen.metadata_parser import (
    MetadataParser,
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


def test_plate_check_success(base_plate):
    """Test successful plate check"""
    plate_id = base_plate.getId()
    print(f"Plate name: {base_plate.getName()}")
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_plate()


# def test_excel_file_check_failure(test_plate):
#     """Test that None is returned when no Excel file is present"""
#     plate_id = test_plate.getId()
#     conn = test_plate._conn
#     parser = MetadataParser(conn, plate_id)
#     assert parser._check_excel_file() is None


# def test_excel_file_check_success(test_plate_with_excel, request):
#     """Test successful Excel file check with single file"""
#     if request.node.callspec.params["test_plate_with_excel"] == "multiple":
#         pytest.skip("This test is for single file only")

#     plate_id = test_plate_with_excel.getId()
#     conn = test_plate_with_excel._conn
#     parser = MetadataParser(conn, plate_id)
#     assert parser._check_excel_file() is not None


# def test_excel_file_check_multiple_files_error(test_plate_with_excel, request):
#     """Test that error is raised when multiple Excel files are present"""
#     if request.node.callspec.params["test_plate_with_excel"] == "single":
#         pytest.skip("This test is for multiple files only")

#     plate_id = test_plate_with_excel.getId()
#     conn = test_plate_with_excel._conn
#     parser = MetadataParser(conn, plate_id)

#     with pytest.raises(MetadataParsingError) as exc_info:
#         parser._check_excel_file()
#     assert str(exc_info.value) == "Multiple Excel files found on plate"


# # --------------------TEST Validate Excel Data--------------------


# def test_validate_excel_data_success(test_plate_with_excel, request):
#     """Test successful validation of Excel data"""
#     if request.node.callspec.params["test_plate_with_excel"] == "multiple":
#         pytest.skip("This test is for single file only")

#     plate_id = test_plate_with_excel.getId()
#     conn = test_plate_with_excel._conn
#     parser = MetadataParser(conn, plate_id)
#     file_annotation = parser._check_excel_file()
#     if file_annotation:
#         excel_file = parse_excel_data(file_annotation)
#     assert parser._validate_excel_data(excel_file)


# def test_validate_excel_data_missing_sheets(test_plate):
#     plate_id = test_plate.getId()
#     conn = test_plate._conn
#     parser = MetadataParser(conn, plate_id)

#     # Test with empty dict
#     invalid_data = {}
#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._validate_excel_data(invalid_data)
#     assert "Missing required sheet: 'Sheet1'" in str(exc_info.value)


# def test_validate_excel_data_missing_sheet1_columns(test_plate):
#     plate_id = test_plate.getId()
#     conn = test_plate._conn
#     parser = MetadataParser(conn, plate_id)
#     # Missing required columns in Sheet1
#     invalid_data = {
#         "Sheet1": pd.DataFrame({"WrongColumn": [1, 2, 3]}),
#         "Sheet2": pd.DataFrame(
#             {"Well": ["A1", "B1"], "cell_line": ["A", "B"]}
#         ),
#     }
#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._validate_excel_data(invalid_data)
#     assert "Validation error: 2 validation errors" in str(exc_info.value)


# def test_validate_excel_data_missing_nuclei_channel(test_plate):
#     plate_id = test_plate.getId()
#     conn = test_plate._conn
#     parser = MetadataParser(conn, plate_id)

#     # Data without DAPI/Hoechst/RFP channel
#     invalid_data = {
#         "Sheet1": pd.DataFrame({"Channels": ["GFP", "YFP"], "Index": [0, 1]}),
#         "Sheet2": pd.DataFrame(
#             {"Well": ["A1", "B1"], "cell_line": ["A", "B"]}
#         ),
#     }

#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._validate_excel_data(invalid_data)
#     assert "At least one nuclei channel (DAPI/HOECHST/RFP) is required" in str(
#         exc_info.value
#     )


# def test_validate_excel_data_missing_sheet2_columns(test_plate):
#     plate_id = test_plate.getId()
#     conn = test_plate._conn
#     parser = MetadataParser(conn, plate_id)

#     # Missing required column in Sheet2
#     invalid_data = {
#         "Sheet1": pd.DataFrame({"Channels": ["DAPI"], "Index": [1]}),
#         "Sheet2": pd.DataFrame({"wrong_column": ["A", "B"]}),
#     }
#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._validate_excel_data(invalid_data)
#     assert "Missing required sheet: 'Well'" in str(exc_info.value)


# # --------------------excel data formatting--------------------


# def test_format_channel_data(test_plate_with_excel, request):
#     """Test formatting of channel data from Excel"""
#     if request.node.callspec.params["test_plate_with_excel"] == "multiple":
#         pytest.skip("This test is for single file only")

#     plate_id = test_plate_with_excel.getId()
#     conn = test_plate_with_excel._conn
#     parser = MetadataParser(conn, plate_id)
#     file_annotation = parser._check_excel_file()
#     if file_annotation:
#         excel_file = parse_excel_data(file_annotation)
#     assert parser._format_channel_data(excel_file) == {
#         "DAPI": 0,
#         "Tub": 1,
#         "p21": 2,
#         "EdU": 3,
#     }


# def test_format_well_data(test_plate_with_excel, request):
#     """Test formatting of well data from Excel"""
#     if request.node.callspec.params["test_plate_with_excel"] == "multiple":
#         pytest.skip("This test is for single file only")

#     plate_id = test_plate_with_excel.getId()
#     conn = test_plate_with_excel._conn
#     parser = MetadataParser(conn, plate_id)
#     file_annotation = parser._check_excel_file()
#     if file_annotation:
#         excel_file = parse_excel_data(file_annotation)
#     assert parser._format_well_data(excel_file) == {
#         "C2": {"cell_line": "RPE-1", "condition": "ctr"},
#         "C5": {"cell_line": "RPE-1", "condition": "CDK4"},
#     }


# # --------------------TEST Plate Annotations--------------------


# def test_parse_plate_annotations_valid_dapi(test_plate_with_map_annotations):
#     """Test successful parsing of plate annotations with DAPI channel."""
#     plate, should_pass = test_plate_with_map_annotations
#     annotations = list(plate.listAnnotations())
#     if (
#         not should_pass
#         or not annotations
#         or "DAPI" not in dict(annotations[0].getValue())
#     ):
#         pytest.skip("This test is for valid DAPI case only")

#     parser = MetadataParser(plate._conn, plate.getId())
#     result = parser._parse_plate_annotations()
#     assert isinstance(result, dict)
#     assert "DAPI" in result
#     assert result["DAPI"] == 0


# def test_parse_plate_annotations_valid_hoechst(
#     test_plate_with_map_annotations,
# ):
#     """Test successful parsing of plate annotations with HOECHST channel."""
#     plate, should_pass = test_plate_with_map_annotations
#     annotations = list(plate.listAnnotations())
#     if (
#         not should_pass
#         or not annotations
#         or "HOECHST" not in dict(annotations[0].getValue())
#     ):
#         pytest.skip("This test is for valid HOECHST case only")

#     parser = MetadataParser(plate._conn, plate.getId())
#     result = parser._parse_plate_annotations()
#     assert isinstance(result, dict)
#     assert "HOECHST" in result
#     assert result["HOECHST"] == 0


# def test_parse_plate_annotations_valid_rfp(test_plate_with_map_annotations):
#     """Test successful parsing of plate annotations with RFP channel."""
#     plate, should_pass = test_plate_with_map_annotations
#     annotations = list(plate.listAnnotations())
#     if (
#         not should_pass
#         or not annotations
#         or "RFP" not in dict(annotations[0].getValue())
#     ):
#         pytest.skip("This test is for valid RFP case only")

#     parser = MetadataParser(plate._conn, plate.getId())
#     result = parser._parse_plate_annotations()
#     assert isinstance(result, dict)
#     assert "RFP" in result
#     assert result["RFP"] == 0


# def test_parse_plate_annotations_missing_nuclei_channel(
#     test_plate_with_map_annotations,
# ):
#     """Test that validation fails when no nuclei channel is present."""
#     plate, should_pass = test_plate_with_map_annotations
#     annotations = list(plate.listAnnotations())
#     if (
#         should_pass
#         or not annotations
#         or any(
#             ch in dict(annotations[0].getValue())
#             for ch in ["DAPI", "HOECHST", "RFP"]
#         )
#     ):
#         pytest.skip("This test is for missing nuclei channel case only")

#     parser = MetadataParser(plate._conn, plate.getId())
#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._parse_plate_annotations()
#     assert "At least one nuclei channel (DAPI/HOECHST/RFP) is required" in str(
#         exc_info.value
#     )


# def test_parse_plate_annotations_non_integer_index(
#     test_plate_with_map_annotations,
# ):
#     """Test that validation fails when channel index is not an integer."""
#     plate, should_pass = test_plate_with_map_annotations
#     annotations = list(plate.listAnnotations())
#     if should_pass:
#         pytest.skip("This test is for non-integer index case only")

#     if not annotations:
#         pytest.skip("No annotations found")

#     ann_values = dict(annotations[0].getValue())
#     if all(str(value).isdigit() for value in ann_values.values()):
#         pytest.skip("This test is for non-integer index case only")

#     parser = MetadataParser(plate._conn, plate.getId())
#     with pytest.raises(MetadataValidationError) as exc_info:
#         parser._parse_plate_annotations()
#     assert "must be an integer" in str(exc_info.value)
