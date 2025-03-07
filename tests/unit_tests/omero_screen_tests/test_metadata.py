import pandas as pd
import pytest

from omero_screen.metadata_parser import (
    ChannelAnnotationError,
    ExcelParsingError,
    MetadataParser,
    MetadataValidationError,
    PlateNotFoundError,
    WellAnnotationError,
)


def test_plate_validation_failure(omero_conn):
    """Test that plate validation raises an exception when the plate doesn't exist."""
    # Create a string buffer to capture the output
    # Attempt to create a parser with an invalid plate ID
    with pytest.raises(PlateNotFoundError) as exc_info:
        MetadataParser(omero_conn, 5000)
    # Test the exception message
    assert str(exc_info.value) == "A plate with id 5000 was not found!"


def test_plate_validation_success(base_plate):
    """Test that plate validation works correctly when the plate exists."""
    # Arrange
    plate_id = base_plate.getId()
    conn = base_plate._conn

    # Act - creating the parser should validate the plate
    parser = MetadataParser(conn, plate_id)

    # Assert
    assert parser.plate is not None, "Plate should be set after initialization"
    assert parser.plate.getId() == plate_id, (
        "Plate ID should match the input ID"
    )
    assert parser.plate == base_plate, (
        "Plate object should match the base_plate fixture"
    )


def test_excel_file_check_no_excel(base_plate):
    """Test that None is returned when no Excel file is present."""
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_excel_file() is None


def test_excel_file_check_success(base_plate, attach_excel):
    """Test successful Excel file check with single file"""
    # Variable used implicitly by the test fixture, ignore unused warning
    # ruff: noqa: F841
    file1 = attach_excel(
        base_plate, {"Sheet1": pd.DataFrame({"A": [1, 2], "B": [3, 4]})}
    )
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)
    assert parser._check_excel_file() == file1


def test_load_data_from_excel(base_plate, attach_excel, standard_excel_data):
    """Test successful loading of data from Excel file."""
    # ruff: noqa: F841
    file1 = attach_excel(base_plate, standard_excel_data)
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    assert file_annotation is not None, "Excel file should be found"
    channel_data, well_data = parser._load_data_from_excel(file_annotation)
    assert channel_data["Channels"] == ["DAPI", "Tub", "EdU"]
    assert channel_data["Index"] == [0, 1, 2]
    assert well_data["Well"] == ["C2", "C5"]
    assert well_data["cell_line"] == ["RPE-1", "RPE-1"]
    assert well_data["condition"] == ["Ctr", "Cdk4"]


def test_load_data_from_excel_failure(base_plate, attach_excel):
    """Test that an error is raised when Excel file has invalid format."""
    # ruff: noqa: F841
    file1 = attach_excel(
        base_plate, {"Sheet1": pd.DataFrame({"A": [1, 2], "B": [3, 4]})}
    )
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)
    file_annotation = parser._check_excel_file()
    assert file_annotation is not None, "Excel file should be found"
    with pytest.raises(ExcelParsingError) as exc_info:
        parser._load_data_from_excel(file_annotation)
    assert (
        str(exc_info.value)
        == "Invalid excel file format - expected Sheet1 and Sheet2"
    )


def test_excel_file_check_multiple_files(base_plate, attach_excel):
    """Test that error is raised when multiple Excel files are present"""
    # Attach two Excel files
    attach_excel(base_plate, {"Sheet1": pd.DataFrame({"A": [1, 2]})})
    attach_excel(base_plate, {"Sheet1": pd.DataFrame({"B": [3, 4]})})

    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)

    with pytest.raises(ExcelParsingError) as exc_info:
        parser._check_excel_file()
    assert str(exc_info.value) == "Multiple Excel files found on plate"


def test_parse_channel_annotations_no_annotations(base_plate):
    """Test that appropriate error is raised when no channel annotations exist."""
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)

    with pytest.raises(ChannelAnnotationError) as exc_info:
        parser._parse_channel_annotations()
    assert str(exc_info.value) == "No channel annotations found on plate"


def test_parse_channel_annotations_success(base_plate_with_annotations):
    """Test that channel annotations are correctly parsed."""
    plate = base_plate_with_annotations
    parser = MetadataParser(plate._conn, plate.getId())
    channel_data = parser._parse_channel_annotations()

    # Check that we got all expected channels with correct indices
    expected_channels = {"DAPI": "0", "Tub": "1", "EdU": "2"}
    print(channel_data)
    assert channel_data == expected_channels, (
        f"Expected channel data {expected_channels}, got {channel_data}"
    )


def test_parse_well_annotations_success(base_plate_with_annotations):
    """Test that well annotations are correctly parsed."""
    plate = base_plate_with_annotations
    parser = MetadataParser(plate._conn, plate.getId())
    well_data = parser._parse_well_annotations()
    assert well_data["Well"] == ["C5", "C2"]
    assert well_data["cell_line"] == ["RPE-1", "RPE-1"]
    assert well_data["condition"] == ["Cdk4", "Ctr"]


def test_parse_well_annotations_failure(base_plate):
    """Test that appropriate error is raised when no well annotations exist."""
    for well in base_plate.listChildren():
        for ann in well.listAnnotations():
            base_plate._conn.deleteObject(ann._obj)

    for ann in base_plate.listAnnotations():
        base_plate._conn.deleteObject(ann._obj)
    plate_id = base_plate.getId()
    conn = base_plate._conn
    parser = MetadataParser(conn, plate_id)

    with pytest.raises(WellAnnotationError) as exc_info:
        parser._parse_well_annotations()
    assert str(exc_info.value) == "No well annotations found for well C5"


# --------------------Validation Checks--------------------
# Im mocking the parser class here to test the validation methods


class MockParser(MetadataParser):
    """Mock parser class for testing channel data validation."""

    def __init__(self, channel_data=None, well_data=None):
        # Skip parent class initialization by not calling super().__init__()
        # This avoids the need for OMERO connection objects
        self.channel_data = channel_data if channel_data is not None else {}
        self.well_data = well_data if well_data is not None else {}


# --------------------TEST Validate Metadata Structure--------------------


def test_validate_metadata_structure_success():
    """Test that valid metadata structure passes validation."""
    parser = MockParser(
        channel_data={"DAPI": 0, "GFP": 1},
        well_data={"Well": ["A1", "A2"], "condition": ["ctrl", "treat"]},
    )
    parser._validate_metadata_structure()  # Should not raise any exceptions


def test_validate_metadata_structure_missing_channel_data():
    """Test that missing channel data raises an error."""
    parser = MockParser(well_data={"Well": ["A1", "A2"]})
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_metadata_structure()
    assert str(exc_info.value) == "No channel data found"


def test_validate_metadata_structure_missing_well_data():
    """Test that missing well data raises an error."""
    parser = MockParser(channel_data={"DAPI": 0, "GFP": 1})
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_metadata_structure()
    assert str(exc_info.value) == "No well data found"


def test_validate_metadata_structure_invalid_channel_keys():
    """Test that non-string channel keys raise an error."""
    parser = MockParser(
        channel_data={1: 0, "GFP": 1}, well_data={"Well": ["A1"]}
    )
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_metadata_structure()
    assert (
        str(exc_info.value)
        == "Channel data must be a dictionary with string keys"
    )


def test_validate_metadata_structure_invalid_channel_values():
    """Test that non-integer channel values raise an error."""
    parser = MockParser(
        channel_data={"DAPI": "0", "GFP": 1},  # "0" is a string, not an int
        well_data={"Well": ["A1"]},
    )
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_metadata_structure()
    assert (
        str(exc_info.value)
        == "Channel data must be a dictionary with integer values"
    )


# --------------------TEST Validate Channel Data--------------------


def test_validate_channel_data_with_dapi():
    """Test that DAPI channel passes validation and remains unchanged."""
    parser = MockParser({"DAPI": 0, "GFP": 1})
    parser._validate_channel_data()
    assert parser.channel_data == {"DAPI": 0, "GFP": 1}


def test_validate_channel_data_normalize_hoechst():
    """Test that Hoechst is normalized to DAPI."""
    parser = MockParser({"Hoechst": 0, "GFP": 1})
    parser._validate_channel_data()
    assert "DAPI" in parser.channel_data
    assert parser.channel_data["DAPI"] == 0
    assert "Hoechst" not in parser.channel_data
    assert parser.channel_data["GFP"] == 1


def test_validate_channel_data_normalize_dna():
    """Test that DNA is normalized to DAPI."""
    parser = MockParser({"DNA": 0, "GFP": 1})
    parser._validate_channel_data()
    assert "DAPI" in parser.channel_data
    assert parser.channel_data["DAPI"] == 0
    assert "DNA" not in parser.channel_data
    assert parser.channel_data["GFP"] == 1


def test_validate_channel_data_normalize_rfp():
    """Test that RFP is normalized to DAPI."""
    parser = MockParser({"RFP": 0, "GFP": 1})
    parser._validate_channel_data()
    assert "DAPI" in parser.channel_data
    assert parser.channel_data["DAPI"] == 0
    assert "RFP" not in parser.channel_data
    assert parser.channel_data["GFP"] == 1


def test_validate_channel_data_case_insensitive():
    """Test that nuclei channel detection is case insensitive."""
    parser = MockParser({"dapi": 0, "GFP": 1})
    parser._validate_channel_data()
    assert "DAPI" in parser.channel_data
    assert parser.channel_data["DAPI"] == 0
    assert parser.channel_data["GFP"] == 1


def test_validate_channel_data_no_nuclei_channel():
    """Test that validation fails when no nuclei channel is present."""
    parser = MockParser({"GFP": 0, "YFP": 1})
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_channel_data()
    assert (
        "At least one nuclei channel (DAPI/Hoechst/DNA/RFP) is required"
        in str(exc_info.value)
    )


# --------------------TEST Validate Well Data--------------------


def test_validate_well_data_success():
    """Test that valid well data passes validation."""
    parser = MockParser(
        well_data={
            "Well": ["A1", "A2"],
            "cell_line": ["RPE1", "RPE1"],
            "condition": ["ctrl", "treat"],
        }
    )
    parser._validate_well_data()  # Should not raise any exceptions


def test_validate_well_data_missing_required_key():
    """Test that missing required key raises an error."""
    parser = MockParser(
        well_data={
            "Well": ["A1", "A2"],
            "condition": ["ctrl", "treat"],  # Missing cell_line
        }
    )
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_well_data()
    assert "Missing required keys in well data: cell_line" in str(
        exc_info.value
    )


def test_validate_well_data_non_list_values():
    """Test that non-list values raise an error."""
    parser = MockParser(
        well_data={
            "Well": ["A1", "A2"],
            "cell_line": "RPE1",  # Should be a list
            "condition": ["ctrl", "treat"],
        }
    )
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_well_data()
    assert "Values must be lists for all keys" in str(exc_info.value)
    assert "cell_line" in str(exc_info.value)


def test_validate_well_data_inconsistent_lengths():
    """Test that lists of different lengths raise an error."""
    parser = MockParser(
        well_data={
            "Well": ["A1", "A2"],
            "cell_line": ["RPE1", "RPE1", "RPE1"],  # One extra value
            "condition": ["ctrl", "treat"],
        }
    )
    with pytest.raises(MetadataValidationError) as exc_info:
        parser._validate_well_data()
    assert "All well data lists must have the same length" in str(
        exc_info.value
    )
    assert "Well: 2" in str(exc_info.value)
    assert "cell_line: 3" in str(exc_info.value)
