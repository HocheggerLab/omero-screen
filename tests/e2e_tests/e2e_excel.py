from omero_utils.map_anns import parse_annotations

from omero_screen.metadata_parser import MetadataParser


def missing_plate(conn=None):
    """Test basic metadata parsing functionality"""
    parser = MetadataParser(conn, 5000)
    parser._parse_metadata()
    print(parser.well_data)


def run_plate_with_correct_excel(conn, plate_id: int):
    """Test the excel file handling functionality"""
    parser = MetadataParser(conn, plate_id)
    parser.manage_metadata()
    plate = conn.getObject("Plate", plate_id)
    print(parse_annotations(plate))
