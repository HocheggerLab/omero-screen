from omero_utils.map_anns import parse_annotations
from omero_utils.omero_connect import omero_connect

from omero_screen.metadata_parser import MetadataParser
from tests.e2e_tests.e2e_setup import e2e_excel_setup


@omero_connect
def missing_plate(conn=None):
    """Test basic metadata parsing functionality"""
    parser = MetadataParser(conn, 5000)
    parser._parse_metadata()
    print(parser.well_data)


@omero_connect
def run_plate_with_correct_excel(conn=None):
    """Test the excel file handling functionality"""
    if conn:
        plate_id = e2e_excel_setup(conn)
        print(plate_id)
        parser = MetadataParser(conn, plate_id)
        parser.manage_metadata()
        plate = conn.getObject("Plate", plate_id)
        print(parse_annotations(plate))
