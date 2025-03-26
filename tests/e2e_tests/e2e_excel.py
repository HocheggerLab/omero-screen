from omero.gateway import BlitzGateway
from omero_utils.map_anns import parse_annotations

from omero_screen.metadata_parser import MetadataParser


def missing_plate(conn: BlitzGateway, plate_id: int) -> None:
    """Test basic metadata parsing functionality"""
    assert plate_id  # the e2erun has to pass a plaet id here!
    from omero_screen.metadata_parser import MetadataParser

    parser = MetadataParser(conn, 5000)
    parser._parse_metadata()
    print(parser.well_data)


def run_plate_with_correct_excel(conn: BlitzGateway, plate_id: int) -> None:
    """Test the excel file handling functionality in a specific environment.

    Args:
        conn: OMERO connection
        plate_id: ID of the plate to test
        env: Environment to test in. One of:
            - "interactive": Normal terminal (default)
            - "slurm": SLURM batch job environment
    """
    parser = MetadataParser(conn, plate_id)
    parser.manage_metadata()
    plate = conn.getObject("Plate", plate_id)
    print(parse_annotations(plate))
