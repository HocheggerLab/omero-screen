from omero_utils.omero_connect import omero_connect

from omero_screen.metadata_parser import MetadataParser


def test_e2e_metadata():
    @omero_connect
    def get_well_data(conn):
        parser = MetadataParser(conn, 53)
        parser.parse_metadata()
        well_data = parser._parse_well_annotations()
        print(well_data)

    get_well_data()
