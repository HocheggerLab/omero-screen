from omero_screen.metadata_parser import MetadataParser
from omero_utils.omero_connect import omero_connect


@omero_connect
def metadata_integration(conn=None):
    parser = MetadataParser(conn, 5000)
    parser._parse_metadata()
    print(parser.well_data)


if __name__ == "__main__":
    metadata_integration()
