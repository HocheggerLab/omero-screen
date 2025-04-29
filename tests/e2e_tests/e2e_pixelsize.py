from typing import Any

from omero.gateway import BlitzGateway

from omero_screen.metadata_parser import MetadataParser


def run_pixel_size_test(
    conn: BlitzGateway, teardown: bool = True
) -> dict[str, Any]:
    """Test the pixel size of a plate."""
    plate_id = 1
    metadata_parser = MetadataParser(conn, plate_id)
    metadata_parser.manage_metadata()
    print(f"pixel_size is {metadata_parser.pixel_size}")
    return {"pixel_size": metadata_parser.pixel_size}
