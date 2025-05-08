from omero.gateway import BlitzGateway, PlateWrapper
from omero_utils.images import delete_mip, parse_mip

from omero_screen.plate_dataset import PlateDataset
from tests.e2e_tests.e2e_excel import (
    clean_plate_annotations,
    get_channel_test_data,
)
from tests.e2e_tests.e2e_setup import excel_file_handling


def run_mip_test(conn: BlitzGateway, teardown: bool = True, plate_id: int = 1):
    """Generate the maximum intensity projection (MIP) for the plate."""
    # metadata for the test OMERO data is the same for each plate
    df = get_channel_test_data()
    excel_file_handling(conn, plate_id, df)
    dataset_id = None

    try:
        dataset_id = PlateDataset(conn, plate_id).dataset_id
        plate = conn.getObject("Plate", plate_id)
        for well in plate.listChildren():
            for image in well.listChildren():
                mip = parse_mip(conn, image.getId(), dataset_id)
                print(
                    f"Image ID: {image.getId()}. MIP: {mip.shape} {mip.dtype}"
                )
    finally:
        # Cleanup if requested
        if teardown:
            # Remove metadata
            plate = conn.getObject("Plate", plate_id)
            clean_mip_results(conn, plate)
            clean_plate_annotations(conn, plate)


def clean_mip_results(conn: BlitzGateway, plate: PlateWrapper):
    """Clean the plate flatfield correction masks"""
    if plate is not None:
        print("Cleaning up MIP images")
        for well in plate.listChildren():
            for image in well.listChildren():
                delete_mip(conn, image.getId())
