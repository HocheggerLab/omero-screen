from omero.gateway import BlitzGateway
from omero_utils.images import delete_masks

from omero_screen.loops import plate_loop
from omero_screen.plate_dataset import PlateDataset
from tests.e2e_tests.e2e_excel import (
    clean_plate_annotations,
    get_channel_test_data,
)
from tests.e2e_tests.e2e_flatfield_corr import clean_flatfield_results
from tests.e2e_tests.e2e_mip import clean_mip_results
from tests.e2e_tests.e2e_setup import excel_file_handling


def run_omero_screen_test(
    conn: BlitzGateway, teardown: bool = True, plate_id: int = 1
):
    """Run OMERO Screen on the plate."""
    # metadata for the test OMERO data is the same for each plate
    df = get_channel_test_data()
    excel_file_handling(conn, plate_id, df)

    try:
        plate_loop(conn, plate_id)
    finally:
        # Cleanup if requested
        if teardown:
            # Remove metadata
            plate = conn.getObject("Plate", plate_id)
            dataset_id = PlateDataset(conn, plate_id).dataset_id
            delete_masks(conn, dataset_id)
            clean_mip_results(conn, plate)
            clean_flatfield_results(conn, plate_id, dataset_id)
            clean_plate_annotations(conn, plate)
