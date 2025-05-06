from omero.gateway import BlitzGateway
from omero_utils.attachments import delete_file_attachment

from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.metadata_parser import MetadataParser
from omero_screen.plate_dataset import PlateDataset
from tests.e2e_tests.e2e_excel import (
    clean_plate_annotations,
    get_channel_test_data,
)
from tests.e2e_tests.e2e_setup import excel_file_handling


def run_flatfield_corr_test(
    conn: BlitzGateway, teardown: bool = True, plate_id: int = 1
):
    """Generate the flatfield correction masks for the plate."""
    # metadata for the test OMERO data is the same for each plate
    df = get_channel_test_data()
    excel_file_handling(conn, plate_id, df)
    dataset_id = None

    try:
        metadata = MetadataParser(conn, plate_id)
        metadata.manage_metadata()
        dataset_id = PlateDataset(conn, plate_id).dataset_id
        flatfield_dict = flatfieldcorr(conn, metadata, dataset_id)
        for k, v in flatfield_dict.items():
            print(f"{k}: {v.shape}, {v.dtype}")
    finally:
        # Cleanup if requested
        if teardown:
            # Remove metadata
            plate = conn.getObject("Plate", plate_id)
            clean_plate_annotations(conn, plate)
            clean_flatfield_results(conn, plate_id, dataset_id)


def clean_flatfield_results(
    conn: BlitzGateway, plate_id: int, dataset_id: int
):
    """Clean the plate flatfield correction masks"""
    if dataset_id:
        print("Cleaning up flatfield masks")
        image_name = f"{plate_id}_flatfield_masks"
        dataset = conn.getObject("Dataset", dataset_id)
        if dataset is not None:
            for image in dataset.listChildren():
                if image.getName() == image_name:
                    conn.deleteObject(image._obj)
                    break
        # Clean examples
        delete_file_attachment(conn, dataset, ends_with="flatfield_check.pdf")
