from omero.gateway import BlitzGateway, ImageWrapper, ScreenWrapper
from omero_utils.message import OmeroError
import pandas as pd
import numpy as np
import numpy.typing as npt
import omero
from omero.rtypes import rint, rstring
from typing import Any
from typing_extensions import Generator
from omero_utils.omero_plate import base_plate, cleanup_plate
from skimage.draw import ellipse

from omero_screen.plate_dataset import PlateDataset
from omero_screen.flatfield_corr import upload_images
from tests.e2e_tests.e2e_setup import excel_file_handling
from tests.e2e_tests.e2e_flatfield_corr import clean_flatfield_results


def test_plate_image(omero_conn):
    """Test a plate with a single image per well."""
    plate = base_plate(omero_conn, ["A1", "B1"])
    plate_id = plate.getId()
    print(f"Created plate {plate_id}")

    # add metadata to the screen
    df = _get_channel_test_data()
    excel_file_handling(omero_conn, plate_id, df)

    # add flatfield correction image.
    # assume the standard plate image size.
    dataset_id = PlateDataset(omero_conn, plate_id).dataset_id
    dataset = omero_conn.getObject("Dataset", dataset_id)
    image_name = f"{plate_id}_flatfield_masks"
    image_dict = {str(k): np.ones((1080,1080)) for k in df["Sheet1"]["Channels"]}
    upload_images(omero_conn, dataset, image_name, image_dict)

    # TODO:
    # run OMERO screen
    # check the final results files

    # Clean up
    cleanup_plate(omero_conn, plate)
    _cleanup_dataset(omero_conn, dataset_id)


def _get_channel_test_data(tub: bool = True) -> dict[str, pd.DataFrame]:
    """Return standard test data with DAPI, Tub, EdU channels

    Args:
        tub: If False then rename the Tub channel to NoTub
    """
    tub_name = "Tub" if tub else "NoTub"
    return {
        "Sheet1": pd.DataFrame(
            {"Channels": ["DAPI", tub_name, "EdU"], "Index": [0, 1, 2]}
        ),
        "Sheet2": pd.DataFrame(
            {
                "Well": ["A1", "B1"],
                "cell_line": ["RPE-1", "RPE-1"],
                "condition": ["Ctr", "Cdk4"],
            }
        ),
    }


def _cleanup_dataset(conn: BlitzGateway, dataset_id: int) -> None:
    """Delete a dataset and all its contents.

    Args:
        conn: The BlitzGateway connection
        dataset_id: The dataset to delete
    """
    try:
        # Use the deleteObjects method which is part of the BlitzGateway API
        # wait=True ensures the deletion completes before returning
        conn.deleteObjects(
            "Dataset",
            [dataset_id],
            deleteAnns=True,
            deleteChildren=True,
            wait=True,
        )
        print(f"Successfully deleted dataset {dataset_id}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete dataset: {e}")
