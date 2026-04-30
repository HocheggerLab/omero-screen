from omero.gateway import BlitzGateway
from omero_screen.loops import plate_loop
import os
import pandas as pd
import numpy as np
from omero.rtypes import rstring
from omero.model import ProjectI
from omero_utils.omero_plate import base_plate, cleanup_plate

from omero_screen.plate_dataset import PlateDataset
from omero_screen.flatfield_corr import upload_images
from tests.e2e_tests.e2e_setup import excel_file_handling
from unittest.mock import MagicMock, patch


def test_plate_image(omero_conn):
    """Test a plate with a single 2D image per well."""
    _run_plate_image(omero_conn, 1, 1)


def test_plate_nd_image(omero_conn):
    """Test a plate with a single time-series 3D image per well."""
    _run_plate_image(omero_conn, 3, 2)


def _run_plate_image(omero_conn, size_z, size_t):
    """Run a plate with a single image per well."""
    plate = None
    project_id = 0
    try:
        os.environ["TEST_IMAGE_SIZE"] = "100"
        plate = base_plate(omero_conn, ["A1", "B1"], size_z=size_z, size_t=size_t)
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
        size = int(os.getenv("TEST_IMAGE_SIZE", "100"))
        image_dict = {str(k): np.ones((size, size)) for k in df["Sheet1"]["Channels"]}
        upload_images(omero_conn, dataset, image_name, image_dict)

        # Mock CellposeModel to avoid downloading models
        with patch("cellpose.models.CellposeModel") as MockCellposeModel:
            # Setup the mock to return a dummy model
            mock_model = MagicMock()
            MockCellposeModel.return_value = mock_model

            # Setup eval method to return dummy masks, flows, styles
            def side_effect_eval(x, **kwargs):
                # Return a mask of the same shape as input image (minus channels)
                # x is typically (C, Y, X) or (Y, X) or list of images
                # Check if input has multiple channels (cell segmentation)
                is_cell_seg = False
                if isinstance(x, list):
                    if len(x) > 1:
                        is_cell_seg = True
                    shape = x[0].shape[-2:]
                else:
                    # Check for (C, Y, X)
                    if x.ndim == 3 and x.shape[0] < 5:
                        is_cell_seg = True
                        shape = x.shape[-2:]
                    # Check for (Y, X, C) created by dstack
                    elif x.ndim == 3 and x.shape[2] < 5:
                        is_cell_seg = True
                        shape = x.shape[:2]
                    else:
                        shape = x.shape[-2:]

                # Create a simple dummy mask with a few labels
                mask = np.zeros(shape, dtype=np.uint16)

                if is_cell_seg:
                    # Larger cells
                    mask[5:35, 5:35] = 1
                    mask[35:65, 35:65] = 2
                else:
                    # Smaller nuclei
                    mask[10:30, 10:30] = 1
                    mask[40:60, 40:60] = 2

                return mask, None, None

            mock_model.eval.side_effect = side_effect_eval

            # run OMERO screen
            df_final, df_final_cc, df_quality_control, dict_gallery = plate_loop(omero_conn, plate_id)
            # check the final results
            assert np.all([x in df_final.columns.values for x in ["plate_id", "well", "image_id", "label"]])

    finally:
        # Clean up
        if plate:
            cleanup_plate(omero_conn, plate)
            if project_id:
                _cleanup_project(omero_conn, project_id)


def _get_channel_test_data(tub: bool = True) -> dict[str, pd.DataFrame]:
    """Return standard test data with DAPI, Tub, EdU channels.

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


def _cleanup_project(conn: BlitzGateway, project_id: int) -> None:
    """Delete a project and all its contents.

    Args:
        conn: The BlitzGateway connection
        project_id: The project to delete
    """
    try:
        # Use the deleteObjects method which is part of the BlitzGateway API
        # wait=True ensures the deletion completes before returning
        conn.deleteObjects(
            "Project",
            [project_id],
            deleteAnns=True,
            deleteChildren=True,
            wait=True,
        )
        print(f"Successfully deleted project {project_id}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete project: {e}")
