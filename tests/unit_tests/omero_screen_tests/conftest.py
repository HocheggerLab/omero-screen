import os
import tempfile

import omero.callbacks
import pandas as pd
import pytest
from omero.cmd import Delete2  # noqa
from omero.rtypes import rstring


@pytest.fixture(scope="session")
def test_plate_with_excel(omero_conn, request: pytest.FixtureRequest):
    """
    Session-scoped fixture to create a test plate and attach an Excel file to it.
    Creates the plate once for all tests and cleans up after all tests are complete.
    Returns the plate object.
    """
    file_ann = None
    plate = None
    try:
        # Create a new plate
        update_service = omero_conn.getUpdateService()
        plate = omero.model.PlateI()
        plate.name = rstring("Test Plate")
        plate = update_service.saveAndReturnObject(plate)
        plate = omero_conn.getObject("Plate", plate.getId().getValue())

        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "metadata.xlsx")

            # Create Excel file with two sheets
            with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
                # Sheet1 - Channels
                df1 = pd.DataFrame(
                    {
                        "Channels": ["DAPI", "Tub", "p21", "EdU"],
                        "Index": [0, 1, 2, 3],
                    }
                )
                df1.to_excel(writer, sheet_name="Sheet1", index=False)

                # Sheet2 - Experimental conditions
                df2 = pd.DataFrame(
                    {
                        "Well": ["C2", "C5"],
                        "cell_line": ["RPE-1", "RPE-1"],
                        "condition": ["ctr", "CDK4"],
                    }
                )
                df2.to_excel(writer, sheet_name="Sheet2", index=False)

            # Attach Excel file to plate
            file_ann = omero_conn.createFileAnnfromLocalFile(
                temp_path,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            plate.linkAnnotation(file_ann)

        return plate

    finally:
        # Move cleanup to session teardown
        def cleanup():
            if file_ann:
                try:
                    file_ann.delete()
                    print(f"Deleted file annotation: {file_ann.getId()}")
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to delete file annotation: {e}")
            if plate:
                try:
                    plate.delete()
                    print(f"Deleted plate: {plate.getId()}")
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to delete plate: {e}")

        # Register cleanup to run at the end of the session
        request.addfinalizer(cleanup)


@pytest.fixture(scope="session")
def test_plate(omero_conn, request: pytest.FixtureRequest):
    """
    Session-scoped fixture to create a test plate.
    Creates the plate once for all tests and cleans up after all tests are complete.
    Returns the plate object.
    """
    plate = None
    try:
        # Create a new plate
        update_service = omero_conn.getUpdateService()
        plate = omero.model.PlateI()
        plate.name = rstring("Test Plate")
        plate = update_service.saveAndReturnObject(plate)
        plate = omero_conn.getObject("Plate", plate.getId().getValue())

        return plate

    finally:

        def cleanup():
            if plate:
                try:
                    plate.delete()
                    print(f"Deleted plate: {plate.getId()}")
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to delete plate: {e}")

        request.addfinalizer(cleanup)
