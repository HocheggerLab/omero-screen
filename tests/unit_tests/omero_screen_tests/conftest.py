import os
import tempfile

import omero.callbacks
import pandas as pd
import pytest
from omero.cmd import Delete2  # noqa
from omero.rtypes import rstring


@pytest.fixture(scope="session", params=["single", "multiple"])
def test_plate_with_excel(omero_conn, request: pytest.FixtureRequest):
    """
    Session-scoped fixture to create a test plate and attach Excel file(s) to it.
    Creates the plate once for all tests and cleans up after all tests are complete.

    Parameters:
        request.param: str
            'single' - creates plate with one Excel file
            'multiple' - creates plate with two Excel files

    Returns the plate object.
    """
    file_anns = []
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
            # First Excel file (common for both scenarios)
            temp_path1 = os.path.join(temp_dir, "metadata1.xlsx")
            with pd.ExcelWriter(temp_path1, engine="openpyxl") as writer:
                df1 = pd.DataFrame(
                    {
                        "Channels": ["DAPI", "Tub", "p21", "EdU"],
                        "Index": [0, 1, 2, 3],
                    }
                )
                df1.to_excel(writer, sheet_name="Sheet1", index=False)

                df2 = pd.DataFrame(
                    {
                        "Well": ["C2", "C5"],
                        "cell_line": ["RPE-1", "RPE-1"],
                        "condition": ["ctr", "CDK4"],
                    }
                )
                df2.to_excel(writer, sheet_name="Sheet2", index=False)

            # Attach first Excel file
            file_ann1 = omero_conn.createFileAnnfromLocalFile(
                temp_path1,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            plate.linkAnnotation(file_ann1)
            file_anns.append(file_ann1)

            # Add second Excel file for 'multiple' scenario
            if request.param == "multiple":
                temp_path2 = os.path.join(temp_dir, "metadata2.xlsx")
                with pd.ExcelWriter(temp_path2, engine="openpyxl") as writer:
                    df3 = pd.DataFrame(
                        {
                            "Additional": ["Data1", "Data2"],
                            "Value": [1, 2],
                        }
                    )
                    df3.to_excel(writer, sheet_name="Sheet1", index=False)

                file_ann2 = omero_conn.createFileAnnfromLocalFile(
                    temp_path2,
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                plate.linkAnnotation(file_ann2)
                file_anns.append(file_ann2)

        return plate

    finally:

        def cleanup():
            for file_ann in file_anns:
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
