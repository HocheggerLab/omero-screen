import os
import tempfile

import omero.callbacks
import pandas as pd
import pytest
from omero.cmd import Delete2  # noqa


@pytest.fixture
def test_plate_with_excel(omero_conn):
    """
    Fixture to attach an Excel file to an empty test plate.
    Returns the project object.
    Deletes the project after the test.
    """
    file_ann = None
    try:
        # Setup project
        plate = omero_conn.getObject("Plate", 2)

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

            # Attach Excel file to project
            file_ann = omero_conn.createFileAnnfromLocalFile(
                temp_path,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            plate.linkAnnotation(file_ann)

        yield plate

    finally:
        # Cleanup will run even if test fails
        if file_ann is not None:
            # Create Delete2 command targeting the file annotation

            delete = omero.cmd.Delete2(
                targetObjects={"FileAnnotation": [file_ann.getId()]}
            )

            # Submit the delete command
            handle = omero_conn.c.sf.submit(delete)
            cb = omero.callbacks.CmdCallbackI(omero_conn.c, handle)

            # Wait for deletion to complete
            cb.loop(10, 500)  # Loop 10 times, 500ms between each try

            print(f"Attempted to delete file annotation: {file_ann.getId()}")
