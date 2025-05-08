import os
import tempfile

import pandas as pd
from omero.gateway import FileAnnotationWrapper
from omero_utils.attachments import delete_excel_attachment, parse_excel_data

# def test_get_file_attachments(test_project):
#     file_ann = get_file_attachments(test_project, ".xlsx")
#     assert file_ann is not None, "failed test because no file attachment found"
#     assert file_ann[0].getFile().getName() == "metadata.xlsx", (
#         "failed test because file name does not match"
#     )


def test_parse_excel_data(base_plate, attach_excel, standard_excel_data):
    # Attach Excel file and get the file annotation
    file1 = attach_excel(base_plate, standard_excel_data)

    # Verify file name
    assert file1.getFile().getName() == "metadata.xlsx", (
        "File name should be metadata.xlsx"
    )

    # Parse the Excel data directly from file1
    data = parse_excel_data(file1)
    assert data is not None, "Parsed data should not be None"

    # Verify sheets exist
    assert "Sheet1" in data, "Sheet1 not found in parsed data"
    assert "Sheet2" in data, "Sheet2 not found in parsed data"

    # Get expected values from the fixture
    expected_sheet1 = standard_excel_data["Sheet1"]
    expected_sheet2 = standard_excel_data["Sheet2"]

    # Verify Sheet1 - Channels
    assert (
        data["Sheet1"]["Channels"].tolist()
        == expected_sheet1["Channels"].tolist()
    ), "Sheet1 channels mismatch"
    assert (
        data["Sheet1"]["Index"].tolist() == expected_sheet1["Index"].tolist()
    ), "Sheet1 indices mismatch"

    # Verify Sheet2 - Experimental conditions
    assert (
        data["Sheet2"]["Well"].tolist() == expected_sheet2["Well"].tolist()
    ), "Sheet2 wells mismatch"
    assert (
        data["Sheet2"]["cell_line"].tolist()
        == expected_sheet2["cell_line"].tolist()
    ), "Sheet2 cell lines mismatch"
    assert (
        data["Sheet2"]["condition"].tolist()
        == expected_sheet2["condition"].tolist()
    ), "Sheet2 conditions mismatch"


def test_delete_excel_attachment(base_plate, standard_excel_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "test_data.xlsx")

        # Write all dataframes to Excel file
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            for sheet_name, df in standard_excel_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create and attach file annotation
        file_ann = base_plate._conn.createFileAnnfromLocalFile(
            temp_path,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        base_plate.linkAnnotation(file_ann)

        # Verify file is attached
        assert file_ann in base_plate.listAnnotations(), (
            "Excel file not attached to plate"
        )

        # Delete excel file
        delete_excel_attachment(base_plate._conn, base_plate)

        # Refresh the base_plate object to get updated state
        refreshed_plate = base_plate._conn.getObject(
            "Plate", base_plate.getId()
        )

        # Verify no excel files are attached
        excel_files = [
            ann
            for ann in refreshed_plate.listAnnotations()
            if isinstance(ann, FileAnnotationWrapper)
        ]
        assert not excel_files, (
            "Excel file was not properly deleted from plate"
        )
