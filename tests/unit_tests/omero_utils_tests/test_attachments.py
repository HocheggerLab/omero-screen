from omero_utils.attachments import get_file_attachments, parse_excel_data


def test_get_file_attachments(test_project):
    file_ann = get_file_attachments(test_project, ".xlsx")
    assert file_ann is not None, "failed test because no file attachment found"
    assert file_ann[0].getFile().getName() == "metadata.xlsx", (
        "failed test because file name does not match"
    )


def test_parse_excel_data(test_project):
    file_ann = get_file_attachments(test_project, ".xlsx")
    assert file_ann is not None, "failed test because no file attachment found"

    data = parse_excel_data(file_ann[0])

    # Verify sheets exist
    assert "Sheet1" in data, "Sheet1 not found in parsed data"
    assert "Sheet2" in data, "Sheet2 not found in parsed data"

    # Verify Sheet1 - Channels
    assert data["Sheet1"]["Channels"].tolist() == [
        "DAPI",
        "Tub",
        "EdU",
    ], "Sheet1 channels mismatch"

    # Verify Sheet2 - Experimental conditions
    assert data["Sheet2"]["Well"].tolist() == [
        "C2",
        "C5",
    ], "Sheet2 wells mismatch"
    assert data["Sheet2"]["cell_line"].tolist() == [
        "RPE-1",
        "RPE-1",
    ], "Sheet2 cell lines mismatch"
    assert data["Sheet2"]["condition"].tolist() == [
        "ctr",
        "CDK4",
    ], "Sheet2 conditions mismatch"
