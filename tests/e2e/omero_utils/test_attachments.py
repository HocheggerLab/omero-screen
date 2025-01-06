from omero_utils.attachments import get_named_file_attachment, parse_excel_data


def test_get_named_file_attachment(omero_conn):
    omero_obj = omero_conn.getObject("Project", 251)
    assert omero_obj is not None, "failed test because no object found"

    file_ann = get_named_file_attachment(omero_obj, "metadata.xlsx")
    assert file_ann is not None, "failed test because no file attachment found"
    assert (
        file_ann.getFile().getName() == "metadata.xlsx"
    ), "failed test because file name does not match"


def test_parse_excel_data(omero_conn):
    omero_obj = omero_conn.getObject("Project", 251)
    assert omero_obj is not None, "failed test because no object found"

    file_ann = get_named_file_attachment(omero_obj, "metadata.xlsx")
    assert file_ann is not None, "failed test because no file attachment found"

    data = parse_excel_data(file_ann)
    print(data["Sheet1"])
    print(data["Sheet2"])
