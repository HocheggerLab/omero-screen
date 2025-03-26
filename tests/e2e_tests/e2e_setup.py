import pandas as pd
from omero_utils.attachments import (
    attach_excel_to_plate,
    delete_excel_attachment,
)
from omero_utils.omero_plate import base_plate


def run_excel_file_handling(conn, plate_id):
    plate = conn.getObject("Plate", plate_id)
    dataframes = {
        "Sheet1": pd.DataFrame(
            {"Channels": ["DAPI", "Tub", "EdU"], "Index": [0, 1, 2]}
        ),
        "Sheet2": pd.DataFrame(
            {
                "Well": ["C2", "C5"],
                "cell_line": ["RPE-1", "RPE-1"],
                "condition": ["Ctr", "Cdk4"],
            }
        ),
    }
    attach_excel_to_plate(conn, plate, dataframes)


def delete_excel(conn=None, plate_id=53):
    if conn:
        plate = conn.getObject("Plate", plate_id)
        delete_excel_attachment(conn, plate)
        deleted_plate = conn.getObject("Plate", plate_id)
        assert deleted_plate is None, "Plate was not deleted"


def e2e_excel_setup(conn=None):
    """Test the excel file handling functionality"""
    plate = base_plate(conn, ["C2", "C5"])
    plate_id = plate.getId()
    # print(f"Testrun: Successfully generated plate with id: {plate_id}")
    run_excel_file_handling(conn, plate_id)
    return plate_id
