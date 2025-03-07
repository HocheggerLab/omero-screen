import os
import tempfile

import pandas as pd

from omero_utils.omero_connect import omero_connect


def attach_excel_to_plate(
    conn,
    plate,
    dataframes: dict[str, pd.DataFrame],
    filename: str = "metadata.xlsx",
) -> object:
    """Attach an Excel file with given dataframes to a plate.

    Args:
        conn: OMERO gateway connection
        plate: The plate to attach the file to
        dataframes: Dictionary of sheet_name -> dataframe to write to Excel
        filename: Name of the Excel file to create

    Returns:
        The created file annotation object
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)

        # Write all dataframes to Excel file
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create and attach file annotation
        file_ann = conn.createFileAnnfromLocalFile(
            temp_path,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        plate.linkAnnotation(file_ann)


@omero_connect
def run_excel_file_handling(conn, plate_id=53):
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
