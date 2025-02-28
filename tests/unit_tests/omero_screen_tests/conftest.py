import os
import tempfile

import pandas as pd
import pytest
from omero.cmd import Delete2  # noqa


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

        return file_ann


def cleanup_file_annotation(conn, plate, file_ann):
    """Clean up a file annotation and its link to a plate.

    Args:
        conn: OMERO gateway connection
        plate: The plate the annotation is linked to
        file_ann: The file annotation to clean up
    """
    try:
        update_service = conn.getUpdateService()
        # First unlink the annotation from the plate
        plate._obj.unlink(file_ann._obj)
        update_service.saveObject(plate._obj)
        # Then delete the annotation itself
        update_service.deleteObject(file_ann._obj)
    except Exception as e:  # noqa: BLE001
        print(f"Error during file annotation cleanup: {e}")


@pytest.fixture
def attach_excel(omero_conn):
    """Fixture that provides a function to attach Excel files to a plate.

    The fixture provides a function that can be called multiple times to attach
    different Excel files, and handles cleanup automatically.

    Usage:
        def test_something(base_plate, attach_excel):
            # Attach first Excel file
            file1 = attach_excel(base_plate, {
                "Sheet1": pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            })

            # Attach second Excel file
            file2 = attach_excel(base_plate, {
                "Sheet1": pd.DataFrame({"X": [5, 6], "Y": [7, 8]})
            })
    """
    file_anns = []

    def _attach_excel(
        plate,
        dataframes: dict[str, pd.DataFrame],
        filename: str = "metadata.xlsx",
    ):
        file_ann = attach_excel_to_plate(
            omero_conn, plate, dataframes, filename
        )
        file_anns.append((plate, file_ann))
        return file_ann

    yield _attach_excel

    # Cleanup all created annotations
    for plate, file_ann in file_anns:
        cleanup_file_annotation(omero_conn, plate, file_ann)


# @pytest.fixture(scope="class")
# def plate_with_excel_files(base_plate, omero_conn):
#     """
#     Class-scoped fixture that attaches Excel files to a plate.
#     The Excel files are created once for all tests in a test class,
#     then cleaned up after all tests in that class are complete.
#     Uses the session-scoped base_plate fixture.

#     Returns:
#         tuple: (plate object, list of file annotations)
#     """
#     file_anns = []

#     try:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Create and attach Excel files
#             temp_path1 = os.path.join(temp_dir, "metadata1.xlsx")
#             with pd.ExcelWriter(temp_path1, engine="openpyxl") as writer:
#                 df1 = pd.DataFrame(
#                     {
#                         "Channels": ["DAPI", "Tub", "p21", "EdU"],
#                         "Index": [0, 1, 2, 3],
#                     }
#                 )
#                 df1.to_excel(writer, sheet_name="Sheet1", index=False)

#                 df2 = pd.DataFrame(
#                     {
#                         "Well": ["C2", "C5"],
#                         "cell_line": ["RPE-1", "RPE-1"],
#                         "condition": ["ctr", "CDK4"],
#                     }
#                 )
#                 df2.to_excel(writer, sheet_name="Sheet2", index=False)

#             # Attach first Excel file
#             file_ann1 = omero_conn.createFileAnnfromLocalFile(
#                 temp_path1,
#                 mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             )
#             base_plate.linkAnnotation(file_ann1)
#             file_anns.append(file_ann1)

#             # Create and attach second Excel file
#             temp_path2 = os.path.join(temp_dir, "metadata2.xlsx")
#             with pd.ExcelWriter(temp_path2, engine="openpyxl") as writer:
#                 df3 = pd.DataFrame(
#                     {
#                         "Additional": ["Data1", "Data2"],
#                         "Value": [1, 2],
#                     }
#                 )
#                 df3.to_excel(writer, sheet_name="Sheet1", index=False)

#             file_ann2 = omero_conn.createFileAnnfromLocalFile(
#                 temp_path2,
#                 mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             )
#             base_plate.linkAnnotation(file_ann2)
#             file_anns.append(file_ann2)

#         yield base_plate, file_anns

#     finally:
#         # Clean up file annotations after all tests in the class are done
#         update_service = omero_conn.getUpdateService()
#         for file_ann in file_anns:
#             try:
#                 # First unlink the annotation from the plate
#                 base_plate._obj.unlink(file_ann._obj)
#                 update_service.saveObject(base_plate._obj)
#                 # Then delete the annotation itself
#                 update_service.deleteObject(file_ann._obj)
#             except Exception as e:  # noqa: BLE001
#                 print(f"Error during file annotation cleanup: {e}")


# @pytest.fixture(scope="session", params=["single", "multiple"])
# def test_plate_with_excel(omero_conn, request: pytest.FixtureRequest):
#     """
#     Session-scoped fixture to create a test plate and attach Excel file(s) to it.
#     Creates the plate once for all tests and cleans up after all tests are complete.

#     Parameters:
#         request.param: str
#             'single' - creates plate with one Excel file
#             'multiple' - creates plate with two Excel files

#     Returns the plate object.
#     """
#     file_anns = []
#     plate = None
#     try:
#         # Create a new plate
#         update_service = omero_conn.getUpdateService()
#         plate = omero.model.PlateI()
#         plate.name = rstring("Test Plate")
#         plate = update_service.saveAndReturnObject(plate)
#         plate = omero_conn.getObject("Plate", plate.getId().getValue())

#         # Create temp directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # First Excel file (common for both scenarios)
#             temp_path1 = os.path.join(temp_dir, "metadata1.xlsx")
#             with pd.ExcelWriter(temp_path1, engine="openpyxl") as writer:
#                 df1 = pd.DataFrame(
#                     {
#                         "Channels": ["DAPI", "Tub", "p21", "EdU"],
#                         "Index": [0, 1, 2, 3],
#                     }
#                 )
#                 df1.to_excel(writer, sheet_name="Sheet1", index=False)

#                 df2 = pd.DataFrame(
#                     {
#                         "Well": ["C2", "C5"],
#                         "cell_line": ["RPE-1", "RPE-1"],
#                         "condition": ["ctr", "CDK4"],
#                     }
#                 )
#                 df2.to_excel(writer, sheet_name="Sheet2", index=False)

#             # Attach first Excel file
#             file_ann1 = omero_conn.createFileAnnfromLocalFile(
#                 temp_path1,
#                 mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             )
#             plate.linkAnnotation(file_ann1)
#             file_anns.append(file_ann1)

#             # Add second Excel file for 'multiple' scenario
#             if request.param == "multiple":
#                 temp_path2 = os.path.join(temp_dir, "metadata2.xlsx")
#                 with pd.ExcelWriter(temp_path2, engine="openpyxl") as writer:
#                     df3 = pd.DataFrame(
#                         {
#                             "Additional": ["Data1", "Data2"],
#                             "Value": [1, 2],
#                         }
#                     )
#                     df3.to_excel(writer, sheet_name="Sheet1", index=False)

#                 file_ann2 = omero_conn.createFileAnnfromLocalFile(
#                     temp_path2,
#                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                 )
#                 plate.linkAnnotation(file_ann2)
#                 file_anns.append(file_ann2)

#         return plate

#     finally:

#         def cleanup():
#             for file_ann in file_anns:
#                 try:
#                     file_ann.delete()
#                     print(f"Deleted file annotation: {file_ann.getId()}")
#                 except Exception as e:  # noqa: BLE001
#                     print(f"Failed to delete file annotation: {e}")
#             if plate:
#                 try:
#                     plate.delete()
#                     print(f"Deleted plate: {plate.getId()}")
#                 except Exception as e:  # noqa: BLE001
#                     print(f"Failed to delete plate: {e}")

#         request.addfinalizer(cleanup)


# @pytest.fixture(scope="session")
# def test_plate(omero_conn, request: pytest.FixtureRequest):
#     """
#     Session-scoped fixture to create a test plate.
#     Creates the plate once for all tests and cleans up after all tests are complete.
#     Returns the plate object.
#     """
#     plate = None
#     try:
#         # Create a new plate
#         update_service = omero_conn.getUpdateService()
#         plate = omero.model.PlateI()
#         plate.name = rstring("Test Plate")
#         plate = update_service.saveAndReturnObject(plate)
#         plate = omero_conn.getObject("Plate", plate.getId().getValue())

#         return plate

#     finally:

#         def cleanup():
#             if plate:
#                 try:
#                     plate.delete()
#                     print(f"Deleted plate: {plate.getId()}")
#                 except Exception as e:  # noqa: BLE001
#                     print(f"Failed to delete plate: {e}")

#         request.addfinalizer(cleanup)


# @pytest.fixture(
#     scope="session",
#     params=[
#         # Valid cases
#         {"data": [("DAPI", "0"), ("Tub", "1")], "should_pass": True},
#         {"data": [("HOECHST", "0"), ("GFP", "1")], "should_pass": True},
#         {"data": [("RFP", "0"), ("YFP", "1")], "should_pass": True},
#         # Invalid cases - no nuclei channel
#         {"data": [("GFP", "0"), ("YFP", "1")], "should_pass": False},
#         # Invalid cases - non-integer index
#         {"data": [("DAPI", "abc"), ("Tub", "1")], "should_pass": False},
#         {"data": [("DAPI", "1.5"), ("Tub", "1")], "should_pass": False},
#     ],
# )
# def test_plate_with_map_annotations(
#     omero_conn, request: pytest.FixtureRequest
# ):
#     """
#     Session-scoped fixture to create a test plate with map annotations.
#     Creates a plate with channel index map annotations and cleans up after all tests.

#     Parameters:
#         request.param: dict
#             data: List of (channel, index) tuples
#             should_pass: bool - Whether this combination should pass validation

#     Returns:
#         tuple: (plate, should_pass) - The plate object and whether validation should pass

#     The fixture creates plates with both valid and invalid annotations to test validation:
#     - Valid cases include plates with proper nuclei channels and integer indices
#     - Invalid cases include plates missing nuclei channels or with non-integer indices
#     """
#     plate = None
#     map_ann = None
#     try:
#         # Create a new plate
#         update_service = omero_conn.getUpdateService()
#         plate = omero.model.PlateI()
#         plate.name = rstring("Test Plate with Map Annotations")
#         plate = update_service.saveAndReturnObject(plate)
#         plate = omero_conn.getObject("Plate", plate.getId().getValue())

#         # Create map annotation
#         map_ann = omero.gateway.MapAnnotationWrapper(omero_conn)
#         map_ann.setValue(request.param["data"])
#         map_ann.setNs("openmicroscopy.org/omero/client/mapAnnotation")
#         map_ann.save()

#         # Link map annotation to plate
#         plate.linkAnnotation(map_ann)

#         return plate, request.param["should_pass"]

#     finally:

#         def cleanup():
#             if map_ann:
#                 try:
#                     map_ann.delete()
#                     print("Deleted map annotation")
#                 except Exception as e:  # noqa: BLE001
#                     print(f"Failed to delete map annotation: {e}")
#             if plate:
#                 try:
#                     plate.delete()
#                     print(f"Deleted plate: {plate.getId()}")
#                 except Exception as e:  # noqa: BLE001
#                     print(f"Failed to delete plate: {e}")

#         request.addfinalizer(cleanup)
