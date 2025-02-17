import os
import tempfile

import pandas as pd
import pytest
from omero.gateway import BlitzGateway
from omero.model import ImageI, PlateI, ProjectI, ScreenI, WellI, WellSampleI
from omero.rtypes import rstring


@pytest.fixture
def test_screen_plate_image(omero_conn: BlitzGateway):
    """Create a test screen with a plate and image.

    Returns:
        tuple: (screen_id, plate_id, image_id)
    """
    update = omero_conn.getUpdateService()

    # Create and save Image
    image = ImageI()
    image.name = rstring("Test Image")
    image = update.saveAndReturnObject(image)

    # Create and save Well and WellSample
    well = WellI()
    well_sample = WellSampleI()
    well_sample.image = image
    well.addWellSample(well_sample)

    # Create and save Plate
    plate = PlateI()
    plate.name = rstring("Test Plate")
    plate.addWell(well)
    plate = update.saveAndReturnObject(plate)

    # Create and save Screen
    screen = ScreenI()
    screen.name = rstring("Test Screen")
    screen.linkPlate(plate)
    screen = update.saveAndReturnObject(screen)

    screen_id = screen.getId().getValue()
    plate_id = screen.linkedPlateList()[0].getId().getValue()
    image_id = (
        plate.copyWells()[0].getWellSample(0).getImage().getId().getValue()
    )

    yield screen_id, plate_id, image_id

    # Cleanup - wait for deletion to complete
    omero_conn.deleteObjects("Screen", [screen_id], deleteAnns=True, wait=True)


@pytest.fixture
def test_project(omero_conn):
    """
    Fixture to create a temporary test project and attach an Excel file to it.
    Returns the project object.
    Deletes the project after the test.
    """
    # Setup project
    update_service = omero_conn.getUpdateService()
    project = ProjectI()
    project.setName(rstring("Test Project"))
    project = omero_conn.getObject(
        "Project",
        update_service.saveAndReturnObject(project).getId().getValue(),
    )

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "metadata.xlsx")

        # Create Excel file with two sheets
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            # Sheet1 - Channels
            df1 = pd.DataFrame({"Channels": ["DAPI", "Tub", "EdU"]})
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
        project.linkAnnotation(file_ann)

    yield project

    # Cleanup
    update_service.deleteObject(project._obj)
