import pytest
from omero.gateway import BlitzGateway
from omero.model import ImageI, PlateI, ScreenI, WellI, WellSampleI
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
