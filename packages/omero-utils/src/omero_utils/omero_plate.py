from typing import Optional

from omero.gateway import BlitzGateway
from omero.model import ImageI, PlateAcquisitionI, PlateI, WellI, WellSampleI
from omero.rtypes import rint, rstring


def create_basic_plate(
    conn: BlitzGateway, name: str = "Test Plate"
) -> tuple[PlateI, PlateAcquisitionI]:
    """Create and save a basic plate with a plate acquisition.

    Args:
        conn: OMERO gateway connection
        name: Name for the plate

    Returns:
        tuple: (plate, plate_acquisition)
    """
    update_service = conn.getUpdateService()

    # Create and save plate
    plate = PlateI()
    plate.name = rstring(name)
    plate = update_service.saveAndReturnObject(plate)

    # Create and save plate acquisition
    plate_acq = PlateAcquisitionI()
    plate_acq.plate = plate
    plate_acq = update_service.saveAndReturnObject(plate_acq)

    return plate, plate_acq


def create_well_with_image(
    conn: BlitzGateway,
    plate: PlateI,
    plate_acq: PlateAcquisitionI,
    position: str,
) -> WellI:
    """Create a well at the specified position with a basic image.

    Args:
        conn: OMERO gateway connection
        plate: The parent plate object
        plate_acq: The plate acquisition object
        position: Well position (e.g., 'C2')

    Returns:
        The saved well object
    """
    update_service = conn.getUpdateService()

    # Convert position to row/column
    row = ord(position[0]) - ord("A")
    col = int(position[1]) - 1

    # Create basic image
    image = ImageI()
    image.name = rstring(f"Placeholder Image for {position}")
    image = update_service.saveAndReturnObject(image)

    # Create well
    well = WellI()
    well.row = rint(row)
    well.column = rint(col)
    well.plate = plate

    # Create well sample and link everything
    well_sample = WellSampleI()
    well_sample.setImage(image)
    well_sample.plateAcquisition = plate_acq
    well_sample.well = well
    well.addWellSample(well_sample)

    # Save the well which will cascade save the well sample
    return update_service.saveAndReturnObject(well)


def base_plate(
    omero_conn: BlitzGateway, well_positions: Optional[list[str]] = None
) -> PlateI:
    """
    Session-scoped fixture that creates a plate with two wells (C2 and C5).
    Each well is linked to the plate through a PlateAcquisition.
    Uses helper functions to create the plate, wells, and handle cleanup.

    Returns:
        The created plate object
    """
    if well_positions is None:
        well_positions = ["C2", "C5"]

    # Create the basic plate structure
    plate, plate_acq = create_basic_plate(omero_conn)

    # Create wells with images
    for pos in well_positions:
        create_well_with_image(omero_conn, plate, plate_acq, pos)

    # Get the plate as a BlitzObject for easier manipulation
    plate = omero_conn.getObject("Plate", plate.getId().getValue())
    return plate


def cleanup_plate(conn: BlitzGateway, plate: PlateI) -> None:
    """Delete a plate and all its contents.

    Args:
        conn: The BlitzGateway connection
        plate: The plate to delete
    """
    try:
        # Use the deleteObjects method which is part of the BlitzGateway API
        # wait=True ensures the deletion completes before returning
        conn.deleteObjects(
            "Plate",
            [plate.getId()],
            deleteAnns=True,
            deleteChildren=True,
            wait=True,
        )
        print(f"Successfully deleted plate {plate.getId()}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete plate: {e}")
