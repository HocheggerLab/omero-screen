"""Module for handling OMERO plate creation and management.

This module provides functions for creating and managing OMERO plates,
including basic plate creation, well creation, and cleanup.

Available functions:

- create_basic_plate(conn, name="Test Plate"): Create a basic plate with a plate acquisition.
- create_well_with_image(conn, plate, plate_acq, position): Create a well at the specified position with a basic image.
- base_plate(omero_conn, well_positions=None): Session-scoped fixture that creates a plate with two wells (C2 and C5).
- cleanup_plate(conn, plate): Delete a plate and all its contents.

"""

import itertools
import os
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from omero.gateway import BlitzGateway
from omero.model import (
    ImageI,
    LengthI,
    PlateAcquisitionI,
    PlateI,
    WellI,
    WellSampleI,
)
from omero.model.enums import UnitsLength
from omero.rtypes import rint, rstring
from skimage.draw import ellipse
from typing_extensions import Generator


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
    plate.setName(rstring(name))
    plate = update_service.saveAndReturnObject(plate)

    # Create and save plate acquisition
    plate_acq = None
    plate_acq = PlateAcquisitionI()
    plate_acq.setPlate(plate)
    plate_acq = update_service.saveAndReturnObject(plate_acq)

    return plate, plate_acq


def create_well_with_image(
    conn: BlitzGateway,
    plate: PlateI,
    plate_acq: PlateAcquisitionI,
    position: str,
    size_z: int = 1,
    size_t: int = 1,
) -> PlateI:
    """Create a well at the specified plate position with a basic 3 channel image.

    For convenience this returns the updated plate and not the well
    to allow repeat calls using the same (up-to-date) plate.

    Args:
        conn: OMERO gateway connection
        plate: The parent plate object
        plate_acq: The plate acquisition object
        position: Well position (e.g., 'C2')
        size_z: Number of slices
        size_t: Number of timepoints

    Returns:
        The saved plate object
    """
    # Convert position to row/column
    row = ord(position[0]) - ord("A")
    col = int(position[1]) - 1

    # Create basic image
    size = int(os.getenv("TEST_IMAGE_SIZE", "1080"))
    img = _create_img((size, size), size_z, size_t)
    image_id = _upload_image(conn, img)

    # Create well
    well = WellI()
    well.setRow(rint(row))
    well.setColumn(rint(col))
    plate.addWell(well)

    # Create well sample and link everything
    well_sample = WellSampleI()
    well_sample.setImage(ImageI(image_id, False))
    well_sample.setPlateAcquisition(plate_acq)
    sample_pos = LengthI(0.0, UnitsLength.METER)
    well_sample.setPosX(sample_pos)
    well_sample.setPosY(sample_pos)
    well.addWellSample(well_sample)

    # Save the well which will cascade save the well sample
    update_service = conn.getUpdateService()
    return update_service.saveAndReturnObject(plate)


def _create_img(
    dim: tuple[int, int], size_z: int, size_t: int
) -> npt.NDArray[np.uint8]:
    """Create a cell image (TCZYX) where C=3.

    Nuclei are in the first channel; cells are in the remaining channels.

    Args:
        dim: (Y,X) dimension of planes
        size_z: Number of slices
        size_t: Number of timepoints
    """
    shape = (3,) + dim
    mask = np.zeros(shape, dtype=np.uint8)
    # Draw cells of 30 diameter with 10 diameter nucleus
    cell_radius = 30 // 2
    nucleus_radius = 10 // 2
    rng = np.random.default_rng()
    for x in range(cell_radius, dim[1] - cell_radius, cell_radius * 2):
        for y in range(cell_radius, dim[0] - cell_radius, cell_radius * 2):
            rr, cc = ellipse(  # type: ignore[no-untyped-call]
                x,
                y,
                cell_radius,
                cell_radius * rng.uniform(0.7, 1),
                dim,
                rotation=rng.uniform(-3.14, 3.14),
            )
            mask[1, rr, cc] = 1
            mask[2, rr, cc] = 1
            rr, cc = ellipse(  # type: ignore[no-untyped-call]
                x,
                y,
                nucleus_radius * rng.uniform(0.7, 1.2),
                nucleus_radius * rng.uniform(0.7, 1.2),
                dim,
                rotation=rng.uniform(-3.14, 3.14),
            )
            mask[0, rr, cc] = 1
    # Mask is CYX. Create output image of TCZYX.
    img = np.zeros((size_t, 3, size_z) + dim, dtype=np.uint8)
    for c in range(3):
        # Random pixel values within the mask
        indices = mask[c] != 0
        count = indices.sum()
        for t, z in itertools.product(range(size_t), range(size_z)):
            i = img[t][c][z]
            i[indices] = rng.uniform(128, 235, count)
            # Add noise
            np.clip(
                i + rng.normal(20, 2, size=dim).astype(np.uint16),
                a_min=0,
                a_max=255,
                out=i,
            )
    return img


def _upload_image(conn: BlitzGateway, img: npt.NDArray[Any]) -> int:
    """Upload the image (TCZYX) to OMERO."""
    size_t, size_c, size_z = img.shape[:3]

    def plane_gen() -> Generator[npt.NDArray[Any]]:
        """Generator that yields each plane in the image in order TCZ."""
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    yield img[t, c, z]

    image = conn.createImageFromNumpySeq(
        plane_gen(), "Test image", size_z, size_c, size_t
    )

    # Add pixel size required for OMERO screen.
    # Re-load the image to avoid update conflicts.
    i = conn.getObject("Image", image.getId())
    u = LengthI(1.0, UnitsLength.MICROMETER)
    p = i.getPrimaryPixels()._obj
    p.setPhysicalSizeX(u)
    p.setPhysicalSizeY(u)
    conn.getUpdateService().saveObject(p)

    return image.getId()  # type: ignore[no-any-return]


def base_plate(
    omero_conn: BlitzGateway,
    well_positions: Optional[list[str]] = None,
    size_z: int = 1,
    size_t: int = 1,
) -> PlateI:
    """Session-scoped fixture that creates a plate with two wells (C2 and C5).

    Each well is linked to the plate through a PlateAcquisition.
    Uses helper functions to create the plate, wells, and handle cleanup.

    Args:
        omero_conn: The OMERO connection object
        well_positions: A list of well positions to create on the plate
        size_z: Number of slices
        size_t: Number of timepoints

    Returns:
        The created plate object
    """
    if well_positions is None:
        well_positions = ["C2", "C5"]

    # Create the basic plate structure
    plate, plate_acq = create_basic_plate(omero_conn)

    # Create wells with images
    for pos in well_positions:
        plate = create_well_with_image(
            omero_conn, plate, plate_acq, pos, size_z=size_z, size_t=size_t
        )

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
