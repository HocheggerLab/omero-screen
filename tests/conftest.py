import os
from collections.abc import Generator
from pathlib import Path

import Ice
import pytest
from omero.gateway import BlitzGateway
from omero.rtypes import rint, rstring
from omero_model_ImageI import ImageI
from omero_model_PlateAcquisitionI import PlateAcquisitionI
from omero_model_PlateI import PlateI
from omero_model_WellI import WellI
from omero_model_WellSampleI import WellSampleI


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after tests"""
    # Store original environment variables
    env_vars = ["USERNAME", "PASSWORD", "HOST", "LOG_LEVEL", "LOG_FILE_PATH"]
    original_env = {key: os.environ.get(key) for key in env_vars}

    # Clean up environment variables
    for key in env_vars:
        if key in os.environ:
            del os.environ[key]
    # Set development environment explicitly for tests
    os.environ["ENV"] = "development"

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture
def test_env_files(tmp_path) -> Generator[Path, None, None]:
    """Create temporary environment files for testing"""
    env_dev = tmp_path / ".env.development"
    env_prod = tmp_path / ".env.production"
    env_base = tmp_path / ".env"

    env_dev.write_text(
        """
    LOG_LEVEL=DEBUG
    HOST=localhost
    USERNAME=root
    PASSWORD=omero
    LOG_FILE_PATH=/tmp/omero_screen.log
    """.strip()
    )

    env_prod.write_text(
        """
    LOG_LEVEL=WARNING
    HOST=ome2.hpc.sussex.ac.uk
    USERNAME=prod-user
    PASSWORD=prod-pass
    LOG_FILE_PATH=/var/log/omero_screen.log
    """.strip()
    )

    env_base.write_text("ENV=development")

    yield tmp_path


# Fixture to mock environment variables
@pytest.fixture
def mock_env(mocker):
    mocker.patch(
        "os.getenv",
        side_effect=lambda key: {
            "USERNAME": "test_user",
            "PASSWORD": "test_password",
            "HOST": "test_host",
        }.get(key),
    )


# -------------------------------fixtures to create connections and omero objects-----------------------------------------------
@pytest.fixture(scope="session")
def omero_conn(request: pytest.FixtureRequest):
    """Fixture to provide and cleanup OMERO connection"""
    # Setup connection using environment variables
    conn = BlitzGateway(
        os.getenv("USERNAME"),  # Will be 'helfrid' in CI, 'root' locally
        os.getenv("PASSWORD", "omero"),
        host=os.getenv(
            "HOST"
        ),  # Will be 'omeroserver' in CI, 'localhost' locally
    )
    conn.connect()

    yield conn  # Provide the connection to the test

    # Cleanup after test
    try:
        conn.close(hard=True)
    except Exception as e:  # noqa: BLE001
        print(f"OMERO/Ice cleanup error: {e}")

    request.addfinalizer(conn.close)


@pytest.fixture(scope="session", autouse=True)
def ice_cleanup():
    """Session-scoped fixture to cleanup Ice communicator"""
    yield
    try:
        ic = Ice.initialize()
        ic.destroy()
    except Exception as e:  # noqa: BLE001
        print(f"Ice cleanup error: {e}")


def create_basic_plate(conn, name="Test Plate"):
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


def create_well_with_image(conn, plate, plate_acq, position):
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


def cleanup_plate(conn, plate):
    """Clean up a plate and all its related objects.

    Args:
        conn: OMERO gateway connection
        plate: The plate to delete
    """
    from omero.cmd import Delete2

    try:
        delete_cmd = Delete2(targetObjects={"Plate": [plate.getId()]})
        handle = conn.c.submit(delete_cmd)
        handle.loop(10, 1000)  # Wait up to 10 seconds for deletion
        print(f"Deleted plate: {plate.getId()}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete plate: {e}")


@pytest.fixture(scope="session")
def base_plate(omero_conn, request: pytest.FixtureRequest):
    """
    Session-scoped fixture that creates a plate with two wells (C2 and C5).
    Each well is linked to the plate through a PlateAcquisition.
    Uses helper functions to create the plate, wells, and handle cleanup.

    Returns:
        The created plate object
    """
    try:
        # Create the basic plate structure
        plate, plate_acq = create_basic_plate(omero_conn)

        # Create wells with images
        well_positions = ["C2", "C5"]
        for pos in well_positions:
            create_well_with_image(omero_conn, plate, plate_acq, pos)

        # Get the plate as a BlitzObject for easier manipulation
        plate = omero_conn.getObject("Plate", plate.getId().getValue())
        return plate

    finally:

        def cleanup():
            if plate:
                cleanup_plate(omero_conn, plate)

        request.addfinalizer(cleanup)
