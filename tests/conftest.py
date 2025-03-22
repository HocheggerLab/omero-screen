import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import Ice

# Import command module separately
import pandas as pd
import pytest

# Import omero.callbacks
# Import command module
# Import OMERO modules carefully to avoid circular imports
# First import the gateway module
from omero.gateway import BlitzGateway

# Finally import model classes
# Then import rtypes which doesn't depend on model classes
from omero_utils.omero_plate import (
    cleanup_plate,
    create_basic_plate,
    create_well_with_image,
)


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
    ENABLE_CONSOLE_LOGGING=true
    ENABLE_FILE_LOGGING=true
    LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
    LOG_MAX_BYTES=1048576
    LOG_BACKUP_COUNT=5
    """.strip()
    )

    env_prod.write_text(
        """
    LOG_LEVEL=WARNING
    HOST=ome2.hpc.sussex.ac.uk
    USERNAME=prod-user
    PASSWORD=prod-pass
    LOG_FILE_PATH=/var/log/omero_screen.log
    ENABLE_CONSOLE_LOGGING=false
    ENABLE_FILE_LOGGING=true
    LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
    LOG_MAX_BYTES=5242880
    LOG_BACKUP_COUNT=10
    """.strip()
    )

    env_base.write_text(
        """
    LOG_LEVEL=INFO
    HOST=localhost
    USERNAME=default-user
    PASSWORD=default-pass
    LOG_FILE_PATH=/tmp/omero_screen_default.log
    ENABLE_CONSOLE_LOGGING=true
    ENABLE_FILE_LOGGING=false
    """.strip()
    )

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
        # Try to delete the annotation directly
        # This should cascade and remove links as well
        if file_ann is not None:
            try:
                conn.deleteObject(file_ann._obj)
                print(f"Deleted file annotation: {file_ann.getId()}")
            except Exception as e:  # noqa: BLE001
                print(f"Error deleting file annotation: {e}")

                # If direct deletion fails, try to find and delete the link first
                try:
                    if plate is not None:
                        links = plate.getAnnotationLinks()
                        if links is not None:
                            for link in links:
                                if link.getChild().getId() == file_ann.getId():
                                    conn.deleteObject(link._obj)
                                    print("Deleted annotation link")
                                    break

                            # Try deleting the annotation again
                            conn.deleteObject(file_ann._obj)
                except Exception as nested_e:  # noqa: BLE001
                    print(f"Error during link cleanup: {nested_e}")
    except Exception as e:  # noqa: BLE001
        print(f"Error during file annotation cleanup: {e}")


@pytest.fixture
def standard_excel_data():
    """Fixture providing standard Excel data structure for testing.

    Returns:
        dict: Dictionary containing two DataFrames:
            - Sheet1: Channel data with Channels and Index columns
            - Sheet2: Well data with Well, cell_line, and condition columns
    """
    return {
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


@pytest.fixture
def base_plate_with_annotations(
    omero_conn, base_plate, request: pytest.FixtureRequest
):
    """Fixture that creates a plate with annotations.
    First cleans up any existing annotations, then adds:
    1. Channel annotations to the plate (DAPI:0, Tub:1, EdU:2)
    2. Well annotations to each well (cell_line and condition)
    """
    from omero.constants.metadata import NSCLIENTMAPANNOTATION
    from omero.gateway import MapAnnotationWrapper

    # First clean up any existing annotations
    for well in base_plate.listChildren():
        for ann in well.listAnnotations():
            omero_conn.deleteObject(ann._obj)

    for ann in base_plate.listAnnotations():
        omero_conn.deleteObject(ann._obj)

    # Add channel annotations to plate
    channel_map = [("DAPI", "0"), ("Tub", "1"), ("EdU", "2")]

    # Create map annotation for plate
    map_ann = MapAnnotationWrapper(omero_conn)
    map_ann.setNs(NSCLIENTMAPANNOTATION)
    map_ann.setValue(channel_map)
    map_ann.save()
    base_plate.linkAnnotation(map_ann)

    # Add well annotations
    well_data = {
        "C2": {"cell_line": "RPE-1", "condition": "Ctr"},
        "C5": {"cell_line": "RPE-1", "condition": "Cdk4"},
    }

    for well in base_plate.listChildren():
        well_pos = well.getWellPos()
        if well_pos in well_data:
            # Convert dict to list of tuples for map annotation
            well_map = [(k, v) for k, v in well_data[well_pos].items()]
            map_ann = MapAnnotationWrapper(omero_conn)
            map_ann.setNs(NSCLIENTMAPANNOTATION)
            map_ann.setValue(well_map)
            map_ann.save()
            well.linkAnnotation(map_ann)

    return base_plate
