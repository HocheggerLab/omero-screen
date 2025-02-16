import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omero.gateway import BlitzGateway
from omero.model import ProjectI
from omero.rtypes import rstring


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


# Fixture to mock BlitzGateway
@pytest.fixture
def mock_blitzgateway(mocker):
    mock_blitz = mocker.patch("omero_utils.omero_connect.BlitzGateway")
    mock_instance = MagicMock()
    mock_instance.connect.return_value = True
    mock_blitz.return_value = mock_instance
    return mock_instance


@pytest.fixture
def omero_conn():
    # Setup connection
    conn = BlitzGateway("root", "omero", host="localhost")
    conn.connect()

    yield conn  # Provide the connection to the test

    # Cleanup after test
    conn.close()


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
