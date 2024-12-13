import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


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
