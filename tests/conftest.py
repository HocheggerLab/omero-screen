from typing import Any
from unittest.mock import Mock, patch

import pytest


class MockBlitzObject:
    """Base class for mock OMERO objects"""

    def __init__(self, object_id: int = 1):
        self.id = object_id
        self._details = Mock()
        self._details.getOwner.return_value = Mock(
            getOmeName=Mock(return_value="test_owner")
        )


class MockOMEROConnection:
    """
    Mock OMERO connection with common attributes and methods.
    Extend this class as needed for specific test cases.
    """

    def __init__(self):
        self.connected = False
        self._last_error = "Mock error message"

        # Common OMERO operations
        self.getObject = Mock()
        self.getObjects = Mock()
        self.createObject = Mock()
        self.deleteObject = Mock()
        self.saveObject = Mock()

        # Configure default behaviors
        self._setup_default_behaviors()

    def _setup_default_behaviors(self) -> None:
        """Configure default mock behaviors"""

        # Example: Setup getObject to return a mock image
        def mock_get_object(obj_type: str, obj_id: int) -> Any:
            mock_obj = MockBlitzObject(obj_id)
            if obj_type == "Image":
                mock_obj.getName = Mock(return_value=f"test_image_{obj_id}")
            elif obj_type == "Dataset":
                mock_obj.getName = Mock(return_value=f"test_dataset_{obj_id}")
            return mock_obj

        self.getObject.side_effect = mock_get_object

    def connect(self) -> bool:
        """Mock connection method"""
        self.connected = True
        return True

    def close(self) -> None:
        """Mock close method"""
        self.connected = False

    def getLastError(self) -> str:
        """Mock error retrieval"""
        return self._last_error

    def setLastError(self, error: str) -> None:
        """Set mock error message"""
        self._last_error = error


@pytest.fixture
def mock_blitz_gateway():
    """
    Fixture that provides a mock BlitzGateway with controlled behavior.
    This is the main fixture for OMERO connection mocking.
    """
    mock_conn = MockOMEROConnection()

    with patch(
        "omero_utils.omero_connect.BlitzGateway", return_value=mock_conn
    ):
        yield mock_conn


@pytest.fixture
def mock_env_vars():
    """Fixture that mocks environment variables"""
    env_vars = {
        "USERNAME": "test_user",
        "PASSWORD": "test_pass",
        "HOST": "test_host",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("omero_utils.omero_connect.load_dotenv"),
        patch("omero_utils.omero_connect.set_env_vars"),
    ):
        yield env_vars


@pytest.fixture
def failed_connection(mock_blitz_gateway):
    """Fixture for testing failed connection scenarios"""
    mock_blitz_gateway.connect.return_value = False
    return mock_blitz_gateway
