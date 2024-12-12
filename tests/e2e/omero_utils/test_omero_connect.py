import os

import pytest
from dotenv import load_dotenv
from omero_screen import set_env_vars
from omero_utils.omero_connect import omero_connect

os.environ.pop("USERNAME", None)
os.environ.pop("PASSWORD", None)
os.environ.pop("HOST", None)
os.environ["ENV"] = "development"


def test_set_env_vars_local():
    dotenv_path = set_env_vars()
    load_dotenv(dotenv_path=dotenv_path)

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    assert username == "root", "Username is not correct"
    assert password == "omero", "Password is not correct"


def test_successful_connection():
    # Test function to be decorated
    @omero_connect
    def connect_plate(conn):
        # Simple test to verify connection works
        return conn.getObject("Plate", 53)

    plate = connect_plate()
    assert plate is not None, "Plate is not found"
    assert plate.getName() == "test_plate03", "Plate name is not correct"


def test_connection_failure(capsys):
    # Save original environment variables
    original_username = os.environ.get("USERNAME")
    original_password = os.environ.get("PASSWORD")
    original_host = os.environ.get("HOST")

    try:
        # First clear all relevant environment variables
        for key in ["USERNAME", "PASSWORD", "HOST"]:
            os.environ.pop(key, None)

        # Set wrong credentials
        os.environ["USERNAME"] = "wrong_user"
        os.environ["PASSWORD"] = "wrong_password"
        os.environ["HOST"] = "localhost"  # Keep the host the same

        @omero_connect
        def connect_plate(conn):
            return conn.getObject("Plate", 53)

        with pytest.raises(Exception):  # noqa: B017
            connect_plate()

        # Capture the stdout and stderr
        captured = capsys.readouterr()
        assert (
            "Failed to connect to Omero" in captured.out
        ), "Expected error message not found in stdout"

    finally:
        # Restore original environment variables
        for key, value in [
            ("USERNAME", original_username),
            ("PASSWORD", original_password),
            ("HOST", original_host),
        ]:
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
