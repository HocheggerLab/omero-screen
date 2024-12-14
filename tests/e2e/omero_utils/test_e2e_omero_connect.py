import os

import pytest
from dotenv import load_dotenv
from omero_screen import set_env_vars
from omero_utils.omero_connect import omero_connect


def test_set_env_vars_local():
    dotenv_path = set_env_vars()
    load_dotenv(dotenv_path=dotenv_path)

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    assert username == "root", "Username is not correct"
    assert password == "omero", "Password is not correct"


def test_successful_connection(clean_env):
    os.environ.update(
        {"USERNAME": "root", "PASSWORD": "omero", "HOST": "localhost"}
    )

    @omero_connect
    def connect_plate(conn):
        # Simple test to verify connection works
        return conn.getObject("Plate", 53)

    plate = connect_plate()
    assert plate is not None, "Plate is not found"
    assert plate.getName() == "test_plate03", "Plate name is not correct"


def test_connection_failure(capsys, clean_env):
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
