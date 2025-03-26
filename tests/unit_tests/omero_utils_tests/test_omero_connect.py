import os

import pytest
from omero_utils.omero_connect import omero_connect


def test_successful_connection():
    @omero_connect
    def check_connection(conn):
        # Just check if we can get the server version, which doesn't require any data
        return conn.getSession()

    server_version = check_connection()

    assert server_version is not None, (
        "Failed to get server version - connection may not be established"
    )


def test_connection_failure(caplog):
    """Test that connection failures are properly logged"""
    # Set wrong credentials
    os.environ["USERNAME"] = "wrong_user"
    os.environ["PASSWORD"] = "wrong_password"
    os.environ["HOST"] = "localhost"  # Keep the host the same

    @omero_connect
    def connect_plate(conn):
        return conn.getObject("Plate", 53)

    with pytest.raises(Exception):  # noqa: B017
        connect_plate()

    # Check the log records
    assert any(
        "Failed to establish connection to OMERO server" in record.message
        for record in caplog.records
    ), "Expected error message not found in logs"
