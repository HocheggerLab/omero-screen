import logging
import os

from omero_utils.omero_connect import omero_connect

logger = logging.getLogger(__name__)


def failed_connection():
    """Test behavior with invalid credentials"""
    # Set wrong credentials before attempting connection
    if "USERNAME" in os.environ:
        del os.environ["USERNAME"]
    if "HOST" in os.environ:
        del os.environ["HOST"]

    os.environ["USERNAME"] = "wrong_user"
    os.environ["HOST"] = "wrong_host"

    @omero_connect
    def attempt_connection(conn=None, plate_id=53):
        if conn:
            plate = conn.getObject("Plate", plate_id)
            print(plate.getName())
        else:
            logger.error("No connection")

    # This should raise an authentication error
    attempt_connection()


def successful_connection():
    """Test behavior with valid credentials"""

    # Set correct credentials before attempting connection
    @omero_connect
    def attempt_connection(conn=None, plate_id=53):
        if conn:
            plate = conn.getObject("Plate", plate_id)
            print(plate.getName())
        else:
            logger.error("No connection")

    attempt_connection()
