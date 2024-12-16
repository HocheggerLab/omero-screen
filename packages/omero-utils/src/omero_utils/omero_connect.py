import functools
import os
from collections.abc import Callable
from typing import Any

from omero.gateway import BlitzGateway
from omero_screen.config import setup_logging

logger = setup_logging("omero_utils")


def omero_connect(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that handles OMERO connection lifecycle management.

    This decorator automatically establishes a connection to an OMERO server using
    credentials from environment variables, passes the connection object to the
    decorated function, and ensures proper cleanup by closing the connection
    afterward, even if an exception occurs.

    Args:
        func (Callable[..., Any]): The function to be decorated. The decorated function
            must accept a 'conn' keyword argument that will receive the OMERO connection.

    Returns:
        Callable[..., Any]: A wrapper function that handles the OMERO connection lifecycle.

    Raises:
        Exception: If connection to the OMERO server fails.

    Example:
        @omero_connect
        def my_function(image_id: int, conn=None):
            # Use conn to interact with OMERO
            return conn.getObject("Image", image_id)
    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args: Any, **kwargs: Any) -> Any:
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")
        conn = None
        value = None
        try:
            logger.debug(
                "Connecting to Omero at at host: %s, username: %s, password: %s",
                host,
                username,
                password,
            )
            print(f"Connecting to Omero at at host: {host}")
            conn = BlitzGateway(username, password, host=host)
            conn.connect()
            value = func(*args, **kwargs, conn=conn)
        except Exception as e:
            print(
                f"Failed to connect to Omero with the following error message: {e}"
            )
            logger.error("Failed to connect to Omero: %s", str(e))
            raise
        finally:
            # No side effects if called without a connection
            if conn and conn.isConnected():
                conn.close()
                logger.info("Closing connection to Omero")
                print(f"Closing connection to Omero at host: {host}")

        return value

    return wrapper_omero_connect
