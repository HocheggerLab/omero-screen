import functools
import os
from collections.abc import Callable
from typing import Any

from omero.gateway import BlitzGateway
from omero_screen.config import get_logger

# Initialize logger with the module's name
logger = get_logger(__name__)


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
                "Connecting to Omero at host: %s, username: %s",
                host,
                username,
            )
            conn = BlitzGateway(username, password, host=host)
            conn.connect()
            value = func(*args, **kwargs, conn=conn)
        except Exception as e:
            logger.error("Failed to connect to Omero: %s", str(e))
            raise
        finally:
            if conn and conn.isConnected():
                conn.close(hard=True)
                logger.info("Closing connection to Omero at host: %s", host)

        return value

    return wrapper_omero_connect
