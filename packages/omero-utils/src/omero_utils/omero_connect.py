"""Module for handling OMERO connection lifecycle management.

This module provides a decorator that automatically establishes a connection to an OMERO server using
credentials from environment variables, passes the connection object to the decorated function,
and ensures proper cleanup by closing the connection afterward, even if an exception occurs.

Available functions:

- omero_connect(func): Decorator that handles OMERO connection lifecycle management.

"""

import functools
import os
from collections.abc import Callable
from typing import Any

from omero.gateway import BlitzGateway
from omero_screen.config import get_logger

from omero_utils.message import OmeroConnectionError, log_connection_success

# Initialize logger with the module's name
logger = get_logger(__name__)
SUCCESS_STYLE = "bold green"


def omero_connect(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that handles OMERO connection lifecycle management.

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
        ConnectionError: If connection to the OMERO server fails or if there are credential issues
        Exception: Other exceptions from the decorated function are passed through

    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args: Any, **kwargs: Any) -> Any:
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")
        conn = None
        value = None

        try:
            if not all([username, password, host]):
                raise OmeroConnectionError(
                    f"Missing required credentials. Need USERNAME, PASSWORD, and HOST.\nGot: host={host}, username={username}, password={'*' * len(password) if password else None}",
                    logger,
                )

            logger.debug(
                "Connecting to Omero at host: %s, username: %s",
                host,
                username,
            )
            conn = BlitzGateway(username, password, host=host)
            conn.connect()

            if not conn.isConnected():
                raise OmeroConnectionError(
                    f"Failed to establish connection to OMERO server at {host} as {username}",
                    logger,
                )

            log_connection_success(
                SUCCESS_STYLE,
                f"Connected to OMERO server at {host} as {username}",
                logger,
            )
            value = func(*args, **kwargs, conn=conn)

        except OmeroConnectionError:
            raise
        except Exception as e:
            # Only wrap connection-related errors, let other errors pass through
            if (
                "connection" in str(e).lower()
                or "credentials" in str(e).lower()
            ):
                raise OmeroConnectionError(
                    f"Failed to establish connection to OMERO server at {host} as {username}",
                    logger,
                    original_error=e,
                ) from e
            raise
        finally:
            if conn and conn.isConnected():
                conn.close(hard=True)
                log_connection_success(
                    SUCCESS_STYLE,
                    f"Closed connection to OMERO server at {host}",
                    logger,
                )

        return value

    return wrapper_omero_connect
