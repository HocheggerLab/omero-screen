import functools
import os
import sys
from collections.abc import Callable
from types import TracebackType
from typing import Any

from omero.gateway import BlitzGateway
from omero_screen.config import get_logger
from rich.console import Console
from rich.panel import Panel

# Initialize logger with the module's name
logger = get_logger(__name__)
console = Console()

# Define consistent styling
SUCCESS_STYLE = "bold green"


class OmeroError:
    """Class to store error information for later display"""

    def __init__(
        self,
        message: str,
        error_type: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        # Get the current exception info
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Get the traceback information
        if original_error and original_error.__traceback__:
            # If we have an original error, use its traceback
            frames = self._format_traceback(original_error.__traceback__)
        elif exc_traceback:
            # If we're handling an exception, use its traceback
            # Skip the last frame (this __init__ call)
            tb_to_use = (
                exc_traceback.tb_next
                if exc_traceback.tb_next
                else exc_traceback
            )
            frames = self._format_traceback(tb_to_use)
        else:
            # If no exception context, show the call stack
            frames = self._get_call_stack()

        self.frames = frames

    def _format_traceback(self, tb: TracebackType) -> list[str]:
        """Format a traceback into a list of strings."""
        import traceback

        return traceback.format_tb(tb)

    def _get_call_stack(self) -> list[str]:
        """Get the current call stack as a list of strings."""
        import traceback

        # Exclude the last two frames (this method and __init__)
        return traceback.format_list(traceback.extract_stack()[:-2])

    def display(self) -> None:
        """Display the error with rich formatting"""
        error_msg = [f"[red]{self.error_type}:[/red]\n{self.message}"]

        if self.original_error:
            error_msg.append(
                f"\n[dim]Original error:[/dim] {str(self.original_error)}"
            )

        error_msg.append(f"\n[dim]Location:[/dim]\n{''.join(self.frames)}")

        console.print(
            Panel.fit(
                "\n".join(error_msg),
                title="Error",
                border_style="red",
            )
        )


class OmeroConnectionError(Exception):
    """Raised when there are issues with the OMERO connection"""


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
        OmeroConnectionError: If connection to the OMERO server fails or if there are credential issues
        Exception: Other exceptions from the decorated function are passed through
    """

    @functools.wraps(func)
    def wrapper_omero_connect(*args: Any, **kwargs: Any) -> Any:
        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        host = os.getenv("HOST")
        error = None
        conn = None
        value = None

        try:
            if not all([username, password, host]):
                error = OmeroError(
                    f"Missing required credentials. Need USERNAME, PASSWORD, and HOST.\nGot: host={host}, username={username}, password={'*' * len(password) if password else None}",
                    "Connection Error",
                )
                raise OmeroConnectionError(error.message)

            logger.debug(
                "Connecting to Omero at host: %s, username: %s",
                host,
                username,
            )
            conn = BlitzGateway(username, password, host=host)
            conn.connect()

            if not conn.isConnected():
                error = OmeroError(
                    f"Failed to establish connection to OMERO server at {host} as {username}",
                    "Connection Error",
                )
                raise OmeroConnectionError(error.message)

            logger.info("Successfully connected to Omero at host: %s", host)
            console.print(
                f"[{SUCCESS_STYLE}]✓ Connected to OMERO server at {host} as {username}"
            )
            value = func(*args, **kwargs, conn=conn)

        except OmeroConnectionError:
            if error:
                logger.error("Failed to connect to Omero: %s", error.message)
                error.display()
            raise
        except Exception as e:
            # Only wrap connection-related errors, let other errors pass through
            if (
                "connection" in str(e).lower()
                or "credentials" in str(e).lower()
            ):
                error = OmeroError(
                    f"Failed to connect to OMERO server at {host}",
                    "Connection Error",
                    original_error=e,
                )
                logger.error("Failed to connect to Omero: %s", str(e))
                error.display()
                raise OmeroConnectionError(error.message) from e
            raise
        finally:
            if conn and conn.isConnected():
                conn.close(hard=True)
                logger.info("Closed connection to Omero at host: %s", host)
                console.print(
                    f"[{SUCCESS_STYLE}]✓ Closed connection to OMERO server at {host}"
                )

        return value

    return wrapper_omero_connect
