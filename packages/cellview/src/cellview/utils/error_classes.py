"""
Error classes for the CellView application.
These classes provide rich-formatted error messages and structured error handling.
"""

from pathlib import Path
from typing import Any, Optional

from omero_screen.config import get_logger
from rich.text import Text

from cellview.utils.ui import CellViewUI, Colors

# Initialize logger with the module's name
logger = get_logger(__name__)


class CellViewError(Exception):
    """Base exception class for CellView application.

    All custom exceptions in the application should inherit from this class.
    Provides rich formatting and context handling for error messages.
    """

    def __init__(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
        show_traceback: bool = True,  # Changed default to True
    ) -> None:
        """Initialize the error with a message and optional context.

        Args:
            message: The main error message
            context: Optional dictionary of contextual information
            show_traceback: Whether to show the traceback in rich format
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.show_traceback = show_traceback
        self.ui = CellViewUI()
        self.logger = get_logger(__name__)

    def display(self) -> None:
        """Display the error message with standardized formatting."""
        # Log the error to the file first
        if self.show_traceback:
            self.logger.error(
                "%s: %s",
                self.__class__.__name__,
                self.message,
                exc_info=True,  # This includes the traceback
            )

        # Create error message
        error_text = Text()
        error_text.append(self.message, style=Colors.ERROR.value)

        # Add context information in a more compact format
        if self.context:
            error_text.append(
                "\n\nDetails:", style=f"bold {Colors.WARNING.value}"
            )
            for key, value in self.context.items():
                error_text.append(f"\n• {key}: ", style=Colors.WARNING.value)
                error_text.append(str(value), style=Colors.INFO.value)

        # Add traceback in a cleaner format
        if self.show_traceback:
            import traceback

            tb = traceback.format_exc().strip()
            # Limit traceback length for cleaner display
            tb_lines = tb.split("\n")
            if len(tb_lines) > 10:
                # Keep first 3 and last 6 lines, plus an ellipsis
                tb = "\n".join(tb_lines[:3] + ["..."] + tb_lines[-6:])

            error_text.append(
                "\n\nStacktrace:", style=f"bold {Colors.WARNING.value}"
            )
            error_text.append(f"\n{tb}", style=Colors.ERROR.value)

        # Use the UI's notification panel for consistent styling
        self.ui.notification_panel(error_text, level="error")


class DataError(CellViewError):
    """Errors related to data handling and validation.

    Examples:
        - Missing or invalid CSV columns
        - Invalid data types
        - Duplicate entries
        - Parse errors
    """


class StateError(CellViewError):
    """Errors related to application state.

    Examples:
        - Missing required state (DataFrame, IDs)
        - Invalid state transitions
        - Invalid operation for current state
    """


class DBError(CellViewError):
    """Errors related to database operations.

    Examples:
        - Connection failures
        - Constraint violations
        - Query errors
    """


class FileError(CellViewError):
    """Errors related to file operations.

    Examples:
        - File access issues
        - Missing files
        - Permission issues
    """

    def __init__(
        self,
        message: str,
        file_path: Path,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize with file path information.

        Args:
            message: The main error message
            file_path: Path to the file that caused the error
            context: Optional additional context
        """
        context = context or {}
        context["file_path"] = str(file_path)
        super().__init__(message, context)


class MeasurementError(DataError):
    """Error class for measurement-related issues."""
