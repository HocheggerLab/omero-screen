"""Module for the CellViewUI class and related functions."""

import logging
from enum import Enum
from typing import Any, Literal, Optional

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Define justify method type to match Rich's expected values
JustifyMethod = Literal["default", "left", "center", "right", "full"]


class Colors(Enum):
    """Enum for colors used in the CellViewUI class."""

    # Primary colors
    PRIMARY = "white"
    SECONDARY = "cyan"
    ACCENT = "magenta"

    # Status colors
    SUCCESS = "light_green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "green"

    # UI elements
    HEADER = "bold cyan"
    TITLE = "bold blue"
    SUBTLE = "dim white"
    HIGHLIGHT = "bright_white"


class CellViewUI:
    """Class for the CellViewUI class and related functions."""

    def __init__(
        self,
        console: Optional[Console] = None,
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the CellViewUI class.

        Args:
            console: The console to use.
            enable_logging: Whether to enable logging.
            logger: The logger to use.
        """
        self.console = console or Console()
        self.enable_logging = enable_logging
        self.logger = logger

    def _log(self, message: str, level: str = "info") -> None:
        """Log a message.

        Args:
            message: The message to log.
            level: The level to log the message at.
        """
        if self.enable_logging and self.logger:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(message)

    def section(self, title: str, subtitle: Optional[str] = None) -> None:
        """Draw a formatted section header.

        Args:
            title: The title of the section.
            subtitle: The subtitle of the section.
        """
        self.console.print()
        self.console.print(Rule(title, style=Colors.ACCENT.value))
        if subtitle:
            self.console.print(
                f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
            )
        self.console.print()
        self._log(f"[SECTION] {title} - {subtitle or ''}")

    def info(self, message: str) -> None:
        """Print a standard information message.

        Args:
            message: The message to print.
        """
        self.console.print(
            f"[{Colors.INFO.value}]{message}[/{Colors.INFO.value}]"
        )
        self._log(message, "info")

    def success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: The message to print.
        """
        self.console.print(
            f"[{Colors.SUCCESS.value}]✔ {message}[/{Colors.SUCCESS.value}]"
        )
        self._log(message, "info")

    def error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: The message to print.
        """
        self.console.print(
            f"[{Colors.ERROR.value}]✖ {message}[/{Colors.ERROR.value}]"
        )
        self._log(message, "error")

    def warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: The message to print.
        """
        self.console.print(
            f"[{Colors.WARNING.value}]⚠ {message}[/{Colors.WARNING.value}]"
        )
        self._log(message, "warning")

    def progress(self, message: str) -> None:
        """Print a progress/status update.

        Args:
            message: The message to print.
        """
        self.console.print(
            f"[{Colors.SECONDARY.value}]… {message}[/{Colors.SECONDARY.value}]"
        )
        self._log(message, "debug")

    def highlight(self, label: str, value: str) -> None:
        """Highlight a key-value pair (for displaying status/info blocks).

        Args:
            label: The label to print.
            value: The value to print.
        """
        self.console.print(
            f"[{Colors.TITLE.value}]{label}:[/{Colors.TITLE.value}] {value}"
        )

    def header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Display an application section header.

        Args:
            title: The title of the section.
            subtitle: The subtitle of the section.
        """
        self.console.print()
        self.console.print(Rule(title, style=Colors.ACCENT.value))
        if subtitle:
            self.console.print(
                f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
            )

    def notification_panel(
        self, message: str | Text, level: str = "info"
    ) -> None:
        """Display a notification panel with appropriate styling.

        Args:
            message: The message to print.
            level: The level of the message.
        """
        if level == "success":
            style, border_style = Colors.SUCCESS.value, Colors.SUCCESS.value
            title = "Success"
        elif level == "warning":
            style, border_style = Colors.WARNING.value, Colors.WARNING.value
            title = "Warning"
        elif level == "error":
            style, border_style = Colors.ERROR.value, Colors.ERROR.value
            title = "Error"
        else:  # info
            style, border_style = Colors.INFO.value, Colors.PRIMARY.value
            title = "Information"

        text = (
            Text(message, style=style) if isinstance(message, str) else message
        )
        panel = Panel(text, title=title, border_style=border_style)
        self.console.print(panel)


ui = CellViewUI()


def display_table(
    con: duckdb.DuckDBPyConnection,
    title: str,
    rows: list[tuple[Any, ...]],
    columns: Optional[list[str]] = None,
    subtitle: Optional[str] = None,
    style_header: str = Colors.PRIMARY.value,
    style_columns: Optional[list[int]] = None,
    highlight_style: str = Colors.SECONDARY.value,
    show_lines: bool = True,
) -> None:
    """Print a SQL query result from DuckDB as a styled Rich table.

    Args:
        con: The DuckDB connection.
        title: The title of the table.
        rows: The rows of the table.
        columns: The columns of the table.
        subtitle: The subtitle of the table.
        style_header: The style for headers.
        style_columns: The list of column indices to style.
        highlight_style: The Rich style string to apply to those columns.
        show_lines: Whether to show horizontal lines in the table.

    Raises:
        ValueError: If the query does not return any description.
    """
    section_header(title, subtitle)
    # result = con.execute(query).fetchall()
    # Make sure description exists after execute (for type checker)
    assert con.description is not None, "Query did not return any description"
    if columns is None:
        assert con.description is not None, (
            "Query did not return any description"
        )
        columns = [desc[0] for desc in con.description]

    console = Console()
    table = Table(show_lines=show_lines)

    for col in columns:
        table.add_column(col, header_style=style_header)

    for row in rows:
        styled_row = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if style_columns and i in style_columns:
                cell_str = f"[{highlight_style}]{cell_str}[/{highlight_style}]"
            styled_row.append(cell_str)
        table.add_row(*styled_row)

    console.print(table)


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    """Display a stylized section header with title and optional subtitle.

    Args:
        title: The title of the section.
        subtitle: The subtitle of the section.
    """
    console = Console()

    console.print()  # blank line above

    # Rule with the title in accent color
    console.print(Rule(title, style=Colors.ACCENT.value))

    # Optional subtitle below the line
    if subtitle:
        console.print(
            f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
        )

    console.print()  # blank line below
