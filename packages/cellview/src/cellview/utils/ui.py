# Place this in a new file: src/cellview/utils/ui.py

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
    # Primary colors
    PRIMARY = "green"
    SECONDARY = "cyan"
    ACCENT = "magenta"

    # Status colors
    SUCCESS = "light_green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "white"

    # UI elements
    HEADER = "bold cyan"
    TITLE = "bold blue"
    SUBTLE = "dim white"
    HIGHLIGHT = "bright_white"


class CellViewUI:
    def __init__(
        self,
        console: Optional[Console] = None,
        enable_logging: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.console = console or Console()
        self.enable_logging = enable_logging
        self.logger = logger

    def _log(self, message: str, level: str = "info") -> None:
        if self.enable_logging and self.logger:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(message)

    def section(self, title: str, subtitle: Optional[str] = None) -> None:
        """Draw a formatted section header."""
        self.console.print()
        self.console.print(Rule(title, style=Colors.ACCENT.value))
        if subtitle:
            self.console.print(
                f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
            )
        self.console.print()
        self._log(f"[SECTION] {title} - {subtitle or ''}")

    def info(self, message: str) -> None:
        """Print a standard information message."""
        self.console.print(
            f"[{Colors.INFO.value}]{message}[/{Colors.INFO.value}]"
        )
        self._log(message, "info")

    def success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(
            f"[{Colors.SUCCESS.value}]✔ {message}[/{Colors.SUCCESS.value}]"
        )
        self._log(message, "info")

    def error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(
            f"[{Colors.ERROR.value}]✖ {message}[/{Colors.ERROR.value}]"
        )
        self._log(message, "error")

    def warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(
            f"[{Colors.WARNING.value}]⚠ {message}[/{Colors.WARNING.value}]"
        )
        self._log(message, "warning")

    def progress(self, message: str) -> None:
        """Print a progress/status update."""
        self.console.print(
            f"[{Colors.SECONDARY.value}]… {message}[/{Colors.SECONDARY.value}]"
        )
        self._log(message, "debug")

    def highlight(self, label: str, value: str) -> None:
        """Highlight a key-value pair (for displaying status/info blocks)."""
        self.console.print(
            f"[{Colors.TITLE.value}]{label}:[/{Colors.TITLE.value}] {value}"
        )

    def header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Display an application section header"""
        self.console.print()
        self.console.print(Rule(title, style=Colors.ACCENT.value))
        if subtitle:
            self.console.print(
                f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
            )

    def notification_panel(
        self, message: str | Text, level: str = "info"
    ) -> None:
        """Display a notification panel with appropriate styling"""
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
    """
    Print a SQL query result from DuckDB as a styled Rich table.

    Parameters:
    - db_path: path to the DuckDB file
    - query: SQL query string
    - style_header: style for headers (default: 'bold green')
    - style_columns: list of column indices (0-based) to style
    - highlight_style: Rich style string to apply to those columns
    - show_lines: whether to show horizontal lines in the table
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
    """Display a stylized section header with title and optional subtitle."""
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


# class CellViewUI:
#     def __init__(self) -> None:
#         self.console: Console = Console()

#     def header(self, title: str, subtitle: Optional[str] = None) -> None:
#         """Display an application section header"""
#         self.console.print()
#         self.console.print(
#             f"[{Colors.HEADER.value}]{title}[/{Colors.HEADER.value}]"
#         )
#         if subtitle:
#             self.console.print(
#                 f"[{Colors.SUBTLE.value}]{subtitle}[/{Colors.SUBTLE.value}]"
#             )
#         self.console.print()

#     def success(self, message: str) -> None:
#         """Display a success message"""
#         self.console.print(
#             f"[{Colors.SUCCESS.value}]✓ {message}[/{Colors.SUCCESS.value}]"
#         )

#     def info(self, message: str) -> None:
#         """Display an informational message"""
#         self.console.print(
#             f"[{Colors.INFO.value}]ℹ {message}[/{Colors.INFO.value}]"
#         )

#     def warning(self, message: str) -> None:
#         """Display a warning message"""
#         self.console.print(
#             f"[{Colors.WARNING.value}]⚠ {message}[/{Colors.WARNING.value}]"
#         )

#     def error(self, message: str) -> None:
#         """Display a simple error message (not for exceptions)"""
#         self.console.print(
#             f"[{Colors.ERROR.value}]✗ {message}[/{Colors.ERROR.value}]"
#         )

#     def progress(self, message: str) -> None:
#         """Display a progress indicator"""
#         self.console.print(
#             f"[{Colors.SECONDARY.value}]→ {message}...[/{Colors.SECONDARY.value}]"
#         )

#     def create_table(self, title, columns):
#         """Create a standardized Rich table"""
#         table = Table(title=title, title_style=Colors.TITLE.value)

#         for col_name, col_type in columns:
#             # Map data types to corresponding styles
#             if col_type == "id":
#                 style = Colors.ID.value
#             elif col_type == "text":
#                 style = Colors.TEXT.value
#             elif col_type == "numeric":
#                 style = Colors.NUMERIC.value
#             elif col_type == "date":
#                 style = Colors.DATE.value
#             else:
#                 style = Colors.INFO.value

#             table.add_column(col_name, style=style)

#         return table

#     def create_standard_table(
#         self,
#         title: str,
#         column_names: List[str],
#         justify_list: Optional[List[JustifyMethod]] = None,
#     ) -> Table:
#         """Create a simplified table with purple headers and white text for data.

#         Args:
#             title: The title of the table
#             column_names: List of column header names
#             justify_list: Optional list of justification values ('left', 'right', 'center', etc.)
#                           If None, all columns will use default justification

#         Returns:
#             Table: A Rich Table object ready for adding rows with white text
#         """
#         table = Table(title=title, title_style="bold magenta")

#         for i, name in enumerate(column_names):
#             justify: Optional[JustifyMethod] = None
#             if justify_list and i < len(justify_list):
#                 justify = justify_list[i]

#             # Only pass justify if it's not None
#             if justify:
#                 table.add_column(name, style="bold magenta", justify=justify)
#             else:
#                 table.add_column(name, style="bold magenta")

#         return table

#     def notification_panel(self, message, level="info"):
#         """Display a notification panel with appropriate styling"""
#         if level == "success":
#             style, border_style = Colors.SUCCESS.value, Colors.SUCCESS.value
#             title = "Success"
#         elif level == "warning":
#             style, border_style = Colors.WARNING.value, Colors.WARNING.value
#             title = "Warning"
#         elif level == "error":
#             style, border_style = Colors.ERROR.value, Colors.ERROR.value
#             title = "Error"
#         else:  # info
#             style, border_style = Colors.INFO.value, Colors.PRIMARY.value
#             title = "Information"

#         text = Text(message, style=style)
#         panel = Panel(text, title=title, border_style=border_style)
#         self.console.print(panel)
