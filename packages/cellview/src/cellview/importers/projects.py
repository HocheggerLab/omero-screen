from typing import Literal, Optional, cast

import duckdb
from omero_screen.config import get_logger
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from cellview.utils.error_classes import DBError, StateError
from cellview.utils.state import CellViewState
from cellview.utils.ui import CellViewUI

JustifyMethod = Literal["default", "left", "center", "right", "full"]

# Initialize logger with the module's name
logger = get_logger(__name__)


class ProjectManager:
    """Manages project selection and creation operations."""

    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        self.db_conn: duckdb.DuckDBPyConnection = db_conn
        self.console = Console()
        self.logger = get_logger(__name__)
        self.state = CellViewState.get_instance()
        self.ui = CellViewUI()

    def select_or_create_project(self) -> None:
        """Main method to select an existing project or create a new one."""
        assert self.state.plate_id is not None
        self._check_plate_exists(self.state.plate_id)
        if self.state.project_name:
            self.state.project_id = self._parse_projectid_from_name(
                self.state.project_name
            )
        elif projects := self._fetch_existing_projects():
            self._display_projects_table(projects)
            while True:
                try:
                    result = self._handle_project_selection(projects)
                    if result is not None:
                        self.state.project_id = result
                        return
                except StateError:
                    # Continue the loop if there's an invalid selection
                    continue
        else:
            self.console.print(
                "[yellow]No projects found. Please enter a new project name.[/yellow]"
            )
            name = Prompt.ask("[cyan]New project name[/cyan]")
            self.state.project_id = self._create_new_project(name)

    def _parse_projectid_from_name(self, name: str) -> int:
        """Parse the project ID from the project name."""
        if result := self.db_conn.execute(
            "SELECT project_id FROM projects WHERE project_name = ?",
            [name],
        ).fetchone():
            self.ui.info(
                f"Attaching data from plate {self.state.plate_id} to project '{name}' with ID {result[0]}"
            )
            return cast(int, result[0])
        self._create_new_project(name)
        return self._parse_projectid_from_name(name)

    def _create_table(
        self, title: str, columns: list[tuple[str, JustifyMethod]]
    ) -> Table:
        """Create a rich table with consistent formatting."""
        table = Table(title=title)
        for col_name, justify in columns:
            table.add_column(col_name, justify=justify)
        return table

    def _check_plate_exists(self, plate_id: int) -> None:
        """Check if a plate exists in the database.
        If it does, display information about where it's stored and raise an error.
        """
        if result := self.db_conn.execute(
            """
            SELECT r.plate_id, p.project_name, e.experiment_name
            FROM repeats r
            JOIN experiments e ON r.experiment_id = e.experiment_id
            JOIN projects p ON e.project_id = p.project_id
            WHERE r.plate_id = ?
            """,
            [plate_id],
        ).fetchone():
            table = self._create_table(
                "Plate Already Exists",
                [
                    ("Plate ID", "right"),
                    ("Project", "left"),
                    ("Experiment", "left"),
                ],
            )
            table.add_row(str(result[0]), result[1], result[2])

            self._add_projects_to_table(table)
            raise DBError(
                "Plate already exists",
                context={
                    "plate_id": plate_id,
                    "project": result[1],
                    "experiment": result[2],
                },
            )

    def _fetch_existing_projects(self) -> list[tuple[int, str, str]]:
        """Fetch all existing projects from the database."""
        try:
            result = self.db_conn.execute(
                "SELECT project_id, project_name, description FROM projects ORDER BY project_id"
            ).fetchall()
            return cast(list[tuple[int, str, str]], result)
        except duckdb.Error as err:
            raise DBError(
                "Failed to fetch projects from database",
                context={"error": str(err)},
            ) from err

    def _display_projects_table(
        self, projects: list[tuple[int, str, str]]
    ) -> None:
        """Display a table of existing projects."""
        table = self._create_table(
            "Available Projects",
            [
                ("ID", "right"),
                ("Project Name", "left"),
                ("Description", "left"),
            ],
        )

        for project_id, project_name, description in projects:
            table.add_row(str(project_id), project_name, description)

        self._add_projects_to_table(table)

    def _add_projects_to_table(self, table: Table) -> None:
        """Add project data rows to the table."""
        self.console.print()
        self.console.print(table)
        self.console.print()

    def _create_new_project(self, name: str) -> int:
        """Create a new project and return its ID."""
        try:
            if existing := self.db_conn.execute(
                "SELECT project_id FROM projects WHERE project_name = ?",
                [name],
            ).fetchone():
                raise DBError(
                    "Project name already exists",
                    context={
                        "project_name": name,
                        "existing_id": existing[0],
                    },
                )

            self.db_conn.execute(
                """
                INSERT INTO projects (project_name)
                VALUES (?)
                """,
                [name],
            )
            result = self.db_conn.execute(
                "SELECT currval('project_id_seq')"
            ).fetchone()
            if result is None:
                raise DBError(
                    "Failed to get ID of newly created project",
                    context={"project_name": name},
                )
            new_id = result[0]
            self.ui.info(f"Created new project '{name}' with ID {new_id}")
            return cast(int, new_id)
        except duckdb.Error as err:
            raise DBError(
                "Failed to create new project",
                context={
                    "project_name": name,
                    "error": str(err),
                },
            ) from err

    def _handle_project_selection(
        self, projects: list[tuple[int, str, str]]
    ) -> Optional[int]:
        """Handle user input for project selection."""
        choice = Prompt.ask(
            "[cyan]Enter a project ID to select, or type a new project name to create it[/cyan]"
        )

        try:
            selected_id = int(choice)
            if any(project_id == selected_id for project_id, _, _ in projects):
                return selected_id
            else:
                raise StateError(
                    "Invalid project ID",
                    context={
                        "provided_id": selected_id,
                        "valid_ids": [p[0] for p in projects],
                    },
                )
        except ValueError:
            # User entered a string - check if it matches existing project
            for project_id, project_name, _ in projects:
                if project_name == choice:
                    self.console.print(
                        f"[green]Selected existing project '{project_name}' (ID: {project_id}).[/green]"
                    )
                    return project_id

            return self._create_new_project(choice)


def select_or_create_project(db_conn: duckdb.DuckDBPyConnection) -> None:
    """Legacy function that creates a ProjectManager instance and calls its main method."""
    manager = ProjectManager(db_conn)
    manager.select_or_create_project()
