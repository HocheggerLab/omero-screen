"""Module for importing plate level experiment data into CellView.

This module provides a class for managing experiment selection and creation operations.
"""

from typing import Optional, cast

import duckdb
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from cellview.utils.state import CellViewState, CellViewStateCore
from cellview.utils.ui import CellViewUI


class ExperimentManager:
    """Manages experiment selection and creation operations.

    Attributes:
        db_conn: The DuckDB connection.
        console: The console.
        state: The CellView state.
        ui: The CellView UI.
    """

    def __init__(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        state: Optional[CellViewStateCore] = None,
    ) -> None:
        """Initialize the ExperimentManager.

        Args:
            db_conn: The DuckDB connection.
            state: The CellView state instance (optional, falls back to singleton if not provided).
        """
        self.db_conn: duckdb.DuckDBPyConnection = db_conn
        self.console = Console()
        # Support both dependency injection and backward compatibility with singleton
        self.state = (
            state if state is not None else CellViewState.get_instance()
        )
        self.ui = CellViewUI()

    def select_or_create_experiment(self) -> int:
        """Main method to select an existing experiment or create a new one.

        Returns:
            The ID of the selected experiment.
        """
        if self.state.experiment_name:
            return self._parse_experimentid_from_name(
                self.state.experiment_name
            )
        elif experiments := self._fetch_existing_experiments():
            self._display_experiments_table(experiments)
            while True:
                result = self._handle_experiment_selection(experiments)
                if result is not None:
                    return result
        else:
            self.console.print(
                "[yellow]No experiments found for this project. Please enter a new experiment name.[/yellow]"
            )
            name = Prompt.ask("[cyan]New experiment name[/cyan]")
            return self._create_new_experiment(name)

    def _parse_experimentid_from_name(self, name: str) -> int:
        """Parse the experiment ID from the experiment name.

        Args:
            name: The name of the experiment.

        Returns:
            The ID of the experiment.
        """
        if result := self.db_conn.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_name = ?",
            [name],
        ).fetchone():
            self.ui.info(
                f"Attaching data from plate {self.state.plate_id} to experiment '{name}' with ID {result[0]}"
            )
            return cast(int, result[0])
        self._create_new_experiment(name)
        return self._parse_experimentid_from_name(name)

    def _fetch_existing_experiments(self) -> list[tuple[int, str, str]]:
        """Fetch all existing experiments for the current project from the database.

        Returns:
            A list of tuples containing the experiment ID, name, and description.
        """
        if not self.state.project_id:
            raise ValueError("No project selected")

        result = self.db_conn.execute(
            """
            SELECT experiment_id, experiment_name, description
            FROM experiments
            WHERE project_id = ?
            ORDER BY experiment_id
            """,
            [self.state.project_id],
        ).fetchall()
        return cast(list[tuple[int, str, str]], result)

    def _fetch_project_name(self) -> str:
        """Fetch the name of the current project.

        Returns:
            The name of the project.
        """
        if not self.state.project_id:
            raise ValueError("No project selected")

        result = self.db_conn.execute(
            "SELECT project_name FROM projects WHERE project_id = ?",
            [self.state.project_id],
        ).fetchone()
        if result is None:
            raise ValueError(
                f"Project with ID {self.state.project_id} not found"
            )
        return cast(str, result[0])

    def _display_experiments_table(
        self, experiments: list[tuple[int, str, str]]
    ) -> None:
        """Display a table of existing experiments.

        Args:
            experiments: A list of tuples containing the experiment ID, name, and description.
        """
        project_name = self._fetch_project_name()
        table = Table(
            title=f"Available Experiments for Project: {project_name}"
        )
        table.add_column("ID", justify="right")
        table.add_column("Experiment Name")
        table.add_column("Description")

        for experiment_id, experiment_name, description in experiments:
            table.add_row(str(experiment_id), experiment_name, description)

        self.console.print()  # Add blank line before table
        self.console.print(table)
        self.console.print()  # Add blank line after table

    def _create_new_experiment(
        self, name: str, description: str | None = None
    ) -> int:
        """Create a new experiment and return its ID.

        Args:
            name: The name of the experiment.
            description: The description of the experiment.

        Returns:
            The ID of the new experiment.
        """
        if not self.state.project_id:
            raise ValueError("No project selected")
        self.db_conn.execute(
            """
            INSERT INTO experiments (project_id, experiment_name, description)
            VALUES (?, ?, ?)
            """,
            (self.state.project_id, name, description),
        )
        result = self.db_conn.execute(
            "SELECT currval('experiment_id_seq')"
        ).fetchone()
        if result is None:
            raise RuntimeError("Failed to get ID of newly created experiment")
        new_id = result[0]
        self.console.print(
            f"[green]Created new experiment '{name}' with ID {new_id}.[/green]"
        )
        return cast(int, new_id)

    def _handle_experiment_selection(
        self, experiments: list[tuple[int, str, str]]
    ) -> Optional[int]:
        """Handle user input for experiment selection.

        Args:
            experiments: A list of tuples containing the experiment ID, name, and description.

        Returns:
            The ID of the selected experiment.
        """
        choice = Prompt.ask(
            "[cyan]Enter an experiment ID to select, or type a new experiment name to create it[/cyan]"
        )

        try:
            selected_id = int(choice)
            if any(
                experiment_id == selected_id
                for experiment_id, _, _ in experiments
            ):
                return selected_id
            self.console.print("[red]Invalid experiment ID.[/red]")
            return None
        except ValueError:
            # Check if the name matches an existing experiment
            for experiment_id, experiment_name, _ in experiments:
                if experiment_name == choice:
                    self.console.print(
                        f"[green]Selected existing experiment '{experiment_name}' (ID: {experiment_id}).[/green]"
                    )
                    return experiment_id

            return self._create_new_experiment(choice)


def select_or_create_experiment(
    db_conn: duckdb.DuckDBPyConnection,
    state: Optional[CellViewStateCore] = None,
) -> None:
    """Function that instantiates an ExperimentManager instance and supplies data to state.

    Args:
        db_conn: The DuckDB connection.
        state: The CellView state instance (optional, falls back to singleton if not provided).
    """
    manager = ExperimentManager(db_conn, state)
    state_instance = (
        state if state is not None else CellViewState.get_instance()
    )
    state_instance.experiment_id = manager.select_or_create_experiment()
