"""Module to manage data state in the CellView application.

This module maintains the state of imported data, including dataframes,
various IDs and foreign keys that need to be tracked across different operations
in the application.
"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import duckdb
import pandas as pd
from omero.gateway import BlitzGateway, PlateWrapper, TagAnnotationWrapper
from omero_utils.attachments import get_file_attachments, parse_csv_data
from omero_utils.omero_connect import omero_connect
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from cellview.utils.error_classes import DataError, DBError, StateError
from cellview.utils.ui import CellViewUI
from omero_screen.config import get_logger

# Initialize logger with the module's name
logger = get_logger(__name__)

JustifyMethod = Literal["default", "left", "center", "right", "full"]


@dataclass
class CellViewStateCore:
    """Core state manager for CellView application (dependency-injectable).

    This class maintains the state of data that need to be tracked
    across different operations in the application. This version is designed
    for dependency injection and does not use singleton pattern.

    Attributes:
        ui: The user interface object.
        csv_path: The path to the CSV file for import.
        df: The dataframe loaded from the imported CSV file.
        plate_id: The omero plate ID associated with the imported CSV file.
        project_name: The project name.
        experiment_name: The experiment name.
        project_id: The project ID.
        experiment_id: The experiment ID.
        repeat_id: The repeat ID.
        condition_id_map: The condition ID map.
        lab_member: The lab member.
        date: The date.
        channel_0: The first channel.
        channel_1: The second channel.
        channel_2: The third channel.
        channel_3: The fourth channel.
        db_conn: The database connection.
        console: The console for output.
        logger: The logger instance.
    """

    # Instance attributes with default values
    ui: CellViewUI
    csv_path: Optional[Path] = None
    df: Optional[pd.DataFrame] = None
    plate_id: Optional[int] = None
    project_name: Optional[Any] = None
    experiment_name: Optional[Any] = None
    project_id: Optional[int] = None
    experiment_id: Optional[int] = None
    repeat_id: Optional[int] = None
    condition_id_map: Optional[dict[str, int]] = None
    lab_member: Optional[str] = None
    date: Optional[str] = None
    channel_0: Optional[str] = None
    channel_1: Optional[str] = None
    channel_2: Optional[str] = None
    channel_3: Optional[str] = None
    db_conn: Optional[duckdb.DuckDBPyConnection] = None
    console: Console = Console()
    logger: Any = None
    _omero_import_mode: bool = False

    def __post_init__(self) -> None:
        """Initialize logger and console after dataclass initialization."""
        if self.logger is None:
            self.logger = get_logger(__name__)

    @classmethod
    def create_from_args(
        cls, args: Optional[argparse.Namespace] = None
    ) -> "CellViewStateCore":
        """Create a new CellViewStateCore instance from command line arguments.

        Args:
            args: Command line arguments containing csv or plate_id

        Returns:
            Initialized CellViewStateCore instance

        Raises:
            DataError: If there are issues reading CSV data or processing
        """
        instance = cls(
            ui=CellViewUI(), console=Console(), logger=get_logger(__name__)
        )

        # Initialize from args if provided
        if args and args.csv:
            instance.csv_path = args.csv
            instance.df = pd.read_csv(args.csv)
            instance.date = instance.extract_date_from_filename(args.csv.name)
            instance.plate_id = instance.get_plate_id()
        elif args and args.plate_id:
            instance.plate_id = args.plate_id
            instance._omero_import_mode = (
                True  # Flag to indicate OMERO import mode
            )
            (
                instance.df,
                instance.project_name,
                instance.experiment_name,
                instance.date,
                instance.lab_member,
            ) = instance.parse_omero_data(args.plate_id)

            # For OMERO imports, we always want to show confirmation dialog
            # The --interactive flag is maintained for backward compatibility but OMERO imports are now always interactive
            instance.ui.info(
                "OMERO import detected - will show interactive confirmation for project/experiment metadata"
            )

        # Set up channels if we have data
        if instance.df is not None:
            try:
                channels = instance.get_channels()
                instance.channel_0 = channels[0] if channels else None
                instance.channel_1 = channels[1] if len(channels) > 1 else None
                instance.channel_2 = channels[2] if len(channels) > 2 else None
                instance.channel_3 = channels[3] if len(channels) > 3 else None
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as err:
                raise DataError(
                    f"Error reading CSV file: {err}",
                    context={
                        "csv_path": str(args.csv)
                        if args and args.csv
                        else None
                    },
                ) from err
            except KeyError as err:
                raise DataError(
                    f"Required column missing in CSV: {err}",
                    context={
                        "csv_path": str(args.csv)
                        if args and args.csv
                        else None,
                        "missing_column": str(err),
                    },
                ) from err
            except ValueError as err:
                raise DataError(
                    f"Error processing data: {err}",
                    context={
                        "csv_path": str(args.csv)
                        if args and args.csv
                        else None
                    },
                ) from err

        return instance

    # -----------------methods to get data from Omero-----------------

    @omero_connect
    def parse_omero_data(
        self,
        plate_id: int,
        conn: Optional[BlitzGateway] = None,
    ) -> tuple[pd.DataFrame, Any, Any, Any, Any]:
        """Parse the Omero data for the given plate ID.

        Args:
            plate_id: The omero screen plate ID.
            conn: The omero connection.

        Returns:
            A tuple containing the dataframe, project name, experiment name, and date.

        """
        if conn is None:
            raise StateError(
                "No database connection available",
                context={"current_state": self.get_state_dict()},
            )
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            raise DataError(
                "Plate not found",
                context={"plate_id": plate_id},
            )
        df = self._get_plate_df(plate)
        project, experiment, date, owner_fullname = self._get_project_info(
            plate
        )
        return df, project, experiment, date, owner_fullname

    def _get_plate_df(
        self,
        plate: PlateWrapper,
    ) -> pd.DataFrame:
        """Get the plate dataframe from the Omero database.

        Args:
            plate: The omero plate object.

        Returns:
            A dataframe containing the data from the plate csv file.

        Raises:
            DataError: If no CSV annotations are found for the plate.
        """
        csv_annotations = get_file_attachments(plate, "csv")
        if not csv_annotations:
            raise DataError(
                "No CSV annotations found for plate",
                context={"plate_id": self.plate_id},
            )

        # First try to find final_data_cc.csv
        for ann in csv_annotations:
            file_name = ann.getFile().getName()
            if file_name and file_name.endswith("final_data_cc.csv"):
                with self.console.status(
                    "Downloading and parsing CSV...", spinner="dots"
                ):
                    df = parse_csv_data(ann)
                if df is not None:
                    self.ui.info(
                        f"Found IF data csv file with cellcycle annotations attached to plate {self.plate_id}"
                    )
                    return df

        # If final_data_cc.csv not found, look for final_data.csv
        for ann in csv_annotations:
            file_name = ann.getFile().getName()
            if file_name and file_name.endswith("final_data.csv"):
                with self.console.status(
                    "Downloading and parsing CSV...", spinner="dots"
                ):
                    df = parse_csv_data(ann)
                if df is not None:
                    self.ui.info(
                        f"Found IF data csv file without cellcycle annotations attached to plate {self.plate_id}"
                    )
                    return df

        # If neither file is found, raise error
        raise DataError(
            "Plate does not have IF data attached",
            context={"plate_id": self.plate_id},
        )

    def _get_project_info(
        self,
        plate: PlateWrapper,
    ) -> tuple[Any, Any, Any, Any]:
        """Get the project info for the given plate.

        This method now handles both screen-based and standalone plates:
        - If plate is part of a screen with tag annotations: extracts project/experiment names
        - If plate is standalone or has incomplete screen info: returns None values for interactive selection

        Args:
            plate: The omero plate object.

        Returns:
            A tuple containing the project name, experiment name, date, and owner.
            Project/experiment names may be None if plate is standalone.
        """
        screen = plate.getParent()
        owner = plate.getOwner()
        owner_fullname = owner.getFullName()
        plate_date = plate.getDate().strftime("%Y-%m-%d")

        if not screen:
            # Standalone plate - no screen parent
            self.ui.info(
                f"Plate {self.plate_id} is a standalone plate (not part of a screen)"
            )
            return None, None, plate_date, owner_fullname

        # Screen exists - check for tag annotations
        experiment_name = screen.getName()
        tags = [
            ann
            for ann in screen.listAnnotations()
            if isinstance(ann, TagAnnotationWrapper)
        ]

        if len(tags) == 1:
            # Perfect case - screen with exactly one tag
            tag = tags[0]
            project_name = tag.getValue()
            self.ui.info(
                f"Found screen-based plate: project='{project_name}', experiment='{experiment_name}'"
            )
            return project_name, experiment_name, plate_date, owner_fullname
        elif len(tags) == 0:
            # Screen exists but no project tag
            self.ui.info(
                f"Plate {self.plate_id} is part of screen '{experiment_name}' but has no project tag"
            )
            return None, experiment_name, plate_date, owner_fullname
        else:
            # Multiple tags - ambiguous
            tag_values = [tag.getValue() for tag in tags]
            self.ui.warning(
                f"Plate {self.plate_id} has multiple project tags: {tag_values}"
            )
            self.ui.info("Please select project and experiment interactively")
            return None, None, plate_date, owner_fullname

    def confirm_project_experiment_names(self) -> tuple[str, str]:
        """Confirm or modify project and experiment names when importing from OMERO.

        This method handles the interactive confirmation when project/experiment names
        are extracted from OMERO screen metadata. It provides clear information about
        what was detected and handles all edge cases gracefully.

        When users choose to override detected metadata or when no metadata is available,
        this method shows rich table displays of existing projects and experiments,
        similar to the normal import flow.

        Returns:
            A tuple containing the confirmed project name and experiment name.
        """
        from rich.prompt import Confirm

        # Display plate information
        self.console.print(
            f"\n[bold cyan]OMERO Import - Plate {self.plate_id}[/bold cyan]"
        )

        if self.project_name and self.experiment_name:
            # Perfect case - both names available from screen with single tag
            self.console.print(
                "[bold blue]✓ Detected from OMERO screen metadata:[/bold blue]"
            )
            self.console.print(
                f"  Project: [green]{self.project_name}[/green]"
            )
            self.console.print(
                f"  Experiment: [green]{self.experiment_name}[/green]"
            )
            self.console.print("  Source: Screen with single project tag")

            if Confirm.ask("\nUse these detected names?", default=True):
                return str(self.project_name), str(self.experiment_name)
            else:
                self.console.print(
                    "[yellow]User chose to override detected metadata[/yellow]"
                )
                # Fall through to interactive selection

        elif self.experiment_name:
            # Screen exists but no or ambiguous project tags
            self.console.print(
                "[bold blue]✓ Detected experiment from OMERO screen:[/bold blue]"
            )
            self.console.print(
                f"  Experiment: [green]{self.experiment_name}[/green]"
            )
            if self.project_name is None:
                self.console.print("  Issue: Screen has no project tag")
            else:
                self.console.print("  Issue: Screen has multiple project tags")

            if Confirm.ask(
                "\nUse this detected experiment name?", default=True
            ):
                confirmed_experiment = str(self.experiment_name)
                # Still need to select project interactively
                confirmed_project = self._interactive_project_selection()
                return confirmed_project, confirmed_experiment
            else:
                # Fall through to full interactive selection
                pass

        else:
            # Standalone plate or no usable metadata
            self.console.print(
                "[yellow]⚠ Standalone plate (not part of a screen)[/yellow]"
            )
            self.console.print("  No project/experiment metadata available")
            self.console.print("  Interactive selection required")

        # Interactive selection section - common for override and standalone cases
        self.console.print("\n[bold cyan]Interactive Selection[/bold cyan]")
        confirmed_project = self._interactive_project_selection()
        confirmed_experiment = self._interactive_experiment_selection(
            confirmed_project
        )

        # Show summary of final choices
        self.console.print("\n[bold green]Final Selection:[/bold green]")
        self.console.print(f"  Project: [green]{confirmed_project}[/green]")
        self.console.print(
            f"  Experiment: [green]{confirmed_experiment}[/green]"
        )

        return confirmed_project, confirmed_experiment

    def list_existing_projects(self) -> list[tuple[int, str, str]]:
        """List all existing projects in the database.

        Returns:
            A list of tuples containing (project_id, project_name, description).

        Raises:
            StateError: If no database connection is available.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"current_state": self.get_state_dict()},
            )

        try:
            result = self.db_conn.execute(
                "SELECT project_id, project_name, description FROM projects ORDER BY project_id"
            ).fetchall()
            return [
                (int(row[0]), str(row[1]), str(row[2]) if row[2] else "")
                for row in result
            ]
        except Exception as err:
            raise StateError(
                "Failed to fetch projects from database",
                context={"error": str(err)},
            ) from err

    def list_existing_experiments(
        self, project_id: int
    ) -> list[tuple[int, str, str]]:
        """List all existing experiments for a given project.

        Args:
            project_id: The ID of the project to list experiments for.

        Returns:
            A list of tuples containing (experiment_id, experiment_name, description).

        Raises:
            StateError: If no database connection is available.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"current_state": self.get_state_dict()},
            )

        try:
            result = self.db_conn.execute(
                """
                SELECT experiment_id, experiment_name, description
                FROM experiments
                WHERE project_id = ?
                ORDER BY experiment_id
                """,
                [project_id],
            ).fetchall()
            return [
                (int(row[0]), str(row[1]), str(row[2]) if row[2] else "")
                for row in result
            ]
        except Exception as err:
            raise StateError(
                "Failed to fetch experiments from database",
                context={"project_id": project_id, "error": str(err)},
            ) from err

    def _interactive_project_selection(self) -> str:
        """Interactive project selection with rich table display.

        Shows existing projects in a table and allows selection or creation of new project.
        Reuses logic from ProjectManager for consistency.

        Returns:
            The selected or created project name.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"current_state": self.get_state_dict()},
            )

        # Fetch existing projects
        projects = self.list_existing_projects()

        if projects:
            # Display projects table
            self._display_projects_table(projects)

            while True:
                try:
                    result = self._handle_project_selection(projects)
                    if result is not None:
                        return result
                except StateError:
                    # Continue the loop if there's an invalid selection
                    continue
        else:
            self.console.print(
                "[yellow]No projects found. Please enter a new project name.[/yellow]"
            )
            name = str(Prompt.ask("[cyan]New project name[/cyan]"))
            self._create_project_if_needed(name)
            return name

    def _interactive_experiment_selection(self, project_name: str) -> str:
        """Interactive experiment selection with rich table display for a given project.

        Shows existing experiments for the selected project in a table and allows
        selection or creation of new experiment. Reuses logic from ExperimentManager.

        Args:
            project_name: The name of the project to show experiments for.

        Returns:
            The selected or created experiment name.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"current_state": self.get_state_dict()},
            )

        # Get project_id for the project name
        project_id = self._get_project_id_by_name(project_name)
        if not project_id:
            raise StateError(
                f"Project '{project_name}' not found in database",
                context={"project_name": project_name},
            )

        # Fetch existing experiments for this project
        experiments = self.list_existing_experiments(project_id)

        if experiments:
            # Display experiments table
            self._display_experiments_table(experiments, project_name)

            while True:
                result = self._handle_experiment_selection(experiments)
                if result is not None:
                    return result
        else:
            self.console.print(
                f"[yellow]No experiments found for project '{project_name}'. Please enter a new experiment name.[/yellow]"
            )
            name = str(Prompt.ask("[cyan]New experiment name[/cyan]"))
            self._create_experiment_if_needed(name, project_id)
            return name

    def _display_projects_table(
        self, projects: list[tuple[int, str, str]]
    ) -> None:
        """Display a table of existing projects.

        Args:
            projects: A list of tuples containing the project ID, name, and description.
        """
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

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _display_experiments_table(
        self, experiments: list[tuple[int, str, str]], project_name: str
    ) -> None:
        """Display a table of existing experiments for a project.

        Args:
            experiments: A list of tuples containing the experiment ID, name, and description.
            project_name: The name of the project these experiments belong to.
        """
        table = Table(
            title=f"Available Experiments for Project: {project_name}"
        )
        table.add_column("ID", justify="right")
        table.add_column("Experiment Name")
        table.add_column("Description")

        for experiment_id, experiment_name, description in experiments:
            table.add_row(str(experiment_id), experiment_name, description)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _create_table(
        self, title: str, columns: list[tuple[str, JustifyMethod]]
    ) -> Table:
        """Create a rich table with consistent formatting.

        Args:
            title: The title of the table.
            columns: The columns of the table as (name, justify) tuples.

        Returns:
            The configured table.
        """
        table = Table(title=title)
        for col_name, justify in columns:
            table.add_column(col_name, justify=justify)
        return table

    def _handle_project_selection(
        self, projects: list[tuple[int, str, str]]
    ) -> Optional[str]:
        """Handle user input for project selection.

        Args:
            projects: A list of tuples containing the project ID, name, and description.

        Returns:
            The name of the selected or created project, or None if invalid selection.
        """
        choice = Prompt.ask(
            "[cyan]Enter a project ID to select, or type a new project name to create it[/cyan]"
        )

        try:
            selected_id = int(choice)
            # Find project by ID
            for project_id, project_name, _ in projects:
                if project_id == selected_id:
                    self.console.print(
                        f"[green]Selected existing project '{project_name}' (ID: {project_id}).[/green]"
                    )
                    return project_name

            # Invalid ID
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
                    return project_name

            # New project name
            choice_str = str(choice)
            self._create_project_if_needed(choice_str)
            self.console.print(
                f"[green]Created new project '{choice_str}'.[/green]"
            )
            return choice_str

    def _handle_experiment_selection(
        self, experiments: list[tuple[int, str, str]]
    ) -> Optional[str]:
        """Handle user input for experiment selection.

        Args:
            experiments: A list of tuples containing the experiment ID, name, and description.

        Returns:
            The name of the selected or created experiment, or None if invalid selection.
        """
        choice = Prompt.ask(
            "[cyan]Enter an experiment ID to select, or type a new experiment name to create it[/cyan]"
        )

        try:
            selected_id = int(choice)
            # Find experiment by ID
            for experiment_id, experiment_name, _ in experiments:
                if experiment_id == selected_id:
                    self.console.print(
                        f"[green]Selected existing experiment '{experiment_name}' (ID: {experiment_id}).[/green]"
                    )
                    return experiment_name

            # Invalid ID
            self.console.print("[red]Invalid experiment ID.[/red]")
            return None
        except ValueError:
            # User entered a string - check if it matches existing experiment
            for experiment_id, experiment_name, _ in experiments:
                if experiment_name == choice:
                    self.console.print(
                        f"[green]Selected existing experiment '{experiment_name}' (ID: {experiment_id}).[/green]"
                    )
                    return experiment_name

            # New experiment name - we'll create it
            choice_str = str(choice)
            self.console.print(
                f"[green]Will create new experiment '{choice_str}'.[/green]"
            )
            return choice_str

    def _get_project_id_by_name(self, project_name: str) -> Optional[int]:
        """Get project ID by name.

        Args:
            project_name: The name of the project.

        Returns:
            The project ID if found, None otherwise.
        """
        if not self.db_conn:
            return None

        try:
            result = self.db_conn.execute(
                "SELECT project_id FROM projects WHERE project_name = ?",
                [project_name],
            ).fetchone()
            return int(result[0]) if result else None
        except (duckdb.Error, ValueError, TypeError):
            return None

    def _create_project_if_needed(self, name: str) -> None:
        """Create a new project if it doesn't exist.

        Args:
            name: The name of the project to create.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"project_name": name},
            )

        # Check if project already exists
        if self._get_project_id_by_name(name):
            return  # Project already exists

        try:
            self.db_conn.execute(
                "INSERT INTO projects (project_name) VALUES (?)",
                [name],
            )
        except Exception as err:
            raise StateError(
                f"Failed to create project '{name}'",
                context={"project_name": name, "error": str(err)},
            ) from err

    def _create_experiment_if_needed(self, name: str, project_id: int) -> None:
        """Create a new experiment if it doesn't exist.

        Args:
            name: The name of the experiment to create.
            project_id: The ID of the project this experiment belongs to.
        """
        if not self.db_conn:
            raise StateError(
                "No database connection available",
                context={"experiment_name": name, "project_id": project_id},
            )

        # Check if experiment already exists for this project
        try:
            existing = self.db_conn.execute(
                """
                SELECT experiment_id FROM experiments
                WHERE experiment_name = ? AND project_id = ?
                """,
                [name, project_id],
            ).fetchone()

            if existing:
                return  # Experiment already exists

            self.db_conn.execute(
                """
                INSERT INTO experiments (project_id, experiment_name)
                VALUES (?, ?)
                """,
                [project_id, name],
            )
        except Exception as err:
            raise StateError(
                f"Failed to create experiment '{name}' for project_id {project_id}",
                context={
                    "experiment_name": name,
                    "project_id": project_id,
                    "error": str(err),
                },
            ) from err

    # -----------------methods to get data from CSV-----------------

    def get_plate_id(self) -> int:
        """Get the plate ID from the loaded DataFrame.

        Returns:
            The plate ID.

        Raises:
            StateError: If no dataframe is loaded.
        """
        if self.df is None:
            raise StateError(
                "Cannot get plate ID: no DataFrame loaded",
                context={"current_state": self.get_state_dict()},
            )

        try:
            plate_ids = self.df["plate_id"].unique()
            if len(plate_ids) > 1:
                raise DataError(
                    "Multiple plates found in the CSV file",
                    context={
                        "plate_ids": list(plate_ids),
                        "csv_path": str(self.csv_path),
                    },
                )
            elif len(plate_ids) == 0:
                raise DataError(
                    "No plate ID found in CSV",
                    context={"csv_path": str(self.csv_path)},
                )

            self.plate_id = int(plate_ids[0])
            return self.plate_id
        except KeyError as err:
            raise DataError(
                "plate_id column not found in CSV",
                context={
                    "csv_path": str(self.csv_path),
                    "available_columns": list(self.df.columns),
                },
            ) from err

    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract a date in YYMMDD format from a filename and convert to YYYY-MM-DD.

        If no date is found, returns the current date in YYYY-MM-DD format.

        Args:
            filename: The filename to extract the date from

        Returns:
            The date string in YYYY-MM-DD format

        """
        # Pattern looks for 6 digits where first two start with 2 (for 20s decade)
        pattern = r"2[0-9][01][0-9][0-3][0-9]"
        match = re.search(pattern, filename)
        if not match:
            # Return current date in YYYY-MM-DD format
            return datetime.now().strftime("%Y-%m-%d")

        date_str = match[0]
        # Convert YYMMDD to YYYY-MM-DD
        yy = date_str[:2]
        mm = date_str[2:4]
        dd = date_str[4:6]
        return f"20{yy}-{mm}-{dd}"  # Assuming dates are in the 2000s

    def get_channels(self) -> list[str]:
        """Get the channels from the CSV file.

        Returns:
            A list of channel names.

        Raises:
            StateError: If no dataframe is loaded.
        """
        if self.df is None:
            return []
        pattern = re.compile(
            r"intensity_(?:max|min|mean)_([A-Za-z0-9]+)_[a-z]+"
        )
        names = self.df.columns.tolist()
        seen = set()
        markers = []
        for name in names:
            if match := pattern.match(name):
                marker = match[1]
                if marker not in seen:
                    markers.append(marker)
                    seen.add(marker)
        return markers

    def get_state_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the current state.

        Returns:
            A dictionary representation of the current state.
        """
        return {
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "df_loaded": self.df is not None,
            "plate_id": self.plate_id,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "repeat_id": self.repeat_id,
        }

    # -----------------methods to prepare for measurements import-----------------

    def prepare_for_measurements(self) -> None:
        """Prepare the dataframe for measurements import using the methods below.

        Raises:
            StateError: If no dataframe is loaded.
        """
        meas_cols = self._find_measurement_cols()
        self.df = self._trim_df(meas_cols)
        channels = self._get_channel_list()
        self._validate_channels(channels)
        # Skip channel renaming - keep original antibody names for flexibility
        # self._rename_channel_columns(channels)  # REMOVED
        self._rename_centroid_cols()
        self._optimize_measurement_types()
        self._set_classifier()

    def _find_measurement_cols(self) -> list[str]:
        """Find columns that are likely to be measurements by identifying columns that vary within images.

        Returns:
            A list of measurement columns.

        Raises:
            StateError: If no dataframe is loaded.
        """
        assert isinstance(self.df, pd.DataFrame)

        # Count unique values per image for each column
        nunique_per_well = self.df.groupby("image_id").nunique()

        # For each column, find cols that have more than 1 unique value
        variable_cols = nunique_per_well.columns[
            nunique_per_well.max() > 1
        ].tolist()

        # Filter out Unnamed columns
        measurement_cols = [
            col for col in variable_cols if "Unnamed" not in col
        ]

        self.logger.debug("Found measurement columns: %s", measurement_cols)
        return measurement_cols

    def _trim_df(self, measurement_cols: list[str]) -> pd.DataFrame:
        """Trim the df to the measurement columns plus required identification columns.

        Keeps measurement columns along with well, image_id, and timepoint columns
        for proper data identification. Explicitly excludes plate_id.

        Args:
            measurement_cols: A list of measurement columns.

        Returns:
            A trimmed dataframe.

        Raises:
            StateError: If no dataframe is loaded.
        """
        if self.df is None:
            raise StateError("No dataframe loaded in state")

        # Required columns for identification
        required_cols = ["well", "image_id", "timepoint"]

        # Add required columns that exist in the dataframe
        cols_to_keep = measurement_cols.copy()
        for col in required_cols:
            if col in self.df.columns:
                cols_to_keep.append(col)
            else:
                raise StateError(
                    f"Required column {col} not found in dataframe",
                    context={"available_columns": self.df.columns.tolist()},
                )

        # Explicitly exclude plate_id
        cols_to_keep = [
            col
            for col in cols_to_keep
            if col not in ["plate_id", "integrated_int_DAPI"]
        ]

        # Return trimmed dataframe with only needed columns
        return self.df[cols_to_keep]

    def _get_channel_list(self) -> list[str]:
        """Set up channels for each measurement column.

        Identifies all unique channels from the measurement columns.
        The pattern is 'intensity_<measure_type>_<channel>_<location>'
        where measure_type is max, min, mean
        and location is nucleus, cell, cyto

        Returns:
            list[str]: List of unique channel names found in measurements in order of appearance

        """
        if self.df is None:
            return []

        # Use a list to maintain order of first appearance instead of a set
        channels = []

        for col in self.df.columns:
            if match := re.match(
                r"intensity_(?:max|min|mean)_([A-Za-z0-9]+)_(?:nucleus|cell|cyto)",
                col,
            ):
                channel = match[1]
                if channel not in channels:
                    channels.append(channel)

        self.logger.debug("Found channels: %s", channels)
        return channels

    def _validate_channels(self, channels: list[str]) -> None:
        """Validate that DAPI channel exists and log discovered channels.

        Raises:
            StateError: If DAPI channel is missing.
        """
        # DAPI is required
        if "DAPI" not in channels:
            raise StateError(
                "DAPI channel is required but not found in data",
                context={"channels": channels},
            )

        # Log discovered channels for info
        self.logger.info("Discovered channels: %s", channels)

        # Update state channels dynamically based on discovered channels
        self.channel_0 = channels[0] if len(channels) > 0 else None
        self.channel_1 = channels[1] if len(channels) > 1 else None
        self.channel_2 = channels[2] if len(channels) > 2 else None
        self.channel_3 = channels[3] if len(channels) > 3 else None

    def _rename_channel_columns(self, channels: list[str]) -> None:
        """Renames non-DAPI channel names in DataFrame columns to ch1, ch2, etc.

        DAPI is left unchanged.

        Args:
            channels: A list of channel names.

        Raises:
            StateError: If the channels do not match.
        """
        assert isinstance(self.df, pd.DataFrame)
        non_dapi_channels: list[str] = [ch for ch in channels if ch != "DAPI"]
        channel_map: dict[str, str] = {
            ch: f"ch{i + 1}" for i, ch in enumerate(non_dapi_channels)
        }

        # Create new column names using simple string replacement
        new_columns: list[str] = []
        for col in self.df.columns:
            new_col = col
            for original, replacement in channel_map.items():
                # Use direct string replacement instead of regex with word boundaries
                # which may not work as expected in all contexts
                if original in new_col:
                    new_col = new_col.replace(original, replacement)
            new_columns.append(new_col)

        # Rename columns
        self.df.rename(
            columns=dict(zip(self.df.columns, new_columns, strict=False)),
            inplace=True,
        )

    def _rename_centroid_cols(self) -> None:
        """Rename centroid columns to have consistent names.

        - Always add -nuc to centroid-0 and centroid-1
        - If centroid-0_x or centroid-1_x exist, rename to centroid-0-cell and centroid-1-cell
        - Drop any centroid-*_y columns

        Raises:
            StateError: If the centroid columns do not exist.
        """
        assert isinstance(self.df, pd.DataFrame)

        rename_map: dict[str, str] = {}

        # Rename base nucleus centroids
        if "centroid-0" in self.df.columns:
            rename_map["centroid-0"] = "centroid-0-nuc"
        if "centroid-1" in self.df.columns:
            rename_map["centroid-1"] = "centroid-1-nuc"

        # Rename cell centroids if present
        if "centroid-0_x" in self.df.columns:
            rename_map["centroid-0_x"] = "centroid-0-cell"
        if "centroid-1_x" in self.df.columns:
            rename_map["centroid-1_x"] = "centroid-1-cell"

        # Drop y-centroid columns if present
        drop_cols: list[str] = [
            col for col in self.df.columns if col.endswith("_y")
        ]

        # Apply renaming
        self.df.rename(columns=rename_map, inplace=True)

        # Drop _y columns
        self.df.drop(columns=drop_cols, inplace=True, errors="ignore")

    def _optimize_measurement_types(self) -> None:
        """Optimize data types for measurement columns.

        Converts numeric columns to appropriate types:
        - For columns with median > 10 and all values in 0-65535 range, converts to uint16
        - For other numeric columns, converts to float32

        Raises:
            StateError: If the dataframe is not loaded.
        """
        assert isinstance(self.df, pd.DataFrame)

        for col in self.df.columns:
            # Skip if column is not numeric
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            # Skip if column is already optimized
            if self.df[col].dtype in ["uint16", "float32"]:
                continue

            # Check if values are in uint16 range and median > 10
            if (
                (self.df[col] >= 0).all()
                and (self.df[col] <= 65535).all()
                and self.df[col].median() > 10
            ):
                self.df[col] = self.df[col].astype("uint16")
            else:
                # Convert to float32 for other numeric columns
                self.df[col] = self.df[col].astype("float32")

    def _set_classifier(self) -> None:
        """Set the classifier field in the repeats table based on the first column name.

        Raises:
            StateError: If the repeat_id or database connection is not available.
        """
        assert isinstance(self.df, pd.DataFrame)

        # Check if we have a repeat_id stored
        if not self.repeat_id:
            raise StateError(
                "Cannot set classifier: no repeat_id stored in state",
                context={"state": self.__dict__},
            )

        # Check if we have a database connection
        if not self.db_conn:
            raise StateError(
                "Cannot set classifier: no database connection available",
                context={"state": self.__dict__},
            )

        # Get the first column name
        first_col = self.df.columns[0]
        if first_col != "label":
            # Update the classifier field in the repeats table using the stored connection
            try:
                self.db_conn.execute(
                    """
                    UPDATE repeats
                    SET classifier = ?
                    WHERE repeat_id = ?
                    """,
                    [first_col, self.repeat_id],
                )

                # Inform the user about the classifier
                self.console.print(
                    f"Found classifier column: [green]{first_col}[/green]"
                )

            except duckdb.Error as err:
                raise DBError(
                    "Failed to update classifier in repeats table",
                    context={
                        "repeat_id": self.repeat_id,
                        "classifier": first_col,
                        "error": str(err),
                    },
                ) from err

            self.df.rename(columns={first_col: "classifier"}, inplace=True)
        else:
            self.console.print(
                "No classifier column found in the file", style="yellow"
            )


# Convenience function for creating state instances
def create_cellview_state(
    args: Optional[argparse.Namespace] = None,
) -> CellViewStateCore:
    """Create a new CellViewStateCore instance (dependency-injectable version).

    This is the preferred way to create state instances for dependency injection.

    Args:
        args: Command line arguments containing csv or plate_id

    Returns:
        Initialized CellViewStateCore instance

    Raises:
        DataError: If there are issues reading CSV data or processing
    """
    return CellViewStateCore.create_from_args(args)


# Backward compatibility wrapper for CellViewState
class CellViewState(CellViewStateCore):
    """Backward compatibility wrapper that maintains the same interface as the old singleton."""

    def __init__(self) -> None:
        """Initialize with default values for backward compatibility."""
        super().__init__(ui=CellViewUI())

    @classmethod
    def get_instance(
        cls, args: Optional[argparse.Namespace] = None
    ) -> "CellViewState":
        """Backward compatibility method that creates new instances instead of singleton.

        Note: This no longer returns a singleton - each call creates a new instance.
        For proper dependency injection, use create_cellview_state() instead.
        """
        if args:
            # Use the new create_from_args method
            core_state = CellViewStateCore.create_from_args(args)
            # Wrap in our compatibility class
            state = cls()
            # Copy all attributes from the core state
            for field in core_state.__dataclass_fields__:
                setattr(state, field, getattr(core_state, field))
            return state
        else:
            return cls()

    def reset(self) -> None:
        """Reset state to default values for backward compatibility."""
        self.csv_path = None
        self.df = None
        self.plate_id = None
        self.project_name = None
        self.experiment_name = None
        self.project_id = None
        self.experiment_id = None
        self.repeat_id = None
        self.condition_id_map = None
        self.lab_member = None
        self.date = None
        self.channel_0 = None
        self.channel_1 = None
        self.channel_2 = None
        self.channel_3 = None
        self.db_conn = None
        self._omero_import_mode = False
