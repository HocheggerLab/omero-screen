"""Tests for enhanced ProjectManager OMERO import functionality.

This module tests the ProjectManager enhancements for OMERO imports including:
- Always-interactive OMERO import confirmation flow
- Integration with state confirmation methods
- Project selection and creation during OMERO imports
- Error handling for existing plates
"""

import argparse
from unittest.mock import Mock, patch
from typing import Optional

import pytest
from cellview.importers.projects import ProjectManager
from cellview.utils.error_classes import DBError, StateError
from cellview.utils.state import CellViewStateCore, create_cellview_state
from rich.console import Console


@pytest.fixture
def mock_state_omero():
    """Create a mock state instance configured for OMERO import mode."""
    state = create_cellview_state()
    state.plate_id = 12345
    state._omero_import_mode = True
    state.project_name = "Detected Project"
    state.experiment_name = "Detected Experiment"
    state.console = Console(quiet=True)
    return state


@pytest.fixture
def mock_state_csv():
    """Create a mock state instance configured for CSV import mode."""
    state = create_cellview_state()
    state.plate_id = 67890
    state._omero_import_mode = False
    state.project_name = None
    state.experiment_name = None
    state.console = Console(quiet=True)
    return state


class TestProjectManagerOMEROFlow:
    """Test ProjectManager OMERO import flow."""

    def test_select_or_create_project_omero_with_confirmation(self, db, mock_state_omero):
        """Test OMERO import flow with confirmation dialog."""
        manager = ProjectManager(db.conn, mock_state_omero)

        with patch.object(mock_state_omero, 'confirm_project_experiment_names') as mock_confirm:
            mock_confirm.return_value = ("Confirmed Project", "Confirmed Experiment")

            manager.select_or_create_project()

            # Verify confirmation was called
            mock_confirm.assert_called_once()

            # Verify state was updated with confirmed values
            assert mock_state_omero.project_name == "Confirmed Project"
            assert mock_state_omero.experiment_name == "Confirmed Experiment"
            assert mock_state_omero.project_id is not None

    def test_select_or_create_project_omero_user_override(self, db, mock_state_omero):
        """Test OMERO import flow when user overrides detected metadata."""
        manager = ProjectManager(db.conn, mock_state_omero)

        # Simulate user overriding detected metadata
        with patch.object(mock_state_omero, 'confirm_project_experiment_names') as mock_confirm:
            mock_confirm.return_value = ("Override Project", "Override Experiment")

            manager.select_or_create_project()

            # Verify confirmation was called
            mock_confirm.assert_called_once()

            # Verify state was updated with override values
            assert mock_state_omero.project_name == "Override Project"
            assert mock_state_omero.experiment_name == "Override Experiment"

    def test_select_or_create_project_omero_standalone_plate(self, db):
        """Test OMERO import flow for standalone plate (no metadata detected)."""
        state = create_cellview_state()
        state.plate_id = 12345
        state._omero_import_mode = True
        state.project_name = None
        state.experiment_name = None
        state.console = Console(quiet=True)

        manager = ProjectManager(db.conn, state)

        with patch.object(state, 'confirm_project_experiment_names') as mock_confirm:
            mock_confirm.return_value = ("Interactive Project", "Interactive Experiment")

            manager.select_or_create_project()

            # Verify confirmation was called even for standalone plates
            mock_confirm.assert_called_once()

            # Verify state was updated
            assert state.project_name == "Interactive Project"
            assert state.experiment_name == "Interactive Experiment"

    def test_select_or_create_project_csv_mode(self, db, mock_state_csv):
        """Test that CSV mode doesn't trigger OMERO confirmation flow."""
        # Add existing project for selection
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Existing Project", "Test Description"]
        )

        manager = ProjectManager(db.conn, mock_state_csv)

        with patch('rich.prompt.Prompt.ask') as mock_prompt, \
             patch.object(mock_state_csv, 'confirm_project_experiment_names') as mock_confirm:

            mock_prompt.return_value = "1"  # Select existing project by ID

            manager.select_or_create_project()

            # Verify confirmation was NOT called for CSV mode
            mock_confirm.assert_not_called()

            # Verify normal project selection occurred
            assert mock_state_csv.project_id is not None


class TestPlateExistenceCheck:
    """Test plate existence checking functionality."""

    def test_check_plate_exists_raises_error(self, db):
        """Test that _check_plate_exists raises error for existing plate."""
        # Setup existing plate data
        db.conn.execute(
            "INSERT INTO projects (project_name) VALUES (?)",
            ["Test Project"]
        )
        project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [project_id, "Test Experiment"]
        )
        experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        db.conn.execute(
            "INSERT INTO repeats (experiment_id, plate_id, date, channel_0) VALUES (?, ?, CURRENT_DATE, 'DAPI')",
            [experiment_id, 12345]
        )

        state = create_cellview_state()
        state.plate_id = 12345
        manager = ProjectManager(db.conn, state)

        with pytest.raises(DBError, match="Plate already exists"):
            manager._check_plate_exists(12345)

    def test_check_plate_exists_no_error_for_new_plate(self, db):
        """Test that _check_plate_exists doesn't raise error for new plate."""
        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        # Should not raise any error
        manager._check_plate_exists(99999)


class TestProjectIdParsing:
    """Test project ID parsing and creation."""

    def test_parse_projectid_from_name_existing(self, db):
        """Test parsing project ID from existing project name."""
        # Create existing project
        db.conn.execute(
            "INSERT INTO projects (project_name) VALUES (?)",
            ["Existing Project"]
        )
        expected_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        state = create_cellview_state()
        state.plate_id = 12345
        manager = ProjectManager(db.conn, state)

        result_id = manager._parse_projectid_from_name("Existing Project")

        assert result_id == expected_id

    def test_parse_projectid_from_name_create_new(self, db):
        """Test parsing project ID creates new project if not found."""
        state = create_cellview_state()
        state.plate_id = 12345
        manager = ProjectManager(db.conn, state)

        result_id = manager._parse_projectid_from_name("New Project")

        # Verify project was created
        project_exists = db.conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = ?",
            ["New Project"]
        ).fetchone()[0]
        assert project_exists == 1
        assert isinstance(result_id, int)


class TestProjectSelection:
    """Test interactive project selection functionality."""

    @patch('rich.prompt.Prompt.ask')
    def test_handle_project_selection_by_id(self, mock_prompt, db):
        """Test project selection by ID."""
        # Create test project
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Test Project", "Test Description"]
        )
        project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        projects = [(project_id, "Test Project", "Test Description")]
        mock_prompt.return_value = str(project_id)

        state = create_cellview_state()
        state.console = Console(quiet=True)
        manager = ProjectManager(db.conn, state)

        result = manager._handle_project_selection(projects)

        assert result == project_id

    @patch('rich.prompt.Prompt.ask')
    def test_handle_project_selection_by_name(self, mock_prompt, db):
        """Test project selection by name."""
        # Create test project
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Test Project", "Test Description"]
        )
        project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        projects = [(project_id, "Test Project", "Test Description")]
        mock_prompt.return_value = "Test Project"

        state = create_cellview_state()
        state.console = Console(quiet=True)
        manager = ProjectManager(db.conn, state)

        result = manager._handle_project_selection(projects)

        assert result == project_id

    @patch('rich.prompt.Prompt.ask')
    def test_handle_project_selection_create_new(self, mock_prompt, db):
        """Test project selection creating new project."""
        projects = []  # No existing projects
        mock_prompt.return_value = "New Project"

        state = create_cellview_state()
        state.console = Console(quiet=True)
        manager = ProjectManager(db.conn, state)

        result = manager._handle_project_selection(projects)

        # Verify new project was created
        project_exists = db.conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = ?",
            ["New Project"]
        ).fetchone()[0]
        assert project_exists == 1
        assert isinstance(result, int)

    @patch('rich.prompt.Prompt.ask')
    def test_handle_project_selection_invalid_id(self, mock_prompt, db):
        """Test project selection with invalid ID."""
        # Create test project
        db.conn.execute(
            "INSERT INTO projects (project_name) VALUES (?)",
            ["Test Project"]
        )
        project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        projects = [(project_id, "Test Project", "")]
        mock_prompt.return_value = "999"  # Invalid ID

        state = create_cellview_state()
        state.console = Console(quiet=True)
        manager = ProjectManager(db.conn, state)

        with pytest.raises(StateError, match="Invalid project ID"):
            manager._handle_project_selection(projects)


class TestProjectCreation:
    """Test project creation functionality."""

    def test_create_new_project_success(self, db):
        """Test successful creation of new project."""
        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        result_id = manager._create_new_project("New Test Project")

        # Verify project was created
        result = db.conn.execute(
            "SELECT project_name FROM projects WHERE project_id = ?",
            [result_id]
        ).fetchone()

        assert result is not None
        assert result[0] == "New Test Project"
        assert isinstance(result_id, int)

    def test_create_new_project_duplicate_name(self, db):
        """Test creation of project with duplicate name raises error."""
        # Create existing project
        db.conn.execute(
            "INSERT INTO projects (project_name) VALUES (?)",
            ["Existing Project"]
        )

        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        with pytest.raises(DBError, match="Project name already exists"):
            manager._create_new_project("Existing Project")


class TestProjectTableDisplay:
    """Test project table display functionality."""

    def test_display_projects_table(self, db):
        """Test displaying projects table."""
        # Create test projects
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Project 1", "Description 1"]
        )
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Project 2", "Description 2"]
        )

        projects = [(1, "Project 1", "Description 1"), (2, "Project 2", "Description 2")]

        state = create_cellview_state()
        state.console = Console(quiet=True)
        manager = ProjectManager(db.conn, state)

        # Should not raise any errors
        manager._display_projects_table(projects)

    def test_fetch_existing_projects_success(self, db):
        """Test successful fetching of existing projects."""
        # Create test projects
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Project 1", "Description 1"]
        )
        db.conn.execute(
            "INSERT INTO projects (project_name, description) VALUES (?, ?)",
            ["Project 2", "Description 2"]
        )

        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        projects = manager._fetch_existing_projects()

        assert len(projects) == 2
        assert projects[0][1] == "Project 1"
        assert projects[1][1] == "Project 2"
        assert all(isinstance(p[0], int) for p in projects)


class TestOMEROImportIntegration:
    """Test full OMERO import integration with ProjectManager."""

    def test_omero_import_detection(self, db):
        """Test that OMERO import mode is properly detected."""
        state = create_cellview_state()
        state.plate_id = 12345
        state._omero_import_mode = True
        state.project_name = "Test Project"
        state.experiment_name = "Test Experiment"

        manager = ProjectManager(db.conn, state)

        # Verify OMERO import mode is detected
        assert hasattr(state, '_omero_import_mode')
        assert state._omero_import_mode is True

    @patch('rich.prompt.Prompt.ask')
    def test_fallback_to_interactive_selection(self, mock_prompt, db):
        """Test fallback to interactive selection when no projects exist."""
        mock_prompt.return_value = "New Project"

        state = create_cellview_state()
        state.plate_id = 12345
        state.project_name = None  # No project name set
        manager = ProjectManager(db.conn, state)

        manager.select_or_create_project()

        # Verify interactive creation occurred
        project_exists = db.conn.execute(
            "SELECT COUNT(*) FROM projects WHERE project_name = ?",
            ["New Project"]
        ).fetchone()[0]
        assert project_exists == 1
        assert state.project_id is not None

    def test_backward_compatibility_singleton_state(self, db):
        """Test that ProjectManager works with singleton state for backward compatibility."""
        from cellview.utils.state import CellViewState

        # Create singleton state (old style)
        singleton_state = CellViewState.get_instance()
        singleton_state.plate_id = 12345
        singleton_state.project_name = "Singleton Project"

        # ProjectManager should work with both state types
        manager = ProjectManager(db.conn, None)  # Pass None to trigger singleton fallback

        # Should not raise any errors
        assert manager.state is not None
        assert hasattr(manager.state, 'plate_id')


class TestErrorHandling:
    """Test error handling in ProjectManager."""

    def test_select_or_create_project_no_plate_id(self, db):
        """Test that select_or_create_project requires plate_id."""
        state = create_cellview_state()
        state.plate_id = None  # No plate ID set

        manager = ProjectManager(db.conn, state)

        with pytest.raises(AssertionError):
            manager.select_or_create_project()

    def test_database_error_handling(self, db):
        """Test handling of database errors."""
        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        # Force a database error by closing the connection
        db.conn.close()

        with pytest.raises(Exception):  # Could be DBError or duckdb.Error
            manager._create_new_project("Test Project")  # Should fail with closed connection

    def test_create_table_helper(self, db):
        """Test the create table helper method."""
        state = create_cellview_state()
        manager = ProjectManager(db.conn, state)

        table = manager._create_table(
            "Test Table",
            [("Column 1", "left"), ("Column 2", "right")]
        )

        assert table.title == "Test Table"
        assert len(table.columns) == 2
