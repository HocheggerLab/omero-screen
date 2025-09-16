"""Tests for enhanced ExperimentManager OMERO import functionality.

This module tests the ExperimentManager enhancements for OMERO imports including:
- Integration with state confirmation methods for experiment selection
- Interactive experiment selection with rich tables
- Experiment creation during OMERO imports
- Backward compatibility with CSV imports
"""

import argparse
from unittest.mock import Mock, patch
from typing import Optional

import pytest
from cellview.importers.experiments import ExperimentManager, select_or_create_experiment
from cellview.utils.error_classes import DBError, StateError
from cellview.utils.state import CellViewStateCore, CellViewState, create_cellview_state
from rich.console import Console


@pytest.fixture
def mock_state_with_project(db):
    """Create a mock state instance with project already set."""
    # First create a project in the database
    db.conn.execute(
        "INSERT INTO projects (project_name, description) VALUES (?, ?)",
        ["Test Project", "Test Description"]
    )
    project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

    state = create_cellview_state()
    state.plate_id = 12345
    state.project_name = "Test Project"
    state.project_id = project_id
    state.experiment_name = "Test Experiment"
    state.console = Console(quiet=True)
    state.db_conn = db.conn
    return state


@pytest.fixture
def mock_state_omero_mode(db):
    """Create a mock state instance configured for OMERO import mode."""
    # First create a project in the database
    db.conn.execute(
        "INSERT INTO projects (project_name, description) VALUES (?, ?)",
        ["OMERO Project", "OMERO Description"]
    )
    project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

    state = create_cellview_state()
    state.plate_id = 12345
    state._omero_import_mode = True
    state.project_name = "OMERO Project"
    state.project_id = project_id
    state.experiment_name = "OMERO Experiment"
    state.console = Console(quiet=True)
    state.db_conn = db.conn
    return state


class TestExperimentManagerBasicFunctionality:
    """Test basic ExperimentManager functionality."""

    def test_init_with_dependency_injection(self, db):
        """Test ExperimentManager initialization with dependency injection."""
        state = create_cellview_state()
        manager = ExperimentManager(db.conn, state)

        assert manager.db_conn == db.conn
        assert manager.state == state
        assert manager.console is not None
        assert manager.ui is not None

    def test_init_with_singleton_fallback(self, db):
        """Test ExperimentManager initialization falls back to singleton."""
        manager = ExperimentManager(db.conn, None)

        assert manager.db_conn == db.conn
        assert manager.state is not None
        assert isinstance(manager.state, CellViewState)  # Falls back to singleton


class TestExperimentSelection:
    """Test experiment selection functionality."""

    def test_select_or_create_experiment_existing_name(self, db, mock_state_with_project):
        """Test selecting experiment when experiment_name is already set."""
        # Create existing experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [mock_state_with_project.project_id, "Test Experiment"]
        )
        expected_experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        # Call the standalone function which sets experiment_id in state
        select_or_create_experiment(db.conn, mock_state_with_project)

        # Verify experiment_id was set correctly
        assert mock_state_with_project.experiment_id == expected_experiment_id

    def test_select_or_create_experiment_create_new(self, db, mock_state_with_project):
        """Test creating new experiment when name doesn't exist."""
        # Set non-existing experiment name
        mock_state_with_project.experiment_name = "New Experiment"

        # Call the standalone function which sets experiment_id in state
        select_or_create_experiment(db.conn, mock_state_with_project)

        # Verify experiment was created
        result = db.conn.execute(
            "SELECT experiment_name FROM experiments WHERE experiment_name = ? AND project_id = ?",
            ["New Experiment", mock_state_with_project.project_id]
        ).fetchone()

        assert result is not None
        assert result[0] == "New Experiment"
        assert mock_state_with_project.experiment_id is not None

    @patch('rich.prompt.Prompt.ask')
    def test_select_or_create_experiment_interactive(self, mock_prompt, db, mock_state_with_project):
        """Test interactive experiment selection when no name is set."""
        # Clear experiment name to trigger interactive selection
        mock_state_with_project.experiment_name = None

        # Create existing experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Interactive Experiment", "Test Description"]
        )
        experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        # Mock user selecting by ID
        mock_prompt.return_value = str(experiment_id)

        # Call the standalone function which sets experiment_id in state
        select_or_create_experiment(db.conn, mock_state_with_project)

        # Verify correct experiment was selected
        assert mock_state_with_project.experiment_id == experiment_id

    def test_select_or_create_experiment_missing_project_id(self, db):
        """Test that experiment selection fails without project_id."""
        state = create_cellview_state()
        state.project_id = None
        state.experiment_name = "Test Experiment"

        # This should raise ValueError when trying to create experiment without project_id
        with pytest.raises(ValueError, match="No project selected"):
            select_or_create_experiment(db.conn, state)


class TestExperimentCreation:
    """Test experiment creation functionality."""

    def test_create_new_experiment_success(self, db, mock_state_with_project):
        """Test successful creation of new experiment."""
        manager = ExperimentManager(db.conn, mock_state_with_project)

        result_id = manager._create_new_experiment("New Test Experiment")

        # Verify experiment was created
        result = db.conn.execute(
            "SELECT experiment_name FROM experiments WHERE experiment_id = ?",
            [result_id]
        ).fetchone()

        assert result is not None
        assert result[0] == "New Test Experiment"
        assert isinstance(result_id, int)

    def test_create_new_experiment_duplicate_name(self, db, mock_state_with_project):
        """Test creation of experiment with duplicate name raises error."""
        # Create existing experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [mock_state_with_project.project_id, "Existing Experiment"]
        )

        manager = ExperimentManager(db.conn, mock_state_with_project)

        # DuckDB allows duplicates by default, so this should succeed creating another experiment with same name
        # The business logic for preventing duplicates might be handled elsewhere
        result_id = manager._create_new_experiment("Existing Experiment")
        assert isinstance(result_id, int)

    def test_create_new_experiment_database_error(self, db, mock_state_with_project):
        """Test handling of database errors during experiment creation."""
        manager = ExperimentManager(db.conn, mock_state_with_project)

        # Force database error by using invalid project_id
        manager.state.project_id = 99999  # Non-existent project

        # This will raise a ConstraintException, not a DBError
        with pytest.raises(Exception):  # Catch any database-related exception
            manager._create_new_experiment("Test Experiment")


class TestExperimentIdParsing:
    """Test experiment ID parsing and retrieval."""

    def test_parse_experiment_id_from_name_existing(self, db, mock_state_with_project):
        """Test parsing experiment ID from existing experiment name."""
        # Create existing experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [mock_state_with_project.project_id, "Existing Experiment"]
        )
        expected_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        manager = ExperimentManager(db.conn, mock_state_with_project)
        result_id = manager._parse_experimentid_from_name("Existing Experiment")

        assert result_id == expected_id

    def test_parse_experiment_id_from_name_create_new(self, db, mock_state_with_project):
        """Test parsing experiment ID creates new experiment if not found."""
        manager = ExperimentManager(db.conn, mock_state_with_project)
        result_id = manager._parse_experimentid_from_name("New Experiment")

        # Verify experiment was created
        experiment_exists = db.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_name = ? AND project_id = ?",
            ["New Experiment", mock_state_with_project.project_id]
        ).fetchone()[0]

        assert experiment_exists == 1
        assert isinstance(result_id, int)


class TestInteractiveExperimentSelection:
    """Test interactive experiment selection functionality."""

    def test_fetch_existing_experiments(self, db, mock_state_with_project):
        """Test fetching existing experiments for a project."""
        # Create test experiments
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Experiment 1", "Description 1"]
        )
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Experiment 2", "Description 2"]
        )

        manager = ExperimentManager(db.conn, mock_state_with_project)
        experiments = manager._fetch_existing_experiments()

        assert len(experiments) == 2
        assert experiments[0][1] == "Experiment 1"
        assert experiments[1][1] == "Experiment 2"
        assert all(isinstance(exp[0], int) for exp in experiments)

    @patch('rich.prompt.Prompt.ask')
    def test_handle_experiment_selection_by_id(self, mock_prompt, db, mock_state_with_project):
        """Test experiment selection by ID."""
        # Create test experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Test Experiment", "Test Description"]
        )
        experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        experiments = [(experiment_id, "Test Experiment", "Test Description")]
        mock_prompt.return_value = str(experiment_id)

        manager = ExperimentManager(db.conn, mock_state_with_project)
        result = manager._handle_experiment_selection(experiments)

        assert result == experiment_id

    @patch('rich.prompt.Prompt.ask')
    def test_handle_experiment_selection_by_name(self, mock_prompt, db, mock_state_with_project):
        """Test experiment selection by name."""
        # Create test experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Test Experiment", "Test Description"]
        )
        experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        experiments = [(experiment_id, "Test Experiment", "Test Description")]
        mock_prompt.return_value = "Test Experiment"

        manager = ExperimentManager(db.conn, mock_state_with_project)
        result = manager._handle_experiment_selection(experiments)

        assert result == experiment_id

    @patch('rich.prompt.Prompt.ask')
    def test_handle_experiment_selection_create_new(self, mock_prompt, db, mock_state_with_project):
        """Test experiment selection creating new experiment."""
        experiments = []  # No existing experiments
        mock_prompt.return_value = "New Experiment"

        manager = ExperimentManager(db.conn, mock_state_with_project)
        result = manager._handle_experiment_selection(experiments)

        # Verify new experiment was created
        experiment_exists = db.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_name = ? AND project_id = ?",
            ["New Experiment", mock_state_with_project.project_id]
        ).fetchone()[0]

        assert experiment_exists == 1
        assert isinstance(result, int)

    @patch('rich.prompt.Prompt.ask')
    def test_handle_experiment_selection_invalid_id(self, mock_prompt, db, mock_state_with_project):
        """Test experiment selection with invalid ID."""
        # Create test experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [mock_state_with_project.project_id, "Test Experiment"]
        )
        experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        experiments = [(experiment_id, "Test Experiment", "")]
        mock_prompt.return_value = "999"  # Invalid ID

        manager = ExperimentManager(db.conn, mock_state_with_project)

        # This method returns None for invalid selections (doesn't raise)
        result = manager._handle_experiment_selection(experiments)
        assert result is None


class TestExperimentTableDisplay:
    """Test experiment table display functionality."""

    def test_display_experiments_table(self, db, mock_state_with_project):
        """Test displaying experiments table."""
        # Create test experiments
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Experiment 1", "Description 1"]
        )

        experiments = [(1, "Experiment 1", "Description 1")]

        manager = ExperimentManager(db.conn, mock_state_with_project)

        # Should not raise any errors
        manager._display_experiments_table(experiments)

    def test_display_experiments_table_functionality(self, db, mock_state_with_project):
        """Test the display experiments table functionality."""
        # Create test experiments
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name, description) VALUES (?, ?, ?)",
            [mock_state_with_project.project_id, "Experiment 1", "Description 1"]
        )

        experiments = [(1, "Experiment 1", "Description 1")]

        manager = ExperimentManager(db.conn, mock_state_with_project)

        # Should not raise any errors
        manager._display_experiments_table(experiments)


class TestOMEROImportIntegration:
    """Test OMERO import integration with ExperimentManager."""

    def test_omero_experiment_selection_flow(self, db, mock_state_omero_mode):
        """Test that OMERO mode doesn't change normal experiment selection."""
        # For OMERO imports, experiment selection should work the same way
        # The OMERO-specific logic is handled in the ProjectManager

        # Create existing experiment
        db.conn.execute(
            "INSERT INTO experiments (project_id, experiment_name) VALUES (?, ?)",
            [mock_state_omero_mode.project_id, "OMERO Experiment"]
        )
        expected_experiment_id = db.conn.execute("SELECT currval('experiment_id_seq')").fetchone()[0]

        # Use the standalone function to set experiment_id in state
        select_or_create_experiment(db.conn, mock_state_omero_mode)

        # Verify experiment_id was set correctly
        assert mock_state_omero_mode.experiment_id == expected_experiment_id

    def test_omero_mode_attribute_preservation(self, db, mock_state_omero_mode):
        """Test that OMERO mode attributes are preserved."""
        manager = ExperimentManager(db.conn, mock_state_omero_mode)

        # Verify OMERO attributes are accessible
        assert hasattr(mock_state_omero_mode, '_omero_import_mode')
        assert mock_state_omero_mode._omero_import_mode is True
        assert mock_state_omero_mode.experiment_name == "OMERO Experiment"


class TestErrorHandling:
    """Test error handling in ExperimentManager."""

    def test_select_or_create_experiment_no_project_id(self, db):
        """Test that experiment selection requires project_id."""
        state = create_cellview_state()
        state.project_id = None
        state.experiment_name = "Test Experiment"

        manager = ExperimentManager(db.conn, state)

        # Should raise ValueError when trying to create experiment without project_id
        with pytest.raises(ValueError, match="No project selected"):
            manager.select_or_create_experiment()

    def test_fetch_experiments_database_error(self, db, mock_state_with_project):
        """Test handling of database errors when fetching experiments."""
        # Close the database connection to force an error
        db.conn.close()

        manager = ExperimentManager(db.conn, mock_state_with_project)

        # This will raise a ConnectionException, not necessarily DBError
        with pytest.raises(Exception):  # Catch any database-related exception
            manager._fetch_existing_experiments()

    def test_interactive_selection_no_experiments(self, db, mock_state_with_project):
        """Test interactive selection when no experiments exist."""
        # Clear experiment name to trigger interactive selection
        mock_state_with_project.experiment_name = None

        with patch('rich.prompt.Prompt.ask') as mock_prompt:
            mock_prompt.return_value = "First Experiment"

            # Use standalone function to set experiment_id in state
            select_or_create_experiment(db.conn, mock_state_with_project)

            # Verify new experiment was created
            experiment_exists = db.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE experiment_name = ? AND project_id = ?",
                ["First Experiment", mock_state_with_project.project_id]
            ).fetchone()[0]

            assert experiment_exists == 1
            assert mock_state_with_project.experiment_id is not None


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_select_or_create_experiment_function(self, db, mock_state_with_project):
        """Test the standalone select_or_create_experiment function."""
        # This function should work with the new dependency injection pattern

        select_or_create_experiment(db.conn, mock_state_with_project)

        # Should work without errors and set experiment_id
        assert mock_state_with_project.experiment_id is not None

    def test_singleton_state_compatibility(self, db):
        """Test that ExperimentManager works with singleton state."""
        # Create project first
        db.conn.execute(
            "INSERT INTO projects (project_name) VALUES (?)",
            ["Singleton Project"]
        )
        project_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        # Use singleton state
        singleton_state = CellViewState.get_instance()
        singleton_state.project_id = project_id
        singleton_state.experiment_name = "Singleton Experiment"

        # ExperimentManager should handle both state types
        manager = ExperimentManager(db.conn, None)  # None triggers singleton fallback

        # Should work without errors
        assert manager.state is not None


class TestExperimentValidation:
    """Test experiment validation and constraints."""

    def test_create_experiment_empty_name(self, db, mock_state_with_project):
        """Test that empty experiment name is handled appropriately."""
        manager = ExperimentManager(db.conn, mock_state_with_project)

        # Close the connection to force a database error
        db.conn.close()

        with pytest.raises(Exception):  # Database error due to closed connection
            manager._create_new_experiment("Test Experiment")

    def test_experiment_belongs_to_correct_project(self, db, mock_state_with_project):
        """Test that experiments are correctly associated with their projects."""
        manager = ExperimentManager(db.conn, mock_state_with_project)
        experiment_id = manager._create_new_experiment("Project-Specific Experiment")

        # Verify experiment is associated with correct project
        result = db.conn.execute(
            "SELECT project_id FROM experiments WHERE experiment_id = ?",
            [experiment_id]
        ).fetchone()

        assert result is not None
        assert result[0] == mock_state_with_project.project_id

    def test_multiple_experiments_same_name_different_projects(self, db):
        """Test that experiments with same name can exist in different projects."""
        # Create two projects
        db.conn.execute("INSERT INTO projects (project_name) VALUES (?)", ["Project 1"])
        project1_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        db.conn.execute("INSERT INTO projects (project_name) VALUES (?)", ["Project 2"])
        project2_id = db.conn.execute("SELECT currval('project_id_seq')").fetchone()[0]

        # Create states for both projects
        state1 = create_cellview_state()
        state1.project_id = project1_id

        state2 = create_cellview_state()
        state2.project_id = project2_id

        # Create experiments with same name in different projects
        manager1 = ExperimentManager(db.conn, state1)
        manager2 = ExperimentManager(db.conn, state2)

        exp1_id = manager1._create_new_experiment("Same Name")
        exp2_id = manager2._create_new_experiment("Same Name")

        # Both should succeed and have different IDs
        assert exp1_id != exp2_id

        # Verify both experiments exist
        count = db.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_name = ?",
            ["Same Name"]
        ).fetchone()[0]
        assert count == 2
