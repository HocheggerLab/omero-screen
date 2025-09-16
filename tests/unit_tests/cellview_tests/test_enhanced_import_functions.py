"""Tests for enhanced import functions orchestration.

This module tests the import_data function and its integration with
enhanced OMERO import functionality including:
- State conversion between singleton and dependency-injectable versions
- OMERO import flow orchestration
- Error handling and cleanup
- Integration with all importer components
"""

import argparse
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

import pandas as pd
import pytest
import duckdb
from cellview.db.db import CellViewDB
from cellview.importers.import_functions import import_data
from cellview.utils.error_classes import CellViewError, DataError, StateError
from cellview.utils.state import CellViewState, CellViewStateCore, create_cellview_state


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing."""
    return pd.DataFrame({
        "well": ["A01", "A02", "B01", "B02"],
        "plate_id": [12345, 12345, 12345, 12345],
        "image_id": [1, 2, 3, 4],
        "timepoint": [1, 1, 1, 1],
        "intensity_max_DAPI_nucleus": [100, 200, 300, 400],
        "intensity_mean_GFP_cell": [50, 75, 125, 175],
    })


@pytest.fixture
def mock_state_core(mock_csv_data):
    """Create a mock CellViewStateCore instance for dependency injection."""
    state = create_cellview_state()
    state.plate_id = 12345
    state.df = mock_csv_data
    state.project_name = "Test Project"
    state.experiment_name = "Test Experiment"
    state.project_id = 1
    state.experiment_id = 1
    state.repeat_id = 1
    state.date = "2024-03-26"
    return state


@pytest.fixture
def mock_state_singleton():
    """Create a mock singleton CellViewState instance for backward compatibility."""
    state = CellViewState.get_instance()
    state.plate_id = 12345
    state.project_name = "Test Project"
    state.experiment_name = "Test Experiment"
    state.project_id = 1
    state.experiment_id = 1
    state.repeat_id = 1
    state.date = "2024-03-26"
    return state


class TestImportDataOrchestration:
    """Test the main import_data function orchestration."""

    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.display_plate_summary')
    def test_import_data_success_with_core_state(
        self,
        mock_display,
        mock_project,
        mock_experiment,
        mock_repeat,
        mock_conditions,
        mock_measurements,
        db,
        mock_state_core
    ):
        """Test successful import_data with CellViewStateCore (dependency injection)."""
        # Setup mocks
        mock_project.return_value = None
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        result = import_data(db, mock_state_core, db.conn)

        # Verify success
        assert result == 0

        # Verify all importer functions were called
        mock_project.assert_called_once()
        mock_experiment.assert_called_once()
        mock_repeat.assert_called_once()
        mock_conditions.assert_called_once()
        mock_measurements.assert_called_once()
        mock_display.assert_called_once_with(12345, db.conn)

        # Verify state connection was set
        assert mock_state_core.db_conn is not None

    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.display_plate_summary')
    def test_import_data_success_with_singleton_state(
        self,
        mock_display,
        mock_project,
        mock_experiment,
        mock_repeat,
        mock_conditions,
        mock_measurements,
        db,
        mock_state_singleton
    ):
        """Test successful import_data with singleton CellViewState (backward compatibility)."""
        # Setup mocks
        mock_project.return_value = None
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        result = import_data(db, mock_state_singleton, db.conn)

        # Verify success
        assert result == 0

        # Verify all importer functions were called
        mock_project.assert_called_once()
        mock_experiment.assert_called_once()
        mock_repeat.assert_called_once()
        mock_conditions.assert_called_once()
        mock_measurements.assert_called_once()
        mock_display.assert_called_once_with(12345, db.conn)

    def test_import_data_creates_connection_if_none(self, db, mock_state_core):
        """Test that import_data creates connection if none provided."""
        with patch('cellview.importers.import_functions.select_or_create_project') as mock_project, \
             patch('cellview.importers.import_functions.select_or_create_experiment') as mock_experiment, \
             patch('cellview.importers.import_functions.create_new_repeat') as mock_repeat, \
             patch('cellview.importers.import_functions.import_conditions') as mock_conditions, \
             patch('cellview.importers.import_functions.import_measurements') as mock_measurements, \
             patch('cellview.importers.import_functions.display_plate_summary') as mock_display:

            # Call without providing connection
            result = import_data(db, mock_state_core, conn=None)

            # Verify that connection was established
            assert mock_state_core.db_conn is not None
            assert result == 0

    @patch('cellview.importers.import_functions.clean_up_db')
    def test_import_data_handles_cellview_error(self, mock_cleanup, db, mock_state_core):
        """Test that import_data handles CellViewError gracefully."""
        error = DataError("Test error")

        with patch('cellview.importers.import_functions.select_or_create_project') as mock_project:
            mock_project.side_effect = error

            result = import_data(db, mock_state_core, db.conn)

            # Verify error handling
            assert result == 1  # Error return code
            mock_cleanup.assert_called_once_with(db, db.conn)

    @patch('cellview.importers.import_functions.clean_up_db')
    def test_import_data_reraises_non_cellview_error(self, mock_cleanup, db, mock_state_core):
        """Test that import_data re-raises non-CellViewError exceptions."""
        error = ValueError("Generic error")

        with patch('cellview.importers.import_functions.select_or_create_project') as mock_project:
            mock_project.side_effect = error

            with pytest.raises(ValueError, match="Generic error"):
                import_data(db, mock_state_core, db.conn)

            # Verify cleanup was called
            mock_cleanup.assert_called_once_with(db, db.conn)


class TestStateConversion:
    """Test state conversion between singleton and dependency-injectable versions."""

    @patch('cellview.importers.import_functions.display_plate_summary')
    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_singleton_to_core_conversion(self, mock_project, mock_experiment, mock_repeat, mock_conditions, mock_measurements, mock_display, db, mock_state_singleton):
        """Test conversion from singleton state to CellViewStateCore."""
        # Add some additional attributes to test conversion
        mock_state_singleton.csv_path = "/path/to/data.csv"
        mock_state_singleton.lab_member = "Test User"
        mock_state_singleton.condition_id_map = {"control": 1}

        def capture_state(conn, state):
            # Capture the state that was passed to the project function
            capture_state.captured_state = state

        mock_project.side_effect = capture_state

        # Mock all other import functions to succeed
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        # Call import_data with singleton state
        import_data(db, mock_state_singleton, db.conn)

        # Verify that a CellViewStateCore was created and passed to importers
        captured_state = capture_state.captured_state
        assert isinstance(captured_state, CellViewStateCore)

        # Verify attributes were copied
        assert captured_state.plate_id == mock_state_singleton.plate_id
        assert captured_state.project_name == mock_state_singleton.project_name
        assert captured_state.experiment_name == mock_state_singleton.experiment_name
        assert captured_state.csv_path == mock_state_singleton.csv_path
        assert captured_state.lab_member == mock_state_singleton.lab_member

    @patch('cellview.importers.import_functions.display_plate_summary')
    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_core_state_passed_directly(self, mock_project, mock_experiment, mock_repeat, mock_conditions, mock_measurements, mock_display, db, mock_state_core):
        """Test that CellViewStateCore is passed directly without conversion."""
        def capture_state(conn, state):
            capture_state.captured_state = state

        mock_project.side_effect = capture_state

        # Mock all other import functions to succeed
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        # Call import_data with core state
        import_data(db, mock_state_core, db.conn)

        # Verify the same instance was passed through
        captured_state = capture_state.captured_state
        assert captured_state is mock_state_core
        assert isinstance(captured_state, CellViewStateCore)


class TestOMEROImportIntegration:
    """Test OMERO import integration through import_data."""

    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.display_plate_summary')
    def test_omero_import_flow(
        self,
        mock_display,
        mock_project,
        mock_experiment,
        mock_repeat,
        mock_conditions,
        mock_measurements,
        db,
        mock_csv_data
    ):
        """Test full OMERO import flow through import_data."""
        # Create OMERO state
        args = argparse.Namespace()
        args.csv = None
        args.plate_id = 12345

        with patch.object(CellViewStateCore, 'parse_omero_data') as mock_parse:
            mock_parse.return_value = (
                mock_csv_data,
                "OMERO Project",
                "OMERO Experiment",
                "2024-03-26",
                "OMERO Owner"
            )

            state = create_cellview_state(args)

        # Verify OMERO import mode is set
        assert state._omero_import_mode is True
        assert state.project_name == "OMERO Project"
        assert state.experiment_name == "OMERO Experiment"

        # Mock all import functions
        mock_project.return_value = None
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        result = import_data(db, state, db.conn)

        # Verify successful import
        assert result == 0

        # Verify all functions were called in correct order
        mock_project.assert_called_once()
        mock_experiment.assert_called_once()
        mock_repeat.assert_called_once()
        mock_conditions.assert_called_once()
        mock_measurements.assert_called_once()
        mock_display.assert_called_once()

    @patch('cellview.importers.import_functions.display_plate_summary')
    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_omero_state_attributes_preserved(self, mock_project, mock_experiment, mock_repeat, mock_conditions, mock_measurements, mock_display, db):
        """Test that OMERO-specific state attributes are preserved through import."""
        state = create_cellview_state()
        state.plate_id = 12345
        state._omero_import_mode = True
        state.project_name = "OMERO Project"
        state.experiment_name = "OMERO Experiment"
        state.lab_member = "OMERO Owner"

        def capture_omero_attributes(conn, state_arg):
            # Verify OMERO attributes are preserved
            assert hasattr(state_arg, '_omero_import_mode')
            assert state_arg._omero_import_mode is True
            assert state_arg.project_name == "OMERO Project"
            assert state_arg.experiment_name == "OMERO Experiment"
            assert state_arg.lab_member == "OMERO Owner"

        mock_project.side_effect = capture_omero_attributes

        # Mock all other import functions to succeed
        mock_experiment.return_value = None
        mock_repeat.return_value = None
        mock_conditions.return_value = None
        mock_measurements.return_value = None
        mock_display.return_value = None

        result = import_data(db, state, db.conn)

        assert result == 0


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch('cellview.importers.import_functions.clean_up_db')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_project_selection_error(self, mock_project, mock_cleanup, db, mock_state_core):
        """Test error handling when project selection fails."""
        error = StateError("Project selection failed")
        mock_project.side_effect = error

        result = import_data(db, mock_state_core, db.conn)

        assert result == 1
        mock_cleanup.assert_called_once_with(db, db.conn)

    @patch('cellview.importers.import_functions.clean_up_db')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    def test_experiment_selection_error(self, mock_experiment, mock_project, mock_cleanup, db, mock_state_core):
        """Test error handling when experiment selection fails."""
        mock_project.return_value = None  # Success
        error = StateError("Experiment selection failed")
        mock_experiment.side_effect = error

        result = import_data(db, mock_state_core, db.conn)

        assert result == 1
        mock_cleanup.assert_called_once_with(db, db.conn)

    @patch('cellview.importers.import_functions.clean_up_db')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.create_new_repeat')
    def test_repeat_creation_error(self, mock_repeat, mock_experiment, mock_project, mock_cleanup, db, mock_state_core):
        """Test error handling when repeat creation fails."""
        mock_project.return_value = None
        mock_experiment.return_value = None
        error = DataError("Repeat creation failed")
        mock_repeat.side_effect = error

        result = import_data(db, mock_state_core, db.conn)

        assert result == 1
        mock_cleanup.assert_called_once_with(db, db.conn)

    @patch('cellview.importers.import_functions.clean_up_db')
    def test_missing_plate_id_error(self, mock_cleanup, db):
        """Test error handling when plate_id is missing."""
        state = create_cellview_state()
        state.plate_id = None  # Missing plate_id

        with pytest.raises(AssertionError):
            import_data(db, state, db.conn)

    @patch('cellview.importers.import_functions.clean_up_db')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_database_connection_error(self, mock_project, mock_cleanup, db, mock_state_core):
        """Test error handling for database connection issues."""
        # Simulate database connection error
        error = duckdb.Error("Database connection failed")
        mock_project.side_effect = error

        with pytest.raises(duckdb.Error):
            import_data(db, mock_state_core, db.conn)

        mock_cleanup.assert_called_once_with(db, db.conn)


class TestImportSequence:
    """Test the sequence of import operations."""

    @patch('cellview.importers.import_functions.display_plate_summary')
    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    def test_import_sequence_order(
        self,
        mock_project,
        mock_experiment,
        mock_repeat,
        mock_conditions,
        mock_measurements,
        mock_display,
        db,
        mock_state_core
    ):
        """Test that import operations are called in correct sequence."""
        call_order = []

        def track_project(conn, state):
            call_order.append('project')

        def track_experiment(conn, state):
            call_order.append('experiment')

        def track_repeat(conn, state):
            call_order.append('repeat')

        def track_conditions(conn, state):
            call_order.append('conditions')

        def track_measurements(conn, state):
            call_order.append('measurements')

        def track_display(plate_id, conn):
            call_order.append('display')

        mock_project.side_effect = track_project
        mock_experiment.side_effect = track_experiment
        mock_repeat.side_effect = track_repeat
        mock_conditions.side_effect = track_conditions
        mock_measurements.side_effect = track_measurements
        mock_display.side_effect = track_display

        result = import_data(db, mock_state_core, db.conn)

        # Verify correct sequence
        expected_order = ['project', 'experiment', 'repeat', 'conditions', 'measurements', 'display']
        assert call_order == expected_order
        assert result == 0

    @patch('cellview.importers.import_functions.import_measurements')
    @patch('cellview.importers.import_functions.import_conditions')
    @patch('cellview.importers.import_functions.create_new_repeat')
    @patch('cellview.importers.import_functions.select_or_create_experiment')
    @patch('cellview.importers.import_functions.select_or_create_project')
    @patch('cellview.importers.import_functions.display_plate_summary')
    def test_state_connection_assignment(
        self,
        mock_display,
        mock_project,
        mock_experiment,
        mock_repeat,
        mock_conditions,
        mock_measurements,
        db,
        mock_state_core
    ):
        """Test that database connection is properly assigned to state."""
        # Verify connection is initially None or different
        original_conn = mock_state_core.db_conn

        import_data(db, mock_state_core, db.conn)

        # Verify connection was set
        assert mock_state_core.db_conn == db.conn
        assert mock_state_core.db_conn != original_conn


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_singleton_state_interface(self, db):
        """Test that singleton state interface still works."""
        # This test ensures that old code using singleton pattern still works
        singleton_state = CellViewState.get_instance()
        singleton_state.plate_id = 12345

        with patch('cellview.importers.import_functions.select_or_create_project') as mock_project, \
             patch('cellview.importers.import_functions.select_or_create_experiment') as mock_experiment, \
             patch('cellview.importers.import_functions.create_new_repeat') as mock_repeat, \
             patch('cellview.importers.import_functions.import_conditions') as mock_conditions, \
             patch('cellview.importers.import_functions.import_measurements') as mock_measurements, \
             patch('cellview.importers.import_functions.display_plate_summary') as mock_display:

            result = import_data(db, singleton_state, db.conn)

            # Verify it works without errors
            assert result == 0

    def test_none_connection_handling(self, db, mock_state_core):
        """Test handling when connection is None (triggers db.connect())."""
        with patch.object(db, 'connect') as mock_connect, \
             patch('cellview.importers.import_functions.select_or_create_project') as mock_project, \
             patch('cellview.importers.import_functions.select_or_create_experiment') as mock_experiment, \
             patch('cellview.importers.import_functions.create_new_repeat') as mock_repeat, \
             patch('cellview.importers.import_functions.import_conditions') as mock_conditions, \
             patch('cellview.importers.import_functions.import_measurements') as mock_measurements, \
             patch('cellview.importers.import_functions.display_plate_summary') as mock_display:

            mock_connect.return_value = db.conn

            result = import_data(db, mock_state_core, conn=None)

            # Verify db.connect() was called
            mock_connect.assert_called_once()
            assert result == 0
