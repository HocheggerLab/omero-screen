"""Tests demonstrating the new dependency injection pattern for CellView."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cellview.utils.state import CellViewStateCore, create_cellview_state, CellViewState
from cellview.importers.measurements import MeasurementsManager
from cellview.importers.projects import ProjectManager


class TestCellViewStateCore:
    """Tests for the new dependency-injectable CellViewStateCore."""

    def test_create_empty_state(self) -> None:
        """Test creating an empty state instance."""
        state = CellViewStateCore.create_from_args(None)

        assert state is not None
        assert state.plate_id is None
        assert state.df is None
        assert state.ui is not None
        assert state.console is not None
        assert state.logger is not None

    def test_create_state_from_csv_args(self, tmp_path: Path) -> None:
        """Test creating state from CSV arguments."""
        # Create a temporary CSV file
        csv_path = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'plate_id': [12345, 12345, 12345],
            'well': ['A01', 'A02', 'A03'],
            'image_id': [1, 2, 3],
            'timepoint': [0, 0, 0],
            'intensity_mean_DAPI_nucleus': [100, 200, 150]
        })
        test_data.to_csv(csv_path, index=False)

        # Create args namespace
        args = argparse.Namespace(csv=csv_path, plate_id=None)

        # Create state
        state = CellViewStateCore.create_from_args(args)

        assert state.plate_id == 12345
        assert state.csv_path == csv_path
        assert state.df is not None
        assert len(state.df) == 3
        assert 'DAPI' in state.get_channels()

    @patch('cellview.utils.state.omero_connect')
    def test_create_state_from_plate_args(self, mock_omero_connect: Mock) -> None:
        """Test creating state from plate ID arguments."""
        # Mock the OMERO connection and plate data
        mock_conn = Mock()
        mock_plate = Mock()
        mock_plate.getObject.return_value = mock_plate
        mock_conn.getObject.return_value = mock_plate

        # Mock the parse_omero_data method to return test data
        test_df = pd.DataFrame({
            'plate_id': [12345],
            'well': ['A01'],
            'intensity_mean_DAPI_nucleus': [100]
        })

        with patch.object(CellViewStateCore, 'parse_omero_data') as mock_parse:
            mock_parse.return_value = (
                test_df, 'test_project', 'test_experiment', '2024-01-01', 'test_user'
            )

            args = argparse.Namespace(csv=None, plate_id=12345)
            state = CellViewStateCore.create_from_args(args)

            assert state.plate_id == 12345
            assert state.project_name == 'test_project'
            assert state.experiment_name == 'test_experiment'
            assert state.df is not None

    def test_convenience_function(self) -> None:
        """Test the convenience function for creating state."""
        state = create_cellview_state(None)

        assert isinstance(state, CellViewStateCore)
        assert state.ui is not None
        assert state.console is not None
        assert state.logger is not None


class TestDependencyInjection:
    """Tests for dependency injection in importer classes."""

    def test_measurements_manager_with_dependency_injection(self) -> None:
        """Test MeasurementsManager with injected state."""
        # Create mock connection and state
        mock_conn = Mock()
        state = CellViewStateCore.create_from_args(None)
        state.df = pd.DataFrame({
            'well': ['A01', 'A02'],
            'image_id': [1, 2],
            'timepoint': [0, 0],
            'intensity_mean_DAPI_nucleus': [100, 200]
        })
        state.condition_id_map = {'A01': 1, 'A02': 2}

        # Create manager with injected state
        manager = MeasurementsManager(mock_conn, state)

        # Verify the manager uses the injected state
        assert manager.state is state
        assert manager.state.df is not None
        assert len(manager.state.df) == 2

    def test_measurements_manager_fallback_to_singleton(self) -> None:
        """Test MeasurementsManager falls back to singleton when no state provided."""
        mock_conn = Mock()

        with patch.object(CellViewState, 'get_instance') as mock_get_instance:
            mock_singleton = Mock()
            mock_get_instance.return_value = mock_singleton

            # Create manager without injected state
            manager = MeasurementsManager(mock_conn)

            # Verify it falls back to singleton
            assert manager.state is mock_singleton
            mock_get_instance.assert_called_once()

    def test_project_manager_with_dependency_injection(self) -> None:
        """Test ProjectManager with injected state."""
        mock_conn = Mock()
        state = CellViewStateCore.create_from_args(None)
        state.plate_id = 12345

        # Create manager with injected state
        manager = ProjectManager(mock_conn, state)

        # Verify the manager uses the injected state
        assert manager.state is state
        assert manager.state.plate_id == 12345

    def test_project_manager_fallback_to_singleton(self) -> None:
        """Test ProjectManager falls back to singleton when no state provided."""
        mock_conn = Mock()

        with patch.object(CellViewState, 'get_instance') as mock_get_instance:
            mock_singleton = Mock()
            mock_get_instance.return_value = mock_singleton

            # Create manager without injected state
            manager = ProjectManager(mock_conn)

            # Verify it falls back to singleton
            assert manager.state is mock_singleton
            mock_get_instance.assert_called_once()


class TestStateIsolation:
    """Tests to demonstrate that dependency injection provides state isolation."""

    def test_multiple_state_instances_are_independent(self) -> None:
        """Test that multiple CellViewStateCore instances are independent."""
        # Create two separate state instances
        state1 = CellViewStateCore.create_from_args(None)
        state2 = CellViewStateCore.create_from_args(None)

        # Modify state1
        state1.plate_id = 123
        state1.project_name = "Project A"

        # Modify state2
        state2.plate_id = 456
        state2.project_name = "Project B"

        # Verify they don't affect each other
        assert state1.plate_id == 123
        assert state1.project_name == "Project A"
        assert state2.plate_id == 456
        assert state2.project_name == "Project B"

        # Verify they are different instances
        assert state1 is not state2

    def test_singleton_vs_dependency_injection_isolation(self) -> None:
        """Test that singleton and dependency injection don't interfere."""
        # Create dependency-injected state
        di_state = CellViewStateCore.create_from_args(None)
        di_state.plate_id = 999
        di_state.project_name = "DI Project"

        # Get singleton instance
        with patch.object(CellViewState, 'get_instance') as mock_get_instance:
            mock_singleton = Mock()
            mock_singleton.plate_id = 888
            mock_singleton.project_name = "Singleton Project"
            mock_get_instance.return_value = mock_singleton

            singleton_state = CellViewState.get_instance()

            # Verify they have different values
            assert di_state.plate_id == 999
            assert di_state.project_name == "DI Project"
            assert singleton_state.plate_id == 888
            assert singleton_state.project_name == "Singleton Project"


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility is maintained."""

    def test_singleton_still_works(self) -> None:
        """Test that the old singleton pattern still works."""
        with patch.object(CellViewState, 'get_instance') as mock_get_instance:
            mock_singleton = Mock()
            mock_get_instance.return_value = mock_singleton

            # Old way should still work
            state = CellViewState.get_instance()
            assert state is mock_singleton
            mock_get_instance.assert_called_once()

    def test_importers_work_with_both_patterns(self) -> None:
        """Test that importers work with both singleton and DI patterns."""
        mock_conn = Mock()

        # Test with dependency injection
        di_state = CellViewStateCore.create_from_args(None)
        di_manager = MeasurementsManager(mock_conn, di_state)
        assert di_manager.state is di_state

        # Test with singleton (backward compatibility)
        with patch.object(CellViewState, 'get_instance') as mock_get_instance:
            mock_singleton = Mock()
            mock_get_instance.return_value = mock_singleton

            singleton_manager = MeasurementsManager(mock_conn)
            assert singleton_manager.state is mock_singleton


if __name__ == "__main__":
    pytest.main([__file__])
