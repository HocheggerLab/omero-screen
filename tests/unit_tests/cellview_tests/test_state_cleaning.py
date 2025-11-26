import pandas as pd
import pytest
import numpy as np
from cellview.utils.state import CellViewStateCore

class TestStateCleaning:
    """Tests for the _clean_agg_data method in CellViewStateCore."""

    @pytest.fixture
    def state(self):
        """Fixture to provide a CellViewStateCore instance."""
        # We can instantiate it directly as it's a dataclass
        # Mocks aren't strictly needed for this pure logic method
        return CellViewStateCore(ui=None) # type: ignore

    def test_clean_agg_data_removes_unnamed_columns(self, state):
        """Test that Unnamed columns (index artifacts) are removed."""
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1],
            "valid_col": [1, 2],
            "Unnamed: 0.1": [0, 1]
        })

        cleaned = state._clean_agg_data(df)

        assert "Unnamed: 0" not in cleaned.columns
        assert "Unnamed: 0.1" not in cleaned.columns
        assert "valid_col" in cleaned.columns
        assert cleaned.shape[1] == 1

    def test_clean_agg_data_removes_redundant_metadata(self, state):
        """Test that redundant metadata columns (ending in .0, .1) are removed.

        NOTE: The new logic also drops 'experiment' and 'plate_id' entirely as they are not in DB schema.
        """
        df = pd.DataFrame({
            "experiment": ["exp1", "exp1"],
            "experiment.0": ["exp1", "exp1"],
            "experiment.1": ["exp1", "exp1"],
            "well": ["A1", "A2"],
            "well.15": ["A1", "A2"],
            "other_col": [1, 2] # This should be kept as it's not in the drop list
        })

        cleaned = state._clean_agg_data(df)

        # experiment should be kept (but suffixes dropped)
        assert "experiment" in cleaned.columns
        assert "experiment.0" not in cleaned.columns
        assert "experiment.1" not in cleaned.columns

        assert "well" in cleaned.columns
        assert "well.15" not in cleaned.columns
        assert "other_col" in cleaned.columns

    def test_clean_agg_data_handles_suffixed_measurements(self, state):
        """Test handling of suffixed measurement columns.

        Logic:
        - If base name exists: Drop suffixed version (redundant)
        - If base name missing: Rename suffixed version (unique)
        """
        df = pd.DataFrame({
            # Case 1: Redundant (base exists)
            "intensity_max_DAPI_nucleus": [100, 200],
            "intensity_max_DAPI_nucleus.0": [100, 200],

            # Case 2: Unique (base missing)
            "intensity_max_p21_nucleus.0": [50, 60],

            # Case 3: Multiple suffixes
            "intensity_mean_EdU_nucleus.1": [10, 20],
            "intensity_mean_EdU_nucleus.2": [10, 20],
        })

        cleaned = state._clean_agg_data(df)

        # Case 1
        assert "intensity_max_DAPI_nucleus" in cleaned.columns
        assert "intensity_max_DAPI_nucleus.0" not in cleaned.columns

        # Case 2
        assert "intensity_max_p21_nucleus" in cleaned.columns
        assert "intensity_max_p21_nucleus.0" not in cleaned.columns

        # Case 3
        assert "intensity_mean_EdU_nucleus" in cleaned.columns
        assert "intensity_mean_EdU_nucleus.1" not in cleaned.columns

    def test_clean_agg_data_removes_empty(self, state):
        """Test removal of empty rows and columns."""
        df = pd.DataFrame({
            "col1": [1, np.nan, 3],
            "empty_col": [np.nan, np.nan, np.nan],
            "col2": [4, np.nan, 6]
        })

        cleaned = state._clean_agg_data(df)

        assert "empty_col" not in cleaned.columns
        assert len(cleaned) == 2 # Row 1 should be dropped
        assert 1 not in cleaned.index

    def test_clean_agg_data_integration(self, state):
        """Integration test with a complex dataframe mimicking real data."""
        data = {
            "Unnamed: 0": [0, 1],
            "experiment": ["test", "test"],
            "experiment.0": ["test", "test"],
            "plate_id": [1, 1],
            "plate_id.1": [1, 1],
            "intensity_max_DAPI_nucleus": [1000, 2000],
            "intensity_max_DAPI_nucleus.0": [1000, 2000], # Redundant
            "intensity_max_p21_nucleus.0": [500, 600],    # Unique
            "intensity_mean_EdU_nucleus.1": [100, 200],   # Unique
            "empty": [np.nan, np.nan]
        }
        df = pd.DataFrame(data)

        cleaned = state._clean_agg_data(df)

        expected_cols = {
            "experiment", "plate_id",
            "intensity_max_DAPI_nucleus",
            "intensity_max_p21_nucleus",
            "intensity_mean_EdU_nucleus"
        }

        assert set(cleaned.columns) == expected_cols
        assert cleaned.shape[0] == 2

    def test_clean_agg_data_replaces_spaces(self, state):
        """Test that spaces in column names are replaced with underscores."""
        df = pd.DataFrame({
            "col with spaces": [1, 2],
            "intensity max Cyclin D1 nucleus": [10, 20],
            "normal_col": [3, 4]
        })

        cleaned = state._clean_agg_data(df)

        assert "col_with_spaces" in cleaned.columns
        assert "intensity_max_Cyclin_D1_nucleus" in cleaned.columns
        assert "normal_col" in cleaned.columns
        assert "col with spaces" not in cleaned.columns

    def test_clean_agg_data_renames_and_drops(self, state):
        """Test that columns are renamed to match DB schema and non-schema columns are dropped."""
        df = pd.DataFrame({
            "centroid-0": [1, 2],
            "integrated_int_DAPI": [100, 200],
            "experiment": ["exp1", "exp2"],
            "plate_id": [1, 1],
            "well": ["A1", "A2"],
            "intensity_max_DAPI_nucleus": [10, 20]
        })

        cleaned = state._clean_agg_data(df)

        # Renamed
        assert "centroid-0-nuc" in cleaned.columns
        assert "centroid-0" not in cleaned.columns
        assert "integrated_int_DAPI_norm" in cleaned.columns
        assert "integrated_int_DAPI" not in cleaned.columns

        # Kept (Metadata)
        assert "experiment" in cleaned.columns
        assert "plate_id" in cleaned.columns

        # Kept
        assert "well" in cleaned.columns
        assert "intensity_max_DAPI_nucleus" in cleaned.columns

    def test_clean_agg_data_strict_dropping(self, state):
        """Test that ANY row with NaN values is dropped."""
        df = pd.DataFrame({
            "timepoint": [1.0, 2.0, 3.0, 4.0],
            "image_id": [10, 20, 30, 40],
            "label": [1, 2, 3, 4],
            "measurement": [100, np.nan, 300, 400], # Row 1 has NaN measurement
            "well": ["A1", "A2", "A3", np.nan]      # Row 3 has NaN well
        })

        cleaned = state._clean_agg_data(df)

        # Rows 1 and 3 should be dropped
        assert len(cleaned) == 2
        assert cleaned["timepoint"].tolist() == [1, 3]
        assert cleaned["image_id"].tolist() == [10, 30]

        # Check types
        assert pd.api.types.is_integer_dtype(cleaned["timepoint"])
        assert pd.api.types.is_integer_dtype(cleaned["image_id"])
