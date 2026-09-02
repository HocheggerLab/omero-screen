import numpy as np
import pandas as pd
import pytest
from cellview.utils.state import CellViewStateCore


class TestStateCleaning:
    """Tests for the _clean_agg_data method in CellViewStateCore."""

    @pytest.fixture
    def state(self):
        """Fixture to provide a CellViewStateCore instance."""
        # We can instantiate it directly as it's a dataclass
        # Mocks aren't strictly needed for this pure logic method
        return CellViewStateCore(ui=None)  # type: ignore

    def test_clean_agg_data_removes_unnamed_columns(self, state):
        """Test that Unnamed columns (index artifacts) are removed."""
        df = pd.DataFrame(
            {"Unnamed: 0": [0, 1], "valid_col": [1, 2], "Unnamed: 0.1": [0, 1]}
        )

        cleaned = state._clean_agg_data(df)

        assert "Unnamed: 0" not in cleaned.columns
        assert "Unnamed: 0.1" not in cleaned.columns
        assert "valid_col" in cleaned.columns
        assert cleaned.shape[1] == 1

    def test_clean_agg_data_removes_redundant_metadata(self, state):
        """Test that redundant metadata columns (ending in .0, .1) are removed.

        NOTE: The new logic also drops 'experiment' and 'plate_id' entirely as they are not in DB schema.
        """
        df = pd.DataFrame(
            {
                "experiment": ["exp1", "exp1"],
                "experiment.0": ["exp1", "exp1"],
                "experiment.1": ["exp1", "exp1"],
                "well": ["A1", "A2"],
                "well.15": ["A1", "A2"],
                "other_col": [
                    1,
                    2,
                ],  # This should be kept as it's not in the drop list
            }
        )

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
        df = pd.DataFrame(
            {
                # Case 1: Redundant (base exists)
                "intensity_max_DAPI_nucleus": [100, 200],
                "intensity_max_DAPI_nucleus.0": [100, 200],
                # Case 2: Unique (base missing)
                "intensity_max_p21_nucleus.0": [50, 60],
                # Case 3: Multiple suffixes
                "intensity_mean_EdU_nucleus.1": [10, 20],
                "intensity_mean_EdU_nucleus.2": [10, 20],
            }
        )

        cleaned = state._clean_agg_data(df)

        # Case 1
        assert "intensity_max_DAPI_nucleus" in cleaned.columns
        assert "intensity_max_DAPI_nucleus.0" not in cleaned.columns

        # Case 2
        assert "intensity_max_p21_nucleus" in cleaned.columns
        assert "intensity_max_p21_nucleus.0" not in cleaned.columns

        # Case 3: several suffixed siblings share an absent base. Only the
        # first is unique; the rest are redundant per-round repeats and must be
        # dropped, not all renamed to the same label (which would create
        # duplicate columns and break downstream ``df[float_cols].round(...)``).
        assert "intensity_mean_EdU_nucleus" in cleaned.columns
        assert "intensity_mean_EdU_nucleus.1" not in cleaned.columns
        assert "intensity_mean_EdU_nucleus.2" not in cleaned.columns

        # No duplicate column labels may survive cleaning.
        assert not cleaned.columns.duplicated().any(), (
            f"duplicate columns: "
            f"{cleaned.columns[cleaned.columns.duplicated()].tolist()}"
        )

    def test_clean_agg_data_removes_empty(self, state):
        """Test removal of empty rows and columns."""
        df = pd.DataFrame(
            {
                "col1": [1, np.nan, 3],
                "empty_col": [np.nan, np.nan, np.nan],
                "col2": [4, np.nan, 6],
            }
        )

        cleaned = state._clean_agg_data(df)

        assert "empty_col" not in cleaned.columns
        assert len(cleaned) == 2  # Row 1 should be dropped
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
            "intensity_max_DAPI_nucleus.0": [1000, 2000],  # Redundant
            "intensity_max_p21_nucleus.0": [500, 600],  # Unique
            "intensity_mean_EdU_nucleus.1": [100, 200],  # Unique
            "empty": [np.nan, np.nan],
        }
        df = pd.DataFrame(data)

        cleaned = state._clean_agg_data(df)

        expected_cols = {
            "experiment",
            "plate_id",
            "intensity_max_DAPI_nucleus",
            "intensity_max_p21_nucleus",
            "intensity_mean_EdU_nucleus",
        }

        assert set(cleaned.columns) == expected_cols
        assert cleaned.shape[0] == 2

    def test_clean_agg_data_replaces_spaces(self, state):
        """Test that spaces in column names are replaced with underscores."""
        df = pd.DataFrame(
            {
                "col with spaces": [1, 2],
                "intensity max Cyclin D1 nucleus": [10, 20],
                "normal_col": [3, 4],
            }
        )

        cleaned = state._clean_agg_data(df)

        assert "col_with_spaces" in cleaned.columns
        assert "intensity_max_Cyclin_D1_nucleus" in cleaned.columns
        assert "normal_col" in cleaned.columns
        assert "col with spaces" not in cleaned.columns

    def test_clean_agg_data_renames_and_drops(self, state):
        """Test that columns are renamed to match DB schema and non-schema columns are dropped."""
        df = pd.DataFrame(
            {
                "centroid-0": [1, 2],
                "integrated_int_DAPI": [100, 200],
                "experiment": ["exp1", "exp2"],
                "plate_id": [1, 1],
                "well": ["A1", "A2"],
                "intensity_max_DAPI_nucleus": [10, 20],
            }
        )

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
        df = pd.DataFrame(
            {
                "timepoint": [1.0, 2.0, 3.0, 4.0],
                "image_id": [10, 20, 30, 40],
                "label": [1, 2, 3, 4],
                "measurement": [
                    100,
                    np.nan,
                    300,
                    400,
                ],  # Row 1 has NaN measurement
                "well": ["A1", "A2", "A3", np.nan],  # Row 3 has NaN well
            }
        )

        cleaned = state._clean_agg_data(df)

        # Rows 1 and 3 should be dropped
        assert len(cleaned) == 2
        assert cleaned["timepoint"].tolist() == [1, 3]
        assert cleaned["image_id"].tolist() == [10, 30]

        # Check types
        assert pd.api.types.is_integer_dtype(cleaned["timepoint"])
        assert pd.api.types.is_integer_dtype(cleaned["image_id"])


class TestCyclicIFDnaNormalisation:
    """Regression tests for the 4i normalised-DNA column.

    Cyclic-IF ``agg_data.csv`` files put the cell-cycle-normalised DNA content
    in ``integrated_int_DAPI_norm``, but which round supplies it depends on
    which round carried EdU. Two earlier defects in ``clean_agg_data`` put
    *raw* integrated DAPI (modal peak ~1e6) into that column, which silently
    rescaled the DNA axis of every cell-cycle plot by ~500,000x.
    """

    @pytest.fixture
    def state(self):
        """Fixture to provide a CellViewStateCore instance."""
        return CellViewStateCore(ui=None)  # type: ignore

    @staticmethod
    def _base():
        """Minimum identity columns so nothing is dropped as malformed."""
        return {
            "experiment": ["x"],
            "plate_id": [4126],
            "well": ["A1"],
            "well_id": [1],
            "image_id": [1],
            "cell_line": ["RPE-1"],
            "label": [1],
            "timepoint": [0],
        }

    def test_norm_recovered_when_master_round_lacks_edu(self, state):
        """The master round need not be the one with cell-cycle analysis.

        Plates 4126-4128 have Y15Cdk1/Tub as the master round, so there is no
        unsuffixed ``integrated_int_DAPI_norm`` at all -- it arrives from the
        EdU restain round as ``integrated_int_DAPI_norm.2``. Step 3 used to
        delete that column (its regex matched ``DAPI_norm`` as readily as
        ``DAPI``), after which step 5 promoted the raw column into the
        vacated slot.
        """
        d = self._base()
        d["integrated_int_DAPI"] = [1.0e6]  # raw, master round
        d["integrated_int_DAPI.2"] = [1.05e6]  # raw, restain round
        d["integrated_int_DAPI_norm.2"] = [2.0]  # the real normalised DNA
        d["cell_cycle.2"] = ["G1"]

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert cleaned["integrated_int_DAPI_norm"].iloc[0] == 2.0, (
            "raw DNA content leaked into the normalised column"
        )
        # The raw value is still useful (it is what the modal-peak rescale
        # workaround in the paper repo operated on) and must survive.
        assert cleaned["integrated_int_DAPI"].iloc[0] == 1.0e6
        assert "integrated_int_DAPI_norm.2" not in cleaned.columns
        assert cleaned["cell_cycle"].iloc[0] == "G1"

    def test_existing_norm_column_is_not_renamed_to_norm_norm(self, state):
        """A master round *with* cell-cycle analysis must be left alone.

        ``^integrated_int_([A-Za-z0-9_]+)$`` matches ``integrated_int_DAPI_norm``
        itself, so the normalised column was renamed to
        ``integrated_int_DAPI_norm_norm`` -- a name nothing reads.
        """
        d = self._base()
        d["integrated_int_DAPI"] = [1.0e6]
        d["integrated_int_DAPI_norm"] = [2.0]
        d["integrated_int_DAPI.1"] = [1.1e6]
        d["integrated_int_DAPI_norm.1"] = [2.1]

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert "integrated_int_DAPI_norm_norm" not in cleaned.columns
        assert cleaned["integrated_int_DAPI_norm"].iloc[0] == 2.0
        assert not cleaned.columns.duplicated().any()

    def test_raw_still_promoted_when_no_norm_exists_anywhere(self, state):
        """Legacy behaviour: a plate with no cell-cycle analysis at all.

        Here promoting raw into the ``_norm`` slot is the intended fallback,
        because the schema has nowhere else to put it.
        """
        d = self._base()
        d["integrated_int_DAPI"] = [1.0e6]

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert cleaned["integrated_int_DAPI_norm"].iloc[0] == 1.0e6
        assert "integrated_int_DAPI" not in cleaned.columns

    def test_non_dapi_nucleus_channel(self, state):
        """The same must hold for post-refactor Hoechst/H2B_RFP plates."""
        d = self._base()
        d["integrated_int_Hoechst"] = [1.0e6]
        d["integrated_int_Hoechst_norm.3"] = [2.0]

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert cleaned["integrated_int_Hoechst_norm"].iloc[0] == 2.0
        assert "integrated_int_Hoechst_norm_norm" not in cleaned.columns

    def test_incompletely_matched_cells_are_dropped(self, state):
        """Complete-case filtering is deliberate and must stay.

        A nucleus that failed to match in one restain round has no position
        in the multiplexed feature space, so it cannot enter PCA/UMAP or the
        distance metrics. It is dropped -- but the count is now logged rather
        than being a silent side effect.
        """
        d = {
            "experiment": ["x", "x"],
            "plate_id": [4126, 4126],
            "well": ["A1", "A1"],
            "well_id": [1, 1],
            "image_id": [1, 1],
            "cell_line": ["RPE-1", "RPE-1"],
            "label": [1, 2],
            "timepoint": [0, 0],
            "integrated_int_DAPI": [1.0e6, 1.0e6],
            # cell 2 was never matched in the p21 round
            "intensity_mean_p21_nucleus.2": [500.0, np.nan],
        }

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert len(cleaned) == 1
        assert cleaned["label"].tolist() == [1]

    def test_malformed_identity_row_is_dropped(self, state):
        """A NaN identity column is malformed input, not an unmatched cell."""
        d = {
            "experiment": ["x", "x"],
            "plate_id": [4126, 4126],
            "well": ["A1", np.nan],
            "well_id": [1, 1],
            "image_id": [1, 1],
            "cell_line": ["RPE-1", "RPE-1"],
            "label": [1, 2],
            "timepoint": [0, 0],
            "integrated_int_DAPI": [1.0e6, 1.0e6],
        }

        cleaned = state._clean_agg_data(pd.DataFrame(d))

        assert len(cleaned) == 1
        assert cleaned["well"].tolist() == ["A1"]
