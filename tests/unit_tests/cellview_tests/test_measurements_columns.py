import pandas as pd
import pytest
from cellview.importers.measurements import MeasurementsManager
from cellview.utils.state import CellViewStateCore

class TestMeasurementsColumns:
    """Tests for MeasurementsManager column identification."""

    @pytest.fixture
    def manager(self):
        """Fixture to provide a MeasurementsManager instance."""
        # We need a mock db connection, but for this specific method it's not used.
        # We can pass None and ignore type checks or mock it.
        state = CellViewStateCore(ui=None) # type: ignore
        return MeasurementsManager(db_conn=None, state=state) # type: ignore

    def test_get_measurement_columns_excludes_metadata(self, manager):
        """Test that metadata columns are excluded from measurement columns."""
        df = pd.DataFrame({
            "well": ["A1"],
            "experiment": ["exp1"],
            "plate_id": [1],
            "well_id": ["id1"],
            "cell_line": ["HeLa"],
            "si": ["siRNA"],
            "stimulus": ["drug"],
            "hours": [24],
            "antibody": ["ab1"],
            "antibody_1": ["ab2"],
            "intensity_max_DAPI_nucleus": [100],
            "area_nucleus": [50],
            "image_id": [1],
            "timepoint": [0]
        })

        cols = manager._get_measurement_columns(df)

        # Metadata should be excluded
        assert "well" not in cols
        assert "experiment" not in cols
        assert "plate_id" not in cols
        assert "well_id" not in cols
        assert "cell_line" not in cols
        assert "si" not in cols
        assert "stimulus" not in cols
        assert "hours" not in cols
        assert "antibody" not in cols
        assert "antibody_1" not in cols

        # Measurements should be included
        assert "intensity_max_DAPI_nucleus" in cols
        assert "area_nucleus" in cols
        assert "image_id" in cols
        assert "timepoint" in cols

        # Check total count
        expected_cols = {"intensity_max_DAPI_nucleus", "area_nucleus", "image_id", "timepoint"}
        assert set(cols) == expected_cols
