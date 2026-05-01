"""Tests for classifier filter logic in gallery_api and _welldata_widget."""

import polars as pl
import pytest
from unittest.mock import MagicMock, patch

from omero_screen_napari.gallery_api import CroppedImageParser, OmeroData, UserData
from omero_screen_napari._welldata_widget import _extract_classifier_data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df() -> pl.DataFrame:
    """Well dataframe with two classifier columns and a cell_cycle column."""
    return pl.DataFrame(
        {
            "well": ["A1"] * 6,
            "image_id": [1, 1, 2, 2, 3, 3],
            "cell_cycle": ["G1", "S", "G1", "G2/M", "S", "G1"],
            "classifier_mitosis": ["normal", "normal", "mitotic", "normal", "mitotic", "normal"],
            "classifier_damage": ["low", "high", "low", "low", "high", "low"],
        }
    )


@pytest.fixture
def mock_omero_data(base_df: pl.DataFrame) -> MagicMock:
    mock = MagicMock(spec=OmeroData)
    mock.plate_data = base_df.lazy()
    mock.image_ids = [1, 2, 3]
    mock.cropped_images = []
    mock.cropped_labels = []
    return mock


@pytest.fixture
def mock_user_data() -> UserData:
    ud = UserData()
    ud.well = "A1"
    ud.segmentation = "nucleus"
    ud.cellcycle = "All"
    ud.classifier_filter = ""
    return ud


def make_parser(omero_data: MagicMock, user_data: UserData) -> CroppedImageParser:
    return CroppedImageParser(omero_data, user_data)


# ---------------------------------------------------------------------------
# _select_classifierdata
# ---------------------------------------------------------------------------

class TestSelectClassifierData:
    def test_empty_filter_returns_all_rows(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = ""
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert len(result) == len(base_df)

    def test_whitespace_only_filter_returns_all_rows(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = "   "
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert len(result) == len(base_df)

    def test_filter_on_first_classifier_column(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = "mitotic"
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert len(result) == 2
        assert all(result["classifier_mitosis"] == "mitotic")

    def test_filter_on_second_classifier_column(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = "high"
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert len(result) == 2
        assert all(result["classifier_damage"] == "high")

    def test_unknown_filter_value_returns_all_rows(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = "nonexistent"
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert len(result) == len(base_df)

    def test_filter_on_df_without_classifier_columns(self, mock_omero_data, mock_user_data):
        df_no_classifier = pl.DataFrame({"well": ["A1"], "cell_cycle": ["G1"]})
        mock_user_data.classifier_filter = "mitotic"
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(df_no_classifier)
        assert len(result) == 1

    def test_filter_returns_correct_columns(self, mock_omero_data, mock_user_data, base_df):
        mock_user_data.classifier_filter = "normal"
        parser = make_parser(mock_omero_data, mock_user_data)
        result = parser._select_classifierdata(base_df)
        assert set(result.columns) == set(base_df.columns)


# ---------------------------------------------------------------------------
# _extract_classifier_data
# ---------------------------------------------------------------------------

class TestExtractClassifierData:
    def _make_plate_data(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "well": ["A1", "A1", "A1", "B1", "B1"],
                "classifier_mitosis": ["normal", "mitotic", "normal", "normal", "normal"],
            }
        ).lazy()

    def test_returns_none_for_none_well(self):
        assert _extract_classifier_data(None) is None

    def test_returns_none_when_no_classifier_columns(self):
        plate_data = pl.DataFrame({"well": ["A1"], "area": [100]}).lazy()
        mock_omero = MagicMock()
        mock_omero.plate_data = plate_data
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            assert _extract_classifier_data("A1") is None

    def test_returns_counts_for_well(self):
        mock_omero = MagicMock()
        mock_omero.plate_data = self._make_plate_data()
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            result = _extract_classifier_data("A1")
        assert result is not None
        assert "mitosis" in result
        counts = dict(result["mitosis"])
        assert counts["normal"] == 2
        assert counts["mitotic"] == 1

    def test_counts_are_well_specific(self):
        mock_omero = MagicMock()
        mock_omero.plate_data = self._make_plate_data()
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            result = _extract_classifier_data("B1")
        assert result is not None
        counts = dict(result["mitosis"])
        assert counts["normal"] == 2
        assert "mitotic" not in counts

    def test_returns_none_for_unknown_well(self):
        mock_omero = MagicMock()
        mock_omero.plate_data = self._make_plate_data()
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            result = _extract_classifier_data("Z99")
        assert result is None

    def test_multiple_classifier_columns(self):
        plate_data = pl.DataFrame(
            {
                "well": ["A1", "A1", "A1"],
                "classifier_mitosis": ["normal", "mitotic", "normal"],
                "classifier_damage": ["low", "low", "high"],
            }
        ).lazy()
        mock_omero = MagicMock()
        mock_omero.plate_data = plate_data
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            result = _extract_classifier_data("A1")
        assert result is not None
        assert set(result.keys()) == {"mitosis", "damage"}

    def test_returns_none_on_exception(self):
        mock_omero = MagicMock()
        mock_omero.plate_data.collect_schema.side_effect = RuntimeError("db error")
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            assert _extract_classifier_data("A1") is None

    def test_sorted_by_count_descending(self):
        plate_data = pl.DataFrame(
            {
                "well": ["A1"] * 5,
                "classifier_mitosis": ["normal", "normal", "normal", "mitotic", "collapsed"],
            }
        ).lazy()
        mock_omero = MagicMock()
        mock_omero.plate_data = plate_data
        with patch("omero_screen_napari._welldata_widget.omero_data", mock_omero):
            result = _extract_classifier_data("A1")
        assert result is not None
        counts = result["mitosis"]
        assert counts[0] == ("normal", 3)
