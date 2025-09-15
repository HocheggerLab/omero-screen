"""Comprehensive pytest tests for combplot API functions in omero-screen-plots package.

This module provides comprehensive testing for the combplot_feature and combplot_cellcycle
functions, focusing on testing the main API functions, parameter validation, error
handling, and return value validation without validating visual output.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from omero_screen_plots.combplot_api import combplot_feature, combplot_cellcycle


class TestCombplotFeatureBasicFunctionality:
    """Test basic functionality of combplot_feature API function."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_minimal_parameters(self, mock_show, dna_edu_data):
        """Test combplot_feature with minimal required parameters."""
        conditions = ["control", "treatment1"]
        feature = "intensity_mean_p21_nucleus"
        threshold = 2500.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,  # Disable selector filtering
            save=False  # Don't save during tests
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == len(conditions) * 3  # 3 rows per condition

        # Verify all axes are Axes objects
        for ax in axes:
            assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_with_selector(self, mock_show, dna_edu_data):
        """Test combplot_feature with selector column filtering."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col="cell_line",
            selector_val="MCF10A",
            save=False
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 3  # 3 rows for 1 condition

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_multiple_conditions(self, mock_show, dna_edu_data):
        """Test combplot_feature with multiple conditions."""
        conditions = ["control", "treatment1"]
        feature = "intensity_mean_p21_nucleus"
        threshold = 2500.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 6  # 3 rows × 2 conditions

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_custom_title(self, mock_show, dna_edu_data):
        """Test combplot_feature with custom title."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0
        custom_title = "Custom Test Title"

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            title=custom_title,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        # Check that title is set (would be in suptitle)
        assert fig._suptitle is not None

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_with_cell_number_limit(self, mock_show, dna_edu_data):
        """Test combplot_feature with cell number sampling."""
        conditions = ["control"]
        feature = "intensity_mean_p21_nucleus"
        threshold = 2500.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            cell_number=10,  # Very small sample
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3  # 3 rows for 1 condition


class TestCombplotCellcycleBasicFunctionality:
    """Test basic functionality of combplot_cellcycle API function."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_minimal_parameters(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle with minimal required parameters."""
        conditions = ["control", "treatment1"]

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False  # Don't save during tests
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # 2 conditions × 2 rows + 1 barplot = 5 axes total
        assert len(axes) == len(conditions) * 2 + 1

        # Verify all axes are Axes objects
        for ax in axes:
            assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_with_selector(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle with selector column filtering."""
        conditions = ["control"]

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col="cell_line",
            selector_val="MCF10A",
            save=False
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 3  # 1 condition × 2 rows + 1 barplot

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_custom_parameters(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle with custom parameters."""
        conditions = ["control"]
        custom_title = "Cell Cycle Test"

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            title=custom_title,
            cc_phases=False,  # Use DNA content terminology
            show_error_bars=False,
            cell_number=20,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 3

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_multiple_conditions(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle with multiple conditions."""
        conditions = ["control", "treatment1"]

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        # 2 conditions × 2 rows + 1 barplot = 5 axes
        assert len(axes) == 5


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_combplot_feature_empty_dataframe(self, dna_edu_data):
        """Test combplot_feature with empty DataFrame after filtering."""
        empty_df = dna_edu_data[dna_edu_data['condition'] == 'nonexistent']

        with pytest.raises(ValueError, match="Invalid conditions"):
            combplot_feature(
                df=empty_df,
                conditions=["control"],
                feature="area_nucleus",
                threshold=300.0,
                selector_col=None,
                save=False
            )

    def test_combplot_cellcycle_empty_dataframe(self, dna_edu_data):
        """Test combplot_cellcycle with empty DataFrame after filtering."""
        empty_df = dna_edu_data[dna_edu_data['condition'] == 'nonexistent']

        with pytest.raises(ValueError, match="Invalid conditions"):
            combplot_cellcycle(
                df=empty_df,
                conditions=["control"],
                selector_col=None,
                save=False
            )

    def test_combplot_feature_invalid_selector(self, dna_edu_data):
        """Test combplot_feature with invalid selector filtering."""
        with pytest.raises(ValueError, match="selector_val.*must be provided"):
            combplot_feature(
                df=dna_edu_data,
                conditions=["control"],
                feature="area_nucleus",
                threshold=300.0,
                selector_col="cell_line",
                selector_val=None,  # Missing selector_val
                save=False
            )

    def test_combplot_cellcycle_invalid_selector(self, dna_edu_data):
        """Test combplot_cellcycle with invalid selector filtering."""
        with pytest.raises(ValueError, match="selector_val.*must be provided"):
            combplot_cellcycle(
                df=dna_edu_data,
                conditions=["control"],
                selector_col="cell_line",
                selector_val=None,  # Missing selector_val
                save=False
            )

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_missing_required_columns(self, mock_show, minimal_plate_data):
        """Test combplot_feature behavior with missing required columns."""
        # minimal_plate_data doesn't have the required columns for combplot
        df_missing_cols = minimal_plate_data.copy()

        with pytest.raises(KeyError):
            combplot_feature(
                df=df_missing_cols,
                conditions=["control"],
                feature="area_nucleus",
                threshold=300.0,
                selector_col=None,
                save=False
            )

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_missing_required_columns(self, mock_show, minimal_plate_data):
        """Test combplot_cellcycle behavior with missing required columns."""
        # minimal_plate_data doesn't have the required columns for combplot
        df_missing_cols = minimal_plate_data.copy()

        with pytest.raises(KeyError):
            combplot_cellcycle(
                df=df_missing_cols,
                conditions=["control"],
                selector_col=None,
                save=False
            )


class TestDataFiltering:
    """Test data filtering scenarios."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_condition_filtering(self, mock_show, dna_edu_data):
        """Test that only specified conditions are plotted."""
        # Data has both 'control' and 'treatment1', but only plot 'control'
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3  # 3 rows for 1 condition

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_condition_filtering(self, mock_show, dna_edu_data):
        """Test that only specified conditions are plotted."""
        conditions = ["treatment1"]  # Only plot treatment1

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3  # 1 condition × 2 rows + 1 barplot

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_with_nonexistent_condition(self, mock_show, dna_edu_data):
        """Test behavior with non-existent condition."""
        with pytest.raises(ValueError, match="Invalid conditions"):
            combplot_feature(
                df=dna_edu_data,
                conditions=["nonexistent_condition"],
                feature="area_nucleus",
                threshold=300.0,
                selector_col=None,
                save=False
            )


@pytest.mark.parametrize("size_units,expected_conversion", [
    ("cm", True),  # Should convert from cm to inches
    ("inches", False),  # Should not convert
])
class TestFigureSizing:
    """Test figure sizing and unit conversion."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_size_units(self, mock_show, dna_edu_data, size_units, expected_conversion):
        """Test figure sizing with different units."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0
        fig_size = (10, 7)

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            fig_size=fig_size,
            size_units=size_units,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        # Check that figure size was set (exact value depends on conversion)
        assert fig.get_figwidth() > 0
        assert fig.get_figheight() > 0

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_size_units(self, mock_show, dna_edu_data, size_units, expected_conversion):
        """Test figure sizing with different units."""
        conditions = ["control"]
        fig_size = (12, 7)

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            fig_size=fig_size,
            size_units=size_units,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert fig.get_figwidth() > 0
        assert fig.get_figheight() > 0


@pytest.mark.parametrize("file_format", ["png", "pdf", "svg"])
class TestSaveFunctionality:
    """Test figure saving functionality."""

    def test_combplot_feature_save_formats(self, dna_edu_data, file_format):
        """Test saving combplot_feature in different formats."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            with patch('matplotlib.pyplot.show'):
                fig, axes = combplot_feature(
                    df=dna_edu_data,
                    conditions=conditions,
                    feature=feature,
                    threshold=threshold,
                    save=True,
                    path=save_path,
                    file_format=file_format,
                    selector_col=None
                )

            assert isinstance(fig, Figure)
            assert isinstance(axes, list)

    def test_combplot_cellcycle_save_formats(self, dna_edu_data, file_format):
        """Test saving combplot_cellcycle in different formats."""
        conditions = ["control"]

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            with patch('matplotlib.pyplot.show'):
                fig, axes = combplot_cellcycle(
                    df=dna_edu_data,
                    conditions=conditions,
                    save=True,
                    path=save_path,
                    file_format=file_format,
                    selector_col=None
                )

            assert isinstance(fig, Figure)
            assert isinstance(axes, list)


class TestSpecialCases:
    """Test special cases and edge conditions."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_no_sampling_needed(self, mock_show, dna_edu_data):
        """Test combplot_feature when data size is smaller than cell_number."""
        # Filter to get small dataset
        small_df = dna_edu_data[dna_edu_data['condition'] == 'control'].head(5)

        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        fig, axes = combplot_feature(
            df=small_df,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            cell_number=10,  # Larger than data size
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_no_sampling_needed(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle when data size is smaller than cell_number."""
        # Filter to get a reasonable dataset size that still works with cellcycle analysis
        # The cellcycle_stacked function needs sufficient data for statistical calculations
        control_data = dna_edu_data[dna_edu_data['condition'] == 'control']

        conditions = ["control"]

        fig, axes = combplot_cellcycle(
            df=control_data,
            conditions=conditions,
            cell_number=200,  # Larger than data size to test no sampling
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_none_cell_number(self, mock_show, dna_edu_data):
        """Test combplot_feature with cell_number=None (no sampling)."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            cell_number=None,  # No sampling
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_none_cell_number(self, mock_show, dna_edu_data):
        """Test combplot_cellcycle with cell_number=None (no sampling)."""
        conditions = ["control"]

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            cell_number=None,  # No sampling
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3


class TestReturnValueValidation:
    """Test validation of return values."""

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_return_tuple(self, mock_show, dna_edu_data):
        """Test that combplot_feature returns correct tuple structure."""
        conditions = ["control", "treatment1"]
        feature = "area_nucleus"
        threshold = 300.0

        result = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        # Should return tuple of (Figure, list)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Figure)
        assert isinstance(result[1], list)

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_return_tuple(self, mock_show, dna_edu_data):
        """Test that combplot_cellcycle returns correct tuple structure."""
        conditions = ["control"]

        result = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False
        )

        # Should return tuple of (Figure, list)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Figure)
        assert isinstance(result[1], list)

    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_axes_count(self, mock_show, dna_edu_data):
        """Test that combplot_feature returns correct number of axes."""
        conditions = ["control", "treatment1"]
        feature = "area_nucleus"
        threshold = 300.0

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        # Should have 3 rows × 2 conditions = 6 axes
        assert len(axes) == 6
        # All should be Axes objects
        for ax in axes:
            assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_axes_count(self, mock_show, dna_edu_data):
        """Test that combplot_cellcycle returns correct number of axes."""
        conditions = ["control", "treatment1"]

        fig, axes = combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False
        )

        # Should have 2 conditions × 2 rows + 1 barplot = 5 axes
        assert len(axes) == 5
        # All should be Axes objects
        for ax in axes:
            assert isinstance(ax, Axes)


class TestDifferentFeatures:
    """Test with different feature types and thresholds."""

    @pytest.mark.parametrize("feature,threshold", [
        ("area_nucleus", 300.0),
        ("intensity_mean_p21_nucleus", 2500.0),
        ("area_cell", 500.0),
        ("intensity_mean_DAPI_nucleus", 15000.0),
    ])
    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_different_features(self, mock_show, dna_edu_data, feature, threshold):
        """Test combplot_feature with different features and thresholds."""
        conditions = ["control"]

        fig, axes = combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        assert isinstance(fig, Figure)
        assert len(axes) == 3  # 3 rows for 1 condition


class TestMockingDependencies:
    """Test with mocked dependencies to ensure proper integration."""

    @patch('omero_screen_plots.combplot_api.scatter_plot')
    @patch('omero_screen_plots.combplot_api.sns.histplot')
    @patch('matplotlib.pyplot.show')
    def test_combplot_feature_calls_dependencies(self, mock_show, mock_histogram, mock_scatter, dna_edu_data):
        """Test that combplot_feature calls the expected dependencies."""
        conditions = ["control"]
        feature = "area_nucleus"
        threshold = 300.0

        combplot_feature(
            df=dna_edu_data,
            conditions=conditions,
            feature=feature,
            threshold=threshold,
            selector_col=None,
            save=False
        )

        # Should call scatter_plot for each condition (2 scatter plots per condition)
        expected_scatter_calls = len(conditions) * 2
        assert mock_scatter.call_count == expected_scatter_calls

        # Should call sns.histplot once per condition for histograms
        expected_histogram_calls = len(conditions)
        assert mock_histogram.call_count == expected_histogram_calls

    @patch('omero_screen_plots.combplot_api.cellcycle_stacked')
    @patch('omero_screen_plots.combplot_api.scatter_plot')
    @patch('omero_screen_plots.combplot_api.sns.histplot')
    @patch('matplotlib.pyplot.show')
    def test_combplot_cellcycle_calls_dependencies(self, mock_show, mock_histogram, mock_scatter, mock_cellcycle, dna_edu_data):
        """Test that combplot_cellcycle calls the expected dependencies."""
        conditions = ["control"]

        combplot_cellcycle(
            df=dna_edu_data,
            conditions=conditions,
            selector_col=None,
            save=False
        )

        # Should call scatter_plot for each condition
        assert mock_scatter.call_count == len(conditions)
        # Should call cellcycle_stacked once
        assert mock_cellcycle.call_count == 1
        # Should call sns.histplot once per condition for histograms
        expected_histogram_calls = len(conditions)
        assert mock_histogram.call_count == expected_histogram_calls
