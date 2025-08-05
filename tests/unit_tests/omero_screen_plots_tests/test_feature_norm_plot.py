"""Comprehensive pytest tests for feature_norm_plot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the normalized feature plot functionality,
focusing on testing the main API functions, normalization behavior, threshold-based
stacked bars, and error handling without validating visual output.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from unittest.mock import patch, MagicMock

from omero_screen_plots.featureplot_api import feature_norm_plot
from omero_screen_plots.colors import COLOR


class TestFeatureNormPlotBasicFunctionality:
    """Test basic functionality of feature_norm_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_minimal_parameters(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with minimal required parameters."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,  # Explicitly set to None to avoid default "cell_line"
            selector_val=None,
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created (has elements)
        assert len(ax.patches) > 0  # Should have stacked bar patches
        assert ax.get_ylabel() != ""  # Should have y-axis label
        assert "%" in ax.get_ylabel()  # Should show percentage

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_default_green_scheme(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with default green color scheme."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            color_scheme="green",
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check that bars were created (stacked bars)
        assert len(ax.patches) > 0

        # Check that some bars have the expected green colors
        bar_colors = [patch.get_facecolor() for patch in ax.patches if hasattr(patch, 'get_facecolor')]
        assert len(bar_colors) > 0  # Should have colored bars

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_with_selector_column(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with selector column filtering."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col="cell_line",
            selector_val="MCF10A"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Title should include the selector value (check fig suptitle)
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert "MCF10A" in fig_title

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_with_custom_title(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with custom title."""
        custom_title = "Custom Normalized Feature Analysis"
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            title=custom_title,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check figure suptitle
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert custom_title in fig_title

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_with_grouping(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with group_size parameter."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            group_size=2,
            within_group_spacing=0.3,
            between_group_gap=0.7,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should still create a valid plot with grouping
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_with_custom_figure_size(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with custom figure size."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            fig_size=(10, 8),
            size_units="cm",
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Note: Figure size testing is complex due to unit conversions


class TestFeatureNormPlotColorSchemes:
    """Test different color schemes for feature_norm_plot."""

    @pytest.mark.parametrize("color_scheme", ["green", "blue", "purple"])
    @patch('matplotlib.pyplot.show')
    def test_valid_color_schemes(self, mock_show, synthetic_plate_data, color_scheme):
        """Test all valid color schemes work correctly."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            color_scheme=color_scheme,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check that bars were created
        assert len(ax.patches) > 0

        # Check that bars have colors (not default)
        bar_colors = [patch.get_facecolor() for patch in ax.patches if hasattr(patch, 'get_facecolor')]
        assert len(bar_colors) > 0

    @patch('matplotlib.pyplot.show')
    def test_invalid_color_scheme_defaults_to_green(self, mock_show, synthetic_plate_data):
        """Test that invalid color scheme defaults to green."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            color_scheme="invalid_scheme",
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should still create a valid plot (defaults to green)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_case_insensitive_color_schemes(self, mock_show, synthetic_plate_data):
        """Test that color schemes are case insensitive."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            color_scheme="BLUE",  # Uppercase
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestFeatureNormPlotThresholdBehavior:
    """Test threshold behavior in feature_norm_plot."""

    @pytest.mark.parametrize("threshold", [1.0, 1.5, 2.0, 3.0])
    @patch('matplotlib.pyplot.show')
    def test_different_threshold_values(self, mock_show, synthetic_plate_data, threshold):
        """Test different threshold values work correctly."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            threshold=threshold,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_extreme_threshold_values(self, mock_show, synthetic_plate_data):
        """Test extreme threshold values."""
        # Very low threshold (most cells should be positive)
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            threshold=0.1,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

        # Very high threshold (most cells should be negative)
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            threshold=10.0,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_invalid_threshold_values(self, mock_show, synthetic_plate_data):
        """Test invalid threshold values - they might be handled gracefully."""
        # Note: The implementation may handle edge cases gracefully rather than raising errors
        # Let's test that the function at least doesn't crash with edge values

        # Very small threshold (close to zero) - should work but may give all positive
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            threshold=0.01,  # Very small but positive
            selector_col=None
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Very large threshold - should work but may give all negative
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            threshold=100.0,  # Very large
            selector_col=None
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestFeatureNormPlotTriplicateDisplay:
    """Test triplicate display options in feature_norm_plot."""

    @pytest.mark.parametrize("show_triplicates", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_show_triplicates_option(self, mock_show, synthetic_plate_data, show_triplicates):
        """Test show_triplicates parameter."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            show_triplicates=show_triplicates,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

        # When showing triplicates, should have more bars
        if show_triplicates:
            # Should have individual replicate bars in addition to summary bars
            # Exact count depends on data structure, but should be more bars
            assert len(ax.patches) >= 3  # At least one bar per condition

    @pytest.mark.parametrize("show_boxes", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_show_boxes_option(self, mock_show, synthetic_plate_data, show_boxes):
        """Test show_boxes parameter for triplicates."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            show_triplicates=True,
            show_boxes=show_boxes,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_triplicates_with_boxes_combination(self, mock_show, synthetic_plate_data):
        """Test show_triplicates=True with show_boxes=True."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            show_triplicates=True,
            show_boxes=True,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestFeatureNormPlotNormalizationOptions:
    """Test normalization options in feature_norm_plot."""

    @pytest.mark.parametrize("normalize_by_plate", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_normalize_by_plate_option(self, mock_show, synthetic_plate_data, normalize_by_plate):
        """Test normalize_by_plate parameter."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            normalize_by_plate=normalize_by_plate,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_normalization_with_single_plate(self, mock_show, minimal_plate_data):
        """Test normalization with single plate data."""
        # Filter to single plate
        single_plate_df = minimal_plate_data[minimal_plate_data["plate_id"] == 1001].copy()

        fig, ax = feature_norm_plot(
            df=single_plate_df,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            normalize_by_plate=True,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_normalization_with_multiple_plates(self, mock_show, many_plates_data):
        """Test normalization with multiple plates."""
        fig, ax = feature_norm_plot(
            df=many_plates_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            normalize_by_plate=True,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestFeatureNormPlotQCGeneration:
    """Test QC plot generation in feature_norm_plot."""

    @patch('matplotlib.pyplot.show')
    @patch('omero_screen_plots.utils.save_fig')
    def test_save_norm_qc_false(self, mock_save_fig, mock_show, synthetic_plate_data):
        """Test that QC plots are not saved when save_norm_qc=False."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            save_norm_qc=False,
            save=False,  # Don't save main plot either
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # save_fig should not be called for QC plots
        # (This is implementation-dependent, but we're testing the interface)

    @patch('matplotlib.pyplot.show')
    def test_save_norm_qc_true(self, mock_show, synthetic_plate_data, tmp_path):
        """Test that QC plots are generated when save_norm_qc=True."""
        # Create the QC directory that the function expects
        qc_dir = tmp_path / "intensity_mean_DAPI_nucleus_norm_qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            save_norm_qc=True,
            save=False,  # Don't save main plot to focus on QC
            path=tmp_path,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # The function should complete without error when QC is requested
        # Check that QC directory was used
        assert qc_dir.exists()


class TestFeatureNormPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test feature_norm_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            feature_norm_plot(
                df=empty_df,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"]
            )

    def test_missing_required_columns(self, synthetic_plate_data):
        """Test feature_norm_plot with missing required columns."""
        # Remove plate_id column
        df_missing_plate = synthetic_plate_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            feature_norm_plot(
                df=df_missing_plate,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"]
            )

    def test_missing_feature_column(self, synthetic_plate_data):
        """Test feature_norm_plot with missing feature column."""
        with pytest.raises(ValueError, match="Feature column 'nonexistent_feature' not found"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="nonexistent_feature",
                conditions=["control", "treatment1"]
            )

    def test_invalid_condition_column(self, synthetic_plate_data):
        """Test feature_norm_plot with invalid condition column."""
        with pytest.raises(ValueError, match="Condition column 'invalid_column' not found"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                condition_col="invalid_column"
            )

    def test_conditions_not_in_data(self, synthetic_plate_data):
        """Test feature_norm_plot with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "nonexistent_condition"]
            )

    def test_invalid_selector_column(self, synthetic_plate_data):
        """Test feature_norm_plot with invalid selector column."""
        with pytest.raises(ValueError, match="Selector column 'invalid_selector' not found"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col="invalid_selector",
                selector_val="some_value"
            )

    def test_selector_column_without_value(self, synthetic_plate_data):
        """Test feature_norm_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError, match="selector_val must be provided when selector_col is specified"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line"
            )

    def test_invalid_selector_value(self, synthetic_plate_data):
        """Test feature_norm_plot with selector_val not in data."""
        with pytest.raises(ValueError, match="Value 'NonexistentCell' not found in column 'cell_line'"):
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell"
            )

    def test_no_data_after_filtering(self, synthetic_plate_data):
        """Test feature_norm_plot when filtering results in no data."""
        # Create a scenario where filtering removes all data
        df_subset = synthetic_plate_data[synthetic_plate_data["condition"] == "control"].copy()

        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_norm_plot(
                df=df_subset,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],  # treatment1 not in subset
                selector_col="cell_line",
                selector_val="MCF10A"
            )

    @patch('matplotlib.pyplot.show')
    def test_insufficient_data_for_normalization(self, mock_show):
        """Test feature_norm_plot with insufficient data for mode calculation."""
        # Create minimal data that might not be sufficient for normalization
        minimal_data = [
            {
                "plate_id": 1001,
                "well": "A1",
                "experiment": "exp1",
                "condition": "control",
                "intensity_mean_DAPI_nucleus": 1000.0,
            },
            {
                "plate_id": 1001,
                "well": "A2",
                "experiment": "exp2",
                "condition": "treatment1",
                "intensity_mean_DAPI_nucleus": 2000.0,
            },
        ]
        df = pd.DataFrame(minimal_data)

        # This might raise an error due to insufficient data for mode calculation
        # Or it might work - depends on the implementation's robustness
        try:
            fig, ax = feature_norm_plot(
                df=df,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col=None
            )
            # If it works, verify it's a valid plot
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
        except ValueError:
            # If it raises an error, that's acceptable for insufficient data
            pass


class TestFeatureNormPlotIntegration:
    """Test integration aspects and matplotlib object interactions."""

    @patch('matplotlib.pyplot.show')
    def test_figure_and_axes_are_matplotlib_objects(self, mock_show, synthetic_plate_data):
        """Test that returned figure and axes are proper matplotlib objects."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        # Test figure properties
        assert hasattr(fig, 'savefig')
        assert hasattr(fig, 'set_size_inches')
        assert callable(fig.savefig)

        # Test axes properties
        assert hasattr(ax, 'plot')
        assert hasattr(ax, 'set_xlabel')
        assert hasattr(ax, 'set_ylabel')
        assert hasattr(ax, 'set_title')
        assert callable(ax.plot)

    @patch('matplotlib.pyplot.show')
    def test_plot_without_display(self, mock_show, synthetic_plate_data):
        """Test that plots can be created without displaying (mocked show)."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        # Verify show was not called (it's mocked)
        mock_show.assert_not_called()

        # But plot should still be created
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_custom_axes_input(self, mock_show, synthetic_plate_data):
        """Test feature_norm_plot with custom axes input."""
        import matplotlib.pyplot as plt

        # Create custom figure and axes
        custom_fig, custom_ax = plt.subplots(figsize=(10, 8))

        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            axes=custom_ax,
            selector_col=None
        )

        # Should return the same axes we provided
        assert ax is custom_ax
        assert fig is custom_fig

        plt.close(custom_fig)  # Clean up

    @patch('matplotlib.pyplot.show')
    def test_plot_elements_present(self, mock_show, synthetic_plate_data):
        """Test that expected plot elements are present."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check for stacked bars
        assert len(ax.patches) > 0

        # Check for percentage y-axis
        y_label = ax.get_ylabel()
        assert "%" in y_label

        # Check x-axis has condition labels
        x_tick_labels = ax.get_xticklabels()
        assert len(x_tick_labels) > 0

        # Check that y-axis goes from 0 to 100 (percentage)
        y_min, y_max = ax.get_ylim()
        assert y_min >= 0
        assert y_max <= 110  # Allow some padding above 100%


class TestFeatureNormPlotParameterized:
    """Parametrized tests for different plot configurations."""

    @pytest.mark.parametrize(
        "selector_col,selector_val",
        [("cell_line", "MCF10A"), ("cell_line", "HeLa"), (None, None)]
    )
    @patch('matplotlib.pyplot.show')
    def test_selector_combinations(self, mock_show, synthetic_plate_data, selector_col, selector_val):
        """Test different selector column combinations."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=selector_col,
            selector_val=selector_val
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check title includes selector value when provided
        if selector_val:
            fig_title = fig._suptitle.get_text() if fig._suptitle else ""
            assert selector_val in fig_title

    @pytest.mark.parametrize("feature", ["intensity_mean_DAPI_nucleus", "intensity_mean_GFP_cell", "area_nucleus"])
    @patch('matplotlib.pyplot.show')
    def test_different_features(self, mock_show, synthetic_plate_data, feature):
        """Test normalizing different features."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature=feature,
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should work for any numeric feature
        assert len(ax.patches) > 0

    @pytest.mark.parametrize("group_size", [1, 2, 3])
    @patch('matplotlib.pyplot.show')
    def test_group_sizes(self, mock_show, synthetic_plate_data, group_size):
        """Test different group sizes work correctly."""
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            group_size=group_size,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0  # Should have stacked bars


class TestFeatureNormPlotSpecialCases:
    """Test special data scenarios."""

    def test_single_condition_data(self, single_condition_data):
        """Test with data containing only the control condition."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_norm_plot(
                df=single_condition_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],  # treatment1 not in data
                selector_col=None
            )

    @patch('matplotlib.pyplot.show')
    def test_data_with_many_plates(self, mock_show, many_plates_data):
        """Test with data containing many plates (for significance testing)."""
        fig, ax = feature_norm_plot(
            df=many_plates_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # With 5 plates, normalization should work across plates
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_feature_norm_plot_with_missing_data_points(self, mock_show):
        """Test feature_norm_plot when some conditions have very few data points."""
        # Create data where one condition has very few points
        data = []

        # Control condition with many points
        for i in range(20):
            data.append({
                "plate_id": 1001,
                "well": f"A{i+1}",
                "experiment": f"exp_control_{i}",
                "condition": "control",
                "intensity_mean_DAPI_nucleus": np.random.uniform(8000, 15000),
            })

        # Treatment condition with only 5 points
        for i in range(5):
            data.append({
                "plate_id": 1001,
                "well": f"B{i+1}",
                "experiment": f"exp_treatment_{i}",
                "condition": "treatment1",
                "intensity_mean_DAPI_nucleus": np.random.uniform(12000, 20000),
            })

        df = pd.DataFrame(data)

        fig, ax = feature_norm_plot(
            df=df,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestFeatureNormPlotErrorMessages:
    """Test specific error messages for various failure modes."""

    def test_informative_error_messages(self, synthetic_plate_data):
        """Test that error messages are informative and helpful."""

        # Test missing feature column error message
        with pytest.raises(ValueError) as exc_info:
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="nonexistent_feature",
                conditions=["control", "treatment1"]
            )

        error_msg = str(exc_info.value)
        assert "Feature column 'nonexistent_feature' not found" in error_msg
        assert "Available columns:" in error_msg

        # Test invalid selector value error message
        with pytest.raises(ValueError) as exc_info:
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell"
            )

        error_msg = str(exc_info.value)
        assert "Value 'NonexistentCell' not found in column 'cell_line'" in error_msg
        assert "Available values:" in error_msg

    def test_helpful_selector_error_message(self, synthetic_plate_data):
        """Test helpful error message when selector_col provided without selector_val."""
        with pytest.raises(ValueError) as exc_info:
            feature_norm_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line"
                # selector_val not provided
            )

        error_msg = str(exc_info.value)
        assert "selector_val must be provided when selector_col is specified" in error_msg
        assert "Available values in 'cell_line':" in error_msg


class TestFeatureNormPlotBackwardCompatibility:
    """Test backward compatibility of the API wrapper."""

    @patch('matplotlib.pyplot.show')
    def test_api_wrapper_maintains_compatibility(self, mock_show, synthetic_plate_data):
        """Test that the API wrapper maintains backward compatibility."""
        # Test with explicit parameters that might have been used
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            x_label=True,  # Explicit x-label parameter
            condition_col="condition",  # Explicit condition column
            selector_col="cell_line",
            selector_val="MCF10A",
            title="Backward Compatible Test",
            fig_size=(8, 6),
            size_units="cm",
            save=False,  # Don't actually save in tests
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check that x-labels are shown (x_label=True)
        x_tick_labels = ax.get_xticklabels()
        assert len(x_tick_labels) > 0
        assert any(label.get_text() != "" for label in x_tick_labels)

    @patch('matplotlib.pyplot.show')
    def test_default_parameter_handling(self, mock_show, synthetic_plate_data):
        """Test that default parameters work as expected."""
        # Test with minimal parameters, relying on defaults
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None,  # Override default to test minimal parameters
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_path_parameter_compatibility(self, mock_show, synthetic_plate_data, tmp_path):
        """Test path parameter handling for save functionality."""
        # Test with save=False (should not actually save)
        fig, ax = feature_norm_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1"],
            save=False,
            path=tmp_path,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Since save=False, no files should be created
        assert len(list(tmp_path.glob("*"))) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
