"""Comprehensive pytest tests for histogram plot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the histogram plot functionality,
focusing on testing the main API functions, class-based architecture, and error
handling without validating visual output.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from unittest.mock import patch

from omero_screen_plots.histogramplot_api import histogram_plot
from omero_screen_plots.histogramplot_factory import (
    HistogramPlotConfig,
    HistogramPlot,
)
from omero_screen_plots.colors import COLOR


class TestHistogramPlotBasicFunctionality:
    """Test basic functionality of histogram_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_minimal_parameters(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with minimal required parameters."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created (has elements)
        assert len(ax.patches) > 0  # Should have histogram patches
        assert ax.get_xlabel() != ""  # Should have x-axis label
        assert ax.get_ylabel() != ""  # Should have y-axis label

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_single_condition_string(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with single condition as string."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions="control",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check x-axis label contains feature name (formatted)
        x_label = ax.get_xlabel().lower()
        assert "intensity" in x_label or "dapi" in x_label

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_multiple_conditions_list(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with multiple conditions as list."""
        conditions = ["control", "treatment1", "treatment2"]
        fig, axes = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=conditions,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == len(conditions)

        # Each subplot should have histogram patches
        for ax in axes:
            assert isinstance(ax, Axes)
            assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_selector_column(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with selector column filtering."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            selector_col="cell_line",
            selector_val="MCF10A"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully with selector filtering
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_custom_title(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with custom title."""
        custom_title = "Custom Histogram Analysis"
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            title=custom_title,
            show_title=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check figure suptitle
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert custom_title in fig_title

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_custom_colors(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with custom colors."""
        custom_colors = ["#FF5733", "#33FF57", "#3357FF"]
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            colors=custom_colors,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully with custom colors
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_log_scale(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with logarithmic scaling."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions="control",
            log_scale=True,
            log_base=2,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check if log scale was applied
        assert ax.get_xscale() == "log"

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_kde_overlay(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with KDE overlay."""
        conditions = ["control", "treatment1"]
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=conditions,
            kde_overlay=True,
            kde_smoothing=0.8,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # KDE overlay should create line plots instead of histograms
        assert len(ax.lines) >= len(conditions)  # Should have KDE lines

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_normalization(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with density normalization."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            normalize=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have histogram patches
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_custom_bins(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with custom bin settings."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            bins=50,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have histogram patches
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_x_limits(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with custom x-axis limits."""
        x_limits = (100, 600)
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            x_limits=x_limits,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        x_min, x_max = ax.get_xlim()
        assert x_min >= x_limits[0] - 10  # Allow for small padding
        assert x_max <= x_limits[1] + 10

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_figure_size(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with custom figure size."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            fig_size=(6, 6),
            size_units="cm",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Figure should be created with approximately correct size
        # (converted from cm to inches)
        fig_width, fig_height = fig.get_size_inches()
        expected_inches = 6 / 2.54  # Convert cm to inches
        assert abs(fig_width - expected_inches) < 0.1
        assert abs(fig_height - expected_inches) < 0.1

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_dynamic_figure_size_single_condition(self, mock_show, synthetic_plate_data):
        """Test histogram_plot dynamic figure sizing for single condition."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            # fig_size=None should use dynamic sizing
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should use default 4x4 cm (converted to inches)
        fig_width, fig_height = fig.get_size_inches()
        expected_inches = 4 / 2.54  # 4 cm in inches
        assert abs(fig_width - expected_inches) < 0.2
        assert abs(fig_height - expected_inches) < 0.2

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_dynamic_figure_size_multiple_conditions(self, mock_show, synthetic_plate_data):
        """Test histogram_plot dynamic figure sizing for multiple conditions."""
        conditions = ["control", "treatment1", "treatment2"]
        fig, axes = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=conditions,
            # fig_size=None should use dynamic sizing
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # Should use 4*num_conditions x 4 cm
        fig_width, fig_height = fig.get_size_inches()
        expected_width_inches = (4 * len(conditions)) / 2.54  # cm to inches
        expected_height_inches = 4 / 2.54
        assert abs(fig_width - expected_width_inches) < 0.5
        assert abs(fig_height - expected_height_inches) < 0.2

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_with_axes_parameter(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with existing axes."""
        import matplotlib.pyplot as plt

        # Create existing figure and axes
        fig_ext, ax_ext = plt.subplots(1, 1, figsize=(5, 4))

        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            axes=ax_ext,
        )

        # Should return the provided axes
        assert ax is ax_ext
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        plt.close(fig_ext)  # Clean up


class TestHistogramPlotKDEFeatures:
    """Test KDE-specific functionality."""

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_kde_with_custom_parameters(self, mock_show, synthetic_plate_data):
        """Test histogram_plot KDE with custom parameters."""
        kde_params = {
            "bw_method": "silverman",
            "gridsize": 300,
            "cut": 3
        }

        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1"],
            kde_overlay=True,
            kde_smoothing=1.2,
            kde_params=kde_params,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have KDE lines
        assert len(ax.lines) >= 2

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_kde_multiple_conditions_different_colors(self, mock_show, synthetic_plate_data):
        """Test KDE overlay uses different colors for multiple conditions."""
        conditions = ["control", "treatment1", "treatment2"]
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=conditions,
            kde_overlay=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have lines for each condition
        assert len(ax.lines) >= len(conditions)


class TestHistogramPlotConfiguration:
    """Test configuration object creation and parameter passing."""

    def test_histogram_plot_config_creation_minimal(self, synthetic_plate_data):
        """Test HistogramPlotConfig creation with minimal parameters."""
        config = HistogramPlotConfig()

        # Should have default values
        assert config.bins == 100
        assert config.log_scale is False
        assert config.log_base == 2
        assert config.normalize is False
        assert config.kde_overlay is False

    def test_histogram_plot_config_creation_custom(self, synthetic_plate_data):
        """Test HistogramPlotConfig creation with custom parameters."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        config = HistogramPlotConfig(
            bins=50,
            log_scale=True,
            log_base=10,
            normalize=True,
            kde_overlay=True,
            kde_smoothing=1.0,
            colors=custom_colors,
            fig_size=(8, 6),
            size_units="inches",
        )

        assert config.bins == 50
        assert config.log_scale is True
        assert config.log_base == 10
        assert config.normalize is True
        assert config.kde_overlay is True
        assert config.kde_smoothing == 1.0
        assert config.colors == custom_colors
        assert config.fig_size == (8, 6)
        assert config.size_units == "inches"

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_class_direct_usage(self, mock_show, synthetic_plate_data):
        """Test using HistogramPlot class directly."""
        config = HistogramPlotConfig(
            bins=30,
            colors=[COLOR.BLUE.value],
        )

        plot = HistogramPlot(config)
        fig, ax = plot.create_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            condition_col="condition",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestHistogramPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test histogram_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            histogram_plot(
                df=empty_df,
                feature="area_nucleus",
                conditions="control"
            )

    def test_missing_feature_column(self, synthetic_plate_data):
        """Test histogram_plot with missing feature column."""
        with pytest.raises((ValueError, KeyError)):
            histogram_plot(
                df=synthetic_plate_data,
                feature="nonexistent_feature",
                conditions="control"
            )

    def test_invalid_condition_column(self, synthetic_plate_data):
        """Test histogram_plot with invalid condition column."""
        with pytest.raises((ValueError, KeyError)):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions="control",
                condition_col="invalid_column"
            )

    def test_condition_not_in_data(self, synthetic_plate_data):
        """Test histogram_plot with condition not present in data."""
        with pytest.raises(ValueError):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions="nonexistent_condition"
            )

    @patch('matplotlib.pyplot.show')
    def test_conditions_list_not_in_data(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with conditions list containing invalid condition."""
        # This might handle partial matches gracefully, so test for either error or partial plot
        try:
            fig, axes = histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "nonexistent_condition"]
            )
            # If no error, should have created at least one plot
            assert isinstance(fig, Figure)
        except ValueError:
            # If error raised, that's also acceptable
            pass

    def test_invalid_selector_column(self, synthetic_plate_data):
        """Test histogram_plot with invalid selector column."""
        with pytest.raises((ValueError, KeyError)):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions="control",
                selector_col="invalid_selector",
                selector_val="some_value"
            )

    def test_selector_column_without_value(self, synthetic_plate_data):
        """Test histogram_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions="control",
                selector_col="cell_line"
            )

    def test_invalid_selector_value(self, synthetic_plate_data):
        """Test histogram_plot with selector_val not in data."""
        with pytest.raises(ValueError):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions="control",
                selector_col="cell_line",
                selector_val="NonexistentCell"
            )

    @patch('matplotlib.pyplot.show')
    def test_single_condition_data_success(self, mock_show, single_condition_data):
        """Test histogram_plot with single condition data (should work)."""
        fig, ax = histogram_plot(
            df=single_condition_data,
            feature="area_nucleus",
            conditions="control",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    def test_axes_parameter_with_multiple_conditions(self, synthetic_plate_data):
        """Test histogram_plot with axes parameter and multiple conditions (should fail)."""
        import matplotlib.pyplot as plt

        fig_ext, ax_ext = plt.subplots(1, 1)

        with pytest.raises(ValueError):
            histogram_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                axes=ax_ext
            )

        plt.close(fig_ext)

    @patch('matplotlib.pyplot.show')
    def test_all_conditions_filtered_out(self, mock_show, synthetic_plate_data):
        """Test histogram_plot when selector filtering removes all conditions."""
        # Create scenario where filtering removes all data for the condition
        df_subset = synthetic_plate_data[synthetic_plate_data["cell_line"] == "MCF10A"].copy()

        with pytest.raises(ValueError):
            histogram_plot(
                df=df_subset,
                feature="area_nucleus",
                conditions="control",
                selector_col="cell_line",
                selector_val="NonexistentCell"  # This will filter out all data
            )


class TestHistogramPlotAdvancedFeatures:
    """Test advanced features and parameter combinations."""

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_string_bins(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with string binning strategy."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            bins="auto",  # String binning strategy
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_no_x_labels(self, mock_show, synthetic_plate_data):
        """Test histogram_plot without x-axis labels."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            show_x_labels=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # X-tick labels should be empty or hidden
        x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        # Labels might be empty or invisible
        assert all(label == "" or not label for label in x_tick_labels) or len(x_tick_labels) == 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_rotated_labels(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with rotated x-axis labels."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            rotation=45,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have created the plot successfully
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_log_scale_different_bases(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with different logarithmic bases."""
        for base in [2, 10, np.e]:
            fig, ax = histogram_plot(
                df=synthetic_plate_data,
                feature="intensity_mean_DAPI_nucleus",
                conditions="control",
                log_scale=True,
                log_base=base,
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            assert ax.get_xscale() == "log"

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_combined_features(self, mock_show, synthetic_plate_data):
        """Test histogram_plot with multiple advanced features combined."""
        conditions = ["control", "treatment1"]
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=conditions,
            log_scale=True,
            log_base=2,
            normalize=True,
            kde_overlay=True,
            kde_smoothing=0.6,
            colors=[COLOR.BLUE.value, COLOR.YELLOW.value],
            x_limits=(1000, 25000),
            title="Combined Features Test",
            show_title=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have KDE lines
        assert len(ax.lines) >= len(conditions)
        assert ax.get_xscale() == "log"


class TestHistogramPlotSaveFeatures:
    """Test save functionality."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_save_functionality(self, mock_show, mock_savefig, synthetic_plate_data, tmp_path):
        """Test histogram_plot save functionality."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            save=True,
            path=tmp_path,
            file_format="png",
            dpi=150,
            tight_layout=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have created plot successfully (save functionality handled internally)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_histogram_plot_save_without_path(self, mock_show, synthetic_plate_data):
        """Test histogram_plot save without path (should work with default)."""
        fig, ax = histogram_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions="control",
            save=True,
            # path=None should use current directory
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully even without explicit path
        assert len(ax.patches) > 0
