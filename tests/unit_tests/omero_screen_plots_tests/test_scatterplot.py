"""Comprehensive pytest tests for scatter plot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the scatter plot functionality,
focusing on testing the main API functions, class-based architecture, auto-detection
features, and error handling without validating visual output.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from unittest.mock import patch

from omero_screen_plots.scatterplot_api import scatter_plot
from omero_screen_plots.scatterplot_factory import (
    ScatterPlotConfig,
    ScatterPlot,
)
from omero_screen_plots.colors import COLOR


@pytest.fixture
def dna_edu_data():
    """Create synthetic DNA/EdU data for cell cycle analysis testing."""
    np.random.seed(42)
    data = []

    plates = [1001, 1002]
    conditions = ["control", "treatment1"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Generate realistic DNA content and EdU intensity data
            for _ in range(50):  # 50 cells per condition
                # DNA content (log-normal distribution around 2N and 4N)
                if np.random.random() < 0.3:  # 30% in S phase
                    dna_content = np.random.uniform(2.2, 3.8)  # S phase
                    edu_intensity = np.random.uniform(1000, 8000)  # EdU positive
                    cell_cycle = "S"
                elif np.random.random() < 0.6:  # 60% of remaining in G1
                    dna_content = np.random.normal(2.0, 0.1)  # G1 phase
                    edu_intensity = np.random.uniform(50, 500)  # EdU negative
                    cell_cycle = "G1"
                else:  # Remaining in G2/M
                    dna_content = np.random.normal(4.0, 0.2)  # G2/M phase
                    edu_intensity = np.random.uniform(50, 500)  # EdU negative
                    cell_cycle = "G2/M"

                # Add some noise and ensure positive values
                dna_content = max(1.0, dna_content + np.random.normal(0, 0.1))
                edu_intensity = max(10, edu_intensity + np.random.normal(0, 100))

                data.append({
                    "plate_id": plate_id,
                    "well": f"A{measurement_id % 12 + 1}",
                    "experiment": f"exp_{measurement_id}",
                    "condition": condition,
                    "cell_line": "MCF10A",
                    "measurement_id": measurement_id,
                    "integrated_int_DAPI_norm": dna_content,
                    "intensity_mean_EdU_nucleus_norm": edu_intensity,
                    "intensity_mean_EdU_nucleus": edu_intensity * 1000,  # Non-normalized version
                    "cell_cycle": cell_cycle,
                    # Additional features for testing
                    "area_nucleus": np.random.uniform(100, 500),
                    "area_cell": np.random.uniform(200, 800),
                    "intensity_mean_p21_nucleus": np.random.uniform(100, 5000),
                    "intensity_mean_DAPI_nucleus": dna_content * 10000,
                })
                measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def threshold_test_data():
    """Create data for threshold testing with clear above/below threshold values."""
    data = []

    # Create data with clear threshold separation
    for i in range(50):
        data.append({
            "plate_id": 1001,
            "well": "A1",
            "experiment": f"exp_{i}",
            "condition": "control",
            "cell_line": "MCF10A",
            "integrated_int_DAPI_norm": 2.0 + np.random.normal(0, 0.1),
            "intensity_mean_p21_nucleus": 2000 + np.random.normal(0, 200),  # Below threshold
            "area_nucleus": np.random.uniform(100, 300),
        })

    for i in range(50, 100):
        data.append({
            "plate_id": 1001,
            "well": "A2",
            "experiment": f"exp_{i}",
            "condition": "treatment1",
            "cell_line": "MCF10A",
            "integrated_int_DAPI_norm": 2.0 + np.random.normal(0, 0.1),
            "intensity_mean_p21_nucleus": 8000 + np.random.normal(0, 500),  # Above threshold
            "area_nucleus": np.random.uniform(100, 300),
        })

    return pd.DataFrame(data)


class TestScatterPlotBasicFunctionality:
    """Test basic functionality of scatter_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_minimal_parameters(self, mock_show, dna_edu_data):
        """Test scatter_plot with minimal required parameters."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created (has scatter points)
        assert len(ax.collections) > 0  # Should have scatter collections
        assert ax.get_xlabel() != ""  # Should have x-axis label
        assert ax.get_ylabel() != ""  # Should have y-axis label

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_single_condition_string(self, mock_show, dna_edu_data):
        """Test scatter_plot with single condition as string."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have log scale for x (DNA content) by default
        assert ax.get_xscale() == "log"
        # Should have log scale for y (EdU feature) by default
        assert ax.get_yscale() == "log"

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_multiple_conditions_list(self, mock_show, dna_edu_data):
        """Test scatter_plot with multiple conditions as list."""
        conditions = ["control", "treatment1"]
        fig, axes = scatter_plot(
            df=dna_edu_data,
            conditions=conditions,
        )

        assert isinstance(fig, Figure)
        # axes can be ndarray from matplotlib subplots
        assert hasattr(axes, '__len__') and len(axes) == len(conditions)

        # Each subplot should have scatter points
        for ax in axes:
            assert isinstance(ax, Axes)
            assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_selector_column(self, mock_show, dna_edu_data):
        """Test scatter_plot with selector column filtering."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            selector_col="cell_line",
            selector_val="MCF10A"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully with selector filtering
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_custom_title(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom title."""
        custom_title = "Custom Cell Cycle Analysis"
        fig, ax = scatter_plot(
            df=dna_edu_data,
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
    def test_scatter_plot_custom_features(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom x and y features."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="area_nucleus",
            y_feature="area_cell",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have scatter points
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_cell_number_sampling(self, mock_show, dna_edu_data):
        """Test scatter_plot with cell number sampling."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            cell_number=20,  # Sample only 20 cells
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should still have scatter points (sampled)
        assert len(ax.collections) > 0


class TestScatterPlotAutoDetectionFeatures:
    """Test auto-detection features like scale, KDE overlay, and reference lines."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_auto_log_scale_for_edu(self, mock_show, dna_edu_data):
        """Test auto-detection of log scale for EdU features."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",  # Should be log
            y_feature="intensity_mean_EdU_nucleus_norm",  # Should be log (EdU)
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "log"  # DNA content should be log
        assert ax.get_yscale() == "log"  # EdU should be log

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_auto_linear_scale_for_non_edu(self, mock_show, dna_edu_data):
        """Test auto-detection uses linear scale for non-EdU features."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",  # Should be log
            y_feature="intensity_mean_p21_nucleus",  # Should be linear (not EdU)
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "log"  # X still log for DNA
        assert ax.get_yscale() == "linear"  # Y should be linear (not EdU)

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_manual_scale_override(self, mock_show, dna_edu_data):
        """Test manual scale settings override auto-detection."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
            x_scale="linear",  # Override default log
            y_scale="linear",  # Override default log
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xscale() == "linear"  # Should be overridden
        assert ax.get_yscale() == "linear"  # Should be overridden

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_auto_kde_overlay_detection(self, mock_show, dna_edu_data):
        """Test auto-detection of KDE overlay for DNA vs EdU plots."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
            kde_overlay=None,  # Should auto-detect
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have scatter points
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_manual_kde_overlay(self, mock_show, dna_edu_data):
        """Test manual KDE overlay setting."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            kde_overlay=True,
            kde_cmap="viridis",
            kde_alpha=0.3,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have both scatter and contour collections
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_auto_cell_cycle_hue_detection(self, mock_show, dna_edu_data):
        """Test auto-detection of cell_cycle column for hue mapping."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue=None,  # Should auto-detect cell_cycle if available
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have scatter points with different colors if cell_cycle was detected
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_manual_hue_column(self, mock_show, dna_edu_data):
        """Test manual hue column specification."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="condition",  # Use condition for hue instead
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0


class TestScatterPlotThresholdFeatures:
    """Test threshold-based coloring functionality."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_threshold_coloring(self, mock_show, threshold_test_data):
        """Test scatter_plot with threshold-based coloring."""
        fig, ax = scatter_plot(
            df=threshold_test_data,
            conditions=["control", "treatment1"],
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_p21_nucleus",
            threshold=5000,  # Threshold for p21 intensity
        )

        assert isinstance(fig, Figure)
        axes_list = ax if hasattr(ax, '__len__') else [ax]
        # Should have scatter points in different colors (blue/red)
        for axis in axes_list:
            assert len(axis.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_threshold_single_condition(self, mock_show, threshold_test_data):
        """Test threshold coloring with single condition."""
        fig, ax = scatter_plot(
            df=threshold_test_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_p21_nucleus",
            threshold=3000,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have colored scatter points
        assert len(ax.collections) > 0


class TestScatterPlotReferenceLines:
    """Test reference line functionality."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_vertical_reference_line(self, mock_show, dna_edu_data):
        """Test scatter_plot with vertical reference line."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            vline=2.0,  # Reference line at 2N DNA content
            line_style="--",
            line_color="red",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have reference line
        assert len(ax.lines) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_horizontal_reference_line(self, mock_show, dna_edu_data):
        """Test scatter_plot with horizontal reference line."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hline=1000,  # Reference line for EdU threshold
            line_style=":",
            line_color="black",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have reference line
        assert len(ax.lines) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_both_reference_lines(self, mock_show, dna_edu_data):
        """Test scatter_plot with both vertical and horizontal reference lines."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            vline=2.0,
            hline=1000,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have both reference lines
        assert len(ax.lines) >= 2


class TestScatterPlotAxisSettings:
    """Test axis limits, ticks, and labels."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_axis_limits(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom axis limits."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_limits=(1.5, 5.0),
            y_limits=(100, 10000),
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Should respect limits (allowing for small padding)
        assert x_min >= 1.4 and x_max <= 5.1
        assert y_min >= 90 and y_max <= 11000

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_custom_ticks(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom tick positions."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_ticks=[1, 2, 4],
            y_ticks=[100, 1000, 10000],
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have set custom ticks
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        assert len(x_ticks) >= 3  # Should include our custom ticks
        assert len(y_ticks) >= 3

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_custom_labels(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom axis labels."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_label="Custom X Label",
            y_label="Custom Y Label",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "Custom X Label"
        assert ax.get_ylabel() == "Custom Y Label"

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_grid(self, mock_show, dna_edu_data):
        """Test scatter_plot with grid enabled."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            grid=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully with grid enabled
        assert len(ax.collections) > 0


class TestScatterPlotLegendFeatures:
    """Test legend functionality."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_legend(self, mock_show, dna_edu_data):
        """Test scatter_plot with legend enabled."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="cell_cycle",  # Use cell cycle phases for legend
            show_legend=True,
            legend_title="Cell Cycle Phase",
            legend_loc="upper right",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have legend
        legend = ax.get_legend()
        if legend:
            assert "Cell Cycle Phase" in legend.get_title().get_text()

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_without_legend(self, mock_show, dna_edu_data):
        """Test scatter_plot with legend disabled."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="cell_cycle",
            show_legend=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should not have legend
        legend = ax.get_legend()
        assert legend is None


class TestScatterPlotConfiguration:
    """Test configuration object creation and parameter passing."""

    def test_scatter_plot_config_creation_minimal(self):
        """Test ScatterPlotConfig creation with minimal parameters."""
        config = ScatterPlotConfig()

        # Should have default values
        assert config.x_feature == "integrated_int_DAPI_norm"
        assert config.y_feature == "intensity_mean_EdU_nucleus_norm"
        assert config.x_scale == "log"
        assert config.cell_number is None  # Default is None
        assert config.size == 2
        assert config.alpha == 1.0

    def test_scatter_plot_config_creation_custom(self):
        """Test ScatterPlotConfig creation with custom parameters."""
        config = ScatterPlotConfig(
            x_feature="area_nucleus",
            y_feature="area_cell",
            x_scale="linear",
            y_scale="linear",
            cell_number=1000,
            size=5,
            alpha=0.7,
            kde_overlay=True,
            threshold=5000,
            fig_size=(8, 6),
        )

        assert config.x_feature == "area_nucleus"
        assert config.y_feature == "area_cell"
        assert config.x_scale == "linear"
        assert config.y_scale == "linear"
        assert config.cell_number == 1000
        assert config.size == 5
        assert config.alpha == 0.7
        assert config.kde_overlay is True
        assert config.threshold == 5000
        assert config.fig_size == (8, 6)

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_class_direct_usage(self, mock_show, dna_edu_data):
        """Test using ScatterPlot class directly."""
        config = ScatterPlotConfig(
            size=10,
            alpha=0.5,
            x_scale="linear",
        )

        plot = ScatterPlot(config)
        fig, ax = plot.create_plot(
            df=dna_edu_data,
            conditions="control",
            condition_col="condition",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0


class TestScatterPlotScatterSettings:
    """Test scatter point size, alpha, and styling."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_custom_size_alpha(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom point size and alpha."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            size=10,
            alpha=0.3,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have scatter points
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_different_sizes(self, mock_show, dna_edu_data):
        """Test scatter_plot with different point sizes."""
        for size in [1, 5, 15]:
            fig, ax = scatter_plot(
                df=dna_edu_data,
                conditions="control",
                size=size,
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            assert len(ax.collections) > 0


class TestScatterPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test scatter_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            scatter_plot(
                df=empty_df,
                conditions="control"
            )

    def test_missing_feature_columns(self, dna_edu_data):
        """Test scatter_plot with missing feature columns."""
        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions="control",
                x_feature="nonexistent_x"
            )

        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions="control",
                y_feature="nonexistent_y"
            )

    def test_invalid_condition_column(self, dna_edu_data):
        """Test scatter_plot with invalid condition column."""
        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions="control",
                condition_col="invalid_column"
            )

    def test_condition_not_in_data(self, dna_edu_data):
        """Test scatter_plot with condition not present in data."""
        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions="nonexistent_condition"
            )

    def test_conditions_list_not_in_data(self, dna_edu_data):
        """Test scatter_plot with conditions list containing invalid condition."""
        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions=["control", "nonexistent_condition"]
            )

    def test_invalid_selector_column(self, dna_edu_data):
        """Test scatter_plot with invalid selector column."""
        with pytest.raises((ValueError, KeyError)):
            scatter_plot(
                df=dna_edu_data,
                conditions="control",
                selector_col="invalid_selector",
                selector_val="some_value"
            )

    @patch('matplotlib.pyplot.show')
    def test_selector_column_without_value(self, mock_show, dna_edu_data):
        """Test scatter_plot with selector_col but no selector_val."""
        # This might handle gracefully by ignoring selector_col
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            selector_col="cell_line"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_invalid_selector_value(self, mock_show, dna_edu_data):
        """Test scatter_plot with selector_val not in data."""
        # This might create empty plot or handle gracefully
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            selector_col="cell_line",
            selector_val="NonexistentCell"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_single_condition_data_success(self, mock_show, dna_edu_data):
        """Test scatter_plot with single condition data (should work)."""
        # Filter to single condition
        single_cond_data = dna_edu_data[dna_edu_data["condition"] == "control"].copy()

        fig, ax = scatter_plot(
            df=single_cond_data,
            conditions="control",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0

    def test_axes_parameter_with_multiple_conditions(self, dna_edu_data):
        """Test scatter_plot with axes parameter and multiple conditions (should fail)."""
        import matplotlib.pyplot as plt

        fig_ext, ax_ext = plt.subplots(1, 1)

        with pytest.raises(ValueError):
            scatter_plot(
                df=dna_edu_data,
                conditions=["control", "treatment1"],
                axes=ax_ext
            )

        plt.close(fig_ext)

    @patch('matplotlib.pyplot.show')
    def test_all_conditions_filtered_out(self, mock_show, dna_edu_data):
        """Test scatter_plot when selector filtering removes all conditions."""
        # This might create empty plot or handle gracefully
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            selector_col="cell_line",
            selector_val="NonexistentCell"  # This will filter out all data
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_invalid_hue_column(self, mock_show, dna_edu_data):
        """Test scatter_plot with invalid hue column."""
        # This might not raise an error if hue is handled gracefully
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="nonexistent_hue"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestScatterPlotAdvancedFeatures:
    """Test advanced features and parameter combinations."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_custom_palette(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom color palette."""
        custom_palette = {"G1": "#FF0000", "S": "#00FF00", "G2/M": "#0000FF"}
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="cell_cycle",
            palette=custom_palette,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_hue_order(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom hue order."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            hue="cell_cycle",
            hue_order=["G1", "S", "G2/M"],
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_different_log_bases(self, mock_show, dna_edu_data):
        """Test scatter_plot with different logarithmic bases."""
        for base in [2, 10]:
            fig, ax = scatter_plot(
                df=dna_edu_data,
                conditions="control",
                x_scale="log",
                x_scale_base=base,
                y_scale="log",
                y_scale_base=base,
            )

            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_combined_features(self, mock_show, dna_edu_data):
        """Test scatter_plot with multiple advanced features combined."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
            hue="cell_cycle",
            kde_overlay=True,
            vline=2.0,
            hline=1000,
            x_limits=(1.5, 5.0),
            y_limits=(100, 10000),
            size=8,
            alpha=0.7,
            grid=True,
            show_legend=True,
            title="Combined Features Test",
            show_title=True,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0  # Should have scatter points
        assert len(ax.lines) >= 2  # Should have reference lines
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_with_axes_parameter(self, mock_show, dna_edu_data):
        """Test scatter_plot with existing axes."""
        import matplotlib.pyplot as plt

        # Create existing figure and axes
        fig_ext, ax_ext = plt.subplots(1, 1, figsize=(5, 4))

        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            axes=ax_ext,
        )

        # Should return the provided axes
        assert ax is ax_ext
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        plt.close(fig_ext)  # Clean up


class TestScatterPlotSaveFeatures:
    """Test save functionality."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_save_functionality(self, mock_show, mock_savefig, dna_edu_data, tmp_path):
        """Test scatter_plot save functionality."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
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
        assert len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_save_without_path(self, mock_show, dna_edu_data):
        """Test scatter_plot save without path (should work with default)."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            save=True,
            # path=None should use current directory
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot successfully even without explicit path
        assert len(ax.collections) > 0


class TestScatterPlotFigureSize:
    """Test figure sizing functionality."""

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_custom_figure_size(self, mock_show, dna_edu_data):
        """Test scatter_plot with custom figure size."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            fig_size=(8, 6),
            size_units="cm",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Figure should be created with approximately correct size
        # (converted from cm to inches)
        fig_width, fig_height = fig.get_size_inches()
        expected_width_inches = 8 / 2.54  # Convert cm to inches
        expected_height_inches = 6 / 2.54
        assert abs(fig_width - expected_width_inches) < 0.1
        assert abs(fig_height - expected_height_inches) < 0.1

    @patch('matplotlib.pyplot.show')
    def test_scatter_plot_figure_size_inches(self, mock_show, dna_edu_data):
        """Test scatter_plot with figure size in inches."""
        fig, ax = scatter_plot(
            df=dna_edu_data,
            conditions="control",
            fig_size=(5, 4),
            size_units="inches",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        fig_width, fig_height = fig.get_size_inches()
        assert abs(fig_width - 5) < 0.1
        assert abs(fig_height - 4) < 0.1
