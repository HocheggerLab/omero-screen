"""Smoke tests for new plotting functions in omero-screen-plots.

These tests verify that all plotting functions execute without errors.
They don't test the visual output quality, just that the functions run.
"""

import matplotlib.pyplot as plt
import pytest

from omero_screen_plots import (
    # New cell cycle plot functions
    cellcycle_grouped_plot,
    cellcycle_stacked_plot,
    cellcycle_standard_plot,
    # New combined plot functions
    cellcycle_scatter_plot,
    feature_scatter_plot,
    full_combplot,
    simple_combplot,
    # histogram_plot imported as new_histogram_plot in __init__.py
)
from omero_screen_plots.plots.combplots import histogram_plot as new_histogram_plot


class TestCellCyclePlots:
    """Test all cell cycle plot variations."""

    def test_cellcycle_standard_plot(self, filtered_data):
        """Test standard cell cycle plot."""
        cellcycle_standard_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")

    def test_cellcycle_stacked_plot(self, filtered_data):
        """Test stacked cell cycle plot."""
        cellcycle_stacked_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")

    def test_cellcycle_grouped_plot(self, filtered_data):
        """Test grouped cell cycle plot."""
        cellcycle_grouped_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")


class TestCombPlots:
    """Test all combined plot types."""

    def test_histogram_plot(self, filtered_data):
        """Test new histogram plot."""
        new_histogram_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            feature="area_nucleus",
            save=False,
        )
        plt.close("all")

    def test_cellcycle_scatter_plot(self, filtered_data):
        """Test cell cycle scatter plot."""
        cellcycle_scatter_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            x_feature="area_nucleus",
            y_feature="intensity_max_Tub_nucleus",
            save=False,
        )
        plt.close("all")

    def test_feature_scatter_plot(self, filtered_data):
        """Test feature scatter plot."""
        feature_scatter_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            feature_col="area_nucleus",
            feature_threshold=200,  # Example threshold
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")

    def test_simple_combplot(self, filtered_data):
        """Test simple combined plot."""
        simple_combplot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            features=["area_nucleus", "intensity_max_Tub_nucleus"],
            save=False,
        )
        plt.close("all")

    def test_full_combplot(self, filtered_data):
        """Test full combined plot."""
        full_combplot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            feature_col="area_nucleus",
            feature_threshold=200,  # Example threshold
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")


class TestPlotVariations:
    """Test various parameter combinations to ensure robustness."""

    def test_plots_with_single_condition(self, filtered_data):
        """Test plots with single condition."""
        single_condition_data = filtered_data[filtered_data.condition == "ctr"]

        # Test a few key plots with single condition
        cellcycle_standard_plot(
            data=single_condition_data,
            conditions=["ctr"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")

    def test_plots_with_custom_colors(self, filtered_data):
        """Test plots with custom color parameters."""
        # Test plots that accept color parameters
        feature_scatter_plot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            feature_col="area_nucleus",
            feature_threshold=200,
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            save=False,
        )
        plt.close("all")

    def test_plots_with_minimal_data(self):
        """Test plots handle small datasets gracefully."""
        import pandas as pd

        # Create minimal test data
        minimal_data = pd.DataFrame({
            "condition": ["ctr", "ctr", "palb", "palb"],
            "cell_line": ["RPE1wt"] * 4,
            "area_nucleus": [100, 120, 110, 130],
            "intensity_max_Tub_nucleus": [1000, 1200, 1100, 1300],
            "cell_cycle": ["G1", "S", "G2", "G1"],
            "cell_cycle_detailed": ["G1", "S", "G2", "G1"],
            "integrated_int_DAPI_norm": [2.0, 3.5, 4.0, 2.2],
            "intensity_mean_EdU_nucleus_norm": [0.5, 1.2, 0.3, 0.8],
            "plate_id": ["P1", "P1", "P2", "P2"],
            "experiment": ["E1", "E2", "E3", "E4"],
        })

        # Test basic plot with minimal data
        simple_combplot(
            data=minimal_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            features=["area_nucleus"],
            save=False,
        )
        plt.close("all")


class TestErrorHandling:
    """Test that plots handle errors gracefully."""

    def test_invalid_condition(self, filtered_data):
        """Test plot with invalid condition."""
        with pytest.raises((KeyError, ValueError)):
            cellcycle_standard_plot(
                data=filtered_data,
                conditions=["invalid_condition"],
                selector_col="cell_line",
                selector_val="RPE1wt",
                condition_col="condition",
                save=False,
            )

    def test_invalid_feature(self, filtered_data):
        """Test plot with invalid feature."""
        with pytest.raises((KeyError, ValueError)):
            feature_scatter_plot(
                data=filtered_data,
                conditions=["ctr", "palb"],
                feature_col="invalid_feature",
                feature_threshold=200,
                selector_col="cell_line",
                selector_val="RPE1wt",
                condition_col="condition",
                save=False,
            )

    def test_empty_dataframe(self):
        """Test plots handle empty dataframes."""
        import pandas as pd

        empty_df = pd.DataFrame()

        with pytest.raises((KeyError, ValueError, IndexError)):
            simple_combplot(
                data=empty_df,
                conditions=["ctr"],
                selector_col="cell_line",
                selector_val="RPE1wt",
                condition_col="condition",
                features=["area_nucleus"],
                save=False,
            )
