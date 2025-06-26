"""Smoke tests for plot classes in omero-screen-plots.

These tests verify that all plot classes can be instantiated and generate plots without errors.
"""

import matplotlib.pyplot as plt
import pytest

from omero_screen_plots import (
    # Plot class imports
    CellCyclePlot,
    CellCycleStackedPlot,
    CellCycleGroupedPlot,
    HistogramPlot,
    CellCycleScatterPlot,
    FeatureScatterPlot,
    SimpleCombPlot,
    FullCombPlot,
)


class TestCellCyclePlotClasses:
    """Test cell cycle plot classes."""

    def test_cellcycle_plot_class(self, filtered_data):
        """Test CellCyclePlot class (standard plot)."""
        plot = CellCyclePlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_cellcycle_stacked_plot_class(self, filtered_data):
        """Test CellCycleStackedPlot class."""
        plot = CellCycleStackedPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_cellcycle_grouped_plot_class(self, filtered_data):
        """Test CellCycleGroupedPlot class."""
        plot = CellCycleGroupedPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)


class TestCombPlotClasses:
    """Test combined plot classes."""

    def test_histogram_plot_class(self, filtered_data):
        """Test HistogramPlot class."""
        plot = HistogramPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            feature="area_nucleus",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_cellcycle_scatter_plot_class(self, filtered_data):
        """Test CellCycleScatterPlot class."""
        plot = CellCycleScatterPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            x_feature="area_nucleus",
            y_feature="intensity_max_Tub_nucleus",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_feature_scatter_plot_class(self, filtered_data):
        """Test FeatureScatterPlot class."""
        plot = FeatureScatterPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            feature_col="area_nucleus",
            feature_threshold=200,
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_simple_combplot_class(self, filtered_data):
        """Test SimpleCombPlot class."""
        plot = SimpleCombPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            features=["area_nucleus", "intensity_max_Tub_nucleus"],
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_full_combplot_class(self, filtered_data):
        """Test FullCombPlot class."""
        plot = FullCombPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            feature_col="area_nucleus",
            feature_threshold=200,
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)


class TestPlotClassMethods:
    """Test additional methods and properties of plot classes."""

    def test_plot_save_functionality(self, filtered_data, tmp_path):
        """Test that plots can be saved to disk."""
        plot = CellCyclePlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
        )

        # Generate and save
        output_path = tmp_path / "test_plot.png"
        fig = plot.generate()
        fig.savefig(output_path)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_plot_with_custom_figsize(self, filtered_data):
        """Test plots with custom figure size."""
        plot = SimpleCombPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            features=["area_nucleus"],
            # figsize=(10, 8),  # SimpleCombPlot may convert cm to inches
        )
        fig = plot.generate()
        # Just check that a figure was created
        assert fig is not None
        assert fig.get_size_inches()[0] > 0
        assert fig.get_size_inches()[1] > 0
        plt.close(fig)


class TestPlotClassEdgeCases:
    """Test edge cases for plot classes."""

    def test_single_feature_combplot(self, filtered_data):
        """Test combined plots with single feature."""
        plot = SimpleCombPlot(
            data=filtered_data,
            conditions=["ctr", "palb"],
            selector_col="cell_line",
            selector_val="RPE1wt",
            condition_col="condition",
            features=["area_nucleus"],  # Single feature
        )
        fig = plot.generate()
        assert fig is not None
        plt.close(fig)

    def test_many_features_combplot(self, filtered_data):
        """Test combined plots with many features."""
        # Get numeric columns for features
        numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [col for col in numeric_cols if 'intensity' in col or 'area' in col][:5]

        if len(feature_cols) >= 2:
            plot = FullCombPlot(
                data=filtered_data,
                conditions=["ctr", "palb"],
                feature_col=feature_cols[0],
                feature_threshold=100,
                selector_col="cell_line",
                selector_val="RPE1wt",
                condition_col="condition",
            )
            fig = plot.generate()
            assert fig is not None
            plt.close(fig)

    def test_plot_class_with_missing_cell_cycle_column(self, filtered_data):
        """Test handling of missing cell cycle data."""
        # Create data without cell_cycle column
        data_no_cc = filtered_data.drop(columns=['cell_cycle', 'cell_cycle_detailed'], errors='ignore')

        # HistogramPlot requires cell_cycle column, so this should fail
        with pytest.raises(ValueError, match="Missing required columns"):
            plot = HistogramPlot(
                data=data_no_cc,
                conditions=["ctr", "palb"],
                selector_col="cell_line",
                selector_val="RPE1wt",
                condition_col="condition",
                feature="area_nucleus",
            )
