"""Comprehensive pytest tests for classificationplot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the classification plot functionality,
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

from omero_screen_plots.classificationplot_api import classification_plot
from omero_screen_plots.classificationplot_factory import (
    ClassificationPlotConfig,
    ClassificationDataProcessor,
    ClassificationPlotBuilder,
)
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import COLORS


@pytest.fixture
def classification_data():
    """Create synthetic classification dataset with realistic class distributions.

    Creates a comprehensive dataset with:
    - 3 plates (1001, 1002, 1003)
    - 3 conditions (control, treatment1, treatment2)
    - Multiple wells per plate/condition combination
    - Classification classes: normal, micronuclei, collapsed
    - Realistic class distribution patterns
    """
    np.random.seed(42)  # For reproducible results

    data = []
    plates = [1001, 1002, 1003]
    conditions = ["control", "treatment1", "treatment2"]
    classes = ["normal", "micronuclei", "collapsed"]
    # Realistic class distributions (normal most common, collapsed least)
    class_weights = [0.7, 0.2, 0.1]
    wells = ["A1", "A2", "B1", "B2", "C1"]
    cell_lines = ["MCF10A", "HeLa"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Use 2-3 wells per condition
            wells_for_condition = np.random.choice(wells, size=np.random.randint(2, 4), replace=False)

            for well in wells_for_condition:
                # 30-80 cells per well to simulate realistic cell counts
                n_cells = np.random.randint(30, 81)

                # Generate classes based on realistic distribution
                cell_classes = np.random.choice(classes, size=n_cells, p=class_weights)

                # Modify class distribution based on condition for testing significance
                if condition == "treatment1":
                    # Treatment1 increases micronuclei
                    cell_classes = np.where(
                        np.random.random(n_cells) < 0.15,
                        "micronuclei",
                        cell_classes
                    )
                elif condition == "treatment2":
                    # Treatment2 increases collapsed cells
                    cell_classes = np.where(
                        np.random.random(n_cells) < 0.12,
                        "collapsed",
                        cell_classes
                    )

                for cls in cell_classes:
                    cell_line = np.random.choice(cell_lines)

                    row = {
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": cell_line,
                        "measurement_id": measurement_id,
                        "well_id": measurement_id * 10,
                        "image_id": measurement_id * 100,
                        "Class": cls,  # Standard classification column name
                        # Add some other typical columns for realism
                        "area_nucleus": np.random.uniform(100, 500),
                        "intensity_mean_DAPI_nucleus": np.random.uniform(1000, 20000),
                    }

                    data.append(row)
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset (normal vs abnormal)."""
    np.random.seed(123)

    data = []
    plates = [1001, 1002]
    conditions = ["control", "treatment1"]
    classes = ["normal", "abnormal"]
    class_weights = [0.8, 0.2]  # 80% normal, 20% abnormal

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            for well in ["A1", "A2"]:
                n_cells = 50
                cell_classes = np.random.choice(classes, size=n_cells, p=class_weights)

                # Treatment increases abnormal cells
                if condition == "treatment1":
                    cell_classes = np.where(
                        np.random.random(n_cells) < 0.2,
                        "abnormal",
                        cell_classes
                    )

                for cls in cell_classes:
                    data.append({
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": "MCF10A",
                        "well_id": measurement_id * 10,
                        "Class": cls,
                        "area_nucleus": np.random.uniform(100, 500),
                    })
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def multiclass_classification_data():
    """Create multi-class classification dataset (5 classes)."""
    np.random.seed(456)

    data = []
    plates = [1001, 1002]
    conditions = ["control", "treatment1"]
    classes = ["normal", "binucleated", "micronuclei", "apoptotic", "collapsed"]
    class_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            for well in ["A1", "A2"]:
                n_cells = 40
                cell_classes = np.random.choice(classes, size=n_cells, p=class_weights)

                for cls in cell_classes:
                    data.append({
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": "MCF10A",
                        "well_id": measurement_id * 10,
                        "Class": cls,
                        "area_nucleus": np.random.uniform(100, 500),
                    })
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def minimal_classification_data():
    """Create minimal classification dataset for basic testing."""
    data = [
        {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "control",
         "cell_line": "MCF10A", "Class": "normal", "well_id": 10},
        {"plate_id": 1001, "well": "A1", "experiment": "exp2", "condition": "control",
         "cell_line": "MCF10A", "Class": "normal", "well_id": 10},
        {"plate_id": 1001, "well": "A1", "experiment": "exp3", "condition": "control",
         "cell_line": "MCF10A", "Class": "micronuclei", "well_id": 10},
        {"plate_id": 1001, "well": "A2", "experiment": "exp4", "condition": "treatment1",
         "cell_line": "MCF10A", "Class": "normal", "well_id": 20},
        {"plate_id": 1001, "well": "A2", "experiment": "exp5", "condition": "treatment1",
         "cell_line": "MCF10A", "Class": "micronuclei", "well_id": 20},
        {"plate_id": 1002, "well": "A1", "experiment": "exp6", "condition": "control",
         "cell_line": "MCF10A", "Class": "normal", "well_id": 30},
        {"plate_id": 1002, "well": "A2", "experiment": "exp7", "condition": "treatment1",
         "cell_line": "MCF10A", "Class": "micronuclei", "well_id": 40},
    ]
    return pd.DataFrame(data)


class TestClassificationPlotBasicFunctionality:
    """Test basic functionality of classification_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_minimal_parameters(self, mock_show, classification_data):
        """Test classification_plot with minimal required parameters."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,
            save=False,  # Don't save in tests
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created with content
        assert len(ax.patches) > 0  # Should have bar patches
        assert ax.get_ylabel() == "% of total cells"

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_stacked_mode(self, mock_show, classification_data):
        """Test classification_plot with stacked display mode (default)."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            display_mode="stacked",
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Stacked mode should have stacked bars
        assert len(ax.patches) > 0
        assert ax.get_ylabel() == "% of total cells"

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_triplicates_mode(self, mock_show, classification_data):
        """Test classification_plot with triplicates display mode."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            display_mode="triplicates",
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Triplicates mode should have individual bars for each plate
        assert len(ax.patches) > 0
        assert ax.get_ylabel() == "Percentage"

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_with_selector_column(self, mock_show, classification_data):
        """Test classification_plot with selector column filtering."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            selector_col="cell_line",
            selector_val="MCF10A",
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Title should include the selector value
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert "MCF10A" in fig_title

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_with_custom_title(self, mock_show, classification_data):
        """Test classification_plot with custom title."""
        custom_title = "Custom Classification Analysis"
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            title=custom_title,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check figure suptitle
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert custom_title in fig_title

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_with_grouping(self, mock_show, classification_data):
        """Test classification_plot with group_size parameter in triplicates mode."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            display_mode="triplicates",
            group_size=2,
            within_group_spacing=0.3,
            between_group_gap=0.7,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Should still create a valid plot with grouping
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_custom_colors(self, mock_show, classification_data):
        """Test classification_plot with custom colors."""
        custom_colors = [COLOR.PURPLE.value, COLOR.TURQUOISE.value, COLOR.OLIVE.value]

        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            colors=custom_colors,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Should create valid plots with custom colors
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_custom_figure_size(self, mock_show, classification_data):
        """Test classification_plot with custom figure size."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            fig_size=(10, 8),
            size_units="cm",
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Figure size should be approximately 10/2.54 x 8/2.54 inches (converted from cm)
        expected_width = 10 / 2.54
        expected_height = 8 / 2.54
        actual_size = fig.get_size_inches()
        assert abs(actual_size[0] - expected_width) < 0.1
        assert abs(actual_size[1] - expected_height) < 0.1

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_with_y_limits(self, mock_show, classification_data):
        """Test classification_plot with custom y-axis limits."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            y_lim=(0, 80),
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check y-axis limits
        y_min, y_max = ax.get_ylim()
        assert y_min >= 0
        assert y_max <= 80

    @patch('matplotlib.pyplot.show')
    def test_classification_plot_without_legend(self, mock_show, classification_data):
        """Test classification_plot with legend disabled."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            show_legend=False,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check that no legend was added
        legend = ax.get_legend()
        assert legend is None


class TestClassificationPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test classification_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            classification_plot(
                df=empty_df,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                save=False,
            )

    def test_missing_required_columns(self, classification_data):
        """Test classification_plot with missing required columns."""
        # Remove plate_id column
        df_missing_plate = classification_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            classification_plot(
                df=df_missing_plate,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                save=False,
            )

        # Remove Class column
        df_missing_class = classification_data.drop("Class", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            classification_plot(
                df=df_missing_class,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                save=False,
            )

    def test_invalid_display_mode(self, classification_data):
        """Test classification_plot with invalid display mode."""
        with pytest.raises(ValueError, match="display_mode must be 'stacked' or 'triplicates'"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                display_mode="invalid_mode",
                save=False,
            )

    def test_invalid_condition_column(self, classification_data):
        """Test classification_plot with invalid condition column."""
        with pytest.raises(ValueError, match="Condition column 'invalid_column' not found in dataframe"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                condition_col="invalid_column",
                save=False,
            )

    def test_conditions_not_in_data(self, classification_data):
        """Test classification_plot with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data: \\['nonexistent_condition'\\]"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "nonexistent_condition"],
                save=False,
            )

    def test_classes_not_in_data(self, classification_data):
        """Test classification_plot with classes not present in data."""
        with pytest.raises(ValueError, match="Classes not found in data: \\['nonexistent_class'\\]"):
            classification_plot(
                df=classification_data,
                classes=["normal", "nonexistent_class"],
                conditions=["control", "treatment1"],
                save=False,
            )

    def test_invalid_selector_column(self, classification_data):
        """Test classification_plot with invalid selector column."""
        with pytest.raises(ValueError, match="Selector column 'invalid_selector' not found in dataframe"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                selector_col="invalid_selector",
                selector_val="some_value",
                save=False,
            )

    def test_selector_column_without_value(self, classification_data):
        """Test classification_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError, match="selector_val must be provided when selector_col is specified"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                save=False,
            )

    def test_invalid_selector_value(self, classification_data):
        """Test classification_plot with selector_val not in data."""
        with pytest.raises(ValueError, match="Value 'NonexistentCell' not found in column 'cell_line'"):
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell",
                save=False,
            )

    def test_no_data_after_filtering(self, classification_data):
        """Test classification_plot when filtering results in no data."""
        # Create a scenario where filtering removes all data
        df_subset = classification_data[classification_data["condition"] == "control"].copy()

        with pytest.raises(ValueError, match="Conditions not found in data: \\['treatment1'\\]"):
            classification_plot(
                df=df_subset,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],  # treatment1 not in subset
                selector_col="cell_line",
                selector_val="MCF10A",
                save=False,
            )

    @patch('matplotlib.pyplot.show')
    def test_single_plate_data(self, mock_show, minimal_classification_data):
        """Test classification_plot with single plate (should work)."""
        # Filter to single plate
        single_plate_df = minimal_classification_data[minimal_classification_data["plate_id"] == 1001].copy()

        fig, ax = classification_plot(
            df=single_plate_df,
            classes=["normal", "micronuclei"],
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot even with single plate (no error bars in stacked mode)


class TestClassificationPlotParametrized:
    """Parametrized tests for different plot configurations."""

    @pytest.mark.parametrize("display_mode", ["stacked", "triplicates"])
    @patch('matplotlib.pyplot.show')
    def test_display_modes(self, mock_show, classification_data, display_mode):
        """Test both display modes work correctly."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            display_mode=display_mode,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Should create valid plots regardless of display mode
        assert len(ax.patches) > 0

    @pytest.mark.parametrize("group_size", [1, 2, 3])
    @patch('matplotlib.pyplot.show')
    def test_group_sizes_triplicates(self, mock_show, classification_data, group_size):
        """Test different group sizes in triplicates mode."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            display_mode="triplicates",
            group_size=group_size,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @pytest.mark.parametrize(
        "selector_col,selector_val",
        [("cell_line", "MCF10A"), ("cell_line", "HeLa"), (None, None)]
    )
    @patch('matplotlib.pyplot.show')
    def test_selector_combinations(self, mock_show, classification_data, selector_col, selector_val):
        """Test different selector column combinations."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            selector_col=selector_col,
            selector_val=selector_val,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check title includes selector value when provided
        if selector_val:
            fig_title = fig._suptitle.get_text() if fig._suptitle else ""
            assert selector_val in fig_title

    @pytest.mark.parametrize("show_legend", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_legend_parameter(self, mock_show, classification_data, show_legend):
        """Test legend enable/disable."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            show_legend=show_legend,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check legend presence/absence
        legend = ax.get_legend()
        if show_legend:
            assert legend is not None
        else:
            assert legend is None

    @patch('matplotlib.pyplot.show')
    def test_different_class_counts_binary(self, mock_show, binary_classification_data):
        """Test with binary classification (2 classes)."""
        fig, ax = classification_plot(
            df=binary_classification_data,
            classes=["normal", "abnormal"],
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_different_class_counts_multiclass(self, mock_show, multiclass_classification_data):
        """Test with multi-class classification (5 classes)."""
        fig, ax = classification_plot(
            df=multiclass_classification_data,
            classes=["normal", "binucleated", "micronuclei", "apoptotic", "collapsed"],
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0


class TestClassificationPlotIntegration:
    """Test integration aspects and matplotlib object interactions."""

    @patch('matplotlib.pyplot.show')
    def test_figure_and_axes_are_matplotlib_objects(self, mock_show, classification_data):
        """Test that returned figure and axes are proper matplotlib objects."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,
            save=False,
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
    def test_plot_without_display(self, mock_show, classification_data):
        """Test that plots can be created without displaying (mocked show)."""
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,
            save=False,
        )

        # Verify show was not called (it's mocked)
        mock_show.assert_not_called()

        # But plot should still be created
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_custom_axes_input(self, mock_show, classification_data):
        """Test classification_plot with custom axes input."""
        import matplotlib.pyplot as plt

        # Create custom figure and axes
        custom_fig, custom_ax = plt.subplots(figsize=(10, 8))

        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            axes=custom_ax,
            selector_col=None,
            save=False,
        )

        # Should return the same axes we provided
        assert ax is custom_ax
        assert fig is custom_fig

        plt.close(custom_fig)  # Clean up


class TestClassificationPlotDataProcessor:
    """Test the ClassificationDataProcessor directly."""

    def test_data_processor_quantify_classification(self, classification_data):
        """Test quantify_classification method."""
        processor = ClassificationDataProcessor(classification_data, class_col="Class")

        # Test quantification
        mean_data, std_data = processor.quantify_classification(
            classification_data, condition_col="condition"
        )

        # Should return DataFrames with proper structure
        assert isinstance(mean_data, pd.DataFrame)
        assert isinstance(std_data, pd.DataFrame)

        # Should have condition and Class columns
        assert "condition" in mean_data.columns
        assert "Class" in mean_data.columns
        assert "percentage" in mean_data.columns

        # Percentages should be between 0 and 100
        assert (mean_data["percentage"] >= 0).all()
        assert (mean_data["percentage"] <= 100).all()

    def test_data_processor_process_data(self, classification_data):
        """Test process_data method for stacked plotting."""
        processor = ClassificationDataProcessor(classification_data, class_col="Class")

        classes = ["normal", "micronuclei", "collapsed"]
        conditions = ["control", "treatment1", "treatment2"]

        plot_data, std_data = processor.process_data(
            classification_data,
            condition_col="condition",
            conditions=conditions,
            classes=classes,
        )

        # Should return properly formatted DataFrames
        assert isinstance(plot_data, pd.DataFrame)
        assert isinstance(std_data, pd.DataFrame)

        # Should have classes as columns
        for cls in classes:
            assert cls in plot_data.columns
            assert cls in std_data.columns

        # Should have condition column
        assert "condition" in plot_data.columns
        assert "condition" in std_data.columns

    def test_data_processor_with_custom_class_column(self, classification_data):
        """Test data processor with custom class column name."""
        # Rename Class column
        df_custom = classification_data.rename(columns={"Class": "Cell_Type"})

        processor = ClassificationDataProcessor(df_custom, class_col="Cell_Type")

        mean_data, std_data = processor.quantify_classification(
            df_custom, condition_col="condition"
        )

        # Should work with custom class column
        assert "Cell_Type" in mean_data.columns
        assert "Cell_Type" in std_data.columns


class TestClassificationPlotBuilder:
    """Test the ClassificationPlotBuilder directly."""

    def test_plot_builder_stacked_mode(self, classification_data):
        """Test plot builder in stacked mode."""
        config = ClassificationPlotConfig(display_mode="stacked")
        builder = ClassificationPlotBuilder(config)

        # Create figure
        builder.create_figure(axes=None)
        assert builder.fig is not None
        assert builder.ax is not None

        # Process data
        processor = ClassificationDataProcessor(classification_data, class_col="Class")
        classes = ["normal", "micronuclei", "collapsed"]
        conditions = ["control", "treatment1", "treatment2"]

        plot_data, std_data = processor.process_data(
            classification_data,
            condition_col="condition",
            conditions=conditions,
            classes=classes,
        )

        # Build stacked plot
        builder.build_stacked_plot(plot_data, std_data, conditions, classes, "condition")

        # Should have created plot elements
        assert len(builder.ax.patches) > 0

    def test_plot_builder_triplicates_mode(self, classification_data):
        """Test plot builder in triplicates mode."""
        config = ClassificationPlotConfig(display_mode="triplicates")
        builder = ClassificationPlotBuilder(config)

        # Create figure
        builder.create_figure(axes=None)
        assert builder.fig is not None
        assert builder.ax is not None

        # Build triplicates plot
        classes = ["normal", "micronuclei", "collapsed"]
        conditions = ["control", "treatment1", "treatment2"]

        builder.build_triplicates_plot(
            classification_data, conditions, classes, "condition", "Class"
        )

        # Should have created plot elements
        assert len(builder.ax.patches) > 0

    def test_plot_builder_with_custom_colors(self, classification_data):
        """Test plot builder with custom colors."""
        custom_colors = [COLOR.PURPLE.value, COLOR.TURQUOISE.value, COLOR.OLIVE.value]
        config = ClassificationPlotConfig(colors=custom_colors)
        builder = ClassificationPlotBuilder(config)

        builder.create_figure(axes=None)

        # Process data for stacked plot
        processor = ClassificationDataProcessor(classification_data, class_col="Class")
        classes = ["normal", "micronuclei", "collapsed"]
        conditions = ["control", "treatment1", "treatment2"]

        plot_data, std_data = processor.process_data(
            classification_data,
            condition_col="condition",
            conditions=conditions,
            classes=classes,
        )

        # Build plot with custom colors
        builder.build_stacked_plot(plot_data, std_data, conditions, classes, "condition")

        # Should have created plot with custom colors
        assert len(builder.ax.patches) > 0


class TestClassificationPlotConfig:
    """Test the ClassificationPlotConfig class."""

    def test_config_defaults(self):
        """Test ClassificationPlotConfig default values."""
        config = ClassificationPlotConfig()

        assert config.display_mode == "stacked"
        assert config.y_lim == (0, 100)
        assert config.fig_size == (7, 7)
        assert config.size_units == "cm"
        assert config.dpi == 300
        assert config.save is False
        assert config.file_format == "pdf"
        assert config.bar_width == 0.75
        assert config.show_legend is True
        assert config.group_size == 2
        assert config.repeat_offset == 0.18

    def test_config_custom_values(self):
        """Test ClassificationPlotConfig with custom values."""
        config = ClassificationPlotConfig(
            display_mode="triplicates",
            y_lim=(0, 80),
            fig_size=(10, 8),
            size_units="inches",
            dpi=150,
            group_size=3,
            bar_width=0.5,
            show_legend=False,
        )

        assert config.display_mode == "triplicates"
        assert config.y_lim == (0, 80)
        assert config.fig_size == (10, 8)
        assert config.size_units == "inches"
        assert config.dpi == 150
        assert config.group_size == 3
        assert config.bar_width == 0.5
        assert config.show_legend is False


class TestClassificationPlotSpecialCases:
    """Test special data scenarios."""

    @patch('matplotlib.pyplot.show')
    def test_binary_classification(self, mock_show, binary_classification_data):
        """Test with binary classification data."""
        fig, ax = classification_plot(
            df=binary_classification_data,
            classes=["normal", "abnormal"],
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_multiclass_classification(self, mock_show, multiclass_classification_data):
        """Test with multi-class classification data."""
        fig, ax = classification_plot(
            df=multiclass_classification_data,
            classes=["normal", "binucleated", "micronuclei", "apoptotic", "collapsed"],
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_single_condition_data(self, mock_show):
        """Test with data containing only one condition."""
        data = []
        classes = ["normal", "micronuclei", "collapsed"]

        # Create data with only control condition
        for cls in classes:
            for i in range(10):  # 10 cells per class
                data.append({
                    "plate_id": 1001,
                    "well": f"A{i+1}",
                    "experiment": f"exp_control_{cls}_{i}",
                    "condition": "control",
                    "cell_line": "MCF10A",
                    "well_id": i + 1,
                    "Class": cls,
                })

        df = pd.DataFrame(data)

        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=["control"],  # Only one condition
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestClassificationPlotErrorMessages:
    """Test specific error messages for various failure modes."""

    def test_informative_error_messages(self, classification_data):
        """Test that error messages are informative and helpful."""

        # Test missing condition column error message
        with pytest.raises(ValueError) as exc_info:
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                condition_col="nonexistent_condition_col",
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "Condition column 'nonexistent_condition_col' not found in dataframe" in error_msg
        assert "Available columns:" in error_msg

        # Test invalid selector value error message
        with pytest.raises(ValueError) as exc_info:
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell",
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "Value 'NonexistentCell' not found in column 'cell_line'" in error_msg
        assert "Available values:" in error_msg

    def test_helpful_selector_error_message(self, classification_data):
        """Test helpful error message when selector_col provided without selector_val."""
        with pytest.raises(ValueError) as exc_info:
            classification_plot(
                df=classification_data,
                classes=["normal", "micronuclei"],
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                # selector_val not provided
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "selector_val must be provided when selector_col is specified" in error_msg
        assert "Available values in 'cell_line':" in error_msg


class TestClassificationPlotBackwardCompatibility:
    """Test backward compatibility and API wrapper functionality."""

    @patch('matplotlib.pyplot.show')
    def test_api_wrapper_maintains_compatibility(self, mock_show, classification_data):
        """Test that the API wrapper maintains backward compatibility."""
        # Test with parameters that might have been used in older versions
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1", "treatment2"],
            condition_col="condition",  # Explicit condition column
            class_col="Class",  # Explicit class column
            selector_col="cell_line",
            selector_val="MCF10A",
            title="Backward Compatible Test",
            fig_size=(8, 8),
            size_units="cm",
            save=False,  # Don't actually save in tests
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check that title is applied
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert "Backward Compatible Test" in fig_title

    @patch('matplotlib.pyplot.show')
    def test_default_parameter_handling(self, mock_show, classification_data):
        """Test that default parameters work as expected."""
        # Test with minimal parameters, relying on defaults
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
            conditions=["control", "treatment1"],
            selector_col=None,  # Override default to test minimal parameters
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_path_parameter_compatibility(self, mock_show, classification_data, tmp_path):
        """Test path parameter handling for save functionality."""
        # Test with save=False (should not actually save)
        fig, ax = classification_plot(
            df=classification_data,
            classes=["normal", "micronuclei", "collapsed"],
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
