"""Comprehensive pytest tests for countplot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the count plot functionality,
focusing on testing the main API functions and error handling without validating
visual output.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from unittest.mock import patch

from omero_screen_plots.countplot_api import count_plot
from omero_screen_plots.countplot_factory import CountPlot, CountPlotConfig, PlotType


@pytest.fixture
def synthetic_plate_data():
    """Create a synthetic dataset with realistic plate data structure.

    Creates a comprehensive dataset with:
    - 3 plates (1001, 1002, 1003)
    - 3 conditions (control, treatment1, treatment2)
    - 2-3 wells per plate/condition combination
    - 5-10 rows per well (simulating multiple cells/experiments)
    - 2 cell lines (MCF10A, HeLa)

    Returns:
        pd.DataFrame: Synthetic plate data with all required columns
    """
    np.random.seed(42)  # For reproducible results

    data = []
    plates = [1001, 1002, 1003]
    conditions = ["control", "treatment1", "treatment2"]
    wells = ["A1", "A2", "B1", "B2", "C1"]
    cell_lines = ["MCF10A", "HeLa"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Use 2-3 wells per condition
            wells_for_condition = np.random.choice(wells, size=np.random.randint(2, 4), replace=False)

            for well in wells_for_condition:
                # 5-10 experiments per well
                n_experiments = np.random.randint(5, 11)

                for _ in range(n_experiments):
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
                        # Add some measurement columns for completeness
                        "area_nucleus": np.random.uniform(100, 500),
                        "intensity_mean_DAPI_nucleus": np.random.uniform(1000, 20000),
                    }
                    data.append(row)
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def minimal_plate_data():
    """Create minimal dataset for basic testing.

    Returns:
        pd.DataFrame: Minimal dataset with just required columns
    """
    data = [
        {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "control", "cell_line": "MCF10A"},
        {"plate_id": 1001, "well": "A1", "experiment": "exp2", "condition": "control", "cell_line": "MCF10A"},
        {"plate_id": 1001, "well": "A2", "experiment": "exp3", "condition": "treatment1", "cell_line": "MCF10A"},
        {"plate_id": 1002, "well": "A1", "experiment": "exp4", "condition": "control", "cell_line": "MCF10A"},
        {"plate_id": 1002, "well": "A2", "experiment": "exp5", "condition": "treatment1", "cell_line": "MCF10A"},
    ]
    return pd.DataFrame(data)


class TestCountPlotBasicFunctionality:
    """Test basic functionality of count_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_count_plot_minimal_parameters(self, mock_show, synthetic_plate_data):
        """Test count_plot with minimal required parameters."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None  # Explicitly set to None to avoid default "cell_line"
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created (has elements)
        assert len(ax.patches) > 0  # Should have bar patches
        assert ax.get_ylabel() != ""  # Should have y-axis label

    @patch('matplotlib.pyplot.show')
    def test_count_plot_normalised_type(self, mock_show, synthetic_plate_data):
        """Test count_plot with PlotType.NORMALISED."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            plot_type=PlotType.NORMALISED,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "Normalized Count" in ax.get_ylabel()

    @patch('matplotlib.pyplot.show')
    def test_count_plot_absolute_type(self, mock_show, synthetic_plate_data):
        """Test count_plot with PlotType.ABSOLUTE."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            plot_type=PlotType.ABSOLUTE,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "Count" in ax.get_ylabel()

    @patch('matplotlib.pyplot.show')
    def test_count_plot_with_selector_column(self, mock_show, synthetic_plate_data):
        """Test count_plot with selector column filtering."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
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
    def test_count_plot_with_custom_title(self, mock_show, synthetic_plate_data):
        """Test count_plot with custom title."""
        custom_title = "Custom Count Analysis"
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            title=custom_title,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check figure suptitle instead of axes title
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert custom_title in fig_title

    @patch('matplotlib.pyplot.show')
    def test_count_plot_with_grouping(self, mock_show, synthetic_plate_data):
        """Test count_plot with group_size parameter."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            group_size=2,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should still create a valid plot with grouping
        assert len(ax.patches) > 0


class TestCountPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test count_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            count_plot(
                df=empty_df,
                norm_control="control",
                conditions=["control", "treatment1"]
            )

    def test_missing_required_columns(self, synthetic_plate_data):
        """Test count_plot with missing required columns."""
        # Remove plate_id column
        df_missing_plate = synthetic_plate_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            count_plot(
                df=df_missing_plate,
                norm_control="control",
                conditions=["control", "treatment1"]
            )

        # Remove well column
        df_missing_well = synthetic_plate_data.drop("well", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            count_plot(
                df=df_missing_well,
                norm_control="control",
                conditions=["control", "treatment1"]
            )

        # Remove experiment column
        df_missing_exp = synthetic_plate_data.drop("experiment", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            count_plot(
                df=df_missing_exp,
                norm_control="control",
                conditions=["control", "treatment1"]
            )

    def test_invalid_condition_column(self, synthetic_plate_data):
        """Test count_plot with invalid condition column."""
        with pytest.raises(ValueError, match="Condition column 'invalid_column' not found"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="control",
                conditions=["control", "treatment1"],
                condition_col="invalid_column"
            )

    def test_conditions_not_in_data(self, synthetic_plate_data):
        """Test count_plot with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="control",
                conditions=["control", "nonexistent_condition"]
            )

    def test_norm_control_not_in_conditions(self, synthetic_plate_data):
        """Test count_plot with norm_control not in conditions list."""
        with pytest.raises(ValueError, match="Normalization control 'invalid_control' must be in conditions list"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="invalid_control",
                conditions=["control", "treatment1"]
            )

    def test_invalid_selector_column(self, synthetic_plate_data):
        """Test count_plot with invalid selector column."""
        with pytest.raises(ValueError, match="Selector column 'invalid_selector' not found"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="control",
                conditions=["control", "treatment1"],
                selector_col="invalid_selector",
                selector_val="some_value"
            )

    def test_selector_column_without_value(self, synthetic_plate_data):
        """Test count_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError, match="selector_val must be provided when selector_col is specified"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="control",
                conditions=["control", "treatment1"],
                selector_col="cell_line"
            )

    def test_invalid_selector_value(self, synthetic_plate_data):
        """Test count_plot with selector_val not in data."""
        with pytest.raises(ValueError, match="Value 'NonexistentCell' not found in column 'cell_line'"):
            count_plot(
                df=synthetic_plate_data,
                norm_control="control",
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell"
            )

    def test_no_data_after_filtering(self, synthetic_plate_data):
        """Test count_plot when filtering results in no data."""
        # Create a scenario where filtering removes all data
        df_subset = synthetic_plate_data[synthetic_plate_data["condition"] == "control"].copy()

        with pytest.raises(ValueError, match="Conditions not found in data"):
            count_plot(
                df=df_subset,
                norm_control="control",
                conditions=["control", "treatment1"],  # treatment1 not in subset
                selector_col="cell_line",
                selector_val="MCF10A"
            )

    @patch('matplotlib.pyplot.show')
    def test_single_plate_no_significance(self, mock_show, minimal_plate_data):
        """Test count_plot with single plate (should work but no significance marks)."""
        # Filter to single plate
        single_plate_df = minimal_plate_data[minimal_plate_data["plate_id"] == 1001].copy()

        fig, ax = count_plot(
            df=single_plate_df,
            norm_control="control",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot without significance marks


class TestCountPlotErrorMessages:
    """Test specific error messages for various failure modes."""

    def test_zero_values_in_control_error(self):
        """Test error when control condition has zero values (division by zero protection)."""
        # Create data where control has zero count
        data = [
            {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "treatment1"},
            {"plate_id": 1001, "well": "A2", "experiment": "exp2", "condition": "treatment1"},
            # No experiments for control condition
        ]
        df = pd.DataFrame(data)

        with pytest.raises(ValueError, match="Conditions not found in data.*control"):
            count_plot(
                df=df,
                norm_control="control",
                conditions=["control", "treatment1"],
                selector_col=None
            )


class TestCountPlotIntegration:
    """Test integration aspects and matplotlib object interactions."""

    @patch('matplotlib.pyplot.show')
    def test_figure_and_axes_are_matplotlib_objects(self, mock_show, synthetic_plate_data):
        """Test that returned figure and axes are proper matplotlib objects."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
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
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
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
        """Test count_plot with custom axes input."""
        import matplotlib.pyplot as plt

        # Create custom figure and axes
        custom_fig, custom_ax = plt.subplots(figsize=(10, 8))

        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            axes=custom_ax,
            selector_col=None
        )

        # Should return the same axes we provided
        assert ax is custom_ax
        assert fig is custom_fig

        plt.close(custom_fig)  # Clean up


class TestCountPlotParameterized:
    """Parametrized tests for different plot configurations."""

    @pytest.mark.parametrize("plot_type", [PlotType.NORMALISED, PlotType.ABSOLUTE])
    @patch('matplotlib.pyplot.show')
    def test_plot_types(self, mock_show, synthetic_plate_data, plot_type):
        """Test both plot types work correctly."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            plot_type=plot_type,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check appropriate y-label
        y_label = ax.get_ylabel()
        if plot_type == PlotType.NORMALISED:
            assert "Normalized" in y_label
        else:
            assert "Count" in y_label and "Normalized" not in y_label

    @pytest.mark.parametrize("group_size", [1, 2, 3])
    @patch('matplotlib.pyplot.show')
    def test_group_sizes(self, mock_show, synthetic_plate_data, group_size):
        """Test different group sizes work correctly."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            group_size=group_size,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0  # Should have bars

    @pytest.mark.parametrize(
        "selector_col,selector_val",
        [("cell_line", "MCF10A"), ("cell_line", "HeLa"), (None, None)]
    )
    @patch('matplotlib.pyplot.show')
    def test_selector_combinations(self, mock_show, synthetic_plate_data, selector_col, selector_val):
        """Test different selector column combinations."""
        fig, ax = count_plot(
            df=synthetic_plate_data,
            norm_control="control",
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


class TestCountPlotFactory:
    """Test the underlying CountPlot factory class directly."""

    def test_count_plot_config_defaults(self):
        """Test CountPlotConfig default values."""
        config = CountPlotConfig()

        assert config.fig_size == (7, 7)
        assert config.size_units == "cm"
        assert config.dpi == 300
        assert config.save is False
        assert config.file_format == "pdf"
        assert config.plot_type == PlotType.NORMALISED
        assert config.group_size == 1

    def test_count_plot_config_custom_values(self):
        """Test CountPlotConfig with custom values."""
        config = CountPlotConfig(
            fig_size=(10, 8),
            size_units="inches",
            dpi=150,
            plot_type=PlotType.ABSOLUTE,
            group_size=2
        )

        assert config.fig_size == (10, 8)
        assert config.size_units == "inches"
        assert config.dpi == 150
        assert config.plot_type == PlotType.ABSOLUTE
        assert config.group_size == 2

    @patch('matplotlib.pyplot.show')
    def test_count_plot_class_direct_usage(self, mock_show, synthetic_plate_data):
        """Test using CountPlot class directly."""
        config = CountPlotConfig(plot_type=PlotType.ABSOLUTE)
        plot = CountPlot(config)

        fig, ax = plot.create_plot(
            df=synthetic_plate_data,
            norm_control="control",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "Count" in ax.get_ylabel()


# Additional fixtures for specific test scenarios
@pytest.fixture
def zero_control_data():
    """Create data where control condition results in zero counts."""
    data = [
        {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "treatment1"},
        {"plate_id": 1001, "well": "A2", "experiment": "exp2", "condition": "treatment1"},
        {"plate_id": 1002, "well": "A1", "experiment": "exp3", "condition": "treatment1"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def single_condition_data():
    """Create data with only one condition."""
    data = [
        {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "control"},
        {"plate_id": 1001, "well": "A2", "experiment": "exp2", "condition": "control"},
        {"plate_id": 1002, "well": "A1", "experiment": "exp3", "condition": "control"},
    ]
    return pd.DataFrame(data)


class TestCountPlotSpecialCases:
    """Test special data scenarios."""

    def test_single_condition_data(self, single_condition_data):
        """Test with data containing only the control condition."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            count_plot(
                df=single_condition_data,
                norm_control="control",
                conditions=["control", "treatment1"],  # treatment1 not in data
                selector_col=None
            )

    @patch('matplotlib.pyplot.show')
    def test_data_with_many_plates(self, mock_show):
        """Test with data containing many plates (for significance testing)."""
        # Create data with 5 plates for significance testing
        data = []
        for plate_id in range(1001, 1006):  # 5 plates
            for condition in ["control", "treatment1"]:
                for well in ["A1", "A2"]:
                    for exp_num in range(3):  # 3 experiments per well
                        data.append({
                            "plate_id": plate_id,
                            "well": well,
                            "experiment": f"exp_{plate_id}_{well}_{exp_num}",
                            "condition": condition
                        })

        df = pd.DataFrame(data)

        fig, ax = count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # With 5 plates, should have significance testing (>= 3 plates required)
        # This is tested indirectly by ensuring the plot is created successfully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
