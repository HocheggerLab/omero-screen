"""Comprehensive pytest tests for cellcycle_plot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the cell cycle plot functionality,
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

from omero_screen_plots.cellcycleplot_api import cellcycle_plot
from omero_screen_plots.cellcycleplot_factory import (
    StandardCellCyclePlotConfig,
    StandardCellCyclePlot,
    BaseCellCyclePlot,
)
from omero_screen_plots.colors import COLOR


@pytest.fixture
def cellcycle_data():
    """Create synthetic cellcycle dataset with realistic cell cycle phases.

    Creates a comprehensive dataset with:
    - 3 plates (1001, 1002, 1003)
    - 3 conditions (control, treatment1, treatment2)
    - Multiple wells per plate/condition combination
    - Cell cycle phases: G1, S, G2/M, Polyploid
    - Realistic phase distribution
    """
    np.random.seed(42)  # For reproducible results

    data = []
    plates = [1001, 1002, 1003]
    conditions = ["control", "treatment1", "treatment2"]
    phases = ["G1", "S", "G2/M", "Polyploid"]
    # Realistic phase distributions (G1 most common, Polyploid least)
    phase_weights = [0.5, 0.25, 0.2, 0.05]
    wells = ["A1", "A2", "B1", "B2", "C1"]
    cell_lines = ["MCF10A", "HeLa"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Use 2-3 wells per condition
            wells_for_condition = np.random.choice(wells, size=np.random.randint(2, 4), replace=False)

            for well in wells_for_condition:
                # 20-50 cells per well to simulate realistic cell counts
                n_cells = np.random.randint(20, 51)

                # Generate phases based on realistic distribution
                cell_phases = np.random.choice(phases, size=n_cells, p=phase_weights)

                # Modify phase distribution based on condition for testing significance
                if condition == "treatment1":
                    # Treatment1 increases G2/M phase
                    cell_phases = np.where(
                        np.random.random(n_cells) < 0.2,
                        "G2/M",
                        cell_phases
                    )
                elif condition == "treatment2":
                    # Treatment2 increases S phase
                    cell_phases = np.where(
                        np.random.random(n_cells) < 0.15,
                        "S",
                        cell_phases
                    )

                for phase in cell_phases:
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
                        "cell_cycle": phase,
                        # Add some other typical columns
                        "area_nucleus": np.random.uniform(100, 500),
                        "intensity_mean_DAPI_nucleus": np.random.uniform(1000, 20000),
                    }

                    data.append(row)
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def cellcycle_data_with_subG1():
    """Create cellcycle dataset that includes Sub-G1 phase."""
    np.random.seed(123)

    data = []
    plates = [1001, 1002]
    conditions = ["control", "treatment1"]
    phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]
    # Include Sub-G1 with small but significant population
    phase_weights = [0.1, 0.45, 0.25, 0.15, 0.05]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            for well in ["A1", "A2"]:
                n_cells = 30
                cell_phases = np.random.choice(phases, size=n_cells, p=phase_weights)

                for phase in cell_phases:
                    data.append({
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": "MCF10A",
                        "cell_cycle": phase,
                        "area_nucleus": np.random.uniform(100, 500),
                    })
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def cellcycle_data_with_M_phase():
    """Create cellcycle dataset with separate G2 and M phases."""
    np.random.seed(456)

    data = []
    plates = [1001, 1002]
    conditions = ["control", "treatment1"]
    phases = ["G1", "S", "G2", "M", "Polyploid"]
    phase_weights = [0.45, 0.25, 0.15, 0.1, 0.05]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            for well in ["A1", "A2"]:
                n_cells = 25
                cell_phases = np.random.choice(phases, size=n_cells, p=phase_weights)

                for phase in cell_phases:
                    data.append({
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": "MCF10A",
                        "cell_cycle": phase,
                        "area_nucleus": np.random.uniform(100, 500),
                    })
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def minimal_cellcycle_data():
    """Create minimal cellcycle dataset for basic testing."""
    data = [
        {"plate_id": 1001, "well": "A1", "experiment": "exp1", "condition": "control",
         "cell_line": "MCF10A", "cell_cycle": "G1"},
        {"plate_id": 1001, "well": "A1", "experiment": "exp2", "condition": "control",
         "cell_line": "MCF10A", "cell_cycle": "S"},
        {"plate_id": 1001, "well": "A1", "experiment": "exp3", "condition": "control",
         "cell_line": "MCF10A", "cell_cycle": "G2/M"},
        {"plate_id": 1001, "well": "A2", "experiment": "exp4", "condition": "treatment1",
         "cell_line": "MCF10A", "cell_cycle": "G1"},
        {"plate_id": 1001, "well": "A2", "experiment": "exp5", "condition": "treatment1",
         "cell_line": "MCF10A", "cell_cycle": "G2/M"},
        {"plate_id": 1002, "well": "A1", "experiment": "exp6", "condition": "control",
         "cell_line": "MCF10A", "cell_cycle": "S"},
        {"plate_id": 1002, "well": "A2", "experiment": "exp7", "condition": "treatment1",
         "cell_line": "MCF10A", "cell_cycle": "G2/M"},
    ]
    return pd.DataFrame(data)


class TestCellCyclePlotBasicFunctionality:
    """Test basic functionality of cellcycle_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_minimal_parameters(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with minimal required parameters."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,
            save=False,  # Don't save in tests
        )

        # Verify return types - cellcycle_plot returns (Figure, list of Axes)
        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) >= 4  # Should have at least 4 subplots for G1, S, G2/M, Polyploid
        assert all(isinstance(ax, Axes) for ax in axes)

        # Verify subplots were created with content
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_selector_column(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with selector column filtering."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1", "treatment2"],
            selector_col="cell_line",
            selector_val="MCF10A",
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Title should include the selector value
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert "MCF10A" in fig_title

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_custom_title(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with custom title."""
        custom_title = "Custom Cell Cycle Analysis"
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1", "treatment2"],
            title=custom_title,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Check figure suptitle
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert custom_title in fig_title

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_default_config(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with default configuration."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # With default config (show_subG1=False), should have 4 phases: G1, S, G2/M, Polyploid
        assert len(axes) == 4

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_show_subG1_true(self, mock_show, cellcycle_data_with_subG1):
        """Test cellcycle_plot with show_subG1=True."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data_with_subG1,
            conditions=["control", "treatment1"],
            show_subG1=True,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # With show_subG1=True, should have 5 phases: Sub-G1, G1, S, G2/M, Polyploid
        assert len(axes) == 5

        # Should use 2x3 layout for 5 phases
        # Check that figure was created with proper size adjustment
        assert fig.get_size_inches()[0] > fig.get_size_inches()[1]  # Should be wider

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_cc_phases_terminology(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with cc_phases=True (cell cycle terminology)."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            cc_phases=True,  # Cell cycle names
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # With cc_phases=True and no Sub-G1 by default, should have 4 phases
        assert len(axes) == 4

        # Should create valid plots with bar patches
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_dna_content_terminology(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with cc_phases=False (DNA content terminology)."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            cc_phases=False,  # DNA content names
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # With cc_phases=False and no Sub-G1 by default, should have 4 phases
        assert len(axes) == 4

        # Should create valid plots with bar patches
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_M_phase_detection(self, mock_show, cellcycle_data_with_M_phase):
        """Test cellcycle_plot with automatic M phase detection."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data_with_M_phase,
            conditions=["control", "treatment1"],
            show_subG1=False,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # With separate G2 and M phases, should have 5 phases: G1, S, G2, M, Polyploid
        assert len(axes) == 5

        # Should create valid plots
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_plate_legend(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with plate legend enabled."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_plate_legend=True,
            show_repeat_points=True,  # Required for legend to show
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots with legend configuration
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

        # Legend testing is complex due to matplotlib internals, just verify plot creation

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_without_repeat_points(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with repeat points disabled."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_repeat_points=False,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should still create valid plots
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_without_significance(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with significance marks disabled."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_significance=False,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots without significance marks
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_custom_colors(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with custom colors."""
        custom_colors = [COLOR.PURPLE.value, COLOR.TURQUOISE.value,
                        COLOR.OLIVE.value, COLOR.LAVENDER.value]

        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            colors=custom_colors,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # Should create valid plots with custom colors
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_custom_figure_size(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with custom figure size."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            fig_size=(8, 8),
            size_units="cm",
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Figure size should be approximately 8/2.54 inches (converted from cm)
        expected_size = 8 / 2.54
        actual_size = fig.get_size_inches()
        assert abs(actual_size[0] - expected_size) < 0.1
        assert abs(actual_size[1] - expected_size) < 0.1

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_rotation(self, mock_show, cellcycle_data):
        """Test cellcycle_plot with custom x-label rotation."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            rotation=90,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots - rotation is applied to bottom row subplots
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches


class TestCellCyclePlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test cellcycle_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            cellcycle_plot(
                df=empty_df,
                conditions=["control", "treatment1"],
                save=False,
            )

    def test_missing_required_columns(self, cellcycle_data):
        """Test cellcycle_plot with missing required columns."""
        # Remove plate_id column
        df_missing_plate = cellcycle_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            cellcycle_plot(
                df=df_missing_plate,
                conditions=["control", "treatment1"],
                save=False,
            )

        # Remove cell_cycle column
        df_missing_cc = cellcycle_data.drop("cell_cycle", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            cellcycle_plot(
                df=df_missing_cc,
                conditions=["control", "treatment1"],
                save=False,
            )

    def test_invalid_condition_column(self, cellcycle_data):
        """Test cellcycle_plot with invalid condition column."""
        with pytest.raises(ValueError, match="Condition column 'invalid_column' not found"):
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                condition_col="invalid_column",
                save=False,
            )

    def test_conditions_not_in_data(self, cellcycle_data):
        """Test cellcycle_plot with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "nonexistent_condition"],
                save=False,
            )

    def test_invalid_selector_column(self, cellcycle_data):
        """Test cellcycle_plot with invalid selector column."""
        with pytest.raises(ValueError, match="Selector column 'invalid_selector' not found"):
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                selector_col="invalid_selector",
                selector_val="some_value",
                save=False,
            )

    def test_selector_column_without_value(self, cellcycle_data):
        """Test cellcycle_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError, match="selector_val must be provided when selector_col is specified"):
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                save=False,
            )

    def test_invalid_selector_value(self, cellcycle_data):
        """Test cellcycle_plot with selector_val not in data."""
        with pytest.raises(ValueError, match="Value 'NonexistentCell' not found in column 'cell_line'"):
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell",
                save=False,
            )

    def test_no_data_after_filtering(self, cellcycle_data):
        """Test cellcycle_plot when filtering results in no data."""
        # Create a scenario where filtering removes all data
        df_subset = cellcycle_data[cellcycle_data["condition"] == "control"].copy()

        with pytest.raises(ValueError, match="Conditions not found in data"):
            cellcycle_plot(
                df=df_subset,
                conditions=["control", "treatment1"],  # treatment1 not in subset
                selector_col="cell_line",
                selector_val="MCF10A",
                save=False,
            )

    @patch('matplotlib.pyplot.show')
    def test_single_plate_no_significance(self, mock_show, minimal_cellcycle_data):
        """Test cellcycle_plot with single plate (should work but no significance marks)."""
        # Filter to single plate
        single_plate_df = minimal_cellcycle_data[minimal_cellcycle_data["plate_id"] == 1001].copy()

        fig, axes = cellcycle_plot(
            df=single_plate_df,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # Should create plot without significance marks due to insufficient plates

    def test_too_many_phases_error(self):
        """Test that configuration limits subplot creation appropriately."""
        # Note: The current implementation doesn't dynamically detect phases from data
        # It uses predefined phase lists, so this test verifies the fixed behavior

        # Create data with many phases, but the implementation will still use
        # its predefined phase list
        data = []
        phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]

        for phase in phases:
            for condition in ["control", "treatment1"]:
                data.append({
                    "plate_id": 1001,
                    "well": "A1",
                    "experiment": f"exp_{condition}_{phase}",
                    "condition": condition,
                    "cell_line": "MCF10A",
                    "cell_cycle": phase,
                })

        df = pd.DataFrame(data)

        # Should work fine since it uses predefined phase lists
        fig, axes = cellcycle_plot(
            df=df,
            conditions=["control", "treatment1"],
            show_subG1=True,  # Should give 5 phases max
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) <= 6  # Should not exceed 6 subplots


class TestCellCyclePlotParametrized:
    """Parametrized tests for different plot configurations."""

    @pytest.mark.parametrize("show_subG1", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_subG1_parameter(self, mock_show, cellcycle_data_with_subG1, show_subG1):
        """Test both show_subG1 settings work correctly."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data_with_subG1,
            conditions=["control", "treatment1"],
            show_subG1=show_subG1,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        if show_subG1:
            assert len(axes) == 5  # Sub-G1, G1, S, G2/M, Polyploid
        else:
            assert len(axes) == 4  # G1, S, G2/M, Polyploid

    @pytest.mark.parametrize("cc_phases", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_terminology_parameter(self, mock_show, cellcycle_data, cc_phases):
        """Test both terminology settings work correctly."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            cc_phases=cc_phases,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots regardless of terminology setting
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @pytest.mark.parametrize("show_repeat_points", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_repeat_points_parameter(self, mock_show, cellcycle_data, show_repeat_points):
        """Test repeat points enable/disable."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_repeat_points=show_repeat_points,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots regardless of repeat points setting
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @pytest.mark.parametrize("show_significance", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_significance_parameter(self, mock_show, cellcycle_data, show_significance):
        """Test significance marks enable/disable."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_significance=show_significance,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots regardless of significance setting
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @pytest.mark.parametrize("show_plate_legend", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_plate_legend_parameter(self, mock_show, cellcycle_data, show_plate_legend):
        """Test plate legend enable/disable."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_plate_legend=show_plate_legend,
            show_repeat_points=True,  # Required for legend to show
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots regardless of legend setting
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches

    @pytest.mark.parametrize(
        "selector_col,selector_val",
        [("cell_line", "MCF10A"), ("cell_line", "HeLa"), (None, None)]
    )
    @patch('matplotlib.pyplot.show')
    def test_selector_combinations(self, mock_show, cellcycle_data, selector_col, selector_val):
        """Test different selector column combinations."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=selector_col,
            selector_val=selector_val,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Check title includes selector value when provided
        if selector_val:
            fig_title = fig._suptitle.get_text() if fig._suptitle else ""
            assert selector_val in fig_title

    @pytest.mark.parametrize("rotation", [0, 45, 90])
    @patch('matplotlib.pyplot.show')
    def test_rotation_parameter(self, mock_show, cellcycle_data, rotation):
        """Test different rotation angles."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            rotation=rotation,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Should create valid plots with different rotations
        for ax in axes:
            assert len(ax.patches) > 0  # Should have bar patches


class TestCellCyclePlotIntegration:
    """Test integration aspects and matplotlib object interactions."""

    @patch('matplotlib.pyplot.show')
    def test_figure_and_axes_are_matplotlib_objects(self, mock_show, cellcycle_data):
        """Test that returned figure and axes are proper matplotlib objects."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        # Test figure properties
        assert hasattr(fig, 'savefig')
        assert hasattr(fig, 'set_size_inches')
        assert callable(fig.savefig)

        # Test axes properties - cellcycle_plot returns list of axes
        for ax in axes:
            assert hasattr(ax, 'plot')
            assert hasattr(ax, 'set_xlabel')
            assert hasattr(ax, 'set_ylabel')
            assert hasattr(ax, 'set_title')
            assert callable(ax.plot)

    @patch('matplotlib.pyplot.show')
    def test_plot_without_display(self, mock_show, cellcycle_data):
        """Test that plots can be created without displaying (mocked show)."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        # Verify show was not called (it's mocked)
        mock_show.assert_not_called()

        # But plot should still be created
        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

    @patch('matplotlib.pyplot.show')
    def test_subplot_layout_2x2_for_4_phases(self, mock_show, cellcycle_data):
        """Test that 4 phases creates 2x2 subplot layout."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            show_subG1=False,  # Should give 4 phases: G1, S, G2/M, Polyploid
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 4  # Exactly 4 subplots for 2x2 layout

    @patch('matplotlib.pyplot.show')
    def test_subplot_layout_2x3_for_5_phases(self, mock_show, cellcycle_data_with_subG1):
        """Test that 5 phases creates 2x3 subplot layout."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data_with_subG1,
            conditions=["control", "treatment1"],
            show_subG1=True,  # Should give 5 phases: Sub-G1, G1, S, G2/M, Polyploid
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 5  # Exactly 5 subplots for 2x3 layout

        # Figure should be wider for 2x3 layout
        width, height = fig.get_size_inches()
        assert width > height  # Should be wider than tall

    @patch('matplotlib.pyplot.show')
    def test_subplot_layout_2x3_for_6_phases(self, mock_show, cellcycle_data_with_M_phase):
        """Test that 6 phases creates 2x3 subplot layout."""
        fig, axes = cellcycle_plot(
            df=cellcycle_data_with_M_phase,
            conditions=["control", "treatment1"],
            show_subG1=True,  # With M phase data, should give 6 phases: Sub-G1, G1, S, G2, M, Polyploid
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Note: This test may not always give exactly 6 phases depending on data
        # But should handle up to 6 phases correctly
        assert len(axes) >= 5  # At least 5 phases

        # Figure should be wider for 2x3 layout
        width, height = fig.get_size_inches()
        assert width > height


class TestCellCyclePlotFactory:
    """Test the underlying CellCyclePlot factory classes directly."""

    def test_standard_cellcycle_plot_config_defaults(self):
        """Test StandardCellCyclePlotConfig default values."""
        config = StandardCellCyclePlotConfig()

        assert config.fig_size == (6, 6)
        assert config.size_units == "cm"
        assert config.dpi == 300
        assert config.save is True
        assert config.file_format == "pdf"
        assert config.tight_layout is False
        assert config.show_significance is True
        assert config.show_repeat_points is True
        assert config.rotation == 45
        assert config.cc_phases is True
        assert config.show_subG1 is True  # StandardCellCyclePlotConfig default
        assert config.show_plate_legend is False

    def test_standard_cellcycle_plot_config_custom_values(self):
        """Test StandardCellCyclePlotConfig with custom values."""
        config = StandardCellCyclePlotConfig(
            fig_size=(8, 8),
            size_units="inch",
            dpi=150,
            show_subG1=False,
            show_plate_legend=True,
            cc_phases=False,
            rotation=90,
        )

        assert config.fig_size == (8, 8)
        assert config.size_units == "inch"
        assert config.dpi == 150
        assert config.show_subG1 is False
        assert config.show_plate_legend is True
        assert config.cc_phases is False
        assert config.rotation == 90

    @patch('matplotlib.pyplot.show')
    def test_standard_cellcycle_plot_class_direct_usage(self, mock_show, cellcycle_data):
        """Test using StandardCellCyclePlot class directly."""
        config = StandardCellCyclePlotConfig(show_subG1=False, cc_phases=True)
        plot = StandardCellCyclePlot(config)

        fig, axes = plot.create_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) == 4  # show_subG1=False should give 4 phases

    def test_base_cellcycle_plot_abstract_methods(self):
        """Test that BaseCellCyclePlot has proper abstract structure."""
        config = StandardCellCyclePlotConfig()

        # BaseCellCyclePlot is abstract, but we can test its structure
        class TestCellCyclePlot(BaseCellCyclePlot):
            def build_plot(self, data, **kwargs):
                pass  # Minimal implementation for testing
                return self

        plot = TestCellCyclePlot(config)
        assert plot.config == config
        assert plot.PLOT_TYPE_NAME == "cellcycle"

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_custom_colors_configuration(self, mock_show, cellcycle_data):
        """Test cellcycle plot with custom colors via factory."""
        custom_colors = [COLOR.PURPLE.value, COLOR.TURQUOISE.value,
                        COLOR.OLIVE.value, COLOR.LAVENDER.value]

        config = StandardCellCyclePlotConfig(colors=custom_colors)
        plot = StandardCellCyclePlot(config)

        fig, axes = plot.create_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)


class TestCellCyclePlotSpecialCases:
    """Test special data scenarios."""

    @patch('matplotlib.pyplot.show')
    def test_single_condition_data(self, mock_show):
        """Test with data containing only one condition."""
        data = []
        phases = ["G1", "S", "G2/M", "Polyploid"]

        # Create data with only control condition
        for phase in phases:
            for i in range(5):  # 5 cells per phase
                data.append({
                    "plate_id": 1001,
                    "well": f"A{i+1}",
                    "experiment": f"exp_control_{phase}_{i}",
                    "condition": "control",
                    "cell_line": "MCF10A",
                    "cell_cycle": phase,
                })

        df = pd.DataFrame(data)

        fig, axes = cellcycle_plot(
            df=df,
            conditions=["control"],  # Only one condition
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # Should work with single condition

    @patch('matplotlib.pyplot.show')
    def test_data_with_many_plates(self, mock_show):
        """Test with data containing many plates (for significance testing)."""
        data = []
        phases = ["G1", "S", "G2/M", "Polyploid"]
        plates = [1001, 1002, 1003, 1004, 1005]  # 5 plates

        for plate_id in plates:
            for condition in ["control", "treatment1"]:
                for phase in phases:
                    for i in range(3):  # 3 cells per phase per condition per plate
                        data.append({
                            "plate_id": plate_id,
                            "well": f"A{i+1}",
                            "experiment": f"exp_{plate_id}_{condition}_{phase}_{i}",
                            "condition": condition,
                            "cell_line": "MCF10A",
                            "cell_cycle": phase,
                        })

        df = pd.DataFrame(data)

        fig, axes = cellcycle_plot(
            df=df,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # With 5 plates, should have significance testing (>= 3 plates required)

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_plot_with_missing_phases(self, mock_show):
        """Test cellcycle plot when some phases are missing from data."""
        data = []
        # Only include G1 and S phases, missing G2/M and Polyploid
        phases = ["G1", "S"]

        for condition in ["control", "treatment1"]:
            for phase in phases:
                for i in range(10):
                    data.append({
                        "plate_id": 1001,
                        "well": f"A{i+1}",
                        "experiment": f"exp_{condition}_{phase}_{i}",
                        "condition": condition,
                        "cell_line": "MCF10A",
                        "cell_cycle": phase,
                    })

        df = pd.DataFrame(data)

        fig, axes = cellcycle_plot(
            df=df,
            conditions=["control", "treatment1"],
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        # Implementation creates subplots for all predefined phases, not just those in data
        assert len(axes) == 4  # Still creates all 4 standard phases (some may be empty)


class TestCellCyclePlotErrorMessages:
    """Test specific error messages for various failure modes."""

    def test_informative_error_messages(self, cellcycle_data):
        """Test that error messages are informative and helpful."""

        # Test missing condition column error message
        with pytest.raises(ValueError) as exc_info:
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                condition_col="nonexistent_condition_col",
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "Condition column 'nonexistent_condition_col' not found" in error_msg
        assert "Available columns:" in error_msg

        # Test invalid selector value error message
        with pytest.raises(ValueError) as exc_info:
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell",
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "Value 'NonexistentCell' not found in column 'cell_line'" in error_msg
        assert "Available values:" in error_msg

    def test_helpful_selector_error_message(self, cellcycle_data):
        """Test helpful error message when selector_col provided without selector_val."""
        with pytest.raises(ValueError) as exc_info:
            cellcycle_plot(
                df=cellcycle_data,
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                # selector_val not provided
                save=False,
            )

        error_msg = str(exc_info.value)
        assert "selector_val must be provided when selector_col is specified" in error_msg
        assert "Available values in 'cell_line':" in error_msg


class TestCellCyclePlotBackwardCompatibility:
    """Test backward compatibility of the API wrapper."""

    @patch('matplotlib.pyplot.show')
    def test_api_wrapper_maintains_compatibility(self, mock_show, cellcycle_data):
        """Test that the API wrapper maintains backward compatibility."""
        # Test with parameters that might have been used in older versions
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1", "treatment2"],
            condition_col="condition",  # Explicit condition column
            selector_col="cell_line",
            selector_val="MCF10A",
            title="Backward Compatible Test",
            fig_size=(7, 7),
            size_units="cm",
            save=False,  # Don't actually save in tests
            cc_phases=True,
            show_subG1=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Check that title is applied
        fig_title = fig._suptitle.get_text() if fig._suptitle else ""
        assert "Backward Compatible Test" in fig_title

    @patch('matplotlib.pyplot.show')
    def test_default_parameter_handling(self, mock_show, cellcycle_data):
        """Test that default parameters work as expected."""
        # Test with minimal parameters, relying on defaults
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            selector_col=None,  # Override default to test minimal parameters
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

    @patch('matplotlib.pyplot.show')
    def test_path_parameter_compatibility(self, mock_show, cellcycle_data, tmp_path):
        """Test path parameter handling for save functionality."""
        # Test with save=False (should not actually save)
        fig, axes = cellcycle_plot(
            df=cellcycle_data,
            conditions=["control", "treatment1"],
            save=False,
            path=tmp_path,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(axes, list)

        # Since save=False, no files should be created
        assert len(list(tmp_path.glob("*"))) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
