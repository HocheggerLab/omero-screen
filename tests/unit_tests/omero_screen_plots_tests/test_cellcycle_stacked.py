"""Comprehensive pytest tests for cellcycle_stacked functionality in omero-screen-plots package.

This module provides high-level smoke testing for the stacked cell cycle plot functionality,
focusing on testing the main API functions, class-based architecture, and error
handling without validating visual output.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from unittest.mock import patch

from omero_screen_plots.cellcycleplot_api import cellcycle_stacked, cellcycle_grouped
from omero_screen_plots.cellcycleplot_factory import (
    StackedCellCyclePlotConfig,
    StackedCellCyclePlot,
    BaseCellCyclePlot,
)
from omero_screen_plots.colors import COLOR


@pytest.fixture
def cellcycle_stacked_data():
    """Create synthetic cellcycle dataset optimized for stacked plot testing."""
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
                # 30-60 cells per well for good statistical power
                n_cells = np.random.randint(30, 61)

                # Generate phases based on realistic distribution
                cell_phases = np.random.choice(phases, size=n_cells, p=phase_weights)

                # Modify phase distribution based on condition for testing
                if condition == "treatment1":
                    # Treatment1 shifts towards G2/M (mitotic arrest)
                    cell_phases = np.where(
                        np.random.random(n_cells) < 0.2,
                        "G2/M",
                        cell_phases
                    )
                elif condition == "treatment2":
                    # Treatment2 increases S phase (replication stress)
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
def minimal_cellcycle_stacked_data():
    """Create minimal cellcycle dataset for basic stacked plot testing."""
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
        {"plate_id": 1003, "well": "A1", "experiment": "exp8", "condition": "control",
         "cell_line": "MCF10A", "cell_cycle": "Polyploid"},
        {"plate_id": 1003, "well": "A2", "experiment": "exp9", "condition": "treatment1",
         "cell_line": "MCF10A", "cell_cycle": "Polyploid"},
    ]
    return pd.DataFrame(data)


class TestCellCycleStackedBasicFunctionality:
    """Test basic functionality of cellcycle_stacked API function."""

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_stacked_minimal_parameters(self, mock_show, cellcycle_stacked_data):
        """Test cellcycle_stacked with minimal required parameters (summary mode)."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,  # Don't use selector filtering
            save=False,  # Don't save in tests
        )

        # Verify return types - cellcycle_stacked returns (Figure, Axes)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify stacked bars were created
        assert len(ax.patches) > 0  # Should have bar patches

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_stacked_with_selector_column(self, mock_show, cellcycle_stacked_data):
        """Test cellcycle_stacked with selector column filtering."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
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
    def test_cellcycle_stacked_summary_mode(self, mock_show, cellcycle_stacked_data):
        """Test cellcycle_stacked in summary mode with error bars."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            show_triplicates=False,  # Summary mode
            show_error_bars=True,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Should have stacked bars
        assert len(ax.patches) > 0

    @patch('matplotlib.pyplot.show')
    def test_cellcycle_stacked_with_axes_parameter(self, mock_show, cellcycle_stacked_data):
        """Test cellcycle_stacked with external axes provided."""
        # Create external figure and axes
        external_fig, external_ax = plt.subplots(figsize=(8, 6))

        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            axes=external_ax,
            selector_col=None,
            save=False,
        )

        # Should return the same figure and axes objects
        assert fig is external_fig
        assert ax is external_ax

        # Should still create bars on the provided axes
        assert len(ax.patches) > 0


class TestCellCycleStackedEdgeCases:
    """Test edge cases and error conditions for cellcycle_stacked."""

    def test_empty_dataframe(self):
        """Test cellcycle_stacked with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            cellcycle_stacked(
                df=empty_df,
                conditions=["control", "treatment1"],
                selector_col=None,
                save=False,
            )

    def test_missing_required_columns(self, cellcycle_stacked_data):
        """Test cellcycle_stacked with missing required columns."""
        # Remove plate_id column
        df_missing_plate = cellcycle_stacked_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            cellcycle_stacked(
                df=df_missing_plate,
                conditions=["control", "treatment1"],
                selector_col=None,
                save=False,
            )

    def test_conditions_not_in_data(self, cellcycle_stacked_data):
        """Test cellcycle_stacked with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            cellcycle_stacked(
                df=cellcycle_stacked_data,
                conditions=["control", "nonexistent_condition"],
                selector_col=None,
                save=False,
            )


class TestCellCycleStackedFactory:
    """Test the underlying StackedCellCyclePlot factory classes directly."""

    def test_stacked_cellcycle_plot_config_defaults(self):
        """Test StackedCellCyclePlotConfig default values."""
        config = StackedCellCyclePlotConfig()

        # Base config defaults
        assert config.fig_size == (6, 6)
        assert config.size_units == "cm"
        assert config.dpi == 300
        assert config.save is True
        assert config.file_format == "pdf"
        assert config.tight_layout is False
        assert config.rotation == 45
        assert config.cc_phases is True

        # Stacked-specific defaults
        assert config.show_triplicates is False
        assert config.show_error_bars is True
        assert config.show_boxes is True
        assert config.group_size == 1
        assert config.within_group_spacing == 0.2
        assert config.between_group_gap == 0.5
        assert config.bar_width == 0.5
        assert config.repeat_offset == 0.18
        assert config.max_repeats == 3
        assert config.show_legend is True
        assert config.y_max == 110

    @patch('matplotlib.pyplot.show')
    def test_stacked_cellcycle_plot_class_direct_usage(self, mock_show, cellcycle_stacked_data):
        """Test using StackedCellCyclePlot class directly."""
        config = StackedCellCyclePlotConfig(
            show_triplicates=False,
            show_error_bars=True,
            cc_phases=True
        )
        plot = StackedCellCyclePlot(config)

        fig, ax = plot.create_plot(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    def test_base_cellcycle_plot_inheritance(self):
        """Test that StackedCellCyclePlot inherits from BaseCellCyclePlot properly."""
        config = StackedCellCyclePlotConfig()
        plot = StackedCellCyclePlot(config)

        assert isinstance(plot, BaseCellCyclePlot)
        assert plot.config == config
        assert plot.PLOT_TYPE_NAME == "cellcycle_stacked"


class TestCellCycleStackedBackwardCompatibility:
    """Test backward compatibility, especially cellcycle_grouped deprecation."""

    def test_cellcycle_grouped_deprecation_warning(self, cellcycle_stacked_data):
        """Test that cellcycle_grouped raises deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            fig, ax = cellcycle_grouped(
                df=cellcycle_stacked_data,
                conditions=["control", "treatment1"],
                selector_col=None,
                save=False,
            )

            # Should raise deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "cellcycle_grouped is deprecated" in str(w[0].message)
            assert "cellcycle_stacked" in str(w[0].message)

            # Should still work
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)


class TestCellCycleStackedParametrized:
    """Parametrized tests for different cellcycle_stacked configurations."""

    @pytest.mark.parametrize("show_triplicates", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_display_modes(self, mock_show, cellcycle_stacked_data, show_triplicates):
        """Test both display modes (summary and triplicates)."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            show_triplicates=show_triplicates,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @pytest.mark.parametrize("show_error_bars", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_error_bars_parameter(self, mock_show, cellcycle_stacked_data, show_error_bars):
        """Test error bars enable/disable in summary mode."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            show_triplicates=False,  # Summary mode
            show_error_bars=show_error_bars,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @pytest.mark.parametrize("cc_phases", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_terminology_parameter(self, mock_show, cellcycle_stacked_data, cc_phases):
        """Test both terminology settings."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            cc_phases=cc_phases,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

    @pytest.mark.parametrize("show_legend", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_legend_parameter(self, mock_show, cellcycle_stacked_data, show_legend):
        """Test legend enable/disable."""
        fig, ax = cellcycle_stacked(
            df=cellcycle_stacked_data,
            conditions=["control", "treatment1"],
            show_legend=show_legend,
            selector_col=None,
            save=False,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0

        if show_legend:
            assert ax.get_legend() is not None
        else:
            assert ax.get_legend() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
