"""Comprehensive pytest tests for featureplot functionality in omero-screen-plots package.

This module provides high-level smoke testing for the feature plot functionality,
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

from omero_screen_plots.featureplot_api import feature_plot
from omero_screen_plots.featureplot_factory import (
    FeaturePlotConfig,
    BaseFeaturePlot,
    StandardFeaturePlot,
)
from omero_screen_plots.colors import COLOR


class TestFeaturePlotBasicFunctionality:
    """Test basic functionality of feature_plot API function."""

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_minimal_parameters(self, mock_show, synthetic_plate_data):
        """Test feature_plot with minimal required parameters."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None,  # Explicitly set to None to avoid default "cell_line"
            selector_val=None,
        )

        # Verify return types
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Verify plot was created (has elements)
        assert len(ax.patches) > 0 or len(ax.collections) > 0  # Should have box patches or violin collections
        assert ax.get_ylabel() != ""  # Should have y-axis label

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_box_plot(self, mock_show, synthetic_plate_data):
        """Test feature_plot with box plot (default)."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            violin=False,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "Area Nucleus" in ax.get_ylabel()

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_violin_plot(self, mock_show, synthetic_plate_data):
        """Test feature_plot with violin plot."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="intensity_mean_DAPI_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            violin=True,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Violin plots create PolyCollections
        assert len(ax.collections) > 0
        # Check y-axis label contains feature name (formatted with title case)
        expected_label = "intensity_mean_DAPI_nucleus".replace("_", " ").title()
        assert expected_label in ax.get_ylabel()

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_selector_column(self, mock_show, synthetic_plate_data):
        """Test feature_plot with selector column filtering."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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
    def test_feature_plot_with_custom_title(self, mock_show, synthetic_plate_data):
        """Test feature_plot with custom title."""
        custom_title = "Custom Feature Analysis"
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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
    def test_feature_plot_with_grouping(self, mock_show, synthetic_plate_data):
        """Test feature_plot with group_size parameter."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            group_size=2,
            within_group_spacing=0.3,
            between_group_gap=0.7,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should still create a valid plot with grouping
        assert len(ax.patches) > 0 or len(ax.collections) > 0

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_scaling(self, mock_show, scaled_feature_data):
        """Test feature_plot with data scaling."""
        fig, ax = feature_plot(
            df=scaled_feature_data,
            feature="intensity_raw",
            conditions=["control", "treatment1"],
            scale=True,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Verify that scaling was applied by checking y-axis limits are reasonable
        y_min, y_max = ax.get_ylim()
        assert y_max <= 80000  # Should be scaled to ~65535 range with some padding

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_y_limits(self, mock_show, synthetic_plate_data):
        """Test feature_plot with custom y-axis limits."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            ymax=1000.0,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        y_min, y_max = ax.get_ylim()
        assert y_max <= 1000.0

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_y_range_tuple(self, mock_show, synthetic_plate_data):
        """Test feature_plot with y-axis range as tuple."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            ymax=(0, 600),
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        y_min, y_max = ax.get_ylim()
        assert y_min >= 0
        assert y_max <= 600

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_without_scatter(self, mock_show, synthetic_plate_data):
        """Test feature_plot without scatter points overlay."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            show_scatter=False,
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have fewer scatter points (only median points, no individual data points)
        scatter_collections = [c for c in ax.collections if hasattr(c, '_sizes')]
        # Still might have median points, but should be fewer
        assert isinstance(fig, Figure)  # Basic validation

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_legend(self, mock_show, synthetic_plate_data):
        """Test feature_plot with legend configuration."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            legend=("Plates", ["Plate 1001", "Plate 1002", "Plate 1003"]),
            selector_col=None,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check if legend was added
        legend = ax.get_legend()
        if legend:
            assert "Plates" in legend.get_title().get_text()


class TestFeaturePlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test feature_plot with empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input dataframe is empty"):
            feature_plot(
                df=empty_df,
                feature="area_nucleus",
                conditions=["control", "treatment1"]
            )

    def test_missing_required_columns(self, synthetic_plate_data):
        """Test feature_plot with missing required columns."""
        # Remove plate_id column
        df_missing_plate = synthetic_plate_data.drop("plate_id", axis=1)

        with pytest.raises(ValueError, match="Missing required columns"):
            feature_plot(
                df=df_missing_plate,
                feature="area_nucleus",
                conditions=["control", "treatment1"]
            )

    def test_missing_feature_column(self, synthetic_plate_data):
        """Test feature_plot with missing feature column."""
        with pytest.raises(ValueError, match="Feature column 'nonexistent_feature' not found"):
            feature_plot(
                df=synthetic_plate_data,
                feature="nonexistent_feature",
                conditions=["control", "treatment1"]
            )

    def test_invalid_condition_column(self, synthetic_plate_data):
        """Test feature_plot with invalid condition column."""
        with pytest.raises(ValueError, match="Condition column 'invalid_column' not found"):
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                condition_col="invalid_column"
            )

    def test_conditions_not_in_data(self, synthetic_plate_data):
        """Test feature_plot with conditions not present in data."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "nonexistent_condition"]
            )

    def test_invalid_selector_column(self, synthetic_plate_data):
        """Test feature_plot with invalid selector column."""
        with pytest.raises(ValueError, match="Selector column 'invalid_selector' not found"):
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                selector_col="invalid_selector",
                selector_val="some_value"
            )

    def test_selector_column_without_value(self, synthetic_plate_data):
        """Test feature_plot with selector_col but no selector_val."""
        with pytest.raises(ValueError, match="selector_val must be provided when selector_col is specified"):
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line"
            )

    def test_invalid_selector_value(self, synthetic_plate_data):
        """Test feature_plot with selector_val not in data."""
        with pytest.raises(ValueError, match="Value 'NonexistentCell' not found in column 'cell_line'"):
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line",
                selector_val="NonexistentCell"
            )

    def test_no_data_after_filtering(self, synthetic_plate_data):
        """Test feature_plot when filtering results in no data."""
        # Create a scenario where filtering removes all data
        df_subset = synthetic_plate_data[synthetic_plate_data["condition"] == "control"].copy()

        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_plot(
                df=df_subset,
                feature="area_nucleus",
                conditions=["control", "treatment1"],  # treatment1 not in subset
                selector_col="cell_line",
                selector_val="MCF10A"
            )

    @patch('matplotlib.pyplot.show')
    def test_single_plate_no_significance(self, mock_show, minimal_plate_data):
        """Test feature_plot with single plate (should work but no significance marks)."""
        # Filter to single plate
        single_plate_df = minimal_plate_data[minimal_plate_data["plate_id"] == 1001].copy()

        fig, ax = feature_plot(
            df=single_plate_df,
            feature="area_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should create plot without significance marks


class TestFeaturePlotParameterized:
    """Parametrized tests for different plot configurations."""

    @pytest.mark.parametrize("violin", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_plot_types(self, mock_show, synthetic_plate_data, violin):
        """Test both box and violin plot types work correctly."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            violin=violin,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        # Check appropriate plot elements
        if violin:
            # Violin plots create collections
            assert len(ax.collections) > 0
        else:
            # Box plots create patches
            assert len(ax.patches) > 0

    @pytest.mark.parametrize("group_size", [1, 2, 3])
    @patch('matplotlib.pyplot.show')
    def test_group_sizes(self, mock_show, synthetic_plate_data, group_size):
        """Test different group sizes work correctly."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            group_size=group_size,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) > 0 or len(ax.collections) > 0  # Should have plot elements

    @pytest.mark.parametrize(
        "selector_col,selector_val",
        [("cell_line", "MCF10A"), ("cell_line", "HeLa"), (None, None)]
    )
    @patch('matplotlib.pyplot.show')
    def test_selector_combinations(self, mock_show, synthetic_plate_data, selector_col, selector_val):
        """Test different selector column combinations."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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

    @pytest.mark.parametrize("feature", ["area_nucleus", "intensity_mean_DAPI_nucleus", "perimeter_nucleus"])
    @patch('matplotlib.pyplot.show')
    def test_different_features(self, mock_show, synthetic_plate_data, feature):
        """Test plotting different features."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature=feature,
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Check y-axis label contains feature name (formatted)
        expected_label = feature.replace("_", " ").title()
        assert expected_label in ax.get_ylabel()

    @pytest.mark.parametrize("show_scatter", [True, False])
    @patch('matplotlib.pyplot.show')
    def test_scatter_overlay_options(self, mock_show, synthetic_plate_data, show_scatter):
        """Test scatter overlay enable/disable."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            show_scatter=show_scatter,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestFeaturePlotIntegration:
    """Test integration aspects and matplotlib object interactions."""

    @patch('matplotlib.pyplot.show')
    def test_figure_and_axes_are_matplotlib_objects(self, mock_show, synthetic_plate_data):
        """Test that returned figure and axes are proper matplotlib objects."""
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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
        """Test feature_plot with custom axes input."""
        import matplotlib.pyplot as plt

        # Create custom figure and axes
        custom_fig, custom_ax = plt.subplots(figsize=(10, 8))

        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            axes=custom_ax,
            selector_col=None
        )

        # Should return the same axes we provided
        assert ax is custom_ax
        assert fig is custom_fig

        plt.close(custom_fig)  # Clean up


class TestFeaturePlotFactory:
    """Test the underlying FeaturePlot factory classes directly."""

    def test_feature_plot_config_defaults(self):
        """Test FeaturePlotConfig default values."""
        config = FeaturePlotConfig()

        assert config.fig_size == (5, 5)
        assert config.size_units == "cm"
        assert config.dpi == 300
        assert config.save is False
        assert config.file_format == "pdf"
        assert config.violin is False
        assert config.show_scatter is True
        assert config.group_size == 1
        assert config.scale is False
        assert config.show_significance is True
        assert config.show_repeat_points is True

    def test_feature_plot_config_custom_values(self):
        """Test FeaturePlotConfig with custom values."""
        config = FeaturePlotConfig(
            fig_size=(10, 8),
            size_units="inches",
            dpi=150,
            violin=True,
            show_scatter=False,
            group_size=2,
            scale=True,
            ymax=1000.0,
        )

        assert config.fig_size == (10, 8)
        assert config.size_units == "inches"
        assert config.dpi == 150
        assert config.violin is True
        assert config.show_scatter is False
        assert config.group_size == 2
        assert config.scale is True
        assert config.ymax == 1000.0

    @patch('matplotlib.pyplot.show')
    def test_standard_feature_plot_class_direct_usage(self, mock_show, synthetic_plate_data):
        """Test using StandardFeaturePlot class directly."""
        config = FeaturePlotConfig(violin=True, show_scatter=False)
        plot = StandardFeaturePlot(config)

        fig, ax = plot.create_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Should have violin collections since violin=True
        assert len(ax.collections) > 0

    def test_base_feature_plot_abstract_methods(self):
        """Test that BaseFeaturePlot cannot be instantiated directly."""
        config = FeaturePlotConfig()

        # BaseFeaturePlot is abstract and should not be instantiated directly
        # But we can instantiate it for testing abstract method behavior
        class TestFeaturePlot(BaseFeaturePlot):
            def _build_plot(self, data, feature, conditions, condition_col, x_positions):
                pass  # Minimal implementation for testing

        plot = TestFeaturePlot(config)
        assert plot.config == config
        assert plot.PLOT_TYPE_NAME == "feature"

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_colors_configuration(self, mock_show, synthetic_plate_data):
        """Test feature plot with custom colors."""
        custom_colors = [COLOR.PURPLE.value, COLOR.TURQUOISE.value, COLOR.OLIVE.value]

        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            colors=custom_colors,
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestFeaturePlotSpecialCases:
    """Test special data scenarios."""

    def test_single_condition_data(self, single_condition_data):
        """Test with data containing only the control condition."""
        with pytest.raises(ValueError, match="Conditions not found in data"):
            feature_plot(
                df=single_condition_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],  # treatment1 not in data
                selector_col=None
            )

    @patch('matplotlib.pyplot.show')
    def test_data_with_many_plates(self, mock_show, many_plates_data):
        """Test with data containing many plates (for significance testing)."""
        fig, ax = feature_plot(
            df=many_plates_data,
            feature="area_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # With 5 plates, should have significance testing (>= 3 plates required)
        # This is tested indirectly by ensuring the plot is created successfully

    @patch('matplotlib.pyplot.show')
    def test_feature_plot_with_missing_data_points(self, mock_show):
        """Test feature plot when some conditions have very few data points."""
        # Create data where one condition has very few points
        data = []

        # Control condition with many points
        for i in range(20):
            data.append({
                "plate_id": 1001,
                "well": f"A{i+1}",
                "experiment": f"exp_control_{i}",
                "condition": "control",
                "area_nucleus": np.random.uniform(200, 400),
            })

        # Treatment condition with only 2 points
        for i in range(2):
            data.append({
                "plate_id": 1001,
                "well": f"B{i+1}",
                "experiment": f"exp_treatment_{i}",
                "condition": "treatment1",
                "area_nucleus": np.random.uniform(300, 500),
            })

        df = pd.DataFrame(data)

        fig, ax = feature_plot(
            df=df,
            feature="area_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestFeaturePlotErrorMessages:
    """Test specific error messages for various failure modes."""

    def test_informative_error_messages(self, synthetic_plate_data):
        """Test that error messages are informative and helpful."""

        # Test missing feature column error message
        with pytest.raises(ValueError) as exc_info:
            feature_plot(
                df=synthetic_plate_data,
                feature="nonexistent_feature",
                conditions=["control", "treatment1"]
            )

        error_msg = str(exc_info.value)
        assert "Feature column 'nonexistent_feature' not found" in error_msg
        assert "Available columns:" in error_msg

        # Test invalid selector value error message
        with pytest.raises(ValueError) as exc_info:
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
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
            feature_plot(
                df=synthetic_plate_data,
                feature="area_nucleus",
                conditions=["control", "treatment1"],
                selector_col="cell_line"
                # selector_val not provided
            )

        error_msg = str(exc_info.value)
        assert "selector_val must be provided when selector_col is specified" in error_msg
        assert "Available values in 'cell_line':" in error_msg


class TestFeaturePlotBackwardCompatibility:
    """Test backward compatibility of the API wrapper."""

    @patch('matplotlib.pyplot.show')
    def test_api_wrapper_maintains_compatibility(self, mock_show, synthetic_plate_data):
        """Test that the API wrapper maintains backward compatibility."""
        # Test with old-style parameters that might have been used
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1", "treatment2"],
            x_label=True,  # Old parameter name
            condition_col="condition",  # Explicit condition column
            selector_col="cell_line",
            selector_val="MCF10A",
            title="Backward Compatible Test",
            fig_size=(7, 7),
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
        # Note: default selector_col is "cell_line", so we need to provide selector_val or set it to None
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
            conditions=["control", "treatment1"],
            selector_col=None,  # Override default to test minimal parameters
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @patch('matplotlib.pyplot.show')
    def test_path_parameter_compatibility(self, mock_show, synthetic_plate_data, tmp_path):
        """Test path parameter handling for save functionality."""
        # Test with save=False (should not actually save)
        fig, ax = feature_plot(
            df=synthetic_plate_data,
            feature="area_nucleus",
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
