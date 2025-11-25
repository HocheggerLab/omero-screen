"""Scatter plot factory with unified configuration and base class architecture."""

from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from omero_screen_plots.base import (
    BaseDataProcessor,
    BasePlotBuilder,
    XYPlotConfig,
)
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import prepare_plot_data


@dataclass
class ScatterPlotConfig(XYPlotConfig):
    """Configuration for scatter plots."""

    # Plot features (inherited from XYPlotConfig, but setting defaults)
    x_feature: str = "integrated_int_DAPI_norm"
    y_feature: str = "intensity_mean_EdU_nucleus_norm"

    # Data sampling
    cell_number: int | None = None  # Number of cells to sample per condition
    random_state: int = 42  # Random seed for reproducible sampling

    # Hue settings
    hue: str | None = None
    hue_order: list[str] | None = None
    palette: list[str] | dict[str, str] | None = None

    # Scale settings (inherited from XYPlotConfig, setting defaults)
    x_scale: Literal["linear", "log"] = "log"
    x_scale_base: int = 2
    y_scale: Literal["linear", "log"] = "log"
    y_scale_base: int = 10

    # Axis limits (inherited from XYPlotConfig)
    x_limits: tuple[float, float] | None = (1, 16)
    y_limits: tuple[float, float] | None = None

    # Scatter plot settings
    size: float = 2
    alpha: float = 1.0

    # KDE overlay settings
    kde_overlay: bool | None = False
    kde_params: dict[str, Any] = field(
        default_factory=lambda: {
            "fill": True,
            "alpha": 0.3,
            "cmap": "rocket_r",
        }
    )

    # Reference lines
    vline: float | None = None
    hline: float | None = None
    line_style: str = "--"
    line_color: str = "black"

    # Grid settings (inherited)
    grid: bool = False

    # Title and labels (inherited)
    show_title: bool = False

    # Legend settings
    show_legend: bool = True
    legend_loc: str = "best"
    legend_title: str | None = None

    # Threshold settings (for categorical coloring)
    threshold: float | None = None
    threshold_colors: dict[str, str] = field(
        default_factory=lambda: {
            "below": COLOR.LIGHT_BLUE.value,
            "above": COLOR.BLUE.value,
        }
    )


class ScatterDataProcessor(BaseDataProcessor):
    """Processes data for scatter plots."""

    def validate_dataframe(self) -> None:
        """Validate required columns exist."""
        # Basic validation is handled by prepare_plot_data for now
        # We can add more specific validation here if needed

    def process_data(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Process data for scatter plot.

        Args:
            df: Input DataFrame
            **kwargs:
                x_feature: str
                y_feature: str
                cell_number: int | None
                random_state: int

        Returns:
            Processed DataFrame
        """
        x_feature = kwargs.get("x_feature")
        y_feature = kwargs.get("y_feature")
        cell_number = kwargs.get("cell_number")
        random_state = kwargs.get("random_state", 42)

        if not x_feature or not y_feature:
            raise ValueError("x_feature and y_feature are required")

        # Validate features exist
        if x_feature not in df.columns:
            raise ValueError(f"x_feature '{x_feature}' not found in dataframe")
        if y_feature not in df.columns:
            raise ValueError(f"y_feature '{y_feature}' not found in dataframe")

        data = df.copy()

        # Sample data if cell_number is specified
        if cell_number and len(data) > cell_number:
            data = data.sample(
                n=cell_number,
                random_state=random_state,
            )

        return data


class ScatterPlot(BasePlotBuilder):
    """Builder for scatter plots."""

    def __init__(self, config: ScatterPlotConfig):
        """Initialize with specific config type."""
        super().__init__(config)
        self.config: ScatterPlotConfig = config  # Type narrowing

    def build_plot(self, data: pd.DataFrame, **kwargs: Any) -> "ScatterPlot":
        """Build scatter plot.

        Args:
            data: Processed DataFrame
            **kwargs:
                hue: str | None
        """
        if self.ax is None:
            raise RuntimeError("Must create figure before building plot")

        # Extract plot parameters
        x_feature = self.config.x_feature
        y_feature = self.config.y_feature
        hue = kwargs.get("hue", self.config.hue)

        # Apply threshold if specified (overrides other hue settings)
        plot_data = data.copy()
        if (
            self.config.threshold is not None
            and y_feature
            and y_feature in plot_data.columns
        ):
            plot_data["threshold_category"] = plot_data[y_feature].apply(
                lambda x: "below" if x < self.config.threshold else "above"
            )
            hue = "threshold_category"

        # Validate hue column exists (after potential threshold creation)
        if hue and hue not in plot_data.columns:
            raise ValueError(f"hue column '{hue}' not found in dataframe")

        # Create scatter plot
        scatter_params = {
            "data": plot_data,
            "x": x_feature,
            "y": y_feature,
            "s": self.config.size,
            "alpha": self.config.alpha,
            "ax": self.ax,
        }

        # Add hue parameters if specified
        if hue:
            scatter_params["hue"] = hue

            # Set hue order and palette based on hue type
            if hue == "cell_cycle":
                # Use standard cell cycle phase order (reverse for display)
                phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
                # Filter to only phases present in data
                available_phases = plot_data[hue].unique()
                scatter_params["hue_order"] = [
                    p for p in phases if p in available_phases
                ]

                # Use colors from matplotlib style (first 5 colors)
                if not self.config.palette:
                    # Get colors from current style
                    prop_cycle = plt.rcParams["axes.prop_cycle"]
                    style_colors = prop_cycle.by_key()["color"]
                    # Use first N colors for N phases
                    n_phases = len(scatter_params["hue_order"])
                    scatter_params["palette"] = style_colors[:n_phases]
                else:
                    scatter_params["palette"] = self.config.palette
            elif hue == "threshold_category":
                scatter_params["hue_order"] = ["below", "above"]
                scatter_params["palette"] = self.config.threshold_colors
            else:
                if self.config.hue_order:
                    scatter_params["hue_order"] = self.config.hue_order
                if self.config.palette:
                    scatter_params["palette"] = self.config.palette

        sns.scatterplot(**scatter_params)

        # Add KDE overlay if requested (always for cell cycle plots)
        if self.config.kde_overlay:
            kde_params = self.config.kde_params.copy()
            # KDE plot should not use hue - it's a density overlay
            sns.kdeplot(
                data=plot_data,
                x=x_feature,
                y=y_feature,
                ax=self.ax,
                **kde_params,
            )

        # Format axes
        if x_feature and y_feature:
            self._format_axes(x_feature, y_feature)

        # Add reference lines
        self._add_reference_lines()

        # Handle legend
        self._configure_legend()

        # Finalize with default title
        default_title = f"Scatter: {x_feature} vs {y_feature}"

        # Only show title if requested and not using provided axes (to avoid double titles)
        if self.config.show_title and not self.axes_provided:
            self.finalize_plot(default_title)
        elif self.fig is not None:
            self._filename = default_title.replace(" ", "_")

        return self

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        x_feature: str,
        y_feature: str,
        condition_col: str,
        conditions: list[str],
    ) -> None:
        """Validate input parameters early to provide helpful error messages."""
        # Check required columns exist
        required_cols = [x_feature, y_feature, condition_col]
        if missing_cols := [
            col for col in required_cols if col not in df.columns
        ]:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        # Validate conditions exist in data
        # Note: We only check if the condition column contains the requested conditions
        # We don't check selector_val here, allowing it to filter to empty
        available_conditions = df[condition_col].unique()
        if invalid_conditions := [
            c for c in conditions if c not in available_conditions
        ]:
            raise ValueError(
                f"Invalid conditions: {invalid_conditions}. "
                f"Available: {list(available_conditions)}"
            )

    def create_plot(
        self,
        df: pd.DataFrame,
        conditions: str | list[str],
        condition_col: str = "condition",
        selector_col: str | None = None,
        selector_val: str | None = None,
        axes: Any | None = None,
    ) -> tuple[Figure, Any]:
        """Create scatter plot using class-based approach.

        This method is maintained for backward compatibility with the API.
        """
        # Handle single vs multiple conditions
        conditions_list = (
            [conditions] if isinstance(conditions, str) else conditions
        )

        # Validate axes usage
        if axes is not None and len(conditions_list) > 1:
            raise ValueError(
                "Cannot use multiple conditions when axes is provided"
            )

        # Get features from config
        x_feature = self.config.x_feature
        y_feature = self.config.y_feature

        # Validate inputs early
        self._validate_inputs(
            df, x_feature, y_feature, condition_col, conditions_list
        )

        # Handle selector edge case: if col provided but val missing, ignore it
        # This matches original behavior where it only filtered if both were present
        if selector_col and not selector_val:
            selector_col = None

        # Prepare data (initial filtering)
        # We catch ValueError from prepare_plot_data for empty data to handle it gracefully
        try:
            plot_data = prepare_plot_data(
                df,
                x_feature,
                conditions_list,
                condition_col,
                selector_col,
                selector_val,
            )
        except AssertionError:
            # prepare_plot_data asserts df is not None
            plot_data = pd.DataFrame(columns=df.columns)

        if plot_data is None or plot_data.empty:
            # Don't raise error, allow empty plot (matches original behavior)
            plot_data = pd.DataFrame(columns=df.columns)

        # Handle hue validation: disable if not found
        if self.config.hue and self.config.hue not in df.columns:
            self.config.hue = None

        # Use DataProcessor for further processing
        # Only process if we have data
        if not plot_data.empty:
            processor = ScatterDataProcessor(plot_data)
            processed_data = processor.process_data(
                plot_data,
                x_feature=x_feature,
                y_feature=y_feature,
                cell_number=self.config.cell_number,
                random_state=self.config.random_state,
            )
        else:
            processed_data = plot_data

        # Handle single condition / existing axes
        if len(conditions_list) == 1 or axes is not None:
            # Filter data for single condition
            if not processed_data.empty:
                cond_data = processed_data[
                    processed_data[condition_col] == conditions_list[0]
                ].copy()
            else:
                cond_data = processed_data

            if axes is not None:
                self.create_figure(axes=axes)
                self.build_plot(cond_data)
                return axes.figure, axes
            else:
                self.create_figure()
                self.build_plot(cond_data)
                fig, ax = self.build()
                self.save_figure()
                return fig, ax

        # Handle multiple conditions using create_subplots
        else:
            fig, axes_list = self.create_subplots(len(conditions_list))

            for i, (ax, cond) in enumerate(
                zip(axes_list, conditions_list, strict=False)
            ):
                # Filter data for this condition
                if not processed_data.empty:
                    cond_data = processed_data[
                        processed_data[condition_col] == cond
                    ].copy()
                else:
                    cond_data = processed_data

                # Create builder for this subplot (reusing config)
                sub_builder = ScatterPlot(self.config)
                sub_builder.create_figure(axes=ax)
                sub_builder.build_plot(cond_data)

                # Add condition as subplot title
                ax.set_title(cond, fontsize=8)

                # Only show y-label on first subplot
                if i > 0:
                    ax.set_ylabel("")

                # Ensure consistent axis formatting for DNA/EdU plots
                is_dna_content = x_feature == "integrated_int_DAPI_norm"
                if is_dna_content:
                    ax.set_xlim(self.config.x_limits or (1, 16))
                    if self.config.x_scale == "log":
                        ax.set_xticks([1, 2, 4, 8, 16])
                        ax.set_xticklabels(["1", "2", "4", "8", "16"])

            # Add suptitle if requested
            if self.config.show_title:
                title = (
                    self.config.title or f"Scatter: {x_feature} vs {y_feature}"
                )
                fig.suptitle(
                    title, fontsize=7, weight="bold", x=0.05, y=1.00, ha="left"
                )

            plt.tight_layout()
            self.save_figure()

            return fig, axes_list

    def _format_axes(self, x_feature: str, y_feature: str) -> None:
        """Format axes labels, scales, and limits."""
        assert self.ax is not None

        # Set scales
        if self.config.x_scale == "log":
            self.ax.set_xscale("log", base=self.config.x_scale_base)
        if self.config.y_scale == "log":
            self.ax.set_yscale("log", base=self.config.y_scale_base)

        # Set limits
        if self.config.x_limits:
            self.ax.set_xlim(self.config.x_limits)
        if self.config.y_limits:
            self.ax.set_ylim(self.config.y_limits)

        # Set ticks and format for log scale
        if self.config.x_scale == "log":
            if self.config.x_ticks:
                self.ax.set_xticks(self.config.x_ticks)
                self.ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: str(int(x)))
                )
            elif x_feature == "integrated_int_DAPI_norm":
                # Default DNA content ticks - only set if within limits
                if self.config.x_limits:
                    # Only include ticks that are within the limits
                    default_ticks = [2, 4, 8]
                    if valid_ticks := [
                        t
                        for t in default_ticks
                        if self.config.x_limits[0]
                        <= t
                        <= self.config.x_limits[1]
                    ]:
                        self.ax.set_xticks(valid_ticks)
                        self.ax.xaxis.set_major_formatter(
                            FuncFormatter(lambda x, pos: str(int(x)))
                        )
                else:
                    self.ax.set_xticks([2, 4, 8])
                    self.ax.xaxis.set_major_formatter(
                        FuncFormatter(lambda x, pos: str(int(x)))
                    )
        elif self.config.x_ticks:
            self.ax.set_xticks(self.config.x_ticks)

        # Set y ticks and format for log scale
        if self.config.y_scale == "log":
            if self.config.y_ticks:
                self.ax.set_yticks(self.config.y_ticks)
            self.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, pos: str(int(y)))
            )
        elif self.config.y_ticks:
            self.ax.set_yticks(self.config.y_ticks)

        # Set labels
        x_label = self.config.x_label or x_feature.replace("_", " ").title()
        y_label = self.config.y_label or y_feature.replace("_", " ").title()
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

        # Configure grid
        self.ax.grid(self.config.grid)

    def _add_reference_lines(self) -> None:
        """Add reference lines if specified."""
        assert self.ax is not None

        if self.config.vline is not None:
            self.ax.axvline(
                x=self.config.vline,
                color=self.config.line_color,
                linestyle=self.config.line_style,
            )

        if self.config.hline is not None:
            self.ax.axhline(
                y=self.config.hline,
                color=self.config.line_color,
                linestyle=self.config.line_style,
            )

    def _configure_legend(self) -> None:
        """Configure legend visibility and positioning."""
        assert self.ax is not None

        if not self.config.show_legend:
            if legend := self.ax.get_legend():
                legend.remove()
        elif self.config.legend_title and (legend := self.ax.get_legend()):
            legend.set_title(self.config.legend_title)

    def _set_positioned_title(self, x_feature: str, y_feature: str) -> None:
        """Set title with specific positioning."""
        assert self.ax is not None

        title = self.config.title or f"{x_feature} vs {y_feature}"
        self.ax.set_title(title, fontsize=10, loc="left", pad=10)


def create_scatter_plot(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    condition: str | list[str],
    condition_col: str = "condition",
    selector_col: str | None = None,
    selector_val: str | None = None,
    axes: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a scatter plot with specified features.

    This is the main factory function that handles single or multiple conditions.

    Args:
        df: DataFrame containing the data
        x_feature: Column name for x-axis
        y_feature: Column name for y-axis
        condition: Single condition or list of conditions
        condition_col: Column containing condition labels
        selector_col: Optional column for filtering
        selector_val: Optional value for filtering
        axes: Optional existing Axes to plot on
        **kwargs: Additional configuration options

    Returns:
        Tuple of (Figure, Axes or list of Axes)
    """
    # Handle single vs multiple conditions
    conditions = [condition] if isinstance(condition, str) else condition

    # If axes provided, must be single condition
    if axes is not None and len(conditions) > 1:
        raise ValueError(
            "Cannot use multiple conditions when axes is provided"
        )

    # Prepare data (initial filtering)
    plot_data = prepare_plot_data(
        df, x_feature, conditions, condition_col, selector_col, selector_val
    )

    if plot_data is None or plot_data.empty:
        raise ValueError(f"No data found for conditions: {conditions}")

    # Also get y_feature data if different from x_feature
    if y_feature != x_feature:
        y_data = prepare_plot_data(
            df,
            y_feature,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )
        if y_data is None or y_data.empty:
            raise ValueError(f"No data found for y_feature: {y_feature}")

    # Create config
    config = ScatterPlotConfig(
        x_feature=x_feature,
        y_feature=y_feature,
        **kwargs,
    )

    # Use DataProcessor for further processing (sampling, etc.)
    processor = ScatterDataProcessor(plot_data)
    processed_data = processor.process_data(
        plot_data,
        x_feature=x_feature,
        y_feature=y_feature,
        cell_number=config.cell_number,
        random_state=config.random_state,
    )

    # Create builder
    builder = ScatterPlot(config)

    # Handle single condition / existing axes
    if len(conditions) == 1 or axes is not None:
        # Filter data for single condition
        cond_data = processed_data[
            processed_data[condition_col] == conditions[0]
        ].copy()

        if axes is not None:
            builder.create_figure(axes=axes)
            builder.build_plot(cond_data)
            return axes.figure, axes
        else:
            builder.create_figure()
            builder.build_plot(cond_data)
            fig, ax = builder.build()
            builder.save_figure()
            return fig, ax

    # Handle multiple conditions using create_subplots
    else:
        fig, axes_list = builder.create_subplots(len(conditions))

        for i, (ax, cond) in enumerate(
            zip(axes_list, conditions, strict=False)
        ):
            # Filter data for this condition
            cond_data = processed_data[
                processed_data[condition_col] == cond
            ].copy()

            # Create builder for this subplot (reusing config)
            sub_builder = ScatterPlot(config)
            sub_builder.create_figure(axes=ax)
            sub_builder.build_plot(cond_data)

            # Add condition as subplot title
            ax.set_title(cond, fontsize=8)

            # Only show y-label on first subplot
            if i > 0:
                ax.set_ylabel("")

            # Ensure consistent axis formatting for DNA/EdU plots
            is_dna_content = x_feature == "integrated_int_DAPI_norm"
            if is_dna_content:
                ax.set_xlim(config.x_limits or (1, 16))
                if config.x_scale == "log":
                    ax.set_xticks([1, 2, 4, 8, 16])
                    ax.set_xticklabels(["1", "2", "4", "8", "16"])

        # Add suptitle if requested
        if config.show_title:
            title = config.title or f"Scatter: {x_feature} vs {y_feature}"
            fig.suptitle(
                title, fontsize=7, weight="bold", x=0.05, y=1.00, ha="left"
            )

        plt.tight_layout()
        builder.save_figure()

        return fig, axes_list
