"""Scatter plot factory with unified configuration and base class architecture."""

from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from omero_screen_plots.base import BasePlotBuilder, BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import prepare_plot_data, save_fig


@dataclass
class ScatterPlotConfig(BasePlotConfig):
    """Configuration for scatter plots."""

    # Plot features
    x_feature: str = "integrated_int_DAPI_norm"
    y_feature: str = "intensity_mean_EdU_nucleus_norm"

    # Data sampling
    cell_number: int | None = None  # Number of cells to sample per condition
    random_state: int = 42  # Random seed for reproducible sampling

    # Hue settings
    hue: str | None = None
    hue_order: list[str] | None = None
    palette: list[str] | dict[str, str] | None = None

    # Scale settings
    x_scale: Literal["linear", "log"] = "log"
    x_scale_base: int = 2
    y_scale: Literal["linear", "log"] = "log"
    y_scale_base: int = 10

    # Axis limits
    x_limits: tuple[float, float] | None = (1, 16)
    y_limits: tuple[float, float] | None = None

    # Axis ticks
    x_ticks: list[float] | None = None
    y_ticks: list[float] | None = None

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

    # Grid settings
    grid: bool = False

    # Title and labels
    show_title: bool = False
    x_label: str | None = None
    y_label: str | None = None

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


class ScatterPlot(BasePlotBuilder):
    """Builder for scatter plots."""

    def __init__(self, config: ScatterPlotConfig):
        """Initialize with specific config type."""
        super().__init__(config)
        self.config: ScatterPlotConfig = config  # Type narrowing

    def build_plot(self, data: pd.DataFrame, **kwargs: Any) -> "ScatterPlot":
        """Build scatter plot."""
        if self.ax is None:
            raise RuntimeError("Must create figure before building plot")

        # Extract plot parameters
        x_feature = kwargs.get("x_feature", self.config.x_feature)
        y_feature = kwargs.get("y_feature", self.config.y_feature)
        hue = kwargs.get("hue", self.config.hue)

        # Sample data if cell_number is specified
        if self.config.cell_number and len(data) > self.config.cell_number:
            data = data.sample(
                n=self.config.cell_number,
                random_state=self.config.random_state,
            )

        # Validate features exist
        if x_feature not in data.columns:
            raise ValueError(f"x_feature '{x_feature}' not found in dataframe")
        if y_feature not in data.columns:
            raise ValueError(f"y_feature '{y_feature}' not found in dataframe")
        # Apply threshold if specified (overrides other hue settings)
        plot_data = data.copy()
        if self.config.threshold is not None:
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

    def _prepare_figure_size(
        self,
        fig_size: tuple[float, float] | None,
        n_conditions: int,
        axes: Any | None,
    ) -> tuple[float, float] | None:
        """Determine figure size based on conditions and axes."""
        if fig_size is not None or axes is not None:
            return fig_size

        # Dynamic figure size based on number of conditions
        return (5, 5) if n_conditions == 1 else (4 * n_conditions, 5)

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        x_feature: str,
        y_feature: str,
        condition_col: str,
        conditions: list[str],
        axes: Any | None,
    ) -> None:
        """Validate input parameters early to provide helpful error messages."""
        # Check required columns exist
        required_cols = [x_feature, y_feature, condition_col]
        if missing_cols := [
            col for col in required_cols if col not in df.columns
        ]:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        # Validate conditions exist in data
        available_conditions = df[condition_col].unique()
        if invalid_conditions := [
            c for c in conditions if c not in available_conditions
        ]:
            raise ValueError(
                f"Invalid conditions: {invalid_conditions}. "
                f"Available: {list(available_conditions)}"
            )

        # Validate axes usage
        if axes is not None and len(conditions) > 1:
            raise ValueError(
                "Cannot use multiple conditions when axes is provided"
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
        """Create scatter plot using class-based approach like other plot classes.

        Args:
            df: DataFrame containing the data
            conditions: Single condition string or list of conditions
            condition_col: Column containing condition labels
            selector_col: Optional column for additional filtering
            selector_val: Optional value for selector_col filtering
            axes: Optional existing Axes to plot on

        Returns:
            tuple: (Figure, Axes or list of Axes)
        """
        # Handle single vs multiple conditions
        conditions_list = (
            [conditions] if isinstance(conditions, str) else conditions
        )

        # Get features from config
        x_feature = self.config.x_feature
        y_feature = self.config.y_feature

        # Validate inputs early
        self._validate_inputs(
            df, x_feature, y_feature, condition_col, conditions_list, axes
        )

        # Check if this is the standard DNA vs EdU plot
        is_dna_content = x_feature == "integrated_int_DAPI_norm"
        is_edu_intensity = y_feature == "intensity_mean_EdU_nucleus_norm"
        is_dna_edu = is_dna_content and is_edu_intensity

        # Handle hue settings: threshold coloring or auto-detect cell cycle
        if self.config.threshold is not None:
            # Threshold coloring overrides other hue settings
            self.config.hue = "threshold_category"
        elif self.config.hue is None and "cell_cycle" in df.columns:
            # Auto-detect cell cycle for default hue
            self.config.hue = "cell_cycle"
        elif self.config.hue and self.config.hue not in df.columns:
            # Disable hue if column doesn't exist
            self.config.hue = None

        # Set reference lines for DNA/EdU plots
        if is_dna_content and self.config.vline is None:
            self.config.vline = 3
        if is_edu_intensity and self.config.hline is None:
            self.config.hline = 3

        # Enable KDE overlay for DNA vs EdU plots
        if self.config.kde_overlay is None and is_dna_edu:
            self.config.kde_overlay = True

        # Prepare figure size
        fig_size = self._prepare_figure_size(
            self.config.fig_size, len(conditions_list), axes
        )
        if fig_size:
            self.config.fig_size = fig_size

        # Single condition - single plot
        if len(conditions_list) == 1 or axes is not None:
            # Filter data for single condition
            cond_data = df[df[condition_col] == conditions_list[0]].copy()

            # Apply selector filter if provided
            if selector_col and selector_val:
                cond_data = cond_data[cond_data[selector_col] == selector_val]

            if axes is not None:
                # Use provided axes
                self.create_figure(axes=axes)
                self.build_plot(cond_data)
                return axes.figure, axes
            else:
                # Create new figure
                self.create_figure()
                self.build_plot(cond_data)
                fig, ax = self.build()

                # Save if configured
                if self.config.save and self.config.path:
                    filename = (
                        self.config.title
                        or f"scatter_{x_feature}_vs_{y_feature}"
                    )
                    filename = filename.replace(" ", "_")
                    save_fig(
                        fig,
                        self.config.path,
                        filename,
                        tight_layout=self.config.tight_layout,
                        fig_extension=self.config.file_format,
                        resolution=self.config.dpi,
                    )

                return fig, ax

        # Multiple conditions - subplot grid
        else:
            n_conditions = len(conditions_list)

            # Create figure with subplots
            fig_inches = self.config.fig_size
            if self.config.size_units == "cm":
                fig_inches = (fig_inches[0] / 2.54, fig_inches[1] / 2.54)

            fig, axes_list = plt.subplots(1, n_conditions, figsize=fig_inches)
            if n_conditions == 1:
                axes_list = [axes_list]

            # Create plot for each condition
            for i, (ax, cond) in enumerate(
                zip(axes_list, conditions_list, strict=False)
            ):
                # Filter data for this condition
                cond_data = df[df[condition_col] == cond].copy()

                # Apply selector filter if provided
                if selector_col and selector_val:
                    cond_data = cond_data[
                        cond_data[selector_col] == selector_val
                    ]

                # Create new ScatterPlot instance with same config for each subplot
                builder = ScatterPlot(self.config)
                builder.create_figure(axes=ax)
                builder.build_plot(cond_data)

                # Add condition as subplot title only for multiple conditions
                if n_conditions > 1:
                    ax.set_title(cond, fontsize=8)

                # Only show y-label on first subplot
                if i > 0:
                    ax.set_ylabel("")

                # Ensure consistent axis formatting for DNA/EdU plots
                if is_dna_content:
                    # Set consistent x-axis limits and ticks for DNA content
                    ax.set_xlim(self.config.x_limits or (1, 16))
                    if self.config.x_scale == "log":
                        ax.set_xticks([1, 2, 4, 8, 16])
                        ax.set_xticklabels(["1", "2", "4", "8", "16"])

                if is_edu_intensity and self.config.y_scale == "log":
                    pass

            # Add suptitle if requested for multiple conditions
            if self.config.show_title:
                title = (
                    self.config.title or f"Scatter: {x_feature} vs {y_feature}"
                )
                fig.suptitle(
                    title, fontsize=7, weight="bold", x=0.05, y=1.00, ha="left"
                )

            # Adjust layout
            plt.tight_layout()

            # Save if configured
            if self.config.save and self.config.path:
                filename = (
                    self.config.title or f"scatter_{x_feature}_vs_{y_feature}"
                )
                filename = filename.replace(" ", "_")
                save_fig(
                    fig,
                    self.config.path,
                    filename,
                    tight_layout=self.config.tight_layout,
                    fig_extension=self.config.file_format,
                    resolution=self.config.dpi,
                )

            return fig, axes_list


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

    # Prepare data
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

    # Single condition - single plot
    if len(conditions) == 1 or axes is not None:
        # Filter data for single condition
        cond_data = df[df[condition_col] == conditions[0]].copy()

        # Apply selector filter if provided
        if selector_col and selector_val:
            cond_data = cond_data[cond_data[selector_col] == selector_val]

        builder = ScatterPlot(config)
        if axes is not None:
            # Use provided axes
            builder.create_figure(axes=axes)
            builder.build_plot(cond_data)
            return axes.figure, axes
        else:
            # Create new figure
            builder.create_figure()
            builder.build_plot(cond_data)
            fig, ax = builder.build()

            # Save if configured (for single condition without axes)
            if config.save and config.path:
                filename = (
                    config.title or f"scatter_{x_feature}_vs_{y_feature}"
                )
                # Replace spaces with underscores in filename for consistency
                filename = filename.replace(" ", "_")
                save_fig(
                    fig,
                    config.path,
                    filename,
                    tight_layout=config.tight_layout,
                    fig_extension=config.file_format,
                    resolution=config.dpi,
                )

            return fig, ax

    # Multiple conditions - subplot grid
    else:
        n_conditions = len(conditions)

        # Determine figure size dynamically if not specified
        if "fig_size" not in kwargs:
            # Default: 4cm width per condition, 4cm height
            config.fig_size = (4 * n_conditions, 4)

        # Create figure with subplots
        fig_inches = config.fig_size
        if config.size_units == "cm":
            fig_inches = (fig_inches[0] / 2.54, fig_inches[1] / 2.54)

        fig, axes = plt.subplots(1, n_conditions, figsize=fig_inches)
        if n_conditions == 1:
            axes = [axes]

        # Create plot for each condition
        for i, (ax, cond) in enumerate(zip(axes, conditions, strict=False)):
            # Filter data for this condition
            cond_data = df[df[condition_col] == cond].copy()

            # Apply selector filter if provided
            if selector_col and selector_val:
                cond_data = cond_data[cond_data[selector_col] == selector_val]

            # Create builder with existing axes
            builder = ScatterPlot(config)
            builder.create_figure(axes=ax)
            builder.build_plot(cond_data)

            # Add condition as subplot title only for multiple conditions
            if n_conditions > 1:
                ax.set_title(cond, fontsize=8)

            # Only show y-label on first subplot
            if i > 0:
                ax.set_ylabel("")

        # Add suptitle if requested for multiple conditions
        if config.show_title:
            title = config.title or f"Scatter: {x_feature} vs {y_feature}"
            fig.suptitle(
                title, fontsize=7, weight="bold", x=0.05, y=1.00, ha="left"
            )

        # Adjust layout
        plt.tight_layout()

        # Save if configured
        if config.save and config.path:
            filename = config.title or f"scatter_{x_feature}_vs_{y_feature}"
            # Replace spaces with underscores in filename for consistency
            filename = filename.replace(" ", "_")
            save_fig(
                fig,
                config.path,
                filename,
                tight_layout=config.tight_layout,
                fig_extension=config.file_format,
                resolution=config.dpi,
            )

        return fig, axes
