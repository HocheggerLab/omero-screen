"""Histogram plot factory with unified configuration and base class architecture."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.base import BasePlotBuilder, BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import convert_size_to_inches, prepare_plot_data


@dataclass
class HistogramPlotConfig(BasePlotConfig):
    """Configuration for histogram plots.

    This class extends BasePlotConfig and adds histogram-specific settings.
    The dataclass decorator automatically creates __init__, __repr__, etc.
    """

    # Figure settings (inherited from BasePlotConfig, but we can override defaults)
    fig_size: tuple[float, float] = (7, 7)
    size_units: str = "cm"
    dpi: int = 300

    # Save settings (inherited from BasePlotConfig)
    save: bool = False
    file_format: str = "pdf"
    tight_layout: bool = False
    path: Path | None = None

    # Display settings (inherited from BasePlotConfig)
    title: str | None = None
    show_title: bool = False  # Whether to show title (default False)
    colors: list[str] = field(default_factory=list)

    # Histogram-specific settings
    bins: int | str = (
        100  # Number of bins (default 100) or 'auto', 'sturges', etc.
    )
    log_scale: bool = False  # Whether to use log scale on x-axis
    log_base: float = 2  # Base for log scale (typically 2 for DNA content)
    x_limits: tuple[float, float] | None = None  # X-axis limits (min, max)
    normalize: bool = False  # Whether to show density instead of counts
    kde_overlay: bool = False  # Whether to add KDE curve overlay
    kde_smoothing: float = 0.8  # KDE smoothing factor (bw_adjust)
    kde_params: dict[str, Any] = field(
        default_factory=dict
    )  # KDE styling options

    # Axis formatting
    show_x_labels: bool = True  # Whether to show x-axis tick labels
    rotation: int = 0  # Rotation angle for x-axis labels


class HistogramPlot(BasePlotBuilder):
    """Histogram plot implementation using the base class architecture.

    This class extends BasePlotBuilder and implements histogram-specific plotting logic.
    """

    PLOT_TYPE_NAME = "histogram"
    config: HistogramPlotConfig  # Type annotation for better IDE support

    def __init__(self, config: HistogramPlotConfig | None = None):
        """Initialize the histogram plot builder.

        Args:
            config: Configuration object. If None, uses default HistogramPlotConfig
        """
        super().__init__(config or HistogramPlotConfig())

    def create_plot(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: str | list[str],
        condition_col: str = "condition",
        selector_col: str | None = None,
        selector_val: str | None = None,
        axes: Axes | None = None,
    ) -> tuple[Figure, Axes | list[Axes]]:
        """Create histogram plot(s) for single or multiple conditions.

        This creates histograms for one or more conditions. If KDE overlay is enabled,
        creates a single plot with overlaid KDE lines instead of histograms.

        Args:
            df: Input dataframe
            feature: Column name to create histogram for
            conditions: Single condition (str) or multiple conditions (list)
            condition_col: Column name containing condition labels
            selector_col: Optional column for additional filtering
            selector_val: Value to filter by if selector_col provided
            axes: Optional existing axes to plot on. Only valid for single condition.

        Returns:
            Tuple of (Figure, Axes) for single condition or (Figure, list[Axes]) for multiple
        """
        # Handle KDE overlay mode - always single plot regardless of condition count
        if self.config.kde_overlay:
            conditions_list = (
                [conditions] if isinstance(conditions, str) else conditions
            )
            return self._create_kde_overlay_plot(
                df,
                feature,
                conditions_list,
                condition_col,
                selector_col,
                selector_val,
                axes,
            )

        # Handle single vs multiple conditions for regular histograms
        if isinstance(conditions, str):
            # Single condition - original logic
            return self._create_single_histogram(
                df,
                feature,
                conditions,
                condition_col,
                selector_col,
                selector_val,
                axes,
            )
        else:
            # Multiple conditions - create subplots
            if axes is not None:
                raise ValueError(
                    "axes parameter not supported for multiple conditions"
                )
            return self._create_multiple_histograms(
                df,
                feature,
                conditions,
                condition_col,
                selector_col,
                selector_val,
            )

    def _create_single_histogram(
        self,
        df: pd.DataFrame,
        feature: str,
        condition: str,
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
        axes: Axes | None,
    ) -> tuple[Figure, Axes]:
        """Create histogram for a single condition."""
        # Step 1: Convert single condition to list for prepare_plot_data compatibility
        conditions_list = [condition]

        # Step 2: Process data using the utility function (same as other plots)
        processed_data = prepare_plot_data(
            df,
            feature,
            conditions_list,
            condition_col,
            selector_col,
            selector_val,
            scale=False,  # Histograms don't need data scaling
        )

        # Step 3: Validate the feature column
        if feature not in processed_data.columns:
            raise ValueError(
                f"Feature column '{feature}' not found in dataframe"
            )

        if not pd.api.types.is_numeric_dtype(processed_data[feature]):
            raise ValueError(
                f"Feature column '{feature}' must contain numeric data"
            )

        # Step 4: Remove NaN values for histogram
        processed_data = processed_data.dropna(subset=[feature])

        if processed_data.empty:
            raise ValueError(
                f"No valid data remaining for feature '{feature}'"
            )

        # Step 5: Create figure using base class method
        self.create_figure(axes)

        # Step 6: Build the plot
        self.build_plot(
            processed_data,
            feature=feature,
            condition=condition,
        )

        # Step 7: Format axes
        self._format_axes(feature)

        # Step 8: Finalize with title (sets filename for saving)
        default_title = f"Histogram: {feature}"
        self.finalize_plot(default_title)

        # Step 8.5: Set positioned title if requested (overrides finalize_plot title)
        if self.config.show_title:
            self._set_positioned_title(feature, condition)

        # Step 9: Save if configured
        self.save_figure()

        # Step 10: Return figure and axes
        assert self.fig is not None and self.ax is not None
        return self.fig, self.ax

    def _create_multiple_histograms(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> tuple[Figure, list[Axes]]:
        """Create multiple histograms with unified binning."""
        n_conditions = len(conditions)

        if n_conditions == 0:
            raise ValueError("At least one condition must be provided")

        # Calculate unified bins if using integer bins
        unified_bins = self._calculate_unified_bins(
            df, feature, conditions, condition_col, selector_col, selector_val
        )

        # Create subplots - use user-specified fig_size directly
        fig_inches = convert_size_to_inches(
            self.config.fig_size, self.config.size_units
        )

        fig, axes_array = plt.subplots(1, n_conditions, figsize=fig_inches)

        # Handle single vs multiple axes
        axes_list = (
            [axes_array] if n_conditions == 1 else list(axes_array.flatten())
        )
        # Create histogram for each condition
        for i, condition in enumerate(conditions):
            ax = axes_list[i]

            # Process data for this condition
            processed_data = prepare_plot_data(
                df,
                feature,
                [condition],
                condition_col,
                selector_col,
                selector_val,
                scale=False,
            )

            if processed_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {condition}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Remove NaN values
            processed_data = processed_data.dropna(subset=[feature])

            # Get color for this condition - cycle through available colors
            if self.config.colors:
                color = self.config.colors[i % len(self.config.colors)]
            else:
                color = COLOR.BLUE.value

            # Create histogram with unified bins
            hist_params = {
                "bins": unified_bins
                if unified_bins is not None
                else self.config.bins,
                "stat": "density" if self.config.normalize else "count",
            }

            sns.histplot(
                data=processed_data,
                x=feature,
                ax=ax,
                color=color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
                **hist_params,
            )

            # Add KDE overlay if requested
            if self.config.kde_overlay:
                kde_params = {"alpha": 0.8, "linewidth": 3}
                kde_params.update(self.config.kde_params)

                if self.config.normalize:
                    # If histogram shows density, KDE is already in correct units
                    sns.kdeplot(
                        data=processed_data,
                        x=feature,
                        ax=ax,
                        color=color,
                        **kde_params,
                    )
                else:
                    # If histogram shows counts, scale KDE to match
                    hist_data = processed_data[feature].dropna()

                    # Calculate scaling factor
                    n_points = len(hist_data)
                    if isinstance(self.config.bins, int):
                        data_range = hist_data.max() - hist_data.min()
                        bin_width = data_range / self.config.bins
                    else:
                        bin_width = (hist_data.max() - hist_data.min()) / 30

                    scaling_factor = n_points * bin_width

                    # Create scaled KDE
                    import numpy as np
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(hist_data)

                    # Create x values for KDE line
                    if self.config.x_limits:
                        x_min, x_max = self.config.x_limits
                    else:
                        x_min, x_max = hist_data.min(), hist_data.max()

                    x_kde = np.linspace(x_min, x_max, 200)
                    y_kde = kde(x_kde) * scaling_factor

                    # Plot scaled KDE line with explicit parameters
                    plot_kwargs: dict[str, Any] = {}
                    if "linewidth" in kde_params:
                        plot_kwargs["linewidth"] = kde_params["linewidth"]
                    if "alpha" in kde_params:
                        plot_kwargs["alpha"] = kde_params["alpha"]
                    ax.plot(x_kde, y_kde, color=color, **plot_kwargs)

            # Format this subplot
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel("Density" if self.config.normalize else "Count")
            ax.set_title(condition, fontsize=10)

            # Apply log scale if requested
            if self.config.log_scale:
                ax.set_xscale("log", base=self.config.log_base)
                # Set clean tick labels for common log bases
                self._set_log_tick_labels_for_axis(ax)

            # Set x-axis limits if specified
            if self.config.x_limits:
                ax.set_xlim(self.config.x_limits)

            # Handle x-axis labels
            if not self.config.show_x_labels:
                ax.set_xticklabels([])
            elif self.config.rotation != 0:
                ax.tick_params(axis="x", rotation=self.config.rotation)

        # Overall figure formatting - only add suptitle if requested
        if self.config.show_title:
            title_text = (
                self.config.title
                or f"Histograms: {feature.replace('_', ' ').title()}"
            )
            fig.suptitle(
                title_text, fontsize=10, x=0.01, y=1.1
            )  # Position at left edge

        # Save if configured
        if self.config.save and self.config.path:
            from omero_screen_plots.utils import save_fig

            filename = f"histogram_{feature}_multi"
            save_fig(
                fig,
                self.config.path,
                filename,
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi,
            )

        return fig, axes_list

    def _create_kde_overlay_plot(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
        axes: Axes | None,
    ) -> tuple[Figure, Axes]:
        """Create single plot with overlaid KDE lines for multiple conditions."""
        # Create figure using base class method
        self.create_figure(axes)

        # Process each condition and plot KDE
        for i, condition in enumerate(conditions):
            # Process data for this condition
            processed_data = prepare_plot_data(
                df,
                feature,
                [condition],
                condition_col,
                selector_col,
                selector_val,
                scale=False,
            )

            if processed_data.empty:
                continue

            # Remove NaN values
            processed_data = processed_data.dropna(subset=[feature])

            if processed_data.empty:
                continue

            # Get color for this condition
            if self.config.colors:
                color = self.config.colors[i % len(self.config.colors)]
            else:
                # Use different colors if multiple conditions
                default_colors = [
                    COLOR.BLUE.value,
                    COLOR.YELLOW.value,
                    COLOR.PINK.value,
                    COLOR.LIGHT_GREEN.value,
                ]
                color = default_colors[i % len(default_colors)]

            # Create KDE line with proper thickness and smoothing
            kde_params = {
                "alpha": 0.8,
                "linewidth": 2.5,  # Thicker lines for better visibility
                "bw_adjust": self.config.kde_smoothing,  # User-configurable smoothing
                "gridsize": 300,  # Higher resolution for smoother curves
            }
            kde_params.update(self.config.kde_params)

            sns.kdeplot(
                data=processed_data,
                x=feature,
                ax=self.ax,
                color=color,
                label=condition,  # Add label for legend
                **kde_params,
            )

        # Format axes
        self._format_axes(feature)

        # Set y-label to Density since we're showing KDE
        assert self.ax is not None
        self.ax.set_ylabel("Density")

        # Add legend to distinguish conditions
        if len(conditions) > 1:
            self.ax.legend()

        # Finalize with title (sets filename for saving)
        default_title = f"KDE: {feature}"
        self.finalize_plot(default_title)

        # Set positioned title if requested (overrides finalize_plot title)
        if self.config.show_title:
            if len(conditions) == 1:
                self._set_positioned_title(feature, conditions[0])
            else:
                title_text = (
                    self.config.title
                    or f"KDE: {feature.replace('_', ' ').title()}"
                )
                self.ax.text(
                    0,
                    1.02,
                    title_text,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )

        # Save if configured
        self.save_figure()

        # Return figure and axes
        assert self.fig is not None and self.ax is not None
        return self.fig, self.ax

    def _calculate_unified_bins(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> npt.NDArray[np.float64] | None:
        """Calculate unified bin edges across all conditions."""
        # Only calculate unified bins if bins is an integer
        if not isinstance(self.config.bins, int) or self.config.bins <= 0:
            return None

        # Collect all data points across conditions
        all_data: list[float] = []
        for condition in conditions:
            cond_data = prepare_plot_data(
                df,
                feature,
                [condition],
                condition_col,
                selector_col,
                selector_val,
                scale=False,
            )
            if not cond_data.empty:
                all_data.extend(cond_data[feature].dropna().values)

        if not all_data:
            return None

        # Calculate unified bin edges
        if self.config.log_scale:
            # For log scale, use log-spaced bins
            all_data_positive = [x for x in all_data if x > 0]
            if all_data_positive:
                min_val = np.min(all_data_positive)
                max_val = np.max(all_data_positive)
                return np.logspace(
                    np.log(min_val) / np.log(self.config.log_base),
                    np.log(max_val) / np.log(self.config.log_base),
                    self.config.bins + 1,
                    base=self.config.log_base,
                )
        else:
            # For linear scale, use linearly-spaced bins
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            return np.linspace(
                min_val, max_val, self.config.bins + 1, dtype=np.float64
            )

        return None

    def build_plot(self, data: pd.DataFrame, **kwargs: Any) -> "HistogramPlot":
        """Build histogram plot for a single condition.

        Args:
            data: Processed dataframe ready for plotting (already filtered to single condition)
            **kwargs: Additional arguments (feature, condition)

        Returns:
            Self for method chaining
        """
        feature = kwargs["feature"]
        condition = kwargs["condition"]

        assert self.ax is not None

        # Get histogram parameters from configuration
        hist_params = self._get_histogram_params()

        # Get color - always use first color for single condition histograms
        color = (
            self.config.colors[0] if self.config.colors else COLOR.BLUE.value
        )

        # Create the histogram
        sns.histplot(
            data=data,
            x=feature,
            ax=self.ax,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            **hist_params,
        )

        # Add KDE overlay if requested
        if self.config.kde_overlay:
            self._add_kde_overlay(data, feature, condition)

        return self

    def _get_histogram_params(self) -> dict[str, Any]:
        """Get histogram parameters based on configuration.

        Returns:
            Dictionary of parameters for sns.histplot
        """
        return {
            "bins": self.config.bins,  # Now defaults to 100, can be int or string like "auto"
            "stat": "density" if self.config.normalize else "count",
        }

    def _add_kde_overlay(
        self, data: pd.DataFrame, feature: str, condition: str
    ) -> None:
        """Add KDE overlay to the histogram for a single condition.

        Args:
            data: Data to plot (already filtered to single condition)
            feature: Feature column
            condition: Condition name (for reference, data is already filtered)
        """
        assert self.ax is not None

        # Set up KDE parameters - make more visible by default
        kde_params = {"alpha": 0.8, "linewidth": 3}
        kde_params.update(self.config.kde_params)

        # Get color (same as histogram)
        color = (
            self.config.colors[0] if self.config.colors else COLOR.BLUE.value
        )

        # Add single KDE curve - scale to match histogram
        if self.config.normalize:
            # If histogram shows density, KDE is already in correct units
            sns.kdeplot(
                data=data,
                x=feature,
                ax=self.ax,
                color=color,
                **kde_params,
            )
        else:
            # If histogram shows counts, scale KDE to match
            # First get the bin width to scale KDE properly
            hist_data = data[feature].dropna()

            # Calculate scaling factor: total data points * bin width
            n_points = len(hist_data)
            if isinstance(self.config.bins, int):
                # For fixed number of bins, calculate bin width
                data_range = hist_data.max() - hist_data.min()
                if self.config.log_scale and data_range > 0:
                    # For log scale, use the unified bins if available
                    # This is approximate scaling - exact would require bin edges
                    bin_width = data_range / self.config.bins
                else:
                    bin_width = data_range / self.config.bins
            else:
                # For 'auto' bins, estimate bin width
                bin_width = (
                    hist_data.max() - hist_data.min()
                ) / 30  # rough estimate

            scaling_factor = n_points * bin_width

            # Create scaled KDE
            import numpy as np
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(hist_data)

            # Create x values for KDE line
            if self.config.x_limits:
                x_min, x_max = self.config.x_limits
            else:
                x_min, x_max = hist_data.min(), hist_data.max()

            x_kde = np.linspace(x_min, x_max, 200)
            y_kde = kde(x_kde) * scaling_factor

            # Plot scaled KDE line with explicit parameters
            plot_kwargs: dict[str, Any] = {}
            if "linewidth" in kde_params:
                plot_kwargs["linewidth"] = kde_params["linewidth"]
            if "alpha" in kde_params:
                plot_kwargs["alpha"] = kde_params["alpha"]
            self.ax.plot(x_kde, y_kde, color=color, **plot_kwargs)

    def _format_axes(self, feature: str) -> None:
        """Format axes labels, limits, and styling.

        Args:
            feature: Feature name for axis labels
        """
        assert self.ax is not None

        # Set axis labels
        xlabel = feature.replace("_", " ").title()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Density" if self.config.normalize else "Count")

        # Apply log scale if requested
        if self.config.log_scale:
            self.ax.set_xscale("log", base=self.config.log_base)
            # Set clean tick labels for common log bases
            self._set_log_tick_labels()

        # Set x-axis limits if specified
        if self.config.x_limits:
            self.ax.set_xlim(self.config.x_limits)

        # Handle x-axis label visibility
        if not self.config.show_x_labels:
            self.ax.set_xticklabels([])
        elif self.config.rotation != 0:
            self.ax.tick_params(axis="x", rotation=self.config.rotation)

    def _set_log_tick_labels(self) -> None:
        """Set clean tick labels for log scale on current axes."""
        if self.ax is not None:
            self._set_log_tick_labels_for_axis(self.ax)

    def _set_log_tick_labels_for_axis(self, ax: Axes) -> None:
        """Set clean tick labels for log scale on specified axes."""
        if not self.config.log_scale:
            return

        # Get data range to determine appropriate tick locations
        xlim = ax.get_xlim()

        if self.config.log_base == 2:
            # For base 2, create ticks at powers of 2: 0.5, 1, 2, 4, 8, 16, etc.
            min_power = np.floor(np.log2(xlim[0])) if xlim[0] > 0 else -1
            max_power = np.ceil(np.log2(xlim[1])) if xlim[1] > 0 else 4

            powers = np.arange(min_power, max_power + 1)
            tick_values = 2**powers

            # Filter to data range
            tick_values = tick_values[
                (tick_values >= xlim[0]) & (tick_values <= xlim[1])
            ]

            if len(tick_values) > 0:
                ax.set_xticks(tick_values)
                # Format labels as integers when >= 1, else as decimals
                labels = []
                for val in tick_values:
                    if val >= 1:
                        labels.append(f"{int(val)}")
                    else:
                        labels.append(f"{val:.1f}")
                ax.set_xticklabels(labels)

        elif self.config.log_base == 10:
            # For base 10, create ticks at powers of 10: 0.1, 1, 10, 100, etc.
            min_power = np.floor(np.log10(xlim[0])) if xlim[0] > 0 else -1
            max_power = np.ceil(np.log10(xlim[1])) if xlim[1] > 0 else 3

            powers = np.arange(min_power, max_power + 1)
            tick_values = 10**powers

            # Filter to data range
            tick_values = tick_values[
                (tick_values >= xlim[0]) & (tick_values <= xlim[1])
            ]

            if len(tick_values) > 0:
                ax.set_xticks(tick_values)
                # Format labels as integers when >= 1, else as decimals
                labels = []
                for val in tick_values:
                    if val >= 1:
                        labels.append(f"{int(val)}")
                    else:
                        labels.append(f"{val:.1f}")
                ax.set_xticklabels(labels)

        else:
            # For other bases, use matplotlib's default but with cleaner formatting
            from matplotlib.ticker import FuncFormatter, LogLocator

            def format_func(x: float, pos: int) -> str:
                if x >= 1:
                    return f"{int(x)}"
                else:
                    return f"{x:.1f}"

            ax.xaxis.set_major_locator(LogLocator(base=self.config.log_base))
            ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    def _set_positioned_title(self, feature: str, condition: str) -> None:
        """Set title positioned at y-axis start with fontsize 10."""
        assert self.ax is not None

        # Generate title text
        title_text = (
            self.config.title
            or f"Histogram: {feature.replace('_', ' ').title()}"
        )

        # Position title at the start of the y-axis (left side)
        # x=0 aligns with y-axis, y=1.02 positions it just above the plot
        self.ax.text(
            0,
            1.02,
            title_text,
            transform=self.ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )
