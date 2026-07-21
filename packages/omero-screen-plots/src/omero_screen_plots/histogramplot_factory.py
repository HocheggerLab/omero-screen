"""Histogram plot factory with unified configuration and base class architecture."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from omero_screen_plots.base import (
    BaseDataProcessor,
    BasePlotBuilder,
    PlotRequest,
    XYPlotConfig,
)
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import (
    prepare_plot_data,
    save_fig,
)


@dataclass
class HistogramPlotConfig(XYPlotConfig):
    """Configuration for histogram plots."""

    # Map feature to x_feature (inherited)
    # x_feature will store the feature name

    # Histogram-specific settings
    bins: int | str = 100
    log_scale: bool = False  # Maps to x_scale="log"
    log_base: float = 2  # Maps to x_scale_base
    normalize: bool = False  # Whether to show density instead of counts
    kde_overlay: bool = False
    kde_smoothing: float = 0.8
    kde_params: dict[str, Any] = field(default_factory=dict)

    # Axis formatting (inherited from XYPlotConfig)
    show_x_labels: bool = True
    rotation: int = 0
    show_title: bool = False

    def __post_init__(self) -> None:
        """Sync legacy parameters with XYPlotConfig parameters."""
        if self.log_scale:
            self.x_scale = "log"
        if self.log_base:
            self.x_scale_base = int(self.log_base)

        # Sync x_limits if provided (XYPlotConfig uses x_limits)
        # No action needed as names match


class HistogramDataProcessor(BaseDataProcessor):
    """Processes data for histogram plots."""

    def validate_dataframe(self) -> None:
        """Validate required columns exist."""

    def process_data(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Process data for histogram plot.

        Args:
            df: Input DataFrame
            **kwargs:
                feature: str

        Returns:
            Processed DataFrame with NaNs removed for the feature
        """
        feature = kwargs.get("feature")
        if not feature:
            raise ValueError("feature is required")

        if feature not in df.columns:
            raise ValueError(
                f"Feature column '{feature}' not found in dataframe"
            )

        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(
                f"Feature column '{feature}' must contain numeric data"
            )

        # Remove NaN values
        return df.dropna(subset=[feature])


class HistogramPlot(BasePlotBuilder):
    """Histogram plot implementation using the base class architecture."""

    PLOT_TYPE_NAME = "histogram"
    config: HistogramPlotConfig

    def __init__(self, config: HistogramPlotConfig | None = None):
        """Initialize the histogram plot builder."""
        super().__init__(config or HistogramPlotConfig())
        self.config: HistogramPlotConfig = self.config  # Type narrowing

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        feature: str,
        condition_col: str,
        conditions: list[str],
    ) -> None:
        """Validate input parameters."""
        # Check required columns exist
        required_cols = [feature, condition_col]
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
        """Create histogram plot(s)."""
        conditions_list = (
            [conditions] if isinstance(conditions, str) else conditions
        )

        # Validate axes usage
        if axes is not None and len(conditions_list) > 1:
            raise ValueError(
                "axes parameter not supported for multiple conditions"
            )

        # Validate inputs
        self._validate_inputs(df, feature, condition_col, conditions_list)

        # Prepare data (initial filtering)
        # Note: We do NOT ignore missing selector_val here, as Histogram tests expect ValueError
        try:
            plot_data = prepare_plot_data(
                df,
                feature,
                conditions_list,
                condition_col,
                selector_col,
                selector_val,
                scale=False,
            )
        except ValueError:
            # Re-raise ValueError from prepare_plot_data (e.g. missing selector_val)
            raise
        except AssertionError:
            # prepare_plot_data asserts df is not None
            plot_data = pd.DataFrame(columns=df.columns)

        if plot_data is None or plot_data.empty:
            raise ValueError(f"No data found for conditions: {conditions}")

        # Process data (remove NaNs)
        processor = HistogramDataProcessor(plot_data)
        processed_data = processor.process_data(plot_data, feature=feature)

        if processed_data.empty:
            raise ValueError(
                f"No valid data remaining for feature '{feature}'"
            )

        # Handle KDE overlay mode - always single plot
        if self.config.kde_overlay:
            return self._create_kde_overlay_plot(
                processed_data,
                feature,
                conditions_list,
                condition_col,
                axes,
            )

        # Handle single vs multiple conditions
        if len(conditions_list) == 1:
            return self._create_single_histogram(
                processed_data,
                feature,
                conditions_list[0],
                condition_col,
                axes,
            )
        else:
            return self._create_multiple_histograms(
                processed_data,
                feature,
                conditions_list,
                condition_col,
            )

    def _create_single_histogram(
        self,
        data: pd.DataFrame,
        feature: str,
        condition: str,
        condition_col: str,
        axes: Axes | None,
    ) -> tuple[Figure, Axes]:
        """Create histogram for a single condition."""
        # Filter for condition
        cond_data = data[data[condition_col] == condition].copy()

        if cond_data.empty:
            raise ValueError(
                f"No valid data remaining for feature '{feature}'"
            )

        self.create_figure(axes)
        self.build_plot(
            cond_data, PlotRequest(conditions=[condition], feature=feature)
        )
        self._format_axes(feature)

        default_title = f"Histogram: {feature}"
        self.finalize_plot(default_title)

        if self.config.show_title:
            self._set_positioned_title(feature, condition)

        self.save_figure()
        assert self.fig is not None
        assert self.ax is not None
        return self.fig, self.ax

    def _create_multiple_histograms(
        self,
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
    ) -> tuple[Figure, list[Axes]]:
        """Create multiple histograms."""
        n_conditions = len(conditions)

        # Calculate unified bins
        unified_bins = self._calculate_unified_bins(
            data, feature, conditions, condition_col
        )

        # Create subplots
        fig, axes_list = self.create_subplots(
            n_conditions, self.config.fig_size
        )

        for i, (ax, cond) in enumerate(
            zip(axes_list, conditions, strict=False)
        ):
            if not data.empty:
                cond_data = data[data[condition_col] == cond].copy()
            else:
                cond_data = data

            if cond_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {cond}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Color
            color = (
                self.config.colors[i % len(self.config.colors)]
                if self.config.colors
                else COLOR.BLUE.value
            )

            # Plot
            hist_params = {
                "bins": unified_bins
                if unified_bins is not None
                else self.config.bins,
                "stat": "density" if self.config.normalize else "count",
            }

            sns.histplot(
                data=cond_data,
                x=feature,
                ax=ax,
                color=color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
                **hist_params,
            )

            # Format
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel("Density" if self.config.normalize else "Count")
            ax.set_title(cond, fontsize=10)

            if self.config.log_scale:
                ax.set_xscale("log", base=self.config.log_base)
                self._set_log_tick_labels_for_axis(ax)

            if self.config.x_limits:
                ax.set_xlim(self.config.x_limits)

            if not self.config.show_x_labels:
                ax.set_xticklabels([])
            elif self.config.rotation != 0:
                ax.tick_params(axis="x", rotation=self.config.rotation)

        if self.config.show_title:
            title = (
                self.config.title
                or f"Histograms: {feature.replace('_', ' ').title()}"
            )
            fig.suptitle(title, fontsize=10, x=0.01, y=1.1)

        if self.config.save and self.config.path:
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
        data: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
        axes: Axes | None,
    ) -> tuple[Figure, Axes]:
        """Create KDE overlay plot."""
        self.create_figure(axes)

        for i, condition in enumerate(conditions):
            if not data.empty:
                cond_data = data[data[condition_col] == condition].copy()
            else:
                continue

            if cond_data.empty:
                continue

            # Color
            if self.config.colors:
                color = self.config.colors[i % len(self.config.colors)]
            else:
                default_colors = [
                    COLOR.BLUE.value,
                    COLOR.YELLOW.value,
                    COLOR.PINK.value,
                    COLOR.LIGHT_GREEN.value,
                ]
                color = default_colors[i % len(default_colors)]

            # KDE params
            kde_params = {
                "alpha": 0.8,
                "linewidth": 2.5,
                "bw_adjust": self.config.kde_smoothing,
                "gridsize": 300,
            }
            kde_params.update(self.config.kde_params)

            sns.kdeplot(
                data=cond_data,
                x=feature,
                ax=self.ax,
                color=color,
                label=condition,
                **kde_params,
            )

        self._format_axes(feature)
        assert self.ax is not None
        self.ax.set_ylabel("Density")
        if len(conditions) > 1:
            self.ax.legend()

        default_title = f"KDE: {feature}"
        if self.config.show_title:
            if len(conditions) == 1:
                self._set_positioned_title(feature, conditions[0])
            else:
                title = (
                    self.config.title
                    or f"KDE: {feature.replace('_', ' ').title()}"
                )
                assert self.ax is not None
                self.ax.text(
                    0,
                    1.02,
                    title,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )
        else:
            self.finalize_plot(default_title)

        self.save_figure()
        assert self.fig is not None
        return self.fig, self.ax

    def build_plot(
        self, data: pd.DataFrame, request: PlotRequest
    ) -> "HistogramPlot":
        """Build histogram plot."""
        feature = request.feature
        assert feature is not None, "histogram requires request.feature"

        # Get color
        color = (
            self.config.colors[0] if self.config.colors else COLOR.BLUE.value
        )

        hist_params = {
            "bins": self.config.bins,
            "stat": "density" if self.config.normalize else "count",
        }

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

        if self.config.kde_overlay:
            # Note: This is for single condition overlay on top of histogram
            # Not the _create_kde_overlay_plot mode which is KDE ONLY
            self._add_kde_overlay(data, feature)

        return self

    def _add_kde_overlay(self, data: pd.DataFrame, feature: str) -> None:
        """Add KDE overlay to histogram."""
        kde_params = {"alpha": 0.8, "linewidth": 3}
        kde_params.update(self.config.kde_params)
        color = (
            self.config.colors[0] if self.config.colors else COLOR.BLUE.value
        )

        if self.config.normalize:
            sns.kdeplot(
                data=data, x=feature, ax=self.ax, color=color, **kde_params
            )
        else:
            # Scale KDE to match counts
            hist_data = data[feature]
            n_points = len(hist_data)

            if isinstance(self.config.bins, int):
                data_range = hist_data.max() - hist_data.min()
                bin_width = data_range / self.config.bins
            else:
                bin_width = (hist_data.max() - hist_data.min()) / 30  # Approx

            scaling_factor = n_points * bin_width

            kde = gaussian_kde(hist_data)
            if self.config.x_limits:
                x_min, x_max = self.config.x_limits
            else:
                x_min, x_max = hist_data.min(), hist_data.max()

            x_kde = np.linspace(x_min, x_max, 200)
            y_kde = kde(x_kde) * scaling_factor

            plot_kwargs = {
                k: v
                for k, v in kde_params.items()
                if k in ["linewidth", "alpha"]
            }
            assert self.ax is not None
            # Cast to Any to avoid mypy issues with **kwargs unpacking
            plot_kwargs_any: dict[str, Any] = plot_kwargs
            self.ax.plot(x_kde, y_kde, color=color, **plot_kwargs_any)

    def _calculate_unified_bins(
        self,
        df: pd.DataFrame,
        feature: str,
        conditions: list[str],
        condition_col: str,
    ) -> np.ndarray[Any, np.dtype[np.float64]] | None:
        """Calculate unified bins."""
        if not isinstance(self.config.bins, int) or self.config.bins <= 0:
            return None

        all_data: list[float] = []
        for condition in conditions:
            # We need to filter again here because we need all data to calc bins
            # But we passed processed_data to _create_multiple_histograms?
            # Ah, _create_multiple_histograms receives 'data' which is ALREADY processed (NaNs removed)
            # So we can just use that.
            cond_data = df[df[condition_col] == condition][feature].values
            all_data.extend(cond_data)

        if not all_data:
            return None

        if self.config.log_scale:
            all_data_pos = [x for x in all_data if x > 0]
            if all_data_pos:
                min_val, max_val = np.min(all_data_pos), np.max(all_data_pos)
                return np.logspace(
                    np.log(min_val) / np.log(self.config.log_base),
                    np.log(max_val) / np.log(self.config.log_base),
                    self.config.bins + 1,
                    base=self.config.log_base,
                )
        else:
            min_val, max_val = np.min(all_data), np.max(all_data)
            return np.linspace(min_val, max_val, self.config.bins + 1)  # type: ignore[no-any-return]

        return None

    def _format_axes(self, feature: str) -> None:
        """Format axes."""
        assert self.ax is not None
        xlabel = feature.replace("_", " ").title()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Density" if self.config.normalize else "Count")

        if self.config.log_scale:
            self.ax.set_xscale("log", base=self.config.log_base)
            self._set_log_tick_labels_for_axis(self.ax)

        if self.config.x_limits:
            self.ax.set_xlim(self.config.x_limits)

        if not self.config.show_x_labels:
            self.ax.set_xticklabels([])
        elif self.config.rotation != 0:
            self.ax.tick_params(axis="x", rotation=self.config.rotation)

    def _set_log_tick_labels_for_axis(self, ax: Axes) -> None:
        """Set clean tick labels for log scale."""
        if not self.config.log_scale:
            return

        xlim = ax.get_xlim()
        if self.config.log_base == 2:
            min_p = np.floor(np.log2(xlim[0])) if xlim[0] > 0 else -1
            max_p = np.ceil(np.log2(xlim[1])) if xlim[1] > 0 else 4
            powers = np.arange(min_p, max_p + 1)
            ticks = 2**powers
            ticks = ticks[(ticks >= xlim[0]) & (ticks <= xlim[1])]
            if len(ticks) > 0:
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in ticks]
                )
        elif self.config.log_base == 10:
            min_p = np.floor(np.log10(xlim[0])) if xlim[0] > 0 else -1
            max_p = np.ceil(np.log10(xlim[1])) if xlim[1] > 0 else 3
            powers = np.arange(min_p, max_p + 1)
            ticks = 10**powers
            ticks = ticks[(ticks >= xlim[0]) & (ticks <= xlim[1])]
            if len(ticks) > 0:
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{int(x)}" if x >= 1 else f"{x:.1f}" for x in ticks]
                )

    def _set_positioned_title(self, feature: str, condition: str) -> None:
        """Set positioned title."""
        title = self.config.title or f"{feature} - {condition}"
        assert self.ax is not None
        self.ax.text(
            0,
            1.02,
            title,
            transform=self.ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )


def create_histogram_plot(
    df: pd.DataFrame,
    feature: str,
    condition: str | list[str],
    condition_col: str = "condition",
    selector_col: str | None = None,
    selector_val: str | None = None,
    axes: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes | list[Axes]]:
    """Create histogram plot factory function."""
    # Map 'feature' to 'x_feature' for config
    if "x_feature" not in kwargs:
        kwargs["x_feature"] = feature

    config = HistogramPlotConfig(**kwargs)

    # Override config with explicit args if provided
    # (Already handled by passing kwargs to config init)

    plot = HistogramPlot(config)
    return plot.create_plot(
        df,
        feature,
        condition,
        condition_col,
        selector_col,
        selector_val,
        axes,
    )
