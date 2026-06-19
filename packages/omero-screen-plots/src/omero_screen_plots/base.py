"""Base classes for omero-screen-plots scalable architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.utils import convert_size_to_inches, save_fig


@dataclass
class BasePlotConfig:
    """Base configuration for all plots."""

    # Common figure settings
    fig_size: tuple[float, float] = (7, 7)
    size_units: str = "cm"
    dpi: int = 300

    # Common save settings
    save: bool = False
    file_format: str = "pdf"
    tight_layout: bool = False
    path: Path | None = None

    # Common display settings
    title: str | None = None
    colors: list[str] = field(default_factory=list)

    # Statistics: paired t-test (matched repeats) by default; set False for
    # the unpaired ttest_ind.
    paired: bool = True
    # Write {title}_stats.csv / {title}_medians.csv to ``path``. Independent of
    # ``save`` so it works when plotting onto a shared/composed figure
    # (axes=..., save=False) — just provide a ``path``.
    save_stats: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for kwargs."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseDataProcessor(ABC):
    """Base class for data processing."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the data processor."""
        self.df = df
        self.validate_dataframe()

    @abstractmethod
    def validate_dataframe(self) -> None:
        """Validate required columns exist."""

    def filter_data(
        self,
        condition_col: str,
        conditions: list[str],
        selector_col: str | None = None,
        selector_val: str | None = None,
    ) -> pd.DataFrame:
        """Common filtering logic with validation."""
        # Validation with proper error messages
        if condition_col not in self.df.columns:
            raise ValueError(
                f"Column '{condition_col}' not found in dataframe"
            )

        # Filter by conditions
        filtered = self.df[self.df[condition_col].isin(conditions)].copy()

        # Apply selector filter if provided
        if selector_col and selector_val:
            if selector_col not in filtered.columns:
                raise ValueError(
                    f"Column '{selector_col}' not found in dataframe"
                )
            if selector_val not in filtered[selector_col].unique():
                raise ValueError(
                    f"Value '{selector_val}' not found in column '{selector_col}'"
                )
            filtered = filtered[filtered[selector_col] == selector_val]
        elif selector_col:
            raise ValueError(
                f"selector_val for {selector_col} must be provided"
            )

        if filtered.empty:
            raise ValueError("No data remaining after filtering")

        return filtered

    @abstractmethod
    def process_data(self, df: pd.DataFrame, **kwargs: Any) -> Any:
        """Process data for specific plot type."""


class BasePlotBuilder(ABC):
    """Base class for plot builders."""

    def __init__(self, config: BasePlotConfig):
        """Initialize the plot builder."""
        self.config = config
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.axes_provided: bool = False
        self._filename: str | None = None
        # Captured statistics for CSV export (see _record_stats / save_figure).
        self._stats_tables: list[pd.DataFrame] = []
        self._median_tables: list[pd.DataFrame] = []

    def create_figure(self, axes: Axes | None = None) -> "BasePlotBuilder":
        """Create or use existing figure."""
        if axes:
            self.fig = cast(
                Figure, axes.figure
            )  # Cast SubFigure to Figure for our use case
            self.ax = axes
            self.axes_provided = True
        else:
            fig_inches = convert_size_to_inches(
                self.config.fig_size, self.config.size_units
            )
            self.fig, self.ax = plt.subplots(figsize=fig_inches)
            self.axes_provided = False
        return self

    @abstractmethod
    def build_plot(self, data: Any, **kwargs: Any) -> "BasePlotBuilder":
        """Build the specific plot type."""

    def finalize_plot(
        self, default_title: str | None = None
    ) -> "BasePlotBuilder":
        """Finalize plot with title and store filename.

        Args:
            default_title: Default title to use if none provided
        """
        # Use provided title, config title, or default
        title = self.config.title or default_title

        # Use finalize_plot_with_title utility for consistent formatting
        from omero_screen_plots.utils import finalize_plot_with_title

        if self.fig is not None:
            self._filename = finalize_plot_with_title(
                self.fig,
                title,
                default_title or "plot",  # fallback feature name
                self.axes_provided,
            )
        return self

    def _record_stats(
        self,
        medians_df: pd.DataFrame,
        results: list[Any],
        *,
        value_label: str,
        condition_col: str,
        value_col: str,
        repeat_col: str = "plate_id",
        extra: dict[str, str] | None = None,
    ) -> None:
        """Capture stats + medians tables for later CSV export.

        Called by builders after computing significance. The tables are written
        to ``{title}_stats.csv`` / ``{title}_medians.csv`` by ``save_figure``.
        """
        from omero_screen_plots.stats import (
            medians_to_dataframe,
            stats_results_to_dataframe,
        )

        if results:
            self._stats_tables.append(
                stats_results_to_dataframe(
                    results, value_label=value_label, extra=extra
                )
            )
        if medians_df is not None and not medians_df.empty:
            self._median_tables.append(
                medians_to_dataframe(
                    medians_df,
                    repeat_col=repeat_col,
                    condition_col=condition_col,
                    value_col=value_col,
                    value_label=value_label,
                    extra=extra,
                )
            )

    def save_figure(self, filename: str | None = None) -> "BasePlotBuilder":
        """Save the figure (if ``save``) and stats CSVs (if ``save``/``save_stats``).

        Both outputs go to ``config.path``. Stats export is independent of the
        figure save so it works on a composed figure (``axes=...``, ``save=False``)
        — set ``save_stats=True`` and a ``path``.
        """
        final_filename = filename or self._filename or "plot"
        # Don't write the figure when embedded in a caller-provided axes — the
        # caller owns the composed figure (and its saving). Stats CSVs are still
        # exported, so save_stats works for composed/multi-panel figures.
        # Builders track this on either `axes_provided` (base) or
        # `_axes_provided` (feature/count/cellcycle subclasses).
        embedded = getattr(self, "_axes_provided", False) or self.axes_provided
        if (
            self.config.save
            and self.config.path
            and self.fig is not None
            and not embedded
        ):
            save_fig(
                self.fig,
                self.config.path,
                final_filename,
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi,
            )
        if self.config.save or self.config.save_stats:
            self._save_stats_tables(final_filename)
        return self

    def _save_stats_tables(self, fig_id: str) -> None:
        """Write captured p-value and median tables to ``config.path``."""
        if not self.config.path:
            if self._stats_tables or self._median_tables:
                from loguru import logger

                logger.warning(
                    "save_stats requested but no `path` set; stats CSV not written."
                )
            return
        from omero_screen_plots.stats import write_stats_csv

        if self._stats_tables:
            write_stats_csv(
                pd.concat(self._stats_tables, ignore_index=True),
                self.config.path,
                fig_id,
                "stats",
            )
        if self._median_tables:
            write_stats_csv(
                pd.concat(self._median_tables, ignore_index=True),
                self.config.path,
                fig_id,
                "medians",
            )

    def create_subplots(
        self,
        n_conditions: int,
        fig_size: tuple[float, float] | None = None,
    ) -> tuple[Figure, list[Axes]]:
        """Create figure with subplots for multiple conditions.

        Args:
            n_conditions: Number of conditions (subplots)
            fig_size: Optional figure size override

        Returns:
            Tuple of (Figure, list of Axes)
        """
        if n_conditions < 1:
            raise ValueError("Number of conditions must be at least 1")

        # Determine figure size
        if fig_size is None:
            # Default dynamic sizing if not provided
            # 5x5 for single, 4*N x 5 for multiple
            fig_size = (5, 5) if n_conditions == 1 else (4 * n_conditions, 5)

        # Convert to inches
        fig_inches = convert_size_to_inches(fig_size, self.config.size_units)

        # Create subplots
        self.fig, axes_array = plt.subplots(
            1, n_conditions, figsize=fig_inches
        )
        self.axes_provided = False

        # Handle single vs multiple axes return type
        if n_conditions == 1:
            self.ax = axes_array
            return self.fig, [axes_array]
        else:
            # For multiple, we don't set self.ax as there isn't a single one
            self.ax = None
            return self.fig, list(axes_array.flatten())

    def build(self) -> tuple[Figure, Axes]:
        """Return completed figure and axes."""
        if self.fig is None or self.ax is None:
            raise RuntimeError(
                "Figure and axes must be created before calling build()"
            )
        return self.fig, self.ax


@dataclass
class XYPlotConfig(BasePlotConfig):
    """Configuration for plots with X and Y axes."""

    # Plot features
    x_feature: str | None = None
    y_feature: str | None = None

    # Scale settings
    x_scale: Literal["linear", "log"] = "linear"
    x_scale_base: int = 2
    y_scale: Literal["linear", "log"] = "linear"
    y_scale_base: int = 10

    # Axis limits
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None

    # Axis ticks
    x_ticks: list[float] | None = None
    y_ticks: list[float] | None = None

    # Labels
    x_label: str | None = None
    y_label: str | None = None

    # Grid
    grid: bool = False
