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
class PlotRequest:
    """Typed per-call parameters handed to a builder's ``build_plot``.

    Replaces the former untyped ``build_plot(self, data, **kwargs)`` key-digging
    (``kwargs["conditions"]``, ``kwargs["x_positions"]``, ...). Not every field
    applies to every plot type — each builder reads only the subset it needs —
    but collecting them in one checked, discoverable object removes the silent
    string-keyed contract. Config-level options stay on the ``*PlotConfig``
    dataclasses; ``PlotRequest`` carries the values that vary per call.
    """

    conditions: list[str]
    condition_col: str = "condition"
    # Feature / count / histogram: the measurement column being plotted.
    feature: str | None = None
    # Box/violin grouped x-axis positions, precomputed by the builder.
    x_positions: list[float] = field(default_factory=list)
    # Cell-cycle plots need the unaggregated df for M-phase auto-detection.
    original_df: pd.DataFrame | None = None
    # Classification: the per-cell class column and the classes to stack.
    classes: list[str] | None = None
    class_col: str | None = None


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
    # Draw "ns" markers for non-significant comparisons. Set False to plot only
    # the stars; the CSV export is unaffected either way.
    show_ns: bool = True
    # Require at least this relative change (0.10 = 10%) against the reference
    # before drawing a marker. None disables the gate. CSV export is unaffected.
    min_effect: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for kwargs."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BasePlotBuilder(ABC):
    """Base class for plot builders."""

    # Columns every input DataFrame must contain for this plot type. Subclasses
    # override with their own requirements; ``_validate_common`` enforces it.
    REQUIRED_COLUMNS: tuple[str, ...] = ("plate_id",)

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

    def _validate_common(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
        *,
        feature: str | None = None,
    ) -> None:
        """Shared input validation for condition-based plots.

        Checks, in order: non-empty df, ``REQUIRED_COLUMNS`` present, the
        optional ``feature`` column, the condition column exists, the requested
        conditions exist, and the selector column/value are consistent. Plot
        types pass ``feature`` when they plot a named feature column, and add
        any further checks (e.g. ``norm_control``) around this call.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")

        if missing_cols := set(self.REQUIRED_COLUMNS) - set(df.columns):
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        if feature is not None and feature not in df.columns:
            raise ValueError(
                f"Feature column '{feature}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if condition_col not in df.columns:
            raise ValueError(
                f"Condition column '{condition_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        available_conditions = set(df[condition_col].unique())
        if missing_conditions := set(conditions) - available_conditions:
            raise ValueError(
                f"Conditions not found in data: {missing_conditions}. "
                f"Available conditions: {sorted(available_conditions)}"
            )

        if selector_col and selector_col not in df.columns:
            raise ValueError(
                f"Selector column '{selector_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if selector_col and not selector_val:
            available_values = sorted(df[selector_col].unique())
            raise ValueError(
                f"selector_val must be provided when selector_col is specified. "
                f"Available values in '{selector_col}': {available_values}"
            )

        if (
            selector_col
            and selector_val
            and selector_val not in df[selector_col].unique()
        ):
            available_values = sorted(df[selector_col].unique())
            raise ValueError(
                f"Value '{selector_val}' not found in column '{selector_col}'. "
                f"Available values: {available_values}"
            )

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
    def build_plot(
        self, data: Any, request: "PlotRequest"
    ) -> "BasePlotBuilder":
        """Build the specific plot type from a typed :class:`PlotRequest`.

        Each builder reads the subset of ``request`` fields it needs; this
        replaced the former untyped ``build_plot(self, data, **kwargs)`` pipe.
        """

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

    def _annotate_and_record_stats(
        self,
        ax: Axes,
        stat_df: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        value_col: str,
        x_positions: list[float],
        *,
        value_label: str,
        group_size: int = 1,
        y_top_factor: float = 0.93,
        normalise_within_plate: bool = False,
        extra: dict[str, str] | None = None,
    ) -> None:
        """Compute significance, annotate stars on ``ax``, and record for CSV.

        Bundles the identical ``compute_significance -> annotate_significance ->
        _record_stats`` trio shared by the count, feature and (per-phase)
        standard cell-cycle plots. The ``nunique >= 3`` replicate gate stays in
        the caller, since plot types differ on which frame drives the count.
        ``normalise_within_plate`` forwards to the log fold-change test for
        ratio-scale readouts (count/feature); leave False for proportions.
        """
        from omero_screen_plots.stats import (
            annotate_significance,
            compute_significance,
        )

        medians_df, results = compute_significance(
            stat_df,
            conditions,
            condition_col,
            value_col,
            group_size=group_size,
            paired=self.config.paired,
            normalise_within_plate=normalise_within_plate,
        )
        annotate_significance(
            ax,
            results,
            x_positions,
            ax.get_ylim()[1] * y_top_factor,
            show_ns=self.config.show_ns,
            min_effect=self.config.min_effect,
        )
        self._record_stats(
            medians_df,
            results,
            value_label=value_label,
            condition_col=condition_col,
            value_col=value_col,
            extra=extra,
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
        if (
            self.config.save
            and self.config.path
            and self.fig is not None
            and not self.axes_provided
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
