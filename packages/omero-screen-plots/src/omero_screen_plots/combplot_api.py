"""Combined plot APIs using modern plotting functions."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec

from omero_screen_plots.cellcycleplot_api import cellcycle_stacked
from omero_screen_plots.scatterplot_api import scatter_plot
from omero_screen_plots.utils import (
    find_dna_norm_column,
    save_fig,
    selector_val_filter,
)

# Column holding the normalised EdU intensity used for the cell-cycle scatter.
EDU_COL = "intensity_mean_EdU_nucleus_norm"


def _plot_dna_histogram(
    ax: Axes,
    data: pd.DataFrame,
    dna_col: str,
    condition: str,
    is_first_column: bool,
) -> None:
    """Draw a log-scaled DNA-content histogram on the given axes.

    Shared by the top row of both combplot variants.

    Args:
        ax: Target matplotlib axes.
        data: Single-condition slice of the (already filtered) data.
        dna_col: Name of the normalised DNA-content column.
        condition: Condition name, used as the column title.
        is_first_column: Whether this is the left-most column (keeps the y-axis).
    """
    sns.histplot(data=data, x=dna_col, ax=ax, color="steelblue")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 16)
    ax.set_title(condition, fontsize=6, weight="regular")
    ax.set_xlabel("")
    ax.xaxis.set_visible(False)
    if is_first_column:
        ax.set_ylabel("Freq.", fontsize=6)
    else:
        ax.yaxis.set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=6)


def _plot_edu_scatter(
    ax: Axes,
    data: pd.DataFrame,
    condition: str,
    condition_col: str,
    dna_col: str,
    edu_limits: tuple[float, float],
    is_first_column: bool,
) -> None:
    """Draw a DNA-vs-EdU scatter coloured by cell-cycle phase.

    Shared by the EdU row of both combplot variants.

    Args:
        ax: Target matplotlib axes.
        data: Single-condition (sampled) slice of the data.
        condition: Condition name.
        condition_col: Column holding the condition labels.
        dna_col: Name of the normalised DNA-content column (x-axis).
        edu_limits: (min, max) y-limits shared across conditions.
        is_first_column: Whether this is the left-most column (keeps the y-axis).
    """
    scatter_plot(
        df=data,
        conditions=condition,
        condition_col=condition_col,
        x_feature=dna_col,
        y_feature=EDU_COL,
        hue="cell_cycle",
        size=2,
        alpha=1.0,
        x_scale="log",
        y_scale="log",
        x_limits=(1, 16),
        y_limits=edu_limits,
        kde_overlay=True,
        kde_alpha=0.1,
        vline=3,
        hline=3,
        axes=ax,
        save=False,
    )
    ax.set_xlabel("")
    if is_first_column:
        ax.set_ylabel("norm. EdU int.", fontsize=6)
    else:
        ax.yaxis.set_visible(False)


def combplot_feature(
    df: pd.DataFrame,
    conditions: list[str],
    feature: Union[str, list[str]],
    threshold: Union[float, list[Optional[float]], None] = None,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    cell_number: Optional[int] = 3000,
    fig_size: Optional[tuple[float, float]] = None,
    size_units: str = "cm",
    save: bool = True,
    path: Optional[Path] = None,
    file_format: str = "png",
    dpi: int = 300,
    figure: Optional[Union[Figure, SubFigure]] = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> tuple[Union[Figure, SubFigure], list[Any]]:
    """Create a combined plot with histograms and scatter plots for feature analysis.

    Creates a ``(2 + n_features)×n_conditions`` grid layout that grows
    vertically with the number of features supplied:

    - Top row: DNA content histograms for each condition.
    - Second row: DNA vs EdU scatter plots with cell cycle phases.
    - One additional row per feature: DNA vs feature scatter plots with
      optional threshold colouring.

    Passing a single ``feature`` string reproduces the classic three-row
    layout (histogram → EdU → feature); passing a list of features stacks one
    scatter row per feature underneath the EdU row.

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        DataFrame containing single-cell measurements.
    conditions : list[str]
        List of condition names to plot.
    condition_col : str, default="condition"
        Column name for experimental condition.
    selector_col : Optional[str], default="cell_line"
        Column name for selector (e.g., cell line).
    selector_val : Optional[str], default=None
        Value to filter selector_col by.

    Plot Configuration
    ^^^^^^^^^^^^^^^^^^
    feature : str | list[str]
        Feature column name(s) for the stacked scatter rows. A single string
        creates one feature row; a list creates one row per entry, in order.
    threshold : float | list[float | None] | None, default=None
        Threshold value(s) for feature colouring (below=blue, above=red).
        A scalar (or ``None``) is applied to every feature row; a list must
        match the length of ``feature``, with ``None`` entries disabling the
        threshold for that row.
    cell_number : Optional[int], default=3000
        Number of cells to sample per condition for plotting.

    Styling & Layout
    ^^^^^^^^^^^^^^^^
    title : Optional[str], default=None
        Plot title. If None, generates default title.
    fig_size : Optional[tuple[float, float]], default=None
        Figure size as (width, height). When ``None`` the height scales with
        the number of rows (matching the classic ``(10, 7)`` default for a
        single feature).
    size_units : str, default="cm"
        Units for fig_size ("cm" or "inches").

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=True
        Whether to save the figure.
    path : Optional[Path], default=None
        Directory path for saving the figure.
    file_format : str, default="png"
        File format for saving ("png", "pdf", "svg", etc.).
    dpi : int, default=300
        Resolution for saved figure.

    Returns:
    -------
    tuple[Figure, list]
        The figure object and list of axes objects.

    Examples:
    --------
    Classic single-feature layout:

    >>> fig, axes = combplot_feature(
    ...     df=data,
    ...     conditions=["control", "treatment1", "treatment2"],
    ...     feature="intensity_mean_p21_nucleus",
    ...     threshold=5000,
    ...     selector_col="cell_line",
    ...     selector_val="MCF10A",
    ... )

    Stack several features as extra rows (one threshold per feature, ``None``
    to disable):

    >>> fig, axes = combplot_feature(
    ...     df=data,
    ...     conditions=["control", "treatment1"],
    ...     feature=[
    ...         "intensity_mean_p21_nucleus",
    ...         "intensity_mean_p53_nucleus",
    ...         "area_cell",
    ...     ],
    ...     threshold=[5000, 3000, None],
    ...     selector_col="cell_line",
    ...     selector_val="MCF10A",
    ... )
    """
    # Normalise feature/threshold into aligned lists.
    features = [feature] if isinstance(feature, str) else list(feature)
    if not features:
        raise ValueError("At least one feature must be provided")
    if isinstance(threshold, list):
        if len(threshold) != len(features):
            raise ValueError(
                f"threshold list length ({len(threshold)}) must match the "
                f"number of features ({len(features)})"
            )
        thresholds: list[Optional[float]] = threshold
    else:
        thresholds = [threshold] * len(features)
    n_features = len(features)

    # Default figure size: width fixed, height grows with the number of rows.
    # Row height-ratios are [1, 3, 3, ...] so total units = 1 + 3*(1 + n).
    # For a single feature this reproduces the historical (10, 7) default.
    if fig_size is None:
        fig_size = (10.0, float(1 + 3 * (1 + n_features)))

    # Convert size to inches if needed
    if size_units == "cm":
        fig_size = (fig_size[0] / 2.54, fig_size[1] / 2.54)

    # Filter data
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    if df_filtered is None:
        raise ValueError("No data remaining after filtering")

    # Resolve the actual DNA-content column for this DataFrame (legacy plates:
    # integrated_int_DAPI_norm; new plates: integrated_int_{fluorophore}_norm).
    dna_col = find_dna_norm_column(df)

    # Create figure with GridSpec layout: histogram + EdU + one row per feature.
    n_conditions = len(conditions)
    n_rows = 2 + n_features
    # ``figure`` lets several combplots share one canvas: pass a SubFigure from
    # Figure.subfigures() and this draws into it instead of making its own
    # figure, so a 2x2 of cell lines is one file rather than four. fig_size is
    # then set by the parent and ignored here.
    fig = plt.figure(figsize=fig_size) if figure is None else figure
    gs = fig.add_gridspec(
        n_rows,
        n_conditions,
        height_ratios=[1] + [3] * (1 + n_features),
        hspace=0.05,
    )
    axes = []

    # Calculate y-limits for consistency across conditions (full, unsampled data).
    edu_limits = (
        df[EDU_COL].quantile(0.01) * 0.8,
        df[EDU_COL].quantile(0.99) * 1.5,
    )
    feature_limits = [
        (df[f].quantile(0.01) * 0.8, df[f].quantile(0.99) * 1.5)
        for f in features
    ]

    # Plot for each condition
    for i, condition in enumerate(conditions):
        is_first = i == 0 and show_ylabel
        # Filter data for this condition
        condition_data = df_filtered[df_filtered[condition_col] == condition]

        # Row 0: Histogram of DNA content (use full data, no sampling)
        ax_hist = fig.add_subplot(gs[0, i])
        _plot_dna_histogram(
            ax_hist, condition_data, dna_col, condition, is_first
        )
        axes.append(ax_hist)

        # Sample data for scatter plots only
        if cell_number and len(condition_data) >= cell_number:
            condition_data_sampled = condition_data.sample(
                n=cell_number, random_state=42
            )
        else:
            condition_data_sampled = condition_data

        # Row 1: DNA vs EdU scatter plot with cell cycle phases
        ax_scatter_edu = fig.add_subplot(gs[1, i])
        _plot_edu_scatter(
            ax_scatter_edu,
            condition_data_sampled,
            condition,
            condition_col,
            dna_col,
            edu_limits,
            is_first,
        )
        # The rows sit at hspace=0.05, so tick labels on an interior row are
        # clipped by the axes below and render as half-height digits. Only the
        # bottom row (the last feature scatter) carries them.
        ax_scatter_edu.set_xticklabels([])
        axes.append(ax_scatter_edu)

        # Rows 2..n: one DNA vs feature scatter per feature, with optional
        # threshold colouring.
        for j, (feat, feat_threshold, (feat_min, feat_max)) in enumerate(
            zip(features, thresholds, feature_limits, strict=False)
        ):
            ax_scatter_feature = fig.add_subplot(gs[2 + j, i])
            scatter_plot(
                df=condition_data_sampled,
                conditions=condition,
                condition_col=condition_col,
                x_feature=dna_col,
                y_feature=feat,
                threshold=feat_threshold,
                size=2,
                alpha=1,
                x_scale="log",
                x_limits=(1, 16),
                y_limits=(feat_min, feat_max),
                axes=ax_scatter_feature,
                save=False,
            )
            ax_scatter_feature.set_xlabel("")
            if is_first:
                ax_scatter_feature.set_ylabel(feat, fontsize=6)
            else:
                ax_scatter_feature.yaxis.set_visible(False)
            axes.append(ax_scatter_feature)

    # Set common x-axis label
    if show_xlabel:
        fig.text(0.5, -0.07, "norm. DNA content", ha="center", fontsize=6)

    # Set title
    if not title:
        title = f"combplot_feature_{selector_val or 'all'}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")

    # Save figure. Only meaningful when this call owns the figure: a SubFigure
    # cannot be written on its own, so when composing, the parent saves.
    if save and path:
        if isinstance(fig, SubFigure):
            raise ValueError(
                "save=True is not supported when drawing into a SubFigure; "
                "save the parent figure instead."
            )
        figure_title = title.replace(" ", "_")
        save_fig(
            fig,
            path,
            figure_title,
            tight_layout=False,
            fig_extension=file_format,
        )

    return fig, axes


def combplot_cellcycle(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    cell_number: Optional[int] = 3000,
    cc_phases: bool = True,
    show_error_bars: bool = True,
    fig_size: tuple[float, float] = (12, 7),
    size_units: str = "cm",
    save: bool = True,
    path: Optional[Path] = None,
    file_format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, list[Any]]:
    """Create a combined plot with histograms, scatter plots, and cell cycle barplot.

    Creates a 2×(n_conditions+1) grid layout:
    - Top row: DNA content histograms for each condition + empty space for barplot
    - Bottom row: DNA vs EdU scatter plots for each condition + cell cycle stacked barplot

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        DataFrame containing single-cell measurements with cell cycle data.
    conditions : list[str]
        List of condition names to plot.
    condition_col : str, default="condition"
        Column name for experimental condition.
    selector_col : Optional[str], default="cell_line"
        Column name for selector (e.g., cell line).
    selector_val : Optional[str], default=None
        Value to filter selector_col by.

    Display Options
    ^^^^^^^^^^^^^^^
    cell_number : Optional[int], default=3000
        Number of cells to sample per condition for plotting.
    cc_phases : bool, default=True
        If True, use cell cycle terminology. If False, use DNA content terminology.
    show_error_bars : bool, default=True
        Whether to show error bars on the stacked cell cycle barplot.

    Styling & Layout
    ^^^^^^^^^^^^^^^^
    title : Optional[str], default=None
        Plot title. If None, generates default title.
    fig_size : tuple[float, float], default=(12, 7)
        Figure size as (width, height).
    size_units : str, default="cm"
        Units for fig_size ("cm" or "inches").

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=True
        Whether to save the figure.
    path : Optional[Path], default=None
        Directory path for saving the figure.
    file_format : str, default="png"
        File format for saving ("png", "pdf", "svg", etc.).
    dpi : int, default=300
        Resolution for saved figure.

    Returns:
    -------
    tuple[Figure, list]
        The figure object and list of axes objects.

    Examples:
    --------
    >>> fig, axes = combplot_cellcycle(
    ...     df=data,
    ...     conditions=["control", "treatment1", "treatment2"],
    ...     selector_col="cell_line",
    ...     selector_val="MCF10A"
    ... )
    """
    # Convert size to inches if needed
    if size_units == "cm":
        fig_size = (fig_size[0] / 2.54, fig_size[1] / 2.54)

    # Filter data
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    if df_filtered is None:
        raise ValueError("No data remaining after filtering")

    # Resolve the actual DNA-content column for this DataFrame.
    dna_col = find_dna_norm_column(df)

    # Create figure with GridSpec layout
    n_conditions = len(conditions)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(
        2,
        n_conditions + 1,
        height_ratios=[1, 3],
        width_ratios=[1] * n_conditions + [1.5],
        hspace=0.05,
        wspace=0.5,
    )
    axes = []

    # Calculate y-limits for consistency
    edu_limits = (
        df[EDU_COL].quantile(0.01) * 0.8,
        df[EDU_COL].quantile(0.99) * 1.5,
    )

    # Plot histograms and scatter plots for each condition
    for i, condition in enumerate(conditions):
        is_first = i == 0
        # Filter data for this condition
        condition_data = df_filtered[df_filtered[condition_col] == condition]

        # Row 0: Histogram of DNA content (use full data, no sampling)
        ax_hist = fig.add_subplot(gs[0, i])
        _plot_dna_histogram(
            ax_hist, condition_data, dna_col, condition, is_first
        )
        axes.append(ax_hist)

        # Sample data for scatter plot only
        if cell_number and len(condition_data) >= cell_number:
            condition_data_sampled = condition_data.sample(
                n=cell_number, random_state=42
            )
        else:
            condition_data_sampled = condition_data

        # Row 1: DNA vs EdU scatter plot with cell cycle phases
        ax_scatter = fig.add_subplot(gs[1, i])
        _plot_edu_scatter(
            ax_scatter,
            condition_data_sampled,
            condition,
            condition_col,
            dna_col,
            edu_limits,
            is_first,
        )
        axes.append(ax_scatter)

    # Add cell cycle stacked barplot in the last column
    ax_barplot = fig.add_subplot(gs[:, n_conditions])  # Spans both rows
    cellcycle_stacked(
        df=df_filtered,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        cc_phases=cc_phases,
        show_error_bars=show_error_bars,
        axes=ax_barplot,
        save=False,
    )
    axes.append(ax_barplot)

    # Set common x-axis label
    fig.text(0.5, -0.07, "norm. DNA content", ha="center", fontsize=6)

    # Set title
    if not title:
        title = f"combplot_cellcycle_{selector_val or 'all'}"
    fig.suptitle(title, fontsize=8, weight="bold", x=0, y=1.05, ha="left")

    # Save figure
    if save and path:
        figure_title = title.replace(" ", "_")
        save_fig(
            fig,
            path,
            figure_title,
            tight_layout=False,
            fig_extension=file_format,
        )

    return fig, axes
