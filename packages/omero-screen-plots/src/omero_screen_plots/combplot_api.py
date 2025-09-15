"""Combined plot APIs using modern plotting functions."""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from omero_screen_plots.cellcycleplot_api import cellcycle_stacked
from omero_screen_plots.scatterplot_api import scatter_plot
from omero_screen_plots.utils import save_fig, selector_val_filter


def combplot_feature(
    df: pd.DataFrame,
    conditions: list[str],
    feature: str,
    threshold: float,
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    cell_number: Optional[int] = 3000,
    fig_size: tuple[float, float] = (10, 7),
    size_units: str = "cm",
    save: bool = True,
    path: Optional[Path] = None,
    file_format: str = "png",
    dpi: int = 300,
) -> tuple[Figure, list[Any]]:
    """Create a combined plot with histograms and scatter plots for feature analysis.

    Creates a 3×n_conditions grid layout:
    - Top row: DNA content histograms for each condition
    - Middle row: DNA vs EdU scatter plots with cell cycle phases
    - Bottom row: DNA vs custom feature scatter plots with threshold coloring

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
    feature : str
        Feature column name for bottom row scatter plots.
    threshold : float
        Threshold value for feature coloring (below=blue, above=red).
    cell_number : Optional[int], default=3000
        Number of cells to sample per condition for plotting.

    Styling & Layout
    ^^^^^^^^^^^^^^^^
    title : Optional[str], default=None
        Plot title. If None, generates default title.
    fig_size : tuple[float, float], default=(10, 7)
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
    >>> fig, axes = combplot_feature(
    ...     df=data,
    ...     conditions=["control", "treatment1", "treatment2"],
    ...     feature="intensity_mean_p21_nucleus",
    ...     threshold=5000,
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

    # Create figure with GridSpec layout
    n_conditions = len(conditions)
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(3, n_conditions, height_ratios=[1, 3, 3], hspace=0.05)
    axes = []

    # Calculate y-limits for consistency
    edu_max = df["intensity_mean_EdU_nucleus_norm"].quantile(0.99) * 1.5
    edu_min = df["intensity_mean_EdU_nucleus_norm"].quantile(0.01) * 0.8
    feature_max = df[feature].quantile(0.99) * 1.5
    feature_min = df[feature].quantile(0.01) * 0.8

    # Plot for each condition
    for i, condition in enumerate(conditions):
        # Filter data for this condition
        condition_data = df_filtered[df_filtered[condition_col] == condition]

        # Row 1: Histogram of DNA content (use full data, no sampling)
        ax_hist = fig.add_subplot(gs[0, i])
        # Use seaborn histplot directly, then set log scale after
        sns.histplot(
            data=condition_data,
            x="integrated_int_DAPI_norm",
            ax=ax_hist,
            color="steelblue",
        )
        ax_hist.set_xscale("log", base=2)
        ax_hist.set_xlim(1, 16)
        ax_hist.set_title(condition, fontsize=6, weight="regular")
        ax_hist.set_xlabel("")
        ax_hist.xaxis.set_visible(False)
        if i == 0:
            ax_hist.set_ylabel("Freq.", fontsize=6)
        else:
            ax_hist.yaxis.set_visible(False)
        ax_hist.tick_params(axis="both", which="major", labelsize=6)
        axes.append(ax_hist)

        # Sample data for scatter plots only
        if cell_number and len(condition_data) >= cell_number:
            condition_data_sampled = condition_data.sample(
                n=cell_number, random_state=42
            )
        else:
            condition_data_sampled = condition_data

        # Row 2: DNA vs EdU scatter plot with cell cycle phases
        ax_scatter_edu = fig.add_subplot(gs[1, i])
        scatter_plot(
            df=condition_data_sampled,
            conditions=condition,
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
            hue="cell_cycle",
            size=2,
            alpha=1,
            x_scale="log",
            y_scale="log",
            x_limits=(1, 16),
            y_limits=(edu_min, edu_max),
            kde_overlay=True,
            kde_alpha=0.1,
            vline=3,
            hline=3,
            axes=ax_scatter_edu,
            save=False,
        )
        ax_scatter_edu.set_xlabel("")
        if i == 0:
            ax_scatter_edu.set_ylabel("norm. EdU int.", fontsize=6)
        else:
            ax_scatter_edu.yaxis.set_visible(False)
        axes.append(ax_scatter_edu)

        # Row 3: DNA vs custom feature scatter plot with threshold coloring
        ax_scatter_feature = fig.add_subplot(gs[2, i])
        scatter_plot(
            df=condition_data_sampled,
            conditions=condition,
            x_feature="integrated_int_DAPI_norm",
            y_feature=feature,
            threshold=threshold,
            size=2,
            alpha=1,
            x_scale="log",
            x_limits=(1, 16),
            y_limits=(feature_min, feature_max),
            axes=ax_scatter_feature,
            save=False,
        )
        ax_scatter_feature.set_xlabel("")
        if i == 0:
            ax_scatter_feature.set_ylabel(feature, fontsize=6)
        else:
            ax_scatter_feature.yaxis.set_visible(False)
        axes.append(ax_scatter_feature)

    # Set common x-axis label
    fig.text(0.5, -0.07, "norm. DNA content", ha="center", fontsize=6)

    # Set title
    if not title:
        title = f"combplot_feature_{selector_val or 'all'}"
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
    edu_max = df["intensity_mean_EdU_nucleus_norm"].quantile(0.99) * 1.5
    edu_min = df["intensity_mean_EdU_nucleus_norm"].quantile(0.01) * 0.8

    # Plot histograms and scatter plots for each condition
    for i, condition in enumerate(conditions):
        # Filter data for this condition
        condition_data = df_filtered[df_filtered[condition_col] == condition]

        # Row 1: Histogram of DNA content (use full data, no sampling)
        ax_hist = fig.add_subplot(gs[0, i])
        # Use seaborn histplot directly, then set log scale after
        sns.histplot(
            data=condition_data,
            x="integrated_int_DAPI_norm",
            ax=ax_hist,
            color="steelblue",
        )
        ax_hist.set_xscale("log", base=2)
        ax_hist.set_xlim(1, 16)
        ax_hist.set_title(condition, fontsize=6, weight="regular")
        ax_hist.set_xlabel("")
        ax_hist.xaxis.set_visible(False)
        if i == 0:
            ax_hist.set_ylabel("Freq.", fontsize=6)
        else:
            ax_hist.yaxis.set_visible(False)
        ax_hist.tick_params(axis="both", which="major", labelsize=6)
        axes.append(ax_hist)

        # Sample data for scatter plot only
        if cell_number and len(condition_data) >= cell_number:
            condition_data_sampled = condition_data.sample(
                n=cell_number, random_state=42
            )
        else:
            condition_data_sampled = condition_data

        # Row 2: DNA vs EdU scatter plot with cell cycle phases
        ax_scatter = fig.add_subplot(gs[1, i])
        scatter_plot(
            df=condition_data_sampled,
            conditions=condition,
            x_feature="integrated_int_DAPI_norm",
            y_feature="intensity_mean_EdU_nucleus_norm",
            hue="cell_cycle",
            size=2,
            alpha=1.0,
            x_scale="log",
            y_scale="log",
            x_limits=(1, 16),
            y_limits=(edu_min, edu_max),
            kde_overlay=True,
            kde_alpha=0.1,
            vline=3,
            hline=3,
            axes=ax_scatter,
            save=False,
        )
        ax_scatter.set_xlabel("")  # Remove x-label
        if i == 0:
            ax_scatter.set_ylabel("norm. EdU int.", fontsize=6)
        else:
            ax_scatter.yaxis.set_visible(False)
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
