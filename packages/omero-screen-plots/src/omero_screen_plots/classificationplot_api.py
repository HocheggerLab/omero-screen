"""Classification plot API with flexible display options."""

from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.classificationplot_factory import (
    ClassificationDataProcessor,
    ClassificationPlotBuilder,
    ClassificationPlotConfig,
)
from omero_screen_plots.utils import COLORS


def classification_plot(
    df: pd.DataFrame,
    classes: list[str],
    conditions: list[str],
    condition_col: str = "condition",
    class_col: str = "Class",
    selector_col: str | None = "cell_line",
    selector_val: str | None = None,
    display_mode: str = "stacked",
    # Figure settings
    fig_size: tuple[float, float] = (7, 7),
    size_units: str = "cm",
    y_lim: tuple[int, int] = (0, 100),
    title: str | None = None,
    colors: list[str] = COLORS,
    axes: Optional[Axes] = None,
    # Stacked mode settings
    bar_width: float = 0.75,
    show_legend: bool = True,
    legend_bbox: tuple[float, float] = (1.05, 1.0),
    # Triplicates mode settings
    group_size: int = 1,  # Default to 1 (no grouping)
    repeat_offset: float = 0.18,
    within_group_spacing: float = 0.2,
    between_group_gap: float = 0.4,
    # Save settings
    save: bool = False,
    path: Optional[Path] = None,
    file_format: str = "pdf",
    tight_layout: bool = True,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Create a classification plot with flexible display options.

    The plot behavior changes based on parameters:
    - display_mode="stacked": Shows stacked bars with error bars
    - display_mode="triplicates": Shows individual repeat bars
    - group_size > 1: Groups conditions together (works with triplicates mode)

    Parameters
    ----------
    Data Filtering
    ^^^^^^^^^^^^^^
    df : pd.DataFrame
        DataFrame containing classification data
    classes : list[str]
        List of class names to plot
    conditions : list[str]
        List of condition names to plot
    condition_col : str, default="condition"
        Column name for experimental condition
    class_col : str, default="Class"
        Column name for class/category (dynamic based on model)
    selector_col : str | None, default="cell_line"
        Column name for selector (e.g., cell line)
    selector_val : str | None, default=None
        Value to filter selector_col by

    Display Options
    ^^^^^^^^^^^^^^^
    display_mode : str, default="stacked"
        "stacked" for error bars, "triplicates" for individual repeats
    y_lim : tuple[int, int], default=(0, 100)
        Y-axis limits

    Styling & Colors
    ^^^^^^^^^^^^^^^^
    fig_size : tuple[float, float], default=(7, 7)
        Figure size as (width, height)
    size_units : str, default="cm"
        Units for fig_size ("cm" or "inches")
    title : str | None, default=None
        Plot title
    colors : list[str], default=COLORS
        List of colors for plotting
    axes : Axes | None, default=None
        Existing axes to plot on

    Stacked Mode Settings
    ^^^^^^^^^^^^^^^^^^^^^
    bar_width : float, default=0.75
        Width of bars in stacked mode
    show_legend : bool, default=True
        Whether to show legend
    legend_bbox : tuple[float, float], default=(1.05, 1.0)
        Legend position as (x, y)

    Triplicates Mode Settings
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    group_size : int, default=1
        Number of conditions to group (1 = no grouping)
    repeat_offset : float, default=0.18
        Offset for repeat bars
    within_group_spacing : float, default=0.2
        Spacing within groups
    between_group_gap : float, default=0.4
        Gap between groups

    Save Options
    ^^^^^^^^^^^^
    save : bool, default=False
        Whether to save the plot
    path : Path | None, default=None
        Path to save the plot
    file_format : str, default="pdf"
        File format for saving
    tight_layout : bool, default=True
        Whether to use tight layout
    dpi : int, default=300
        Resolution for saving

    Returns:
        Tuple of (figure, axes)

    Examples:
        # Stacked plot with error bars
        fig, ax = classification_plot(
            df, ["G1", "S", "G2M"], ["Control", "Treatment"],
            display_mode="stacked"
        )

        # Individual triplicates without grouping
        fig, ax = classification_plot(
            df, ["G1", "S", "G2M"], ["Control", "Treatment"],
            display_mode="triplicates"
        )

        # Individual triplicates with grouping
        fig, ax = classification_plot(
            df, ["G1", "S", "G2M"], ["Control", "Treat1", "Treat2", "Treat3"],
            display_mode="triplicates",
            group_size=2  # Groups conditions in pairs
        )
    """
    if display_mode not in ["stacked", "triplicates"]:
        raise ValueError("display_mode must be 'stacked' or 'triplicates'")

    # Create configuration
    config = ClassificationPlotConfig(
        fig_size=fig_size,
        size_units=size_units,
        title=title,
        colors=colors,
        save=save,
        path=path,
        file_format=file_format,
        tight_layout=tight_layout,
        dpi=dpi,
        display_mode=display_mode,
        y_lim=y_lim,
        bar_width=bar_width,
        show_legend=show_legend,
        legend_bbox=legend_bbox,
        group_size=group_size,
        repeat_offset=repeat_offset,
        within_group_spacing=within_group_spacing,
        between_group_gap=between_group_gap,
    )

    # Initialize data processor
    processor = ClassificationDataProcessor(df, class_col)

    # Validate conditions and classes exist in data
    if condition_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Condition column '{condition_col}' not found in dataframe. "
            f"Available columns: {available_cols}"
        )

    available_conditions = df[condition_col].unique().tolist()
    missing_conditions = [
        c for c in conditions if c not in available_conditions
    ]
    if missing_conditions:
        raise ValueError(
            f"Conditions not found in data: {missing_conditions}. "
            f"Available conditions: {available_conditions}"
        )

    available_classes = df[class_col].unique().tolist()
    missing_classes = [c for c in classes if c not in available_classes]
    if missing_classes:
        raise ValueError(
            f"Classes not found in data: {missing_classes}. "
            f"Available classes: {available_classes}"
        )

    # Validate selector parameters
    if selector_col is not None:
        if selector_col not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Selector column '{selector_col}' not found in dataframe. "
                f"Available columns: {available_cols}"
            )
        if selector_val is None:
            available_vals = df[selector_col].unique().tolist()
            raise ValueError(
                f"selector_val must be provided when selector_col is specified. "
                f"Available values in '{selector_col}': {available_vals}"
            )
        if selector_val not in df[selector_col].unique():
            available_vals = df[selector_col].unique().tolist()
            raise ValueError(
                f"Value '{selector_val}' not found in column '{selector_col}'. "
                f"Available values: {available_vals}"
            )

    # Filter data
    filtered_df = processor.filter_data(
        condition_col=condition_col,
        conditions=conditions,
        selector_col=selector_col,
        selector_val=selector_val,
    )

    # Create plot builder
    builder = ClassificationPlotBuilder(config)
    builder.create_figure(axes)

    # Process data and build plot based on display mode
    if display_mode == "stacked":
        # Process data for stacked plot
        plot_data, std_data = processor.process_data(
            filtered_df,
            condition_col=condition_col,
            conditions=conditions,
            classes=classes,
        )

        # Build stacked plot
        builder.build_plot(
            data=(plot_data, std_data),
            conditions=conditions,
            classes=classes,
            condition_col=condition_col,
            class_col=class_col,
        )
    else:  # triplicates
        # Build triplicates plot with original data
        builder.build_plot(
            data=filtered_df,
            conditions=conditions,
            classes=classes,
            condition_col=condition_col,
            class_col=class_col,
        )

    # Generate default title if none provided
    default_title = "Classification Analysis"
    if selector_val:
        default_title += f" {selector_val}"

    # Finalize and save
    builder.finalize_plot(default_title)
    builder.save_figure()

    return builder.build()
