"""Utility functions for OMERO screen plots."""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

current_dir = Path(__file__).parent
style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
plt.style.use(style_path)
prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]


def save_fig(
    fig: Figure,
    path: Path,
    fig_id: str,
    tight_layout: bool = False,
    fig_extension: str = "pdf",
    resolution: int = 300,
) -> None:
    """Save a matplotlib figure to a file.

    Parameters
    ----------
    fig : Figure
        The figure to save.
    path : Path
        The path for saving the figure.
    fig_id : str
        The name of the saved figure.
    tight_layout : bool, optional
        Whether to use tight layout (default is True).
    fig_extension : str, optional
        The file extension for the saved figure (default is 'pdf').
    resolution : int, optional
        The resolution of the saved figure in dpi (default is 300).

    Returns:
    -------
    None
        Saves the figure in the specified format.
    """
    dest = path / f"{fig_id}.{fig_extension}"
    print("Saving figure", fig_id)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(
        str(dest),
        format=fig_extension,
        dpi=resolution,
        facecolor="white",
        edgecolor="white",
    )


def selector_val_filter(
    df: pd.DataFrame,
    selector_col: Optional[str],
    selector_val: Optional[str],
    condition_col: Optional[str],
    conditions: Optional[list[str]],
) -> Optional[pd.DataFrame]:
    """Check if selector_val is provided for selector_col and filter df."""
    if condition_col and conditions:
        df = df[df[condition_col].isin(conditions)].copy()
    if selector_col and selector_val:
        return df[df[selector_col] == selector_val].copy()
    elif selector_col:
        raise ValueError(f"selector_val for {selector_col} must be provided")
    else:
        return df.copy()


def get_repeat_points(
    df: pd.DataFrame, condition_col: str, y_col: str
) -> pd.DataFrame:
    """Get repeat points for the given condition and y column."""
    return df.groupby(["plate_id", condition_col])[y_col].count().reset_index()


def show_repeat_points(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    ax: Axes,
) -> None:
    """Show repeat points."""
    sns.stripplot(
        data=df,
        x=condition_col,
        y=y_col,
        marker="o",
        size=3,
        color="lightgray",
        dodge=True,
        legend=False,
        order=conditions,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )


def show_repeat_points_grouped(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    ax: Axes,
    x_positions: list[float],
    jitter_width: float = 0.07,
) -> None:
    """Show repeat points with custom x-positions for grouped layouts.

    Args:
        df: DataFrame containing the data
        conditions: List of condition names
        condition_col: Column name containing condition values
        y_col: Column name for y-axis values
        ax: Matplotlib axes
        x_positions: Custom x-axis positions for each condition
        jitter_width: Width of jitter for point spreading
    """
    for i, condition in enumerate(conditions):
        condition_data = df[df[condition_col] == condition]
        if not condition_data.empty:
            y_values = condition_data[y_col].values
            # Add jitter to x-position
            x_jittered = np.random.normal(
                x_positions[i], jitter_width, size=len(y_values)
            )

            ax.scatter(
                x_jittered,
                y_values,
                marker="o",
                s=15,  # size=3 in seaborn roughly equals s=9 in scatter
                color="lightgray",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
                zorder=3,
            )


def show_repeat_points_adaptive(
    df: pd.DataFrame,
    conditions: list[str],
    condition_col: str,
    y_col: str,
    ax: Axes,
    group_size: int = 1,
    x_positions: list[float] | None = None,
) -> None:
    """Adaptively show repeat points based on group_size.

    Args:
        df: DataFrame containing the data
        conditions: List of condition names
        condition_col: Column name containing condition values
        y_col: Column name for y-axis values
        ax: Matplotlib axes
        group_size: Number of conditions per group
        x_positions: Optional custom x-axis positions
    """
    if group_size == 1 or x_positions is None:
        # Use standard seaborn stripplot
        show_repeat_points(df, conditions, condition_col, y_col, ax)
    else:
        # Use custom positioning for grouped layout
        show_repeat_points_grouped(
            df, conditions, condition_col, y_col, ax, x_positions
        )


def select_datapoints(
    df: pd.DataFrame, conditions: list[str], condition_col: str, n: int = 30
) -> pd.DataFrame:
    """Select up to n random datapoints per category and plate-id."""
    df_sampled = pd.DataFrame()
    for condition in conditions:
        for plate_id in df.plate_id.unique():
            df_sub = df[
                (df[condition_col] == condition) & (df.plate_id == plate_id)
            ]
            if len(df_sub) > 0:  # Include data if any exists
                if len(df_sub) > n:
                    df_sub = df_sub.sample(n=n, random_state=1)
                df_sampled = pd.concat([df_sampled, df_sub])
    return df_sampled


def scale_data(
    df: pd.DataFrame,
    scale_col: str,
    scale_min: float = 1.0,
    scale_max: float = 99.0,
) -> pd.DataFrame:
    """Scale data in the dataframe by the specified column."""
    p_low = np.percentile(df[scale_col], scale_min)
    p_high = np.percentile(df[scale_col], scale_max)
    df[scale_col] = np.clip(df[scale_col], p_low, p_high)
    df[scale_col] = ((df[scale_col] - p_low) / (p_high - p_low)) * 65535
    return df


def grouped_x_positions(
    n_conditions: int,
    group_size: int = 2,
    bar_width: float = 0.5,
    within_group_spacing: float = 0.5,
    between_group_gap: float = 1.0,
) -> list[float]:
    """Generate x-axis positions for grouped plots.

    Parameters:
    - n_conditions: number of conditions (bars/groups)
    - group_size: number of conditions per group
    - bar_width: width of each bar
    - within_group_spacing: space between conditions in a group (distance between bar edges)
    - between_group_gap: extra space between groups (distance between bar edges)
    Returns a list of x positions for each condition (bar center).
    """
    x_positions: list[float] = []
    pos: float = 0.0
    for i in range(n_conditions):
        x_positions.append(pos)
        if (i + 1) % group_size == 0 and (i + 1) < n_conditions:
            pos += bar_width + between_group_gap
        else:
            pos += bar_width + within_group_spacing
    return x_positions


def draw_triplicate_box(
    ax: Axes,
    trip_xs: list[float],
    height: float,
    *,
    half_width: float,
    y_min: float = 0.0,
    linewidth: float = 0.5,
    edgecolor: str = "black",
    zorder: int = 10,
) -> None:
    """Draw a single box hugging a condition's triplicate bars.

    The box spans from ``min(trip_xs) - half_width`` to
    ``max(trip_xs) + half_width`` horizontally and from ``y_min`` to ``height``
    vertically. Callers compute ``trip_xs``/``height``/``half_width`` themselves
    so the (deliberately different) per-plot-type triplicate geometry is
    preserved; this only removes the duplicated Rectangle boilerplate.

    Args:
        ax: Matplotlib axes to draw on.
        trip_xs: Bar-centre x positions of the condition's replicate bars.
        height: Top edge of the box (bar stack height).
        half_width: Padding added beyond the outermost bar centres.
        y_min: Bottom edge of the box (default 0).
        linewidth: Box edge line width.
        edgecolor: Box edge colour.
        zorder: Draw order (default 10, above the bars).
    """
    from matplotlib.patches import Rectangle

    left = min(trip_xs) - half_width
    right = max(trip_xs) + half_width
    ax.add_patch(
        Rectangle(
            (left, y_min),
            width=right - left,
            height=height - y_min,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            zorder=zorder,
        )
    )


def grouped_or_range_positions(
    n_conditions: int,
    *,
    use_grouped: bool,
    group_size: int = 1,
    within_group_spacing: float = 0.5,
    between_group_gap: float = 1.0,
) -> list[float]:
    """Return per-condition x positions, grouped or plain integer.

    When ``use_grouped`` (manual bars: grouped layout or triplicate mode) the
    positions come from :func:`grouped_x_positions` so ``between_group_gap``
    controls the spacing. Otherwise the layout is categorical (seaborn-drawn)
    and integer positions keep the overlays aligned. This is the single source
    of the "grouped vs integer" branch shared by the count and feature plots.
    """
    if use_grouped:
        return grouped_x_positions(
            n_conditions,
            group_size=group_size,
            within_group_spacing=within_group_spacing,
            between_group_gap=between_group_gap,
        )
    return [float(i) for i in range(n_conditions)]


def convert_size_to_inches(
    fig_size: tuple[float, float], size_units: str = "cm"
) -> tuple[float, float]:
    """Convert figure size from specified units to inches for matplotlib.

    Args:
        fig_size: Figure size as (width, height)
        size_units: Units of the figure size ("cm" or "inches")

    Returns:
        Figure size in inches
    """
    if size_units == "cm":
        return (fig_size[0] / 2.54, fig_size[1] / 2.54)
    return fig_size


def prepare_plot_data(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    condition_col: str,
    selector_col: Optional[str],
    selector_val: Optional[str],
    scale: bool = False,
) -> pd.DataFrame:
    """Prepare and filter data for plotting.

    Args:
        df: Input dataframe
        feature: Feature column to plot
        conditions: List of conditions to include
        condition_col: Column containing condition values
        selector_col: Optional column for filtering
        selector_val: Optional value to filter by
        scale: Whether to scale the data

    Returns:
        Filtered and optionally scaled dataframe
    """
    df_filtered = selector_val_filter(
        df, selector_col, selector_val, condition_col, conditions
    )
    assert df_filtered is not None, "No data found"
    if scale:
        df_filtered = scale_data(df_filtered, feature)
    return df_filtered


def setup_figure(
    axes: Optional[Axes],
    fig_size: tuple[float, float],
    size_units: str = "cm",
) -> tuple[Figure, Axes]:
    """Setup figure and axes for plotting.

    Args:
        axes: Optional existing axes to use
        fig_size: Figure size as (width, height)
        size_units: Units for figure size ("cm" or "inches")

    Returns:
        Figure and axes objects
    """
    fig_size_inches = convert_size_to_inches(fig_size, size_units)

    if axes is None:
        fig, ax = plt.subplots(figsize=fig_size_inches)
    else:
        fig = axes.get_figure()
        ax = axes

    return fig, ax


def create_standard_boxplot(
    ax: Axes,
    data: list[float] | npt.NDArray[np.floating[Any]],
    x_pos: float,
    color: Optional[str] = None,
    width: float = 0.5,
    alpha: float = 0.75,
    linewidth: float = 0.5,
    **kwargs: Any,
) -> None:
    """Create a standardized boxplot with consistent styling.

    Args:
        ax: Matplotlib axes
        data: Data to plot
        x_pos: X-axis position
        color: Box color (defaults to last color in palette)
        width: Box width
        alpha: Box transparency
        linewidth: Box edge line width
        **kwargs: Additional arguments passed to boxplot
    """
    if color is None:
        color = COLORS[-1]

    ax.boxplot(
        [data],
        positions=[x_pos],
        widths=width,
        showfliers=False,
        patch_artist=True,
        boxprops={
            "facecolor": color,
            "edgecolor": "black",
            "linewidth": linewidth,
            "alpha": alpha,
        },
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        **kwargs,
    )


def create_standard_violin(
    ax: Axes,
    data: list[float] | npt.NDArray[np.floating[Any]],
    x_pos: float,
    color: Optional[str] = None,
    width: float = 0.5,
    alpha: float = 0.75,
    linewidth: float = 0.5,
    **kwargs: Any,
) -> None:
    """Create a standardized violin plot with consistent styling.

    Args:
        ax: Matplotlib axes
        data: Data to plot
        x_pos: X-axis position
        color: Violin color (defaults to last color in palette)
        width: Violin width
        alpha: Violin transparency
        linewidth: Violin edge line width
        **kwargs: Additional arguments passed to violinplot
    """
    if color is None:
        color = COLORS[-1]

    vp = ax.violinplot(
        [data],
        positions=[x_pos],
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        **kwargs,
    )

    bodies = vp["bodies"] if isinstance(vp["bodies"], list) else [vp["bodies"]]
    for body in bodies:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(alpha)
        body.set_linewidth(linewidth)

    # Style the median line
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    # Style the whiskers (vertical lines)
    if "cbars" in vp:
        vp["cbars"].set_color("black")
        vp["cbars"].set_linewidth(0.5)

    # Style the caps (horizontal lines at whisker ends)
    if "cmins" in vp:
        vp["cmins"].set_color("black")
        vp["cmins"].set_linewidth(0.5)
    if "cmaxes" in vp:
        vp["cmaxes"].set_color("black")
        vp["cmaxes"].set_linewidth(0.5)


def set_y_limits(ax: Axes, ymax: float | tuple[float, float] | None) -> None:
    """Set y-axis limits handling both single values and tuples.

    Args:
        ax: Matplotlib axes
        ymax: Maximum y-value (single float) or (min, max) tuple
    """
    if ymax is not None:
        if isinstance(ymax, tuple):
            ax.set_ylim(ymax[0], ymax[1])
        else:
            ax.set_ylim(0, ymax)


def format_plot_labels(
    ax: Axes,
    feature: str,
    conditions: list[str],
    x_positions: list[float],
    x_label: bool = True,
) -> None:
    """Format plot labels with consistent styling.

    Args:
        ax: Matplotlib axes
        feature: Feature name for y-label
        conditions: Condition names for x-labels
        x_positions: X-axis positions
        x_label: Whether to show x-labels
    """
    ax.set_ylabel(feature)
    ax.set_xlabel("")
    ax.set_xticks(x_positions)

    if x_label:
        ax.set_xticklabels(conditions, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])


def finalize_plot_with_title(
    fig: Figure,
    title: Optional[str],
    feature: str,
    axes_provided: bool = False,
) -> str:
    """Finalize plot with title and return formatted filename.

    Args:
        fig: Matplotlib figure
        title: Plot title (if None, uses feature name)
        feature: Feature name (used as fallback title)
        axes_provided: Whether axes were provided (affects title placement)

    Returns:
        Formatted filename based on title
    """
    if not title:
        title = feature

    if not axes_provided and title:
        fig.suptitle(title, fontsize=7, weight="bold", x=0, y=1.05, ha="left")

    return title.replace(" ", "_") if title else "plot"


def find_dna_norm_column(df: pd.DataFrame) -> str:
    """Locate the normalised DNA-content column in a single-cell DataFrame.

    Post-channel-roles refactor, the nucleus channel is no longer renamed to
    ``DAPI``; the integrated intensity column is named after the actual
    fluorophore (e.g. ``integrated_int_Hoechst_norm``, ``integrated_int_H2B_RFP_norm``).
    Legacy plates still produce ``integrated_int_DAPI_norm``.

    This helper picks the column matching ``integrated_int_{X}_norm`` where ``X``
    is not ``EdU``. ``EdU`` is excluded because, although it has an
    ``integrated_int_*`` column upstream, it is a cell-cycle marker, not the
    DNA-content channel.

    Args:
        df: Single-cell DataFrame produced by the omero-screen pipeline.

    Returns:
        The name of the normalised DNA-content column.

    Raises:
        KeyError: If no candidate column is found, or if more than one
            non-EdU ``integrated_int_*_norm`` column is present (which would
            mean the DNA channel is ambiguous).
    """
    candidates = [
        col
        for col in df.columns
        if col.startswith("integrated_int_")
        and col.endswith("_norm")
        and not col.startswith("integrated_int_EdU")
    ]
    if not candidates:
        raise KeyError(
            "No normalised DNA-content column found. Expected a column matching "
            "'integrated_int_{channel}_norm' (e.g. 'integrated_int_DAPI_norm', "
            "'integrated_int_Hoechst_norm'). Has cellcycle_analysis run on this data?"
        )
    if len(candidates) > 1:
        raise KeyError(
            f"Ambiguous DNA-content column — multiple candidates found: {candidates}. "
            f"Pass the desired column name explicitly to the plot function."
        )
    return str(candidates[0])
