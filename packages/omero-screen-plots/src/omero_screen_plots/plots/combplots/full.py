"""Full combined plot with histogram, cell cycle scatter, and feature scatter.

This module provides the FullCombPlot class that creates a 3-row combined plot with:
- Top row: Histograms of DAPI intensity for each condition
- Middle row: Cell cycle scatter plots (DAPI vs EdU) for each condition
- Bottom row: Feature scatter plots (DAPI vs specified feature) for each condition
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BaseCombPlot


class FullCombPlot(BaseCombPlot):
    """Full combined plot with histogram, cell cycle scatter, and feature scatter.

    Creates a 3-row combined visualization that includes:
    - Histograms showing DNA content distribution
    - Cell cycle scatter plots (DAPI vs EdU)
    - Feature scatter plots (DAPI vs any specified feature)

    This comprehensive view is ideal for:
    - Detailed cell cycle and feature analysis
    - Multi-parameter comparison across conditions
    - Publication-ready comprehensive figures
    """

    @property
    def plot_type(self) -> str:
        return "full_combplot"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: List[str],
        feature_col: str,
        feature_threshold: float,
        **kwargs: Any,
    ) -> None:
        """Initialize full combined plot.

        Args:
            data: DataFrame containing cell cycle data
            conditions: List of conditions to plot
            feature_col: Column name for the feature to analyze
            feature_threshold: Threshold value for feature categorization
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data, conditions, **kwargs)
        self.feature_col = feature_col
        self.feature_threshold = feature_threshold

        # Validate feature column exists
        if feature_col not in self.data.columns:
            raise ValueError(
                f"Feature column '{feature_col}' not found in data"
            )

        # Calculate feature range
        self.feature_min = data[feature_col].quantile(0.01) * 0.8
        self.feature_max = data[feature_col].quantile(0.99) * 1.5

    def generate(self) -> Figure:
        """Generate the full combined plot.

        Returns:
            Figure containing the combined plot
        """
        n_conditions = len(self.conditions)

        # Setup figure with 3 rows
        fig, gs = self.setup_subplot_grid(
            n_rows=3,
            n_cols=n_conditions,
            height_ratios=[1, 3, 3],
            width_ratios=None,
            hspace=0.05,
        )

        # Create condition list for 3 rows
        condition_list = self.conditions * 3
        ax_list = [(i, j) for i in range(3) for j in range(n_conditions)]

        # Create plots for each position
        for i, (row, col) in enumerate(ax_list):
            condition = condition_list[i]
            data = self.get_condition_data(condition)
            ax = fig.add_subplot(gs[row, col])

            if i < n_conditions:
                # Top row: Histograms (part of combplot, no x-label)
                self.create_histogram(
                    ax, data, col, show_individual_xlabel=False
                )
                ax.set_title(condition, size=6, weight="regular")

            elif i < 2 * n_conditions:
                # Middle row: Cell cycle scatter plots
                self.create_cellcycle_scatter(ax, data, col, n_conditions)
                ax.set_ylim(self.edu_min, self.edu_max)

            else:
                # Bottom row: Feature scatter plots
                self._create_feature_scatter(ax, data, col, n_conditions)
                ax.set_ylim(self.feature_min, self.feature_max)

            # Disable grid for all plots
            ax.grid(visible=False)

        # Add common x-axis label
        self.add_common_x_label(fig)

        # Apply title using centralized method
        self._apply_figure_title(fig)

        return fig

    def _create_feature_scatter(
        self,
        ax: Axes,
        data: pd.DataFrame,
        condition_index: int,
        total_conditions: int,
    ) -> None:
        """Create feature scatter plot for the bottom row.

        Args:
            ax: Matplotlib axes to plot on
            data: Data for this condition
            condition_index: Index of condition
            total_conditions: Total number of conditions
        """
        # Create threshold-based categories
        data = data.copy()
        data.loc[:, "threshold_category"] = data[self.feature_col].apply(
            lambda x: "below" if x < self.feature_threshold else "above"
        )

        # Create scatter plot
        import seaborn as sns

        sns.scatterplot(
            data=data,
            x=self.dapi_col,
            y=self.feature_col,
            hue="threshold_category",
            palette={
                "below": "#87CEEB",
                "above": "#4169E1",
            },  # Light blue, blue
            hue_order=["below", "above"],
            s=2,
            alpha=1,
            ax=ax,
        )

        # Remove legend
        ax.legend().remove()

        # Setup axes using base class method
        self.setup_scatter_axes(
            ax,
            condition_index,
            total_conditions
            * 2,  # Multiply by 2 for the condition index logic
            self.feature_col,
            self.feature_min,
            self.feature_max,
        )

        # Don't set y-scale to log for feature plots (unlike EdU)
        # Just keep linear scale

    def setup_scatter_axes(
        self,
        ax: Axes,
        condition_index: int,
        total_conditions: int,
        y_col: str,
        y_min: float,
        y_max: float,
    ) -> None:
        """Setup scatter plot axes configuration (overridden from base).

        Args:
            ax: Matplotlib axes
            condition_index: Index of current condition
            total_conditions: Total number of conditions
            y_col: Y-axis column name
            y_min: Y-axis minimum
            y_max: Y-axis maximum
        """
        from matplotlib import ticker

        # X-axis (DAPI) configuration
        ax.set_xscale("log")
        ax.set_xlim(self.dapi_min, self.dapi_max)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: str(int(x)))
        )
        ax.set_xticks([2, 4, 8])
        ax.set_xlabel("")

        # Y-axis configuration
        ax.set_ylim(y_min, y_max)

        # Show y-label only on leftmost plot (condition_index == 0)
        if condition_index == 0:
            y_label = self._format_y_label(y_col)
            ax.set_ylabel(y_label, fontsize=6)

            # Only set log scale for EdU columns
            if "edu" in y_col.lower():
                ax.set_yscale("log", base=2)
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: str(int(x)))
                )
        else:
            ax.yaxis.set_visible(False)

        # Add reference lines
        ax.axvline(x=3, color="black", linestyle="--")
        ax.axhline(y=3, color="black", linestyle="--")

        # General styling
        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=6)

    def _format_y_label(self, y_col: str) -> str:
        """Format y-axis label from column name.

        Args:
            y_col: Column name

        Returns:
            Formatted label
        """
        if "edu" in y_col.lower():
            return "norm. EdU int."
        else:
            # Extract meaningful part from column name
            parts = y_col.split("_")
            if len(parts) >= 3:
                return f"{parts[2]} norm."
            return y_col.replace("_", " ").title()

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the full combined plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            feature_part = (
                f"_{self.feature_col.split('_')[-1]}"
                if "_" in self.feature_col
                else f"_{self.feature_col}"
            )
            filename = f"full_combplot{selector_part}{feature_part}.png"

        super().save(path, filename, tight_layout, **kwargs)


def full_combplot(
    data: pd.DataFrame,
    conditions: List[str],
    feature_col: str,
    feature_threshold: float,
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    # Full combplot specific arguments
    cell_number: Optional[int] = None,
    dapi_col: str = "integrated_int_DAPI_norm",
    edu_col: str = "intensity_mean_EdU_nucleus_norm",
    # Integration arguments
    ax: Optional[Axes] = None,
    # Output arguments
    save: bool = False,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,
    # Additional matplotlib/save arguments
    dpi: int = 300,
    format: str = "png",
    tight_layout: bool = False,
    **kwargs: Any,
) -> Figure:
    """Create a full combined plot with histogram, cell cycle scatter, and feature scatter.

    This is the main user-facing function for creating comprehensive combined plots that include
    histograms, cell cycle scatter plots, and feature scatter plots in a 3-row layout.

    Args:
        data: DataFrame containing cell cycle data with required columns:
              - dapi_col: Normalized DAPI intensity (DNA content)
              - edu_col: Normalized EdU intensity (replication activity)
              - feature_col: The feature column to analyze
              - 'cell_cycle': Cell cycle phase annotations
              - condition_col: Column containing experimental conditions
              - selector_col: Column for data selection (e.g., cell_line)
        conditions: List of experimental conditions to plot
        feature_col: Column name for the feature to plot against DAPI
        feature_threshold: Threshold value for feature categorization

        # Data filtering arguments
        condition_col: Name of column containing experimental conditions
        selector_col: Name of column for data filtering (e.g., 'cell_line')
        selector_val: Value to filter by in selector_col (e.g., 'RPE-1')

        # Plot appearance arguments
        title: Overall plot title. If None, auto-generated from selector_val
        colors: Custom color palette. If None, uses default from config
        figsize: Figure size as (width, height) in inches. If None, uses default

        # Full combplot specific arguments
        cell_number: Optional limit on number of cells per condition (for performance)
        dapi_col: Column name for DAPI intensity values
        edu_col: Column name for EdU intensity values

        # Integration arguments
        ax: Optional matplotlib axes to plot on. If provided, creates subplot

        # Output arguments
        save: Whether to save the figure to file
        output_path: Directory or full path for saving. Required if save=True
        filename: Specific filename. If None, auto-generated based on parameters

        # Save quality arguments
        dpi: Resolution for saved figure (dots per inch)
        format: File format ('png', 'pdf', 'svg', etc.)
        tight_layout: Whether to apply tight layout (False recommended for combplots)

        **kwargs: Additional arguments passed to the base class

    Returns:
        matplotlib.figure.Figure: The generated figure object

    Raises:
        ValueError: If required arguments are missing or invalid
        FileNotFoundError: If output_path doesn't exist when save=True

    Examples:
        Basic usage:
        >>> fig = full_combplot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     feature_col='intensity_mean_yH2AX_nucleus',
        ...     feature_threshold=3.0,
        ...     selector_val='RPE-1'
        ... )

        Multi-condition analysis:
        >>> fig = full_combplot(
        ...     data=cell_data,
        ...     conditions=['DMSO', 'CDK4i', 'CDK6i', 'Combination'],
        ...     feature_col='area_nucleus',
        ...     feature_threshold=5000,
        ...     selector_val='RPE-1',
        ...     cell_number=5000
        ... )

        Publication-ready output:
        >>> fig = full_combplot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     feature_col='intensity_mean_yH2AX_nucleus',
        ...     feature_threshold=3.0,
        ...     selector_val='RPE-1',
        ...     title='Comprehensive Cell Cycle and DNA Damage Analysis',
        ...     figsize=(10, 7),
        ...     save=True,
        ...     output_path='figures/',
        ...     dpi=600
        ... )
    """
    from pathlib import Path

    # Validate required arguments
    if data.empty:
        raise ValueError("Input data cannot be empty")

    if not conditions:
        raise ValueError("At least one condition must be specified")

    if feature_col not in data.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in data")

    if save and not output_path:
        raise ValueError("output_path is required when save=True")

    # Auto-generate title if not provided
    if title is None and selector_val:
        feature_name = (
            feature_col.split("_")[-1] if "_" in feature_col else feature_col
        )
        title = (
            f"Cell Cycle & {feature_name.title()} Analysis - {selector_val}"
        )
    elif title is None:
        feature_name = (
            feature_col.split("_")[-1] if "_" in feature_col else feature_col
        )
        title = f"Cell Cycle & {feature_name.title()} Analysis"

    # Create the plot instance
    plot = FullCombPlot(
        data=data,
        conditions=conditions,
        feature_col=feature_col,
        feature_threshold=feature_threshold,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        cell_number=cell_number,
        dapi_col=dapi_col,
        edu_col=edu_col,
        ax=ax,
        **kwargs,
    )

    # Generate the plot
    try:
        fig = plot.generate()

        # Save if requested (only if we own the figure)
        if save and plot._owns_figure:
            if output_path is None:
                raise ValueError("output_path cannot be None when save=True")
            save_path = Path(output_path)

            # Auto-generate filename if not provided
            if filename is None:
                # Create descriptive filename
                selector_part = f"_{selector_val}" if selector_val else ""
                feature_part = (
                    f"_{feature_col.split('_')[-1]}"
                    if "_" in feature_col
                    else f"_{feature_col}"
                )
                sample_part = f"_n{cell_number}" if cell_number else ""
                threshold_part = f"_t{feature_threshold}"
                filename = f"full_combplot{selector_part}{feature_part}{threshold_part}{sample_part}.{format}"

            # Ensure filename has correct extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            plot.save(
                path=save_path,
                filename=filename,
                tight_layout=tight_layout,
                dpi=dpi,
                format=format,
            )

            print(f"Full combined plot saved to: {save_path / filename}")
        elif save:
            print(
                "Warning: Cannot save when using provided axis. Save the parent figure manually."
            )

        return fig

    except Exception as e:
        # Clean up resources in case of error
        if plot._owns_figure:
            plot.close()
        raise e

    finally:
        # Note: We don't automatically close the figure here because the user
        # might want to further customize it. User should call plt.close(fig)
        # or plot.close() when done (if they own the figure).
        pass
