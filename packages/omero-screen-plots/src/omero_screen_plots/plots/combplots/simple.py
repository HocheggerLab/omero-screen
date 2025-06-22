"""Simple combined plot with histogram, cell cycle scatter, and stacked bar.

This module provides the SimpleCombPlot class that creates a 2-row combined plot with:
- Top row: Histograms of DAPI intensity for each condition
- Bottom row: Cell cycle scatter plots for each condition + stacked bar plot summary
"""

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BaseCombPlot


class SimpleCombPlot(BaseCombPlot):
    """Simple combined plot with histogram, scatter, and bar chart.

    Creates a 2-row combined visualization that includes:
    - Histograms showing DNA content distribution
    - Scatter plots showing cell cycle analysis (DAPI vs EdU)
    - Stacked bar chart showing cell cycle phase proportions

    This comprehensive view is ideal for:
    - Complete cell cycle analysis workflow
    - Comparison across experimental conditions
    - Publication-ready multi-panel figures
    """

    @property
    def plot_type(self) -> str:
        return "simple_combplot"

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: List[str],
        show_h3_phases: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize simple combined plot.

        Args:
            data: DataFrame containing cell cycle data
            conditions: List of conditions to plot
            show_h3_phases: Whether to use H3-based cell cycle phases
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(data, conditions, **kwargs)
        self.show_h3_phases = show_h3_phases

        # Validate additional required columns for stacked bar
        if "plate_id" not in self.data.columns:
            warnings.warn(
                "'plate_id' column not found. Stacked bar plot may not work correctly."
            )

    def generate(self) -> Figure:
        """Generate the simple combined plot.

        Returns:
            Figure containing the combined plot
        """
        n_conditions = len(self.conditions)

        # Setup figure with custom width for the bar plot
        width_ratios = [1] * n_conditions + [
            1.2
        ]  # Last column wider for bar plot
        fig, gs = self.setup_subplot_grid(
            n_rows=2,
            n_cols=n_conditions + 1,
            height_ratios=[1, 3],
            width_ratios=width_ratios,
            hspace=0.05,
            wspace=0.25,
        )

        # Create plots for each condition
        scatter_axes = []  # Track scatter plot axes for x-label positioning
        hist_ref = None  # Reference histogram for title alignment

        for i, condition in enumerate(self.conditions):
            data = self.get_condition_data(condition)

            # Top row: Histogram (part of combplot, no x-label)
            hist_ax = fig.add_subplot(gs[0, i])
            self.create_histogram(
                hist_ax, data, i, show_individual_xlabel=False
            )
            hist_ax.set_title(condition, size=6, weight="regular", y=1.00)

            # Save first histogram as reference for bar plot title alignment
            if i == 0:
                hist_ref = hist_ax

            # Bottom row: Cell cycle scatter
            scatter_ax = fig.add_subplot(gs[1, i])
            self.create_cellcycle_scatter(scatter_ax, data, i, n_conditions)
            scatter_axes.append(scatter_ax)  # Track for x-label positioning

        # Right column: Stacked bar plot (spans both rows)
        bar_ax = fig.add_subplot(gs[:, -1])
        self._create_stacked_bar(bar_ax, fig, hist_ref)

        # Add common x-axis label centered under scatter plots only
        self.add_common_x_label(fig, target_axes=scatter_axes)

        # Apply title using centralized method
        self._apply_figure_title(fig)

        return fig

    def _create_stacked_bar(
        self,
        ax: Axes,
        fig: Optional[Figure] = None,
        hist_ref: Optional[Axes] = None,
    ) -> None:
        """Create stacked bar plot of cell cycle phases.

        Args:
            ax: Matplotlib axes for the bar plot
            fig: Figure object (for title positioning)
            hist_ref: Reference histogram axes for title alignment
        """
        try:
            # Calculate cell cycle proportions
            df_mean, df_std = self._calculate_phase_proportions()

            # Reorder columns to match scatter plot phase order
            if self.show_h3_phases:
                phase_order = ["Polyploid", "M", "G2", "S", "G1", "Sub-G1"]
            else:
                phase_order = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]

            # Reorder data to match scatter plot colors
            available_phases = [p for p in phase_order if p in df_mean.columns]
            df_mean = df_mean[available_phases]
            df_std = df_std[available_phases]

            # Create stacked bar plot
            df_mean.plot(
                kind="bar",
                stacked=True,
                yerr=df_std,
                width=0.75,
                ax=ax,
                color=self.colors[: len(available_phases)],
            )

            # Configure axes
            ax.set_ylim(0, 110)
            ax.set_xticklabels(self.conditions, rotation=30, ha="right")
            ax.set_xlabel("")
            ax.set_ylabel("% of population")
            ax.grid(False)

            # Position title to align with histogram titles
            if fig and hist_ref:
                # Get the position of both axes
                bar_bbox = ax.get_position()
                hist_bbox = hist_ref.get_position()

                # Use bar plot's horizontal center and histogram's title height
                title_x = bar_bbox.x0 + bar_bbox.width / 2
                # Position at same height as histogram titles (with same y offset of 1.00)
                title_y = hist_bbox.y1 + 0.02  # Same offset as used in y=1.00

                fig.text(
                    title_x,
                    title_y,
                    "Cell cycle phases",
                    fontsize=6,
                    ha="center",
                    weight="regular",
                )
            else:
                # Fallback to axis title
                ax.set_title("Cell cycle phases", fontsize=6, y=1.00)

            # Configure legend (match scatter plot order)
            if self.show_h3_phases:
                phase_labels = ["Polyploid", "M", "G2", "S", "G1", "Sub-G1"]
            else:
                phase_labels = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]

            # Get legend handles and reverse order to match visual stacking
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = handles[::-1], labels[::-1]

            # Remove default legend
            legend = ax.get_legend()
            if legend:
                legend.remove()

            # Position legend outside plot
            box = ax.get_position()
            ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))
            ax.legend(
                handles,
                phase_labels[::-1],  # Reverse to match visual stacking order
                title="CellCyclePhase",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=6,
                title_fontsize=7,
                frameon=False,
            )

        except Exception as e:
            warnings.warn(f"Could not create stacked bar plot: {e}")
            ax.text(
                0.5,
                0.5,
                "Bar plot\nunavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _calculate_phase_proportions(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate cell cycle phase proportions for stacked bar plot.

        Returns:
            Tuple of (mean_proportions, std_proportions)
        """
        # Import the function from the original module
        # This is a temporary solution - ideally this would be refactored
        try:
            from ...cellcycleplot import prop_pivot

            return prop_pivot(
                self.data,
                self.condition_col,
                self.conditions,
                self.show_h3_phases,
            )
        except ImportError:
            # Fallback implementation
            return self._fallback_phase_proportions()

    def _fallback_phase_proportions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fallback implementation for phase proportion calculation."""
        if self.show_h3_phases:
            phases = ["Polyploid", "M", "G2", "S", "G1", "Sub-G1"]
        else:
            phases = ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]

        # Calculate proportions by condition and replicate
        proportions = []
        for condition in self.conditions:
            condition_data = self.data[
                self.data[self.condition_col] == condition
            ]

            if "plate_id" in condition_data.columns:
                # Group by replicate
                for plate_id in condition_data["plate_id"].unique():
                    plate_data = condition_data[
                        condition_data["plate_id"] == plate_id
                    ]
                    phase_counts = plate_data["cell_cycle"].value_counts()
                    total_cells = len(plate_data)

                    phase_props = {}
                    for phase in phases:
                        phase_props[phase] = (
                            phase_counts.get(phase, 0) / total_cells
                        ) * 100

                    phase_props["condition"] = condition
                    phase_props["plate_id"] = plate_id
                    proportions.append(phase_props)
            else:
                # No replicates, just calculate overall proportions
                phase_counts = condition_data["cell_cycle"].value_counts()
                total_cells = len(condition_data)

                phase_props = {}
                for phase in phases:
                    phase_props[phase] = (
                        phase_counts.get(phase, 0) / total_cells
                    ) * 100

                phase_props["condition"] = condition
                proportions.append(phase_props)

        # Convert to DataFrame
        props_df = pd.DataFrame(proportions)

        # Calculate mean and std
        mean_props = props_df.groupby("condition")[phases].mean()
        std_props = props_df.groupby("condition")[phases].std().fillna(0)

        # Reindex to match condition order
        mean_props = mean_props.reindex(self.conditions)
        std_props = std_props.reindex(self.conditions)

        return mean_props, std_props

    def save(
        self,
        path: Union[str, Path],
        filename: Optional[str] = None,
        tight_layout: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the simple combined plot.

        Args:
            path: Path to save location
            filename: Optional filename. If None, generates descriptive name
            **kwargs: Additional save parameters
        """
        if filename is None:
            selector_part = (
                f"_{self.selector_val}" if self.selector_val else ""
            )
            h3_part = "_h3" if self.show_h3_phases else ""
            filename = f"simple_combplot{selector_part}{h3_part}.png"

        super().save(path, filename, tight_layout, **kwargs)


def simple_combplot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    # Simple combplot specific arguments
    cell_number: Optional[int] = None,
    dapi_col: str = "integrated_int_DAPI_norm",
    edu_col: str = "intensity_mean_EdU_nucleus_norm",
    show_h3_phases: bool = False,
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
    """Create a simple combined plot with histogram, cell cycle scatter, and stacked bar.

    This is the main user-facing function for creating simple combined plots that include
    histograms, cell cycle scatter plots, and a stacked bar chart summary.

    Args:
        data: DataFrame containing cell cycle data with required columns:
              - dapi_col: Normalized DAPI intensity (DNA content)
              - edu_col: Normalized EdU intensity (replication activity)
              - 'cell_cycle': Cell cycle phase annotations
              - condition_col: Column containing experimental conditions
              - selector_col: Column for data selection (e.g., cell_line)
              - 'plate_id': Replicate identifier (optional, for error bars)
        conditions: List of experimental conditions to plot

        # Data filtering arguments
        condition_col: Name of column containing experimental conditions
        selector_col: Name of column for data filtering (e.g., 'cell_line')
        selector_val: Value to filter by in selector_col (e.g., 'RPE-1')

        # Plot appearance arguments
        title: Overall plot title. If None, auto-generated from selector_val
        colors: Custom color palette. If None, uses default from config
        figsize: Figure size as (width, height) in inches. If None, uses default

        # Simple combplot specific arguments
        cell_number: Optional limit on number of cells per condition (for performance)
        dapi_col: Column name for DAPI intensity values
        edu_col: Column name for EdU intensity values
        show_h3_phases: Whether to show H3-based cell cycle phases (6 phases vs 5)

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
        >>> fig = simple_combplot(
        ...     data=cell_data,
        ...     conditions=['Control', 'Treatment'],
        ...     selector_val='RPE-1'
        ... )

        With H3 phases and cell sampling:
        >>> fig = simple_combplot(
        ...     data=cell_data,
        ...     conditions=['Control', 'CDK4i', 'CDK6i'],
        ...     selector_val='RPE-1',
        ...     show_h3_phases=True,
        ...     cell_number=10000
        ... )

        Complete analysis with saving:
        >>> fig = simple_combplot(
        ...     data=cell_data,
        ...     conditions=['DMSO', 'CDK4i_5μM', 'CDK6i_10μM'],
        ...     selector_val='RPE-1',
        ...     title='CDK4/6 Inhibitor Cell Cycle Analysis',
        ...     cell_number=5000,
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

    if save and not output_path:
        raise ValueError("output_path is required when save=True")

    # Auto-generate title if not provided
    if title is None and selector_val:
        title = f"Cell Cycle Analysis - {selector_val}"
    elif title is None:
        title = "Cell Cycle Analysis"

    # Create the plot instance
    plot = SimpleCombPlot(
        data=data,
        conditions=conditions,
        condition_col=condition_col,
        selector_col=selector_col,
        selector_val=selector_val,
        title=title,
        colors=colors,
        figsize=figsize,
        cell_number=cell_number,
        dapi_col=dapi_col,
        edu_col=edu_col,
        show_h3_phases=show_h3_phases,
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
                h3_part = "_h3" if show_h3_phases else ""
                sample_part = f"_n{cell_number}" if cell_number else ""
                filename = f"simple_combplot{selector_part}{h3_part}{sample_part}.{format}"

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

            print(f"Simple combined plot saved to: {save_path / filename}")
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
