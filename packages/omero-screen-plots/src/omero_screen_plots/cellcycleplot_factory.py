"""Cell cycle plot factory with unified configuration and base class architecture."""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.base import BasePlotBuilder, BasePlotConfig
from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import set_significance_marks
from omero_screen_plots.utils import (
    selector_val_filter,
)


@dataclass
class CellCyclePlotConfig(BasePlotConfig):
    """Configuration for cell cycle plots."""

    # Figure settings
    fig_size: tuple[float, float] = (6, 6)
    size_units: str = "cm"
    dpi: int = 300

    # Save settings
    save: bool = True
    file_format: str = "pdf"
    tight_layout: bool = False
    path: Path | None = None

    # Display settings
    title: str | None = None
    colors: list[str] = field(default_factory=list)

    # Cell cycle specific settings
    show_significance: bool = True
    show_repeat_points: bool = True
    rotation: int = 45
    cc_phases: bool = True  # True = {Sub-G1, G1, S, G2/M, Polyploid}, False = {<2N, 2N, S, 4N, >4N}


@dataclass
class StandardCellCyclePlotConfig(CellCyclePlotConfig):
    """Configuration specific to standard cell cycle plots."""

    # Standard plot specific settings
    show_subG1: bool = True  # Whether to include Sub-G1 phase in subplots
    show_plate_legend: bool = (
        False  # Whether to show legend with plate_id shapes
    )


class BaseCellCyclePlot(BasePlotBuilder):
    """Base class for cell cycle plots with common functionality."""

    PLOT_TYPE_NAME = "cellcycle"
    config: CellCyclePlotConfig  # Type annotation for mypy

    def __init__(self, config: CellCyclePlotConfig | None = None):
        """Initialize with configuration."""
        super().__init__(config or CellCyclePlotConfig())
        self.axes: list[Axes] | None = (
            None  # Cell cycle plots use multiple axes
        )

    def create_plot(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        condition_col: str = "condition",
        selector_col: str | None = None,
        selector_val: str | None = None,
    ) -> tuple[Figure, list[Axes]]:
        """Create complete cell cycle plot.

        Note: Cell cycle plots create their own 2x2 subplot figure and
        cannot accept an external axes parameter.

        Args:
            df: Input dataframe
            conditions: List of conditions to plot
            condition_col: Column name containing conditions
            selector_col: Optional column for filtering
            selector_val: Value to filter by if selector_col provided

        Returns:
            Tuple of (Figure, list of Axes) - differs from other plots
        """
        # Validate inputs
        self._validate_inputs(
            df,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Filter and process data
        processed_data = self._process_data(
            df,
            conditions,
            condition_col,
            selector_col,
            selector_val,
        )

        # Build plot (delegated to subclasses), passing original df for phase detection
        self.build_plot(
            processed_data,
            conditions=conditions,
            condition_col=condition_col,
            original_df=df,  # Pass original df for phase detection
        )

        # Finalize
        self._finalize_plot(selector_val)

        # Save if configured
        self._save_plot()

        assert self.fig is not None and self.axes is not None, (
            "Figure and axes should be created"
        )
        return self.fig, self.axes

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> None:
        """Validate all inputs with improved error messages."""
        if df.empty:
            raise ValueError("Input dataframe is empty")

        # Check required columns for cell cycle analysis
        required_cols = ["plate_id", "cell_cycle"]
        if missing_cols := set(required_cols) - set(df.columns):
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Check condition column
        if condition_col not in df.columns:
            raise ValueError(
                f"Condition column '{condition_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Check conditions exist
        available_conditions = set(df[condition_col].unique())
        if missing_conditions := set(conditions) - available_conditions:
            raise ValueError(
                f"Conditions not found in data: {missing_conditions}. "
                f"Available conditions: {sorted(available_conditions)}"
            )

        # Check selector parameters
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

    def _process_data(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        selector_col: str | None,
        selector_val: str | None,
    ) -> pd.DataFrame:
        """Process data for cell cycle plots."""
        # Filter data using existing utility
        filtered_data = selector_val_filter(
            df, selector_col, selector_val, condition_col, conditions
        )
        if filtered_data is None or filtered_data.empty:
            raise ValueError(
                "No data remaining after filtering and processing"
            )

        # Calculate cell cycle phase percentages
        return self._calculate_phase_percentages(filtered_data, condition_col)

    def _calculate_phase_percentages(
        self, df: pd.DataFrame, condition_col: str = "condition"
    ) -> pd.DataFrame:
        """Calculate the percentage of cells in each cell cycle phase for each condition.

        Args:
            df: DataFrame containing cell cycle data.
            condition_col: Column name for experimental condition.

        Returns:
            DataFrame with percentage of cells in each phase per condition.
        """
        return (
            (
                df.groupby(
                    ["plate_id", "cell_line", condition_col, "cell_cycle"]
                )["experiment"].count()
                / df.groupby(["plate_id", "cell_line", condition_col])[
                    "experiment"
                ].count()
                * 100
            )
            .reset_index()
            .rename(columns={"experiment": "percent"})
        )

    def _get_phases_and_colors(
        self, df: pd.DataFrame
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Determine cell cycle phases and their colors based on data and configuration.

        Args:
            df: DataFrame containing cell cycle data

        Returns:
            Tuple of (data_phases_list, display_phase_mapping, color_mapping_dict)
        """
        # Check if M phase exists in data (auto-detection)
        available_phases = (
            set(df["cell_cycle"].unique())
            if "cell_cycle" in df.columns
            else set()
        )
        has_M_phase = "M" in available_phases

        # Define data phases (always use original data column names for filtering)
        if has_M_phase:
            data_phases = ["Sub-G1", "G1", "S", "G2", "M", "Polyploid"]
        else:
            data_phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]

        # Define display phase mapping based on cc_phases setting
        if has_M_phase:
            if self.config.cc_phases:
                display_mapping = {
                    "Sub-G1": "Sub-G1",
                    "G1": "G1",
                    "S": "S",
                    "G2": "G2",
                    "M": "M",
                    "Polyploid": "Polyploid",
                }
            else:
                display_mapping = {
                    "Sub-G1": "<2N",
                    "G1": "2N",
                    "S": "S",
                    "G2": "G2",
                    "M": "M",
                    "Polyploid": ">4N",
                }
        else:
            if self.config.cc_phases:
                display_mapping = {
                    "Sub-G1": "Sub-G1",
                    "G1": "G1",
                    "S": "S",
                    "G2/M": "G2/M",
                    "Polyploid": "Polyploid",
                }
            else:
                display_mapping = {
                    "Sub-G1": "<2N",
                    "G1": "2N",
                    "S": "S",
                    "G2/M": "4N",
                    "Polyploid": ">4N",
                }

        # Define color mapping for data phases (always use data phase names as keys)
        color_mapping = {
            "Sub-G1": COLOR.GREY.value,
            "G1": COLOR.PINK.value,
            "S": COLOR.LIGHT_BLUE.value,
            "G2/M": COLOR.YELLOW.value,
            "G2": COLOR.YELLOW.value,
            "M": COLOR.TURQUOISE.value,
            "Polyploid": COLOR.BLUE.value,
        }

        return data_phases, display_mapping, color_mapping

    @abstractmethod
    def build_plot(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BasePlotBuilder":
        """Build the specific cell cycle plot type. Implemented by subclasses."""

    def _finalize_plot(self, selector_val: str | None) -> None:
        """Finalize plot with title."""
        # Generate default title
        default_title = (
            f"Cellcycle Analysis {selector_val}"
            if selector_val
            else "Cellcycle Analysis"
        )

        # Use provided title, config title, or default
        title = self.config.title or default_title

        # Apply title to figure
        assert self.fig is not None
        self.fig.suptitle(
            title, fontsize=8, weight="bold", x=0, y=1.05, ha="left"
        )
        self._filename = title.replace(" ", "_")

    def _save_plot(self) -> None:
        """Save figure using parent's save_figure method."""
        # Use parent's save_figure method with our filename
        self.save_figure(self._filename or "cellcycleplot")


class StandardCellCyclePlot(BaseCellCyclePlot):
    """Standard cell cycle plot with variable subplot grid showing each phase separately."""

    PLOT_TYPE_NAME = "cellcycle"
    config: StandardCellCyclePlotConfig  # Type annotation for specific config

    def __init__(self, config: StandardCellCyclePlotConfig | None = None):
        """Initialize with standard-specific configuration."""
        super().__init__(config or StandardCellCyclePlotConfig())

    def _setup_subplots(self, n_phases: int) -> None:
        """Setup variable subplot figure for standard cell cycle plot.

        Args:
            n_phases: Number of phases to plot (determines subplot layout)
                     4 phases: 2x2, 5 phases: 2x3, 6 phases: 2x3
        """
        # Convert fig_size if needed
        if self.config.size_units == "cm":
            fig_size = (
                self.config.fig_size[0] / 2.54,
                self.config.fig_size[1] / 2.54,
            )
        else:
            fig_size = self.config.fig_size

        # Determine subplot layout based on number of phases
        if n_phases <= 4:
            nrows, ncols = 2, 2
        elif n_phases <= 6:
            nrows, ncols = 2, 3
            # Adjust figure width for 2x3 layout
            fig_size = (fig_size[0] * 1.5, fig_size[1])  # Make wider
        else:
            raise ValueError(
                f"Cannot handle {n_phases} phases in subplot grid"
            )

        # Create figure with variable subplots
        self.fig, ax_array = plt.subplots(nrows, ncols, figsize=fig_size)

        # Handle different subplot layouts
        if n_phases <= 4:
            self.axes = [
                ax_array[0, 0],
                ax_array[0, 1],
                ax_array[1, 0],
                ax_array[1, 1],
            ][:n_phases]
        else:  # 2x3 layout
            self.axes = [
                ax_array[0, 0],
                ax_array[0, 1],
                ax_array[0, 2],
                ax_array[1, 0],
                ax_array[1, 1],
                ax_array[1, 2],
            ][:n_phases]

            # Hide unused subplots if n_phases < 6
            for i in range(n_phases, 6):
                row, col = divmod(i, 3)
                ax_array[row, col].set_visible(False)

        # Set ax to None since we're using multiple axes
        self.ax = None

    def build_plot(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BasePlotBuilder":
        """Build cell cycle plot with variable subplot grid for each phase."""
        # Extract required parameters from kwargs
        conditions = kwargs["conditions"]
        condition_col = kwargs["condition_col"]
        original_df = kwargs["original_df"]

        # Determine phases and colors based on original data and configuration
        data_phases, display_mapping, color_mapping = (
            self._get_phases_and_colors(original_df)
        )

        # Filter out Sub-G1 if not configured to show it
        if not self.config.show_subG1:
            data_phases = [p for p in data_phases if p != "Sub-G1"]

        # Setup subplots based on final number of phases
        self._setup_subplots(len(data_phases))

        # Determine layout info for subplot formatting
        n_phases = len(data_phases)
        is_2x3_layout = n_phases > 4

        assert self.axes is not None

        # Create a plot for each cell cycle phase
        for i, data_phase in enumerate(data_phases):
            ax = self.axes[i]
            df_phase = data[
                (data.cell_cycle == data_phase)
                & (data[condition_col].isin(conditions))
            ]

            # Get display name for this phase
            display_phase = display_mapping[data_phase]

            # Create bar plot for this phase
            sns.barplot(
                data=df_phase,
                x=condition_col,
                y="percent",
                color=color_mapping[
                    data_phase
                ],  # Use data phase for color lookup
                order=conditions,
                ax=ax,
            )

            # Add repeat points if enabled
            if self.config.show_repeat_points:
                self._add_repeat_points_with_shapes(
                    df_phase=df_phase,
                    conditions=conditions,
                    condition_col=condition_col,
                    ax=ax,
                )

            # Add significance marks if we have enough replicates
            if self.config.show_significance and data.plate_id.nunique() >= 3:
                set_significance_marks(
                    ax,
                    df_phase,
                    conditions,
                    condition_col,
                    "percent",
                    ax.get_ylim()[1],
                )

            # Format this subplot using display name for title
            self._format_subplot(
                ax, display_phase, conditions, i, is_2x3_layout
            )

        # Add plate legend if enabled and we have multiple plates
        if (
            self.config.show_plate_legend
            and self.config.show_repeat_points
            and self._has_multiple_plates(data)
        ):
            self._add_plate_legend_to_figure(data)

        return self

    def _format_subplot(
        self,
        ax: Axes,
        phase: str,
        conditions: list[str],
        subplot_index: int,
        is_2x3_layout: bool = False,
    ) -> None:
        """Format individual subplot in variable grid layout."""
        # Set subplot title
        ax.set_title(f"{phase}", fontsize=6, y=1.05)

        if is_2x3_layout:
            # 2x3 layout: positions are [0,1,2] top row, [3,4,5] bottom row
            # Remove y-label for non-leftmost subplots (1,2,4,5)
            if subplot_index not in {
                0,
                3,
            }:  # Keep y-label only for leftmost subplots
                ax.set_ylabel("")

            # Remove x-tick labels for top row (0,1,2)
            if subplot_index in {0, 1, 2}:
                ax.set_xticklabels([])
            else:
                # Bottom row (3,4,5) get x-tick labels
                ax.set_xticks(range(len(conditions)))
                ax.set_xticklabels(
                    conditions, rotation=self.config.rotation, ha="right"
                )
        else:
            # 2x2 layout: positions are [0,1] top row, [2,3] bottom row
            # Remove y-label for right subplots (1 and 3)
            if subplot_index in {1, 3}:
                ax.set_ylabel("")

            # Remove x-tick labels for top subplots (0 and 1)
            if subplot_index in {0, 1}:
                ax.set_xticklabels([])
            else:
                # Bottom subplots (2 and 3) get x-tick labels
                ax.set_xticks(range(len(conditions)))
                ax.set_xticklabels(
                    conditions, rotation=self.config.rotation, ha="right"
                )

        # Remove x-label for all subplots (common x-axis)
        ax.set_xlabel("")

    def _add_repeat_points_with_shapes(
        self,
        df_phase: pd.DataFrame,
        conditions: list[str],
        condition_col: str,
        ax: Axes,
    ) -> None:
        """Add repeat points with different shapes for each plate."""
        # Define marker shapes for each plate
        markers = ["s", "o", "^"]  # square, circle, triangle
        plate_ids = sorted(df_phase["plate_id"].unique())

        # Add jitter width for spacing
        jitter_width = 0.05

        # Plot points for each plate with different shapes
        for plate_idx, plate_id in enumerate(
            plate_ids[:3]
        ):  # Limit to 3 plates
            plate_data = df_phase[df_phase["plate_id"] == plate_id]
            marker = markers[plate_idx % len(markers)]

            for cond_idx, condition in enumerate(conditions):
                cond_plate_data = plate_data[
                    plate_data[condition_col] == condition
                ]
                if not cond_plate_data.empty:
                    x_base = cond_idx
                    y_values = cond_plate_data["percent"].values

                    # Add jitter for visibility when multiple plates
                    if len(plate_ids) > 1:
                        x_jittered = np.random.normal(
                            x_base, jitter_width, size=len(y_values)
                        )
                    else:
                        x_jittered = np.array([x_base] * len(y_values))

                    ax.scatter(
                        x_jittered,
                        y_values,
                        marker=marker,
                        color="lightgray",
                        s=25,  # size
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.8,
                        zorder=3,
                    )

    def _add_plate_legend_to_figure(self, data: pd.DataFrame) -> None:
        """Add legend showing plate symbols to the figure."""
        plate_ids = sorted(data["plate_id"].unique())
        # Limit to 3 plates maximum for legend
        plate_ids = plate_ids[:3]

        # Define marker shapes for each plate
        markers = ["s", "o", "^"]  # square, circle, triangle

        from matplotlib.lines import Line2D

        # Create legend handles
        handles = []
        for i, plate_id in enumerate(plate_ids):
            marker = markers[i % len(markers)]
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="lightgray",
                markerfacecolor="lightgray",
                markersize=6,
                linestyle="None",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=str(plate_id),
            )
            handles.append(handle)

        # Add legend to the figure (not to individual subplots)
        assert self.fig is not None
        self.fig.legend(
            handles=handles,
            title="Plate ID",
            bbox_to_anchor=(1.02, 0.8),
            loc="upper left",
        )

    def _has_multiple_plates(self, data: pd.DataFrame) -> bool:
        """Check if data contains multiple plates."""
        return bool(data["plate_id"].nunique() > 1)
