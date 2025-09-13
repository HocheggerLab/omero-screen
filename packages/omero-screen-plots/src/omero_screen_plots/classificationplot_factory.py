"""Classification plot factory following the base class architecture."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.patches import Patch, Rectangle

from omero_screen_plots.base import (
    BaseDataProcessor,
    BasePlotBuilder,
    BasePlotConfig,
)
from omero_screen_plots.colors import COLOR
from omero_screen_plots.utils import (
    grouped_x_positions,
)


@dataclass
class ClassificationPlotConfig(BasePlotConfig):
    """Configuration for classification plots."""

    # Display mode: "stacked" for error bars, "triplicates" for individual repeats
    display_mode: str = "stacked"
    y_lim: tuple[int, int] = (0, 100)

    # Triplicate mode settings
    repeat_offset: float = 0.18
    group_size: int = 2
    within_group_spacing: float = 0.6
    between_group_gap: float = 0.7

    # Stacked mode settings
    bar_width: float = 0.75
    show_legend: bool = True
    legend_bbox: tuple[float, float] = (1.05, 1.0)


class ClassificationDataProcessor(BaseDataProcessor):
    """Processes data for classification plots."""

    def __init__(self, df: pd.DataFrame, class_col: str = "Class"):
        """Initialize with dynamic class column."""
        self.class_col = class_col
        super().__init__(df)

    def validate_dataframe(self) -> None:
        """Validate required columns exist."""
        if self.df.empty:
            raise ValueError("Input dataframe is empty")

        required_cols = [
            "plate_id",
            "cell_line",
            "well_id",
            "experiment",
            self.class_col,
        ]
        missing_cols = [
            col for col in required_cols if col not in self.df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def quantify_classification(
        self, df: pd.DataFrame, condition_col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Quantify classification results for each condition and return summary statistics.

        Args:
            df: DataFrame containing classification data
            condition_col: Column name for experimental condition

        Returns:
            Tuple of (mean_percentages, std_percentages) DataFrames
        """
        # Count classifications per well
        df_class = (
            df.groupby(
                [
                    "plate_id",
                    "cell_line",
                    "well_id",
                    condition_col,
                    self.class_col,
                ]
            )["experiment"]
            .count()
            .reset_index()
            .rename(columns={"experiment": "class count"})
        )

        # Calculate percentages per well
        df_class["percentage"] = (
            df_class["class count"]
            / df_class.groupby(["plate_id", "cell_line", "well_id"])[
                "class count"
            ].transform("sum")
            * 100
        )

        # Calculate mean percentages per plate
        df_class_mean = (
            df_class.groupby([condition_col, self.class_col])["percentage"]
            .mean()
            .reset_index()
        )

        # Calculate standard deviations
        if len(df["plate_id"].unique()) > 1:
            df_class_std = (
                df_class.groupby([condition_col, self.class_col])["percentage"]
                .std()
                .reset_index()
            )
        else:
            # Single plate: set std to 0
            df_class_std = df_class_mean.copy()
            df_class_std["percentage"] = 0

        return df_class_mean, df_class_std

    def process_data(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process data for classification plotting.

        Returns:
            Tuple of (mean_data, std_data) ready for plotting
        """
        # Extract required parameters from kwargs
        condition_col = kwargs.get("condition_col")
        conditions = kwargs.get("conditions")
        classes = kwargs.get("classes")

        if not condition_col or not conditions or not classes:
            raise ValueError(
                "process_data requires condition_col, conditions, and classes parameters"
            )
        # Get mean and std data
        df_mean, df_std = self.quantify_classification(df, condition_col)

        # Set categorical dtype to enforce order
        df_mean[condition_col] = pd.Categorical(
            df_mean[condition_col], categories=conditions, ordered=True
        )
        df_std[condition_col] = pd.Categorical(
            df_std[condition_col], categories=conditions, ordered=True
        )

        # Pivot to get classes as columns
        plot_data = df_mean.pivot_table(
            index=condition_col,
            columns=self.class_col,
            values="percentage",
            observed=False,
        ).reset_index()

        std_data = df_std.pivot_table(
            index=condition_col,
            columns=self.class_col,
            values="percentage",
            observed=False,
        ).reset_index()

        return plot_data, std_data


class ClassificationPlotBuilder(BasePlotBuilder):
    """Builds classification plots."""

    def __init__(self, config: ClassificationPlotConfig):
        """Initialize the ClassificationPlotBuilder.

        Parameters
        ----------
        config : ClassificationPlotConfig
            Configuration object containing plot settings.
        """
        super().__init__(config)
        self.config: ClassificationPlotConfig = config

    def build_stacked_plot(
        self,
        plot_data: pd.DataFrame,
        std_data: pd.DataFrame,
        conditions: list[str],
        classes: list[str],
        condition_col: str,
    ) -> "ClassificationPlotBuilder":
        """Build stacked bar plot with error bars."""
        assert self.ax is not None

        # Use colors from config if provided, otherwise default colors
        colors = (
            self.config.colors[: len(classes)] if self.config.colors else None
        )

        # Prepare error bar data
        yerr = std_data[classes].values.T

        # Create stacked bar plot
        plot_data.plot(
            x=condition_col,
            y=classes,
            kind="bar",
            stacked=True,
            yerr=yerr,
            width=self.config.bar_width,
            legend=False,
            color=colors,
            ax=self.ax,
        )

        # Format x-axis
        self.ax.set_xticklabels(
            self.ax.get_xticklabels(), rotation=45, ha="right", fontsize=7
        )
        self.ax.set_xlabel("")
        self.ax.set_ylabel("% of total cells")
        self.ax.set_ylim(self.config.y_lim)

        # Add legend if requested
        if self.config.show_legend:
            self.ax.legend(
                fontsize=7,
                bbox_to_anchor=self.config.legend_bbox,
                loc="upper left",
            )
            self.ax.get_legend().set_title("")

        return self

    def build_triplicates_plot(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        classes: list[str],
        condition_col: str,
        class_col: str,
    ) -> "ClassificationPlotBuilder":
        """Build grouped stacked plot with individual triplicates."""
        assert self.ax is not None

        # Use colors from config if provided, otherwise use default scheme
        if self.config.colors:
            colors = self.config.colors[: len(classes)]
        else:
            # Default color scheme based on number of classes
            if len(classes) == 2:
                colors = [COLOR.LIGHT_GREEN.value, COLOR.OLIVE.value]
            elif len(classes) == 3:
                colors = [
                    COLOR.BLUE.value,
                    COLOR.LIGHT_BLUE.value,
                    COLOR.GREY.value,
                ]
            else:
                colors = [color.value for color in COLOR]

        # Get percentage data per plate_id, condition, class
        df_class = (
            df.groupby(["plate_id", condition_col, class_col])["experiment"]
            .count()
            .reset_index()
            .rename(columns={"experiment": "class count"})
        )
        df_class["percentage"] = (
            df_class["class count"]
            / df_class.groupby(["plate_id", condition_col])[
                "class count"
            ].transform("sum")
            * 100
        )

        # Get repeat IDs (up to 3)
        n_repeats = 3
        repeat_ids = sorted(df_class["plate_id"].unique())[:n_repeats]
        n_conditions = len(conditions)

        # Calculate x positions based on grouping
        if self.config.group_size > 1:
            x_base_positions = grouped_x_positions(
                n_conditions,
                group_size=self.config.group_size,
                within_group_spacing=self.config.within_group_spacing,
                between_group_gap=self.config.between_group_gap,
            )
        else:
            # No grouping: use simple sequential positions
            x_base_positions = list(range(n_conditions))

        bar_width = self.config.repeat_offset * 1.05

        # Plot bars: for each condition, for each repeat, stack classes
        for cond_idx, cond in enumerate(conditions):
            for rep_idx, plate_id in enumerate(repeat_ids):
                xpos = (
                    x_base_positions[cond_idx]
                    + (rep_idx - 1) * self.config.repeat_offset
                )
                plate_data = df_class[
                    (df_class[condition_col] == cond)
                    & (df_class["plate_id"] == plate_id)
                ]

                if not plate_data.empty:
                    bottoms = 0
                    for i, cls in enumerate(classes):
                        val = plate_data[plate_data[class_col] == cls][
                            "percentage"
                        ].sum()
                        self.ax.bar(
                            xpos,
                            val,
                            width=bar_width,
                            bottom=bottoms,
                            color=colors[i],
                            edgecolor="white",
                            linewidth=0.7,
                            label=cls
                            if cond_idx == 0 and rep_idx == 0
                            else None,
                        )
                        bottoms += val

        # Set x-axis
        self.ax.set_xticks(x_base_positions)
        self.ax.set_xticklabels(conditions, rotation=45, ha="right")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Percentage")
        self.ax.set_ylim(0, 100)

        # Draw triplicate boxes
        y_min = 0
        y_max_box = 100
        for cond_idx, _ in enumerate(conditions):
            trip_xs = [
                x_base_positions[cond_idx]
                + (rep_idx - 1) * self.config.repeat_offset
                for rep_idx in range(n_repeats)
            ]
            left = min(trip_xs) - bar_width / 2
            right = max(trip_xs) + bar_width / 2
            rect = Rectangle(
                (left, y_min),
                width=right - left,
                height=y_max_box - y_min,
                linewidth=0.5,
                edgecolor="black",
                facecolor="none",
                zorder=10,
            )
            self.ax.add_patch(rect)

        # Add legend for classes
        if self.config.show_legend:
            legend_handles = [
                Patch(
                    facecolor=colors[i % len(colors)],
                    edgecolor="white",
                    label=cls,
                )
                for i, cls in enumerate(classes)
            ]
            dummy_label = " " * 20
            legend_handles.append(
                Patch(facecolor="none", edgecolor="none", label=dummy_label)
            )
            self.ax.legend(
                handles=legend_handles,
                title="",
                bbox_to_anchor=self.config.legend_bbox,
                loc="upper left",
                frameon=False,
            )

        return self

    def build_plot(
        self,
        data: Any,
        **kwargs: Any,
    ) -> "ClassificationPlotBuilder":
        """Build the classification plot based on display mode."""
        # Extract required parameters from kwargs
        conditions = kwargs.get("conditions")
        classes = kwargs.get("classes")
        condition_col = kwargs.get("condition_col")
        class_col = kwargs.get("class_col")

        if not conditions or not classes or not condition_col or not class_col:
            raise ValueError(
                "build_plot requires conditions, classes, condition_col, and class_col parameters"
            )
        if self.config.display_mode == "stacked":
            # data should be tuple of (plot_data, std_data)
            if not isinstance(data, tuple):
                raise ValueError(
                    "Stacked mode requires tuple of (plot_data, std_data)"
                )
            plot_data, std_data = data
            self.build_stacked_plot(
                plot_data, std_data, conditions, classes, condition_col
            )
        elif self.config.display_mode == "triplicates":
            # data should be the original DataFrame
            if isinstance(data, tuple):
                raise ValueError(
                    "Triplicates mode requires original DataFrame"
                )
            self.build_triplicates_plot(
                data, conditions, classes, condition_col, class_col
            )
        else:
            raise ValueError(
                f"Unknown display mode: {self.config.display_mode}"
            )

        return self
