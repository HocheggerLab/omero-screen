"""Classification plot factory following the base class architecture."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.patches import Patch

from omero_screen_plots.base import (
    BasePlotBuilder,
    BasePlotConfig,
    PlotRequest,
)
from omero_screen_plots.colors import COLOR
from omero_screen_plots.stats import (
    annotate_significance,
    compute_significance,
)
from omero_screen_plots.utils import (
    draw_triplicate_box,
    grouped_x_positions,
)


@dataclass
class ClassificationPlotConfig(BasePlotConfig):
    """Configuration for classification plots."""

    # Show triplicates: False for stacked bars with error bars, True for individual repeats
    show_triplicates: bool = False
    y_lim: tuple[int, int] = (0, 100)

    # Triplicate mode settings
    repeat_offset: float = 0.18
    group_size: int = 2
    within_group_spacing: float = 0.6
    between_group_gap: float = 0.7
    show_boxes: bool = True

    # Significance settings
    show_significance: bool = True
    # Class whose condition-vs-reference test is annotated on the plot
    # (default = first class in `classes`); both CSVs cover every class.
    stats_class: str | None = None

    # Stacked mode settings
    bar_width: float = 0.75
    show_legend: bool = True
    legend_bbox: tuple[float, float] = (0.95, 1)


class ClassificationDataProcessor:
    """Processes data for classification plots.

    Standalone helper (previously subclassed the retired ``BaseDataProcessor``);
    it owns the condition/selector filtering the classification API relies on.
    """

    def __init__(self, df: pd.DataFrame, class_col: str = "Class"):
        """Initialize with dynamic class column."""
        self.df = df
        self.class_col = class_col
        self.validate_dataframe()

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

    def filter_data(
        self,
        condition_col: str,
        conditions: list[str],
        selector_col: str | None = None,
        selector_val: str | None = None,
    ) -> pd.DataFrame:
        """Filter by conditions and an optional selector column/value."""
        if condition_col not in self.df.columns:
            raise ValueError(
                f"Column '{condition_col}' not found in dataframe"
            )

        filtered = self.df[self.df[condition_col].isin(conditions)].copy()

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

        # Triplicate bars are drawn manually, so always use grouped_x_positions
        # (group_size==1 => between_group_gap controls the gap between every
        # condition's triplicate cluster, consistent with the other plots).
        x_base_positions = grouped_x_positions(
            n_conditions,
            group_size=self.config.group_size,
            within_group_spacing=self.config.within_group_spacing,
            between_group_gap=self.config.between_group_gap,
        )

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
        self.ax.set_ylim(self.config.y_lim)

        # Draw triplicate boxes (hugging the actual stacked bar tops rather
        # than a fixed 100% — otherwise a box around a subset of classes that
        # does not sum to 100% leaves a large empty rectangle).
        if self.config.show_boxes:
            y_min = 0
            for cond_idx, cond in enumerate(conditions):
                trip_xs = [
                    x_base_positions[cond_idx]
                    + (rep_idx - 1) * self.config.repeat_offset
                    for rep_idx in range(n_repeats)
                ]
                # Tallest stacked total across this condition's repeats,
                # restricted to the classes actually plotted (df_class holds
                # every class, which would always sum to 100%).
                cond_data = df_class[
                    (df_class[condition_col] == cond)
                    & (df_class[class_col].isin(classes))
                ]
                y_max_box = max(
                    (
                        cond_data[cond_data["plate_id"] == plate_id][
                            "percentage"
                        ].sum()
                        for plate_id in repeat_ids
                    ),
                    default=0,
                )
                if y_max_box <= y_min:
                    continue
                draw_triplicate_box(
                    self.ax,
                    trip_xs,
                    y_max_box,
                    half_width=bar_width / 2,
                    y_min=y_min,
                )

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
            self.ax.legend(
                handles=legend_handles,
                title="",
                bbox_to_anchor=self.config.legend_bbox,
                loc="upper left",
            )

        return self

    def build_plot(
        self,
        data: Any,
        request: PlotRequest,
    ) -> "ClassificationPlotBuilder":
        """Build the classification plot based on display mode."""
        conditions = request.conditions
        classes = request.classes
        condition_col = request.condition_col
        class_col = request.class_col

        if not conditions or not classes or not condition_col or not class_col:
            raise ValueError(
                "build_plot requires conditions, classes, condition_col, and class_col parameters"
            )
        if self.config.show_triplicates:
            # data should be the original DataFrame
            if isinstance(data, tuple):
                raise ValueError(
                    "show_triplicates=True requires original DataFrame"
                )
            self.build_triplicates_plot(
                data, conditions, classes, condition_col, class_col
            )
        else:
            # data should be tuple of (plot_data, std_data)
            if not isinstance(data, tuple):
                raise ValueError(
                    "show_triplicates=False requires tuple of (plot_data, std_data)"
                )
            plot_data, std_data = data
            self.build_stacked_plot(
                plot_data, std_data, conditions, classes, condition_col
            )

        return self

    def add_significance(
        self,
        df: pd.DataFrame,
        conditions: list[str],
        classes: list[str],
        condition_col: str,
        class_col: str,
    ) -> "ClassificationPlotBuilder":
        """Add per-class significance: condition vs first-in-group, paired by plate.

        Computes the per-plate percentage of each class and runs the
        replicate-level t-test for **every** class (recorded for CSV export).
        On-plot stars are only drawn when a **single** class is plotted — with a
        stacked multi-class bar it is ambiguous which class a star refers to,
        so the markers are omitted (the CSV still has all comparisons).
        """
        assert self.ax is not None
        if not self.config.show_significance:
            return self
        if df["plate_id"].nunique() < 3:
            return self

        # Per-plate class percentages (one row per plate/condition/class).
        counts = (
            df.groupby(["plate_id", condition_col, class_col])
            .size()
            .reset_index(name="n")
        )
        counts["percentage"] = (
            counts["n"]
            / counts.groupby(["plate_id", condition_col])["n"].transform("sum")
            * 100
        )

        # On-plot stars are ambiguous when several stacked classes are marked
        # at once, so multi-class bars stay unannotated by default. Naming a
        # ``stats_class`` opts in to a single row of markers for that one
        # class; every class is still exported to CSV.
        annotate_on_plot = (
            len(classes) == 1 or self.config.stats_class is not None
        )
        focus_class = self.config.stats_class or classes[0]
        group_size = self.config.group_size
        n = len(conditions)
        # Match the bar layout: triplicate bars are drawn at grouped positions
        # (gap-aware); the stacked single-bar layout is categorical.
        if self.config.show_triplicates:
            x_positions = grouped_x_positions(
                n,
                group_size=group_size,
                within_group_spacing=self.config.within_group_spacing,
                between_group_gap=self.config.between_group_gap,
            )
        else:
            x_positions = [float(i) for i in range(n)]
        y = self.ax.get_ylim()[1] * 0.95

        for cls in classes:
            cls_df = counts[counts[class_col] == cls]
            medians_df, results = compute_significance(
                cls_df,
                conditions,
                condition_col,
                "percentage",
                group_size=group_size,
                paired=self.config.paired,
            )
            self._record_stats(
                medians_df,
                results,
                value_label=str(cls),
                condition_col=condition_col,
                value_col="percentage",
            )
            if annotate_on_plot and cls == focus_class:
                annotate_significance(
                    self.ax,
                    results,
                    x_positions,
                    y,
                    show_ns=self.config.show_ns,
                    min_effect=self.config.min_effect,
                )

        return self
