#!/usr/bin/env python3
"""Generate example plots for documentation."""

import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from omero_screen_plots import count_plot, feature_plot, feature_norm_plot, cellcycle_plot, cellcycle_stacked, histogram_plot, scatter_plot, classification_plot, save_fig, PlotType
from conf import get_example_data

# Setup paths
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('.'))

def main() -> None:
    try:
        # Create _static directory
        static_dir = Path('_static')
        static_dir.mkdir(exist_ok=True)

        print("Loading example data...")
        df: Optional[pd.DataFrame] = get_example_data()  # type: ignore[no-untyped-call]

        if df is None:
            print("Failed to load example data")
            return

        print("Generating plots...")

        quickstart_examples(df, static_dir)
        count_plot_examples(df, static_dir)
        feature_plot_examples(df, static_dir)
        feature_norm_plot_examples(df, static_dir)
        cellcycle_plot_examples(df, static_dir)
        cellcycle_stacked_examples(df, static_dir)
        classification_plot_examples(df, static_dir)
        histogram_plot_examples(df, static_dir)
        scatter_plot_examples(df, static_dir)

        print(f"✅ All plots generated in {static_dir.absolute()}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the package is installed and paths are correct")
    except Exception as e:
        print(f"❌ Error: {e}")



def quickstart_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate quickstart examples."""
    try:
    # 1. Count Plot
        print("  - Count plot")
        count_plot(
            df=df,
            norm_control="control",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col='cell_line',
            selector_val='MCF10A',
            fig_size=(6, 4),
            title="qs count plot",  # Use simple filename
            save=True,
            file_format="svg",
            path=static_dir
        )

        # 2. Feature Plot
        print("  - Feature plot")
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col='cell_line',
            selector_val='MCF10A',
            title="qs feature plot",  # Use simple filename
            fig_size=(6, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # 3. Cell Cycle Plot
        print("  - Cell cycle plot")

        cellcycle_stacked(
            df=df,
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col='cell_line',
            selector_val='MCF10A',
            show_error_bars=True,
            title="qs cellcycle plot",  # Use simple filename
            fig_size=(7, 4),
            size_units="cm",
            between_group_gap=0.4,
            save=True,
            file_format="svg",
            path=static_dir,
        )
        print(f"✅ All quickstart plots generated in {static_dir.absolute()}")

    except Exception as e:
        print(f"❌ Error: {e}")


def count_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate count plot examples."""
    try:
        print("  - count plot basic")
        # Basic normalized count plot
        fig, ax = count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "cond01", "cond02", "cond03"],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="count plot basic",
            fig_size=(6, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        print("  - count plot absolute")
        # Basic normalized count plot
        fig, ax = count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "cond01", "cond02", "cond03"],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            plot_type=PlotType.ABSOLUTE,
            title="count plot absolute",
            fig_size=(6, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Grouped layout for better visual organization
        print("  - count plot grouped")
        fig, ax = count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "cond01", "cond02", "cond03"],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            group_size=3,  # Group conditions in sets of 3
            within_group_spacing=0.2,
            between_group_gap=0.8,
            fig_size=(6, 4),
            title="count plot grouped",
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Count plot with axes
        print("  - count plot with axes")
        fig, axes = plt.subplots(2, 1, figsize=(2, 4))

        # Plot for cell line 1
        count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "cond01", "cond02", "cond03"],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            axes=axes[0],
            plot_type=PlotType.NORMALISED,
            title="MCF10A Norm Counts",
            x_label=False
        )
        axes[0].set_title("MCF10A Norm Counts", fontsize=8, fontweight="bold", loc="left", y=1.05)

        # Plot for cell line 2
        count_plot(
            df=df,
            norm_control="control",
            conditions=["control", "cond01", "cond02", "cond03"],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            plot_type=PlotType.ABSOLUTE,
            axes=axes[1],
            title="MCF10A Abs. Counts"
        )
        axes[1].set_title("MCF10A Abs. Counts", fontsize=8, fontweight="bold", loc="left", y=1.05)
        suptitle = fig.suptitle("count plot with axes", fontsize=12, fontweight="bold")
        suptitle.set_y(1.02)
        plt.tight_layout()
        save_fig(fig, static_dir, "count_plot_with_axes", fig_extension="svg")

        print(f"✅ All count plot examples generated in {static_dir.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def feature_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate feature plot examples."""
    try:
        # Basic feature plot with boxplots
        print("  - feature plot basic")
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="feature plot basic",
            fig_size=(5, 5),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature plot with grouped layout
        print("  - feature plot grouped")
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="feature plot grouped",
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.5,
            fig_size=(6, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature plot with violin plots
        print("  - feature plot violin")
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="feature plot violin",
            violin=True,
            ymax=20000,
            fig_size=(5, 5),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature plot without scatter points
        print("  - feature plot no scatter")
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="feature plot no scatter",
            violin=True,
            show_scatter=False,
            ymax=(2000, 12000),
            group_size=2,
            fig_size=(5, 5),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Combined feature plots on subplot
        print("  - feature plot comparison")
        fig, axes = plt.subplots(2, 1, figsize=(2, 4))
        fig.suptitle("feature plot comparison", fontsize=8, weight="bold", x=0.2, y=1)

        # First subplot - p21 intensity
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            axes=axes[0],
            group_size=2,
            x_label=False,
        )
        axes[0].set_title("mean nuc. p21 intensity", fontsize=7, y=1.05, x=0, weight="bold")

        # Second subplot - cell area
        feature_plot(
            df=df,
            feature="area_cell",
            conditions=['control', 'cond01', 'cond02', 'cond03'],
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            violin=True,
            show_scatter=False,
            ymax=10000,
            axes=axes[1],
            group_size=2,
        )
        axes[1].set_title("area cell", fontsize=7, y=1.05, x=0, weight="bold")

        save_fig(fig, static_dir, "feature_plot_comparison", tight_layout=False, fig_extension="svg", resolution=300)

        print(f"✅ All feature plot examples generated in {static_dir.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")


def feature_norm_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate feature norm plot examples."""
    try:
        conditions = ['control', 'cond01', 'cond02', 'cond03']

        # Basic feature norm plot (default green scheme)
        print("  - feature norm plot basic")
        feature_norm_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            color_scheme="green",
            title="feature norm plot basic",
            show_error_bars=True,
            fig_size=(5, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature norm plot with triplicates (blue scheme)
        print("  - feature norm plot triplicates")
        feature_norm_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            color_scheme="blue",
            title="feature norm plot triplicates",
            show_triplicates=True,
            show_boxes=True,
            threshold=1.5,
            group_size=1,
            fig_size=(5, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature norm plot with grouped layout (purple scheme)
        print("  - feature norm plot grouped")
        feature_norm_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            color_scheme="purple",
            title="feature norm plot grouped",
            show_triplicates=True,
            show_boxes=True,
            threshold=1.5,
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.4,
            fig_size=(5, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Feature norm plot with different threshold
        print("  - feature norm plot threshold")
        feature_norm_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            color_scheme="green",
            title="feature norm plot threshold",
            threshold=2.0,  # Higher threshold
            show_triplicates=True,
            show_boxes=True,
            fig_size=(5, 4),
            save=True,
            file_format="svg",
            path=static_dir
        )

        # Combined feature plots comparison (like in the notebook)
        print("  - feature norm plot comparison")
        fig, ax = plt.subplots(3, 1, figsize=(2, 6))

        # Standard feature plot
        feature_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            x_label=False,
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.4,
            axes=ax[0],
        )
        ax[0].set_title("p21 feature plot", fontsize=7, y=1.05, x=0, weight="bold")

        # Feature norm plot
        feature_norm_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            x_label=False,
            axes=ax[1],
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.4,
            show_triplicates=True,
            show_boxes=True,
            threshold=1.5,
        )
        ax[1].set_title("p21 feature norm plot", fontsize=7, y=1.05, x=0, weight="bold")

        # Violin plot for comparison
        feature_plot(
            df=df,
            feature="area_cell",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            x_label=True,
            violin=True,
            show_scatter=False,
            ymax=10000,
            axes=ax[2],
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.4,
        )
        ax[2].set_title("area cell violin", fontsize=7, y=1.05, x=0, weight="bold")

        fig.suptitle("feature norm plot comparison", fontsize=8, weight="bold", x=0.2)
        save_fig(fig, static_dir, "feature_norm_plot_comparison", fig_extension="svg", resolution=300)

        print(f"✅ All feature norm plot examples generated in {static_dir.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")


def cellcycle_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate cellcycle plot examples."""
    try:
        conditions = ['control', 'cond01', 'cond02', 'cond03']

        # Basic cellcycle plot (default - show all phases including Sub-G1)
        print("  - cellcycle plot default")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot default",
            show_subG1=True,  # Show all phases including Sub-G1
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # DNA content terminology
        print("  - cellcycle plot DNA content")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot DNA content",
            cc_phases=False,  # Use DNA content terminology
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # No Sub-G1 phase (2x2 layout)
        print("  - cellcycle plot no SubG1")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot no SubG1",
            show_subG1=False,  # Hide Sub-G1 phase
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # With plate legend
        print("  - cellcycle plot with legend")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot with legend",
            show_repeat_points=True,
            show_plate_legend=True,  # Show plate shapes legend
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # No statistics (clean look)
        print("  - cellcycle plot clean")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot clean",
            show_repeat_points=False,  # Hide repeat points
            show_significance=False,   # Hide significance marks
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # Custom colors
        print("  - cellcycle plot custom colors")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot custom colors",
            colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            fig_size=(8, 6),
            rotation=0,  # No rotation for x-labels
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        # Combined features
        print("  - cellcycle plot combined")
        fig, axes = cellcycle_plot(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            title="cellcycle plot combined",
            cc_phases=False,         # DNA terminology
            show_subG1=False,       # No Sub-G1
            show_plate_legend=True, # Show legend
            fig_size=(6, 5),
            save=True,
            path=static_dir,
            file_format="svg",
            dpi=300,
        )

        print(f"✅ All cellcycle plot examples generated in {static_dir.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")


def cellcycle_stacked_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate cellcycle stacked plot examples."""
    try:
        conditions = ['control', 'cond01', 'cond02', 'cond03']

        # Basic stacked plot (summary with error bars)
        print("  - cellcycle stacked default")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="cellcycle stacked default",
            fig_size=(7, 5),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # Without error bars
        print("  - cellcycle stacked no errorbars")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="cellcycle stacked no errorbars",
            show_error_bars=False,
            fig_size=(7, 5),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # Triplicates with boxes
        print("  - cellcycle stacked triplicates")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="cellcycle stacked triplicates",
            show_triplicates=True,
            group_size=2,
            fig_size=(7, 6),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # DNA content terminology
        print("  - cellcycle stacked DNA content")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=["control", "cond01", "cond02"],
            selector_val="MCF10A",
            cc_phases=False,  # Use DNA content terminology
            title="cellcycle stacked DNA content",
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Custom phase selection
        print("  - cellcycle stacked custom phases")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=["control", "cond01", "cond02"],
            selector_val="MCF10A",
            phase_order=["G1", "S", "G2/M"],  # Exclude Sub-G1 and Polyploid
            title="cellcycle stacked custom phases",
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Without legend
        print("  - cellcycle stacked no legend")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=["control", "cond01"],
            selector_val="MCF10A",
            show_legend=False,  # Remove legend
            title="cellcycle stacked no legend",
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Triplicates without boxes
        print("  - cellcycle stacked triplicates no boxes")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=["control", "cond01"],
            selector_val="MCF10A",
            show_triplicates=True,  # Show individual bars
            show_boxes=False,       # But don't draw boxes
            title="cellcycle stacked triplicates no boxes",
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Custom colors
        print("  - cellcycle stacked custom colors")
        custom_colors = [
            "#FFB6C1",  # Light pink for Sub-G1
            "#87CEEB",  # Sky blue for G1
            "#98FB98",  # Pale green for S
            "#F0E68C",  # Khaki for G2/M
            "#DDA0DD",  # Plum for Polyploid
        ]

        fig, ax = cellcycle_stacked(
            df=df,
            conditions=["control", "cond01", "cond02"],
            selector_val="MCF10A",
            colors=custom_colors,
            title="cellcycle stacked custom colors",
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Advanced grouping with triplicates
        print("  - cellcycle stacked advanced grouping")
        fig, ax = cellcycle_stacked(
            df=df,
            conditions=conditions,
            selector_val="MCF10A",
            show_triplicates=True,     # Individual bars per plate
            show_boxes=True,          # Boxes around triplicates
            group_size=2,             # Group in pairs
            within_group_spacing=0.01, # Space between bars within group
            between_group_gap=0.1,    # Gap between groups
            repeat_offset=0.15,       # Closer spacing for triplicates
            title="cellcycle stacked advanced grouping",
            fig_size=(8, 6),         # Wider figure for complex layout
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Combined with feature plot
        print("  - cellcycle stacked combined")
        fig, ax = plt.subplots(nrows=2, figsize=(2, 5))

        feature_norm_plot(
            df=df,
            feature="area_cell",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title="feature norm plot",
            show_triplicates=True,
            group_size=2,
            within_group_spacing=0.1,
            between_group_gap=0.2,
            x_label=False,
            axes=ax[0],
        )
        ax[0].set_title("feature norm plot", fontsize=8, y=1.05, x=0, weight="bold")

        cellcycle_stacked(
            df=df,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            show_triplicates=True,
            group_size=2,
            within_group_spacing=0.1,  # Match feature plot spacing
            between_group_gap=0.2,      # Match feature plot spacing
            axes=ax[1],
        )
        ax[1].set_title("cellcycle stacked plot", fontsize=8, y=1.05, x=0, weight="bold")

        fig.suptitle("cellcycle stacked combined", fontsize=8, weight="bold", x=0.2)
        save_fig(fig, static_dir, "cellcycle_stacked_combined", fig_extension="svg", resolution=300)

        print(f"✅ All cellcycle stacked examples generated in {static_dir.absolute()}")
    except Exception as e:
        print(f"❌ Error: {e}")


def histogram_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate histogram plot examples."""
    try:
        conditions = ["control", "cond01", "cond02", "cond03"]

        # Basic histogram
        print("  - histogram basic")
        fig, ax = histogram_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions="control",
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            bins=100,
            title="histogram plot basic",
            fig_size=(6, 4),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # Multiple conditions
        print("  - histogram multiple conditions")
        fig, axes = histogram_plot(
            df=df,
            feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            bins=50,
            title="histogram plot multiple",
            fig_size=(16, 4),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # DNA content with log scale
        print("  - histogram DNA content")
        fig, axes = histogram_plot(
            df=df,
            feature="integrated_int_DAPI_norm",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            bins=100,
            log_scale=True,
            log_base=2,
            x_limits=(1, 16),
            title="histogram plot DNA content",
            fig_size=(16, 4),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # KDE overlay
        print("  - histogram KDE overlay")
        fig, ax = histogram_plot(
            df=df,
            feature="integrated_int_DAPI_norm",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            kde_overlay=True,
            kde_smoothing=0.8,
            log_scale=True,
            log_base=2,
            x_limits=(1, 16),
            title="histogram plot KDE overlay",
            fig_size=(8, 5),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        # Normalized histogram
        print("  - histogram normalized")
        fig, axes = histogram_plot(
            df=df,
            feature="area_cell",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            bins=50,
            normalize=True,
            title="histogram plot normalized",
            fig_size=(16, 4),
            save=True,
            path=static_dir,
            file_format="svg",
        )

        print("  ✅ Histogram plot examples generated")

    except Exception as e:
        print(f"  ❌ Error generating histogram examples: {e}")


def scatter_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate scatter plot examples."""
    try:
        conditions = ['control', 'cond01', 'cond02', 'cond03']

        print("  - scatter plot basic")
        # Basic cell cycle scatter plot (DNA vs EdU with auto-detection)
        fig, ax = scatter_plot(
            df=df,
            conditions="control",
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=3000,
            title="scatter_plot_basic",
            show_title=False,
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - scatter plot multiple conditions")
        # Multiple conditions comparison
        fig, axes = scatter_plot(
            df=df,
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=2000,
            title="scatter_plot_multiple",
            show_title=False,
            fig_size=(16, 4),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - scatter plot threshold analysis")
        # Threshold-based coloring (blue below, red above)
        fig, axes = scatter_plot(
            df=df,
            y_feature="intensity_mean_p21_nucleus",
            conditions=conditions,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=3000,
            threshold=5000,  # Threshold for blue/red coloring
            y_limits=(1000, 12000),
            title="scatter_plot_threshold",
            show_title=False,
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - scatter plot custom scales")
        # Custom scales with reference lines
        fig, ax = scatter_plot(
            df=df,
            x_feature="area_cell",
            y_feature="intensity_mean_p21_nucleus",
            conditions="control",
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=5000,
            x_scale="log",  # Force log scale
            y_scale="log",
            x_limits=(100, 10000),
            y_limits=(1000, 15000),
            vline=2000,  # Custom reference lines
            hline=5000,
            kde_overlay=True,
            title="scatter_plot_custom",
            show_title=False,
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - scatter plot KDE overlay")
        # KDE overlay customization
        fig, ax = scatter_plot(
            df=df,
            conditions="control",
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=4000,
            kde_overlay=True,
            kde_alpha=0.2,
            kde_cmap="viridis",
            title="scatter_plot_kde",
            show_title=False,
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - scatter plot cell cycle phases")
        # Manual cell cycle phase control
        fig, ax = scatter_plot(
            df=df,
            conditions="control",
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            cell_number=4000,
            hue="cell_cycle",
            hue_order=["Sub-G1", "G1", "S", "G2/M", "Polyploid"],
            show_legend=True,
            legend_title="Cell Cycle Phase",
            title="scatter_plot_phases",
            show_title=False,
            fig_size=(6, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  ✅ Scatter plot examples generated")

    except Exception as e:
        print(f"  ❌ Error generating scatter examples: {e}")


def classification_plot_examples(df: pd.DataFrame, static_dir: Path) -> None:
    """Generate classification plot examples."""
    try:
        conditions = ['control', 'cond01', 'cond02', 'cond03']
        classes = ["normal", "micronuclei", "collapsed"]

        print("  - classification plot stacked")
        # Basic stacked classification plot
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            title="Cell Morphology Classification - Stacked",
            fig_size=(8, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - classification plot triplicates")
        # Individual triplicates
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="triplicates",
            group_size=1,
            title="Cell Morphology Classification - Triplicates",
            fig_size=(10, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - classification plot grouped")
        # Grouped triplicates
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="triplicates",
            group_size=2,
            within_group_spacing=0.2,
            between_group_gap=0.4,
            title="classification_grouped",
            fig_size=(12, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - classification plot custom colors")
        # Custom colors
        from omero_screen_plots.colors import COLOR
        custom_colors = [COLOR.GREY.value, COLOR.LIGHT_GREEN.value, COLOR.OLIVE.value]

        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=['control', 'cond01', 'cond02'],
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            colors=custom_colors,
            title="classification_colors",
            fig_size=(8, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("  - classification plot comparison")
        # Display mode comparison (multi-panel)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Stacked
        classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            colors=custom_colors,
            show_legend=False,
            axes=axes[0]
        )
        axes[0].set_title("Stacked Mode", fontweight='bold')

        # Individual triplicates
        classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="triplicates",
            group_size=1,
            colors=custom_colors,
            show_legend=False,
            axes=axes[1]
        )
        axes[1].set_title("Individual Triplicates", fontweight='bold')

        # Grouped triplicates
        classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="triplicates",
            group_size=2,
            colors=custom_colors,
            axes=axes[2]
        )
        axes[2].set_title("Grouped Triplicates", fontweight='bold')

        fig.suptitle("Classification Plot Display Modes", fontsize=14, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, static_dir, "classification_comparison", fig_extension="svg")
        plt.close(fig)

        # Example 5: Dynamic classification column (cell cycle example)
        print("  - classification plot cell cycle")
        # Check if we have cell cycle data, if not create a mock example
        if 'cell_cycle' in df.columns:
            cell_cycle_classes = ['G1', 'S', 'G2/M']
            class_col = 'cell_cycle'
        else:
            # Use classifier with different classes for demo
            cell_cycle_classes = ["normal", "micronuclei", "collapsed"]
            class_col = "classifier"

        fig, ax = classification_plot(
            df=df,
            classes=cell_cycle_classes,
            conditions=conditions,
            condition_col="condition",
            class_col=class_col,
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            colors=[COLOR.LIGHT_BLUE.value, COLOR.BLUE.value, COLOR.GREY.value],
            title="Dynamic Classification Column",
            fig_size=(8, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Example 6: Bar width customization
        print("  - classification plot bar width")
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=['control', 'cond01'],  # Fewer conditions
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            bar_width=0.5,  # Narrow bars
            title="Bar Width Customization",
            fig_size=(6, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Example 7: Y-axis customization
        print("  - classification plot y-axis")
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            y_lim=(70, 100),  # Focused view on 70-100%
            title="Y-axis Range Customization",
            fig_size=(8, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        # Example 8: Legend customization
        print("  - classification plot legend")
        fig, ax = classification_plot(
            df=df,
            classes=classes,
            conditions=conditions,
            condition_col="condition",
            class_col="classifier",
            selector_col="cell_line",
            selector_val="MCF10A",
            display_mode="stacked",
            show_legend=True,
            legend_bbox=(0.5, 1.15),  # Top center position
            title="Legend Position Customization",
            fig_size=(8, 6),
            save=True,
            path=static_dir,
            file_format="svg"
        )

        print("✅ All classification plot examples generated in", static_dir)

    except Exception as e:
        print(f"  ❌ Error generating classification examples: {e}")


if __name__ == "__main__":
    main()
