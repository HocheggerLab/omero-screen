#!/usr/bin/env python3
"""Generate example plots for documentation."""

import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from omero_screen_plots import count_plot, feature_plot, cellcycle_stacked, save_fig, PlotType
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
            y_err=True,
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


if __name__ == "__main__":
    main()
