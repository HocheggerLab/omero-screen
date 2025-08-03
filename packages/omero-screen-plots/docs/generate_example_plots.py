#!/usr/bin/env python3
"""Generate example plots for documentation."""

import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd

# Setup paths
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('.'))

def main() -> None:
    try:
        # Import functions
        from conf import get_example_data
        from omero_screen_plots import count_plot, feature_plot, cellcycle_stacked

        # Create _static directory
        static_dir = Path('_static')
        static_dir.mkdir(exist_ok=True)

        print("Loading example data...")
        df: Optional[pd.DataFrame] = get_example_data(subset='feature', n_samples=1000)  # type: ignore[no-untyped-call]

        if df is None:
            print("Failed to load example data")
            return

        print("Generating plots...")

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


        print(f"✅ All plots generated in {static_dir.absolute()}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the package is installed and paths are correct")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
