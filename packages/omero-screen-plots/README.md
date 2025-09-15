# omero-screen-plots

Plotting Functions for Omero-Screen Immuno-Fluorescence Data.

## Recent Updates

**v0.1.3+**: The package has been completely refactored with a modern API architecture:
- **New Combined Plot Functions**: `combplot_feature` and `combplot_cellcycle` provide comprehensive multi-panel visualizations
- **Simplified Architecture**: Single-class design for better performance and maintainability
- **Comprehensive Documentation**: Full API reference with examples at [hocheggerlab.github.io/omero-screen/](https://hocheggerlab.github.io/omero-screen/)
- **Legacy Cleanup**: Removed outdated plotting functions in favor of the new API

## Status

Version: ![version](https://img.shields.io/badge/version-0.2.0-blue)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

**📖 Complete documentation and examples available at**: [hocheggerlab.github.io/omero-screen/](https://hocheggerlab.github.io/omero-screen/)

**💡 Interactive examples**: Jupyter notebooks in [`examples/`](examples/) directory

### Combined Plots (New!)

The package now provides two powerful combined plotting functions for comprehensive data analysis:

#### combplot_feature
Multi-panel feature analysis with DNA content context. Creates a 3-row grid:
- **Top row**: DNA content histograms
- **Middle row**: DNA vs EdU scatter plots with cell cycle phases
- **Bottom row**: DNA vs custom feature scatter plots with threshold coloring

```python
from omero_screen_plots import combplot_feature

fig, axes = combplot_feature(
    df=df,
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    feature="intensity_mean_p21_nucleus",
    threshold=5000,
    selector_col="cell_line",
    selector_val="MCF10A",
    title="p21 Intensity Analysis",
    cell_number=3000,
    save=True,
    file_format="svg"
)
```

#### combplot_cellcycle
Comprehensive cell cycle analysis with integrated barplot. Creates a 2-row grid:
- **Top row**: DNA content histograms
- **Bottom row**: DNA vs EdU scatter plots
- **Right column**: Stacked cell cycle phase barplot

```python
from omero_screen_plots import combplot_cellcycle

fig, axes = combplot_cellcycle(
    df=df,
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell Cycle Distribution",
    cc_phases=True,
    show_error_bars=True,
    save=True
)
```

### Cell Cycle Analysis

#### cellcycle_plot
Quantitative analysis of cell cycle phases in a 2×2 subplot grid. Each phase (G1, S, G2/M, Polyploid) is shown separately with statistical analysis when ≥3 plates are present.

```python
from omero_screen_plots import cellcycle_plot

fig, axes = cellcycle_plot(
    df=df,
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell Cycle Analysis",
    save=True
)
```

#### cellcycle_stacked
Stacked barplot showing cell cycle phase proportions with flexible display modes (summary with error bars or individual triplicates).

```python
from omero_screen_plots import cellcycle_stacked

fig, ax = cellcycle_stacked(
    df=df,
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    show_error_bars=True,
    cc_phases=True,
    save=True
)
```

### Feature Analysis

#### feature_plot
Box/violin plots for comparing quantitative features across experimental conditions with optional scatter overlays and statistical significance testing.

```python
from omero_screen_plots import feature_plot

fig, ax = feature_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="p21 Intensity Analysis",
    violin=True,
    save=True
)
```

#### feature_norm_plot
Normalized feature plots with threshold-based analysis for identifying treatment effects.

```python
from omero_screen_plots import feature_norm_plot

fig, ax = feature_norm_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    norm_feature="area_cell",
    threshold=0.5,
    selector_val="MCF10A",
    save=True
)
```

### Count Analysis

#### count_plot
Provides normalized or absolute cell counts with flexible grouping and statistical analysis.

```python
from omero_screen_plots import count_plot, PlotType

# Normalized counts (default)
fig, ax = count_plot(
    df=df,
    norm_control="control",
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Normalized Cell Counts",
    save=True
)

# Absolute counts
fig, ax = count_plot(
    df=df,
    norm_control="control",
    conditions=['control', 'cond01', 'cond02', 'cond03'],
    plot_type=PlotType.ABSOLUTE,
    selector_val="MCF10A",
    save=True
)
```

### Individual Plot Types

#### histogram_plot
Flexible histogram visualization with support for log scaling, KDE overlays, and multiple conditions.

```python
from omero_screen_plots import histogram_plot

fig, ax = histogram_plot(
    df=df,
    feature="integrated_int_DAPI_norm",
    conditions=['control', 'treatment'],
    log_scale=True,
    kde_overlay=True,
    save=True
)
```

#### scatter_plot
Comprehensive scatter plots with cell cycle coloring, KDE overlays, and threshold analysis.

```python
from omero_screen_plots import scatter_plot

fig, ax = scatter_plot(
    df=df,
    conditions="control",
    x_feature="integrated_int_DAPI_norm",
    y_feature="intensity_mean_EdU_nucleus_norm",
    hue="cell_cycle",
    kde_overlay=True,
    save=True
)
```

#### classification_plot
Visualization of categorical classification results with stacked percentages or individual replicates.

```python
from omero_screen_plots import classification_plot

fig, ax = classification_plot(
    df=df,
    classes=["normal", "micronuclei", "collapsed"],
    conditions=['control', 'treatment1', 'treatment2'],
    class_col="classifier",
    display_mode="stacked",
    save=True
)
```

## Documentation & Examples

- **📚 Complete API Documentation**: [hocheggerlab.github.io/omero-screen/](https://hocheggerlab.github.io/omero-screen/)
- **💻 Interactive Examples**: [`examples/`](examples/) directory with Jupyter notebooks
- **🔬 Sample Data**: Example datasets for testing and learning

## Key Features

- **Publication-Ready Plots**: Consistent, high-quality figures with customizable styling
- **Statistical Analysis**: Built-in statistical testing and significance marking
- **Flexible Data Filtering**: Support for experimental conditions and cell line selection
- **Modern Architecture**: Clean, maintainable codebase with comprehensive error handling
- **Performance Optimized**: Efficient data sampling and memory management
- **Extensive Customization**: Colors, sizing, grouping, and output format options

## Installation

```bash
uv pip install omero-screen
```

## Quick Start

```python
import pandas as pd
from omero_screen_plots import combplot_cellcycle, feature_plot, count_plot

# Load your data
df = pd.read_csv("your_screening_data.csv")

# Create a comprehensive cell cycle analysis
fig, axes = combplot_cellcycle(
    df=df,
    conditions=['control', 'treatment1', 'treatment2'],
    selector_col="cell_line",
    selector_val="MCF10A"
)
```

## Requirements

- Python 3.12 or greater
- pandas, matplotlib, seaborn, scipy
- Optional: OMERO integration for direct server access

## Authors

Created by Helfrid Hochegger
Email: hh65@sussex.ac.uk

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the contributors of matplotlib, pandas, seaborn, and scipy!
