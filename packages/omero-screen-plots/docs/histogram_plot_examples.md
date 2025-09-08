# Histogram Plot Usage Examples

This guide provides comprehensive examples for using the `histogram_plot` function for distribution analysis in high-content screening data.

## Basic Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omero_screen_plots import histogram_plot
from omero_screen_plots.colors import COLOR

# Load example data
df = pd.read_csv("sample_plate_data.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Define conditions for analysis
conditions = ['control', 'cond01', 'cond02', 'cond03']
```

## Example 1: Basic Single Condition Histogram

The simplest use case - visualizing the distribution of a feature for one condition.

```python
# Basic histogram for single condition
fig, ax = histogram_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions="control",  # Single condition as string
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    bins=100,  # Default: 100 bins for good resolution
    title="p21 Expression Distribution - Control",
    show_title=True,
    save=True,
    path=output_dir,
    file_format="pdf"
)

plt.show()
```

**Use Case**: Analyzing the distribution of a specific marker in control cells to understand baseline heterogeneity.

## Example 2: Multiple Conditions with Subplots

Compare distributions across multiple conditions using automatic subplot layout.

```python
# Multiple conditions create separate subplots
fig, axes = histogram_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,  # List of conditions
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    bins=50,  # Fewer bins for clearer visualization
    title="p21 Expression Across Conditions",
    show_title=True,
    fig_size=(16, 4),  # Automatically sized: 4cm per condition
    save=True,
    path=output_dir
)

print(f"Created {len(axes)} subplots for comparison")
```

**Use Case**: Comparing how different treatments affect the distribution of a biomarker.

## Example 3: DNA Content Analysis with Log Scale

Essential for cell cycle analysis - using log2 scale for DNA content visualization.

```python
# DNA content histogram with log2 scale
fig, axes = histogram_plot(
    df=df,
    feature="integrated_int_DAPI_norm",
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    bins=100,
    log_scale=True,
    log_base=2,  # Base 2 for DNA content (2N, 4N, 8N)
    x_limits=(1, 16),  # Typical DNA content range
    title="DNA Content Distribution",
    show_title=True,
    save=True,
    path=output_dir
)
```

**Use Case**: Analyzing cell cycle distribution where DNA content doubles during S-phase (2N → 4N).

## Example 4: KDE Overlay for Distribution Comparison

Compare multiple conditions on a single plot using smooth density curves.

```python
# KDE overlay mode: single plot with overlaid curves
fig, ax = histogram_plot(
    df=df,
    feature="integrated_int_DAPI_norm",
    conditions=conditions,  # All conditions overlaid
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    kde_overlay=True,  # Enables KDE-only mode
    kde_smoothing=0.8,  # Smoothing factor (0.5-2.0)
    log_scale=True,
    log_base=2,
    x_limits=(1, 16),
    title="DNA Content Comparison",
    show_title=True,
    fig_size=(8, 5),
    save=True,
    path=output_dir
)

# Note: Returns single Axes for KDE mode, not list
print(f"Created single plot with {len(conditions)} overlaid curves")
```

**Use Case**: Direct comparison of distributions to identify shifts in population dynamics.

## Example 5: Normalized Histograms for Sample Size Independence

Use density normalization to compare distributions with different sample sizes.

```python
# Normalized histograms show probability density
fig, axes = histogram_plot(
    df=df,
    feature="area_cell",
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    bins=50,
    normalize=True,  # Show density instead of counts
    title="Cell Area Distribution (Normalized)",
    show_title=True,
    save=True,
    path=output_dir
)
```

**Use Case**: Comparing distributions when conditions have different cell counts due to toxicity or proliferation effects.

## Example 6: Custom Binning Strategies

Different binning approaches for optimal visualization.

```python
# Compare different binning strategies
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
bin_strategies = [30, 100, 'auto', 'sturges']
titles = ['30 bins', '100 bins (default)', 'Auto', 'Sturges']

for ax, bins, title in zip(axes.flat, bin_strategies, titles):
    histogram_plot(
        df=df,
        feature="intensity_mean_p21_nucleus",
        conditions="control",
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        bins=bins,
        axes=ax
    )
    ax.set_title(title)

fig.suptitle("Binning Strategy Comparison", fontsize=12)
fig.tight_layout()
plt.savefig(output_dir / "binning_comparison.pdf")
```

**Use Case**: Finding the optimal bin size for your specific data distribution.

## Example 7: KDE Smoothing Optimization

Fine-tune KDE smoothing for optimal visualization.

```python
# Compare different smoothing levels
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
smoothing_values = [0.5, 0.8, 1.5]
titles = ["Very Smooth (0.5)", "Default (0.8)", "More Detail (1.5)"]

for ax, smooth, title in zip(axes, smoothing_values, titles):
    histogram_plot(
        df=df,
        feature="intensity_mean_p21_nucleus",
        conditions=['control', 'cond02'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        kde_overlay=True,
        kde_smoothing=smooth,
        axes=ax
    )
    ax.set_title(title)

fig.suptitle("KDE Smoothing Comparison", fontsize=12)
fig.tight_layout()
```

**Use Case**: Balancing between smooth curves and preserving distribution details.

## Example 8: Multi-Feature Analysis

Analyze multiple features in a single figure.

```python
# Create comprehensive figure with multiple features
fig = plt.figure(figsize=(12, 8))
features = {
    "integrated_int_DAPI_norm": "DNA Content",
    "intensity_mean_p21_nucleus": "p21 Expression",
    "area_cell": "Cell Area",
    "intensity_mean_EdU_nucleus_norm": "EdU Incorporation"
}

for i, (feature, label) in enumerate(features.items(), 1):
    # Histograms for individual conditions
    ax = plt.subplot(2, 4, i)
    histogram_plot(
        df=df,
        feature=feature,
        conditions="control",
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        bins=50,
        axes=ax
    )
    ax.set_title(f"{label} - Control", fontsize=10)

    # KDE overlay for comparison
    ax = plt.subplot(2, 4, i+4)
    histogram_plot(
        df=df,
        feature=feature,
        conditions=['control', 'cond02'],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        kde_overlay=True,
        axes=ax
    )
    ax.set_title(f"{label} - Comparison", fontsize=10)

fig.suptitle("Multi-Feature Distribution Analysis", fontsize=14)
fig.tight_layout()
plt.savefig(output_dir / "multi_feature_analysis.pdf")
```

**Use Case**: Comprehensive phenotypic profiling across multiple cellular features.

## Example 9: Custom Color Schemes

Apply custom colors for publication-specific requirements.

```python
# Define custom color palette
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Regular histograms (all use first color)
fig1, axes1 = histogram_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    colors=['#FF6B6B'],  # Single color for all histograms
    bins=50,
    title="Custom Color Histograms",
    show_title=True
)

# KDE overlay with custom colors (different color per condition)
fig2, ax2 = histogram_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    kde_overlay=True,
    colors=custom_colors,  # Different colors for each curve
    title="Custom Color KDE Overlay",
    show_title=True,
    fig_size=(8, 5)
)
```

**Use Case**: Matching journal or presentation color requirements.

## Example 10: Advanced KDE Parameters

Fine control over KDE generation for specific analysis needs.

```python
# Advanced KDE customization
fig, ax = histogram_plot(
    df=df,
    feature="intensity_mean_p21_nucleus",
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    kde_overlay=True,
    kde_smoothing=0.6,  # Extra smooth
    kde_params={
        'gridsize': 500,      # High resolution for smooth curves
        'bw_method': 'scott', # Bandwidth selection method
        'cut': 3,            # Extend KDE beyond data range
        'alpha': 0.9,        # Transparency
        'linewidth': 3       # Thicker lines
    },
    title="Advanced KDE Parameters",
    show_title=True,
    fig_size=(10, 5)
)
```

**Use Case**: Publication-quality density plots with specific visual requirements.

## Key Parameters Reference

### Basic Parameters
- `conditions`: Single string or list of strings for conditions to plot
- `feature`: Column name containing the data to plot
- `bins`: Number of bins (default: 100) or strategy ('auto', 'sturges', etc.)

### Scaling Options
- `log_scale`: Enable logarithmic x-axis
- `log_base`: Base for log scale (typically 2 for DNA content)
- `x_limits`: Tuple of (min, max) for x-axis range
- `normalize`: Show density instead of counts

### KDE Options
- `kde_overlay`: Enable KDE-only mode (single plot, no histograms)
- `kde_smoothing`: Smoothness factor (0.5-2.0, default: 0.8)
- `kde_params`: Dictionary of additional KDE parameters

### Display Options
- `show_title`: Whether to display title (default: False)
- `fig_size`: Figure size as (width, height) in cm or inches
- `colors`: List of colors for plots

## Tips for Effective Use

1. **DNA Content Analysis**: Always use `log_scale=True` with `log_base=2` for cell cycle data
2. **Comparing Conditions**: Use KDE overlay for direct comparison on single plot
3. **Sample Size Differences**: Enable `normalize=True` when comparing groups with different n
4. **Smoothing**: Lower `kde_smoothing` values (0.5-0.7) for smoother curves, higher (1.5-2.0) for more detail
5. **Binning**: Start with default 100 bins, adjust based on data range and sample size
6. **Figure Size**: Default is 4×4 cm per condition, adjust as needed

## Common Use Cases

### Cell Cycle Analysis
```python
histogram_plot(df, "integrated_int_DAPI_norm", conditions,
               log_scale=True, log_base=2, x_limits=(1, 16))
```

### Biomarker Expression
```python
histogram_plot(df, "intensity_mean_p21_nucleus", conditions,
               kde_overlay=True, kde_smoothing=0.7)
```

### Morphological Features
```python
histogram_plot(df, "area_cell", conditions,
               normalize=True, bins=50)
```
