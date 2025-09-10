# Scatter Plot Examples

The `scatter_plot` function provides intelligent defaults and extensive customization options for creating scatter plots, particularly optimized for cell cycle analysis and biomarker visualization.

## Key Features

### Smart Auto-Detection
- **DNA Content (x-axis)**: Automatically applies log scale, limits (1,16), reference line at 3, cell cycle coloring
- **EdU Intensity (y-axis)**: Automatically applies log scale, reference line at 3
- **DNA vs EdU**: Full cell cycle plot with KDE overlay
- **Threshold Override**: When threshold is set, uses blue (below) and red (above) coloring

### Core Functionality
- **Flexible input**: Single condition (string) or multiple conditions (list)
- **Cell sampling**: Control performance with `cell_number` parameter
- **Multiple scales**: Linear or log (with auto-detection)
- **KDE overlays**: Density contours with customizable appearance
- **Reference lines**: Automatic or custom positioning
- **Color schemes**: Cell cycle phases, threshold-based, or custom palettes

## Basic Usage

### 1. Default Cell Cycle Analysis

The most common use case - DNA content vs EdU intensity with automatic cell cycle detection:

```python
import pandas as pd
from omero_screen_plots import scatter_plot

# Load your data
df = pd.read_csv("your_data.csv")

# Basic cell cycle scatter plot
fig, ax = scatter_plot(
    df=df,
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=3000,  # Sample for performance
    title="Cell Cycle Analysis",
    show_title=True
)
```

**Auto-detected features:**
- Log scales (base 2) for both axes
- X-limits: (1, 16) for DNA content
- Reference lines at x=3, y=3 for cell cycle gating
- Cell cycle phase coloring (if available)
- KDE density overlay

### 2. Multiple Conditions Comparison

Compare multiple treatment conditions side-by-side:

```python
conditions = ["control", "treatment1", "treatment2", "treatment3"]

fig, axes = scatter_plot(
    df=df,
    conditions=conditions,  # List creates subplots
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=2000,  # Fewer cells for multiple plots
    title="Treatment Comparison - Cell Cycle",
    show_title=True,
    fig_size=(16, 4)  # 4cm per condition
)
```

## Advanced Usage

### 3. Biomarker Analysis with Threshold

Analyze protein expression levels with threshold-based coloring:

```python
# DNA content vs protein expression with threshold
fig, axes = scatter_plot(
    df=df,
    y_feature="intensity_mean_p21_nucleus",  # Change y-axis
    conditions=conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=3000,
    threshold=5000,  # Threshold for p21 expression
    y_limits=(1000, 12000),
    title="p21 Expression Threshold Analysis",
    show_title=True
)
```

**Result:**
- Blue points: Below threshold (low p21)
- Red points: Above threshold (high p21)
- Maintains DNA content auto-detection (log scale, limits, reference line)

### 4. Linear Scale Analysis

For non-DNA/EdU features, linear scales are used automatically:

```python
# Cell morphology analysis
fig, ax = scatter_plot(
    df=df,
    x_feature="area_cell",
    y_feature="intensity_mean_p21_nucleus",
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=5000,
    x_limits=(0, 8000),  # Cell area range
    y_limits=(1000, 12000),  # Protein range
    title="Cell Size vs Protein Expression",
    show_title=True
)
```

### 5. Custom Scales and Reference Lines

Override auto-detection when needed:

```python
# Force log scales and custom reference lines
fig, ax = scatter_plot(
    df=df,
    x_feature="area_cell",
    y_feature="intensity_mean_p21_nucleus",
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=5000,
    x_scale="log",  # Override auto-detection
    y_scale="log",
    x_limits=(100, 10000),
    y_limits=(1000, 15000),
    vline=2000,  # Custom vertical line
    hline=5000,  # Custom horizontal line
    kde_overlay=True,  # Force KDE overlay
    title="Custom Scales with Reference Lines",
    show_title=True
)
```

## Customization Options

### 6. Cell Cycle Phase Control

Explicit control over cell cycle coloring:

```python
# Manual cell cycle settings
fig, ax = scatter_plot(
    df=df,
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=4000,
    hue="cell_cycle",  # Explicit hue
    hue_order=["Sub-G1", "G1", "S", "G2/M", "Polyploid"],
    show_legend=True,
    legend_title="Cell Cycle Phase",
    title="Manual Cell Cycle Control",
    show_title=True,
    fig_size=(6, 6)  # Square plot
)
```

### 7. KDE Overlay Customization

Control density overlay appearance:

```python
# Subtle KDE overlay
fig, ax = scatter_plot(
    df=df,
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=3000,
    kde_overlay=True,
    kde_alpha=0.1,  # Very subtle
    kde_cmap="viridis",  # Different colormap
    title="Custom KDE Overlay",
    show_title=True
)
```

### 8. Point Styling for Data Density

Optimize visualization for different data densities:

```python
# High-density data visualization
fig, ax = scatter_plot(
    df=df,
    conditions="control",
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    cell_number=10000,  # Many points
    size=1,  # Small points
    alpha=0.3,  # Transparent
    kde_overlay=False,  # Focus on points
    title="High-Density Visualization",
    show_title=True
)
```

### 9. Performance vs Quality Trade-offs

Balance speed and detail with cell sampling:

```python
# Fast preview (1000 cells)
fig, ax = scatter_plot(df, "control", cell_number=1000)

# Standard analysis (3000 cells)
fig, ax = scatter_plot(df, "control", cell_number=3000)

# Publication quality (10000+ cells)
fig, ax = scatter_plot(df, "control", cell_number=15000)
```

### 10. Integration with Matplotlib

Use with existing matplotlib figures:

```python
import matplotlib.pyplot as plt

# Create custom figure layout
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Different analysis on each subplot
scatter_plot(df, "control", axes=axes[0,0])
scatter_plot(df, "treatment", axes=axes[0,1])
scatter_plot(df, "control", threshold=5000, axes=axes[1,0])
scatter_plot(df, "treatment", threshold=5000, axes=axes[1,1])

fig.suptitle("Comprehensive Analysis", fontsize=14)
fig.tight_layout()
```

## Common Use Cases

### Cell Cycle Analysis
- **DNA vs EdU**: Classic cell cycle plot with phase identification
- **Multi-condition**: Compare treatments effects on cell cycle
- **Time course**: Track cell cycle changes over time

### Biomarker Analysis
- **Expression thresholds**: Identify high/low expressing populations
- **DNA vs protein**: Correlate cell cycle with protein levels
- **Morphology correlation**: Link cell size with biomarkers

### Quality Control
- **Data distribution**: Visualize measurement quality with KDE
- **Outlier detection**: Identify problematic cells or measurements
- **Batch effects**: Compare data consistency across experiments

## Performance Guidelines

### Cell Number Recommendations
- **Fast preview**: 500-1000 cells
- **Standard analysis**: 2000-5000 cells
- **Publication quality**: 8000-15000 cells
- **Maximum detail**: 20000+ cells (may be slow)

### Optimization Tips
- Use lower `alpha` (0.1-0.5) for high cell numbers
- Disable KDE overlay for very large datasets
- Use `cell_number` parameter to control performance
- Consider subplot layouts for multiple conditions

### Memory Considerations
- Large datasets (>100k cells): Always use `cell_number` sampling
- Multiple conditions: Reduce `cell_number` per condition
- KDE overlay: Adds computational overhead but improves interpretation

## Parameter Reference

### Essential Parameters
- `df`: DataFrame with your data
- `conditions`: Single condition (str) or multiple (list)
- `condition_col`: Column name for experimental conditions
- `selector_col/selector_val`: Additional data filtering

### Feature Control
- `x_feature`: X-axis column (default: DNA content)
- `y_feature`: Y-axis column (default: EdU intensity)
- `cell_number`: Number of cells to sample per condition

### Automatic vs Manual Control
- **Auto-detected**: DNA content → log scale, limits (1,16), cell cycle colors
- **Manual override**: Set `x_scale`, `x_limits`, `hue`, etc. explicitly
- **Threshold mode**: Set `threshold` → blue/red coloring overrides cell cycle

### Styling Options
- `size`: Point size (1-5, default: 2)
- `alpha`: Transparency (0-1, default: 1.0)
- `kde_overlay`: Density contours (auto for DNA vs EdU)
- `kde_alpha`: KDE transparency (0.1-0.6)
- `kde_cmap`: KDE colormap ("rocket_r", "viridis", etc.)
