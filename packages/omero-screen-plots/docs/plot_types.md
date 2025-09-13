# Plot Types

OmeroScreen Plots provides several specialized plot types for high-content screening data analysis. Each plot type is designed for specific analytical needs.

## FeaturePlot

**Purpose**: Compare any measured feature across conditions using box plots, violin plots, or boxen plots.

**Use cases**:
- Protein expression levels
- Cell morphology parameters
- Intensity measurements
- Any quantitative feature

### Basic Usage

```python
from omero_screen_plots.featureplot import FeaturePlot

plot = FeaturePlot(
    data_path="sample_plate_data.csv",
    y_feature="intensity_mean_p21_nucleus",
    conditions=["DMSO", "Nutlin"],
    condition_col="condition",
    plot_type="box"  # Options: "box", "violin", "boxen"
)
```

### Key Parameters

- **y_feature**: The feature to plot on y-axis
- **plot_type**: Visual representation style
- **test_group_pairs**: Pairs for statistical comparisons
- **grouped**: Enable visual grouping of conditions
- **show_points**: Overlay individual data points

### Output

- Box/violin plots with statistical annotations
- Optional scatter overlay
- Automatic or custom statistics

## ClassificationPlot

**Purpose**: Visualize categorical classification results as stacked bars or individual triplicates.

**Use cases**:
- Cell morphology classification
- Phenotype distribution analysis
- Quality control assessment
- Drug toxicity evaluation

### Basic Usage

```python
from omero_screen_plots import classification_plot

fig, ax = classification_plot(
    df=df,
    classes=["normal", "micronuclei", "collapsed"],
    conditions=["control", "treatment1", "treatment2"],
    class_col="classifier",  # Dynamic classification column
    display_mode="stacked"    # or "triplicates"
)
```

### Key Features

- **Dynamic class column**: Adapt to any categorical classification
- **Two display modes**: Stacked bars with error bars or individual triplicates
- **Flexible grouping**: Group conditions for easier comparison
- **Custom colors**: Apply meaningful color schemes to categories

## CellCyclePlot

**Purpose**: Analyze cell cycle distribution based on DNA content (DAPI) and S-phase marker (EdU).

**Use cases**:
- Cell cycle profiling
- Drug effect on proliferation
- p21-based cell cycle arrest analysis

### Basic Usage

```python
from omero_screen_plots.cellcycleplot import CellCyclePlot

cc_plot = CellCyclePlot(
    data_path="sample_plate_data.csv",
    conditions=["DMSO", "Nutlin"],
    condition_col="condition",
    dapi_col="intensity_integrated_dapi_nucleus",
    edu_col="intensity_mean_edu_nucleus",
    p21_col="intensity_mean_p21_nucleus"  # Optional
)
```

### Key Parameters

- **dapi_col**: Column with integrated DAPI intensity
- **edu_col**: Column with EdU incorporation data
- **p21_col**: Optional p21 expression for arrest analysis
- **plot_type**: "standard", "grouped", or "stacked"
- **normalise**: Normalize to percentage or absolute counts

### Output

- Cell cycle phase distribution (G1, S, G2)
- Optional p21+ arrested cells
- Statistical comparisons between conditions

## CombPlot

**Purpose**: Combined visualizations showing relationships between features with marginal distributions.

**Use cases**:
- Feature correlation analysis
- Scatter plots with histograms
- Distribution comparisons
- Multi-dimensional data exploration

### Plot Types Available

1. **Scatter Plot**
   ```python
   plot = CombPlot(
       data_path="sample_plate_data.csv",
       plot_type="scatter",
       x_feature="intensity_integrated_dapi_nucleus",
       y_feature="intensity_mean_edu_nucleus"
   )
   ```

2. **Histogram**
   ```python
   plot = CombPlot(
       data_path="sample_plate_data.csv",
       plot_type="histogram",
       x_feature="area_nucleus"
   )
   ```

3. **Hexbin Plot**
   ```python
   plot = CombPlot(
       data_path="sample_plate_data.csv",
       plot_type="hexbin",
       x_feature="area_nucleus",
       y_feature="intensity_mean_dapi_nucleus"
   )
   ```

### Key Parameters

- **x_feature**, **y_feature**: Features for x and y axes
- **plot_type**: "scatter", "histogram", "hexbin", "kde"
- **show_regression**: Add regression line to scatter
- **marginal_type**: Type of marginal plots

## CountPlot

**Purpose**: Analyze and visualize cell counts per well or condition.

**Use cases**:
- Toxicity assessment
- Proliferation analysis
- Quality control
- Well-to-well variation

### Basic Usage

```python
from omero_screen_plots.countplot import CountPlot

count_plot = CountPlot(
    data_path="sample_plate_data.csv",
    conditions=["DMSO", "Nutlin", "Etop", "Noc"],
    condition_col="condition"
)
```

### Key Parameters

- **normalise_to**: Normalize counts to specific condition
- **show_plate_layout**: Display as plate heatmap
- **aggregate_by**: Group by well, condition, or plate

### Output

- Bar plots of cell counts
- Statistical comparisons
- Optional plate layout visualization

## Specialized Analysis Functions

### Normalise

**Purpose**: Data normalization utilities for immunofluorescence data.

```python
from omero_screen_plots.normalise import NormalisePlot

norm_plot = NormalisePlot(
    data_path="sample_plate_data.csv",
    feature="intensity_mean_p21_nucleus",
    control_condition="DMSO"
)
```

Features:
- Z-score normalization
- Percent of control
- Robust scaling
- Plate-wise normalization

### Stats Module

**Purpose**: Statistical analysis utilities used across all plot types.

```python
from omero_screen_plots.stats import perform_statistical_tests

# Automatic statistical testing
results = perform_statistical_tests(
    data,
    groups=["DMSO", "Nutlin"],
    test_type="auto"  # Automatically selects appropriate test
)
```

## Choosing the Right Plot Type

| Analysis Goal | Recommended Plot Type |
|--------------|---------------------|
| Compare single feature across conditions | FeaturePlot |
| Analyze cell cycle distribution | CellCyclePlot |
| Explore feature relationships | CombPlot (scatter) |
| Assess cell viability/counts | CountPlot |
| Examine distributions | CombPlot (histogram) |
| Normalize data | NormalisePlot |

## Common Workflow

1. **Start with CountPlot** to check data quality and cell numbers
2. **Use FeaturePlot** to explore individual features
3. **Apply CellCyclePlot** for proliferation analysis
4. **Explore with CombPlot** for feature relationships
5. **Normalize data** when comparing across plates

## Tips

- Always check cell counts first to identify problematic wells
- Use violin plots for large datasets to see distribution shapes
- Enable `show_points` in FeaturePlot for small sample sizes
- Use grouped layouts when comparing related conditions
- Save plots in vector format (PDF/SVG) for publications
