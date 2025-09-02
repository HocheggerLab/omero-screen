# Count Plot Usage Examples

This guide provides comprehensive examples for using the `count_plot` function with different scenarios commonly encountered in high-content screening analysis.

## Basic Setup

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omero_screen_plots.countplot_api import count_plot
from omero_screen_plots.countplot_factory import PlotType

# Load example data
df = pd.read_csv("sample_plate_data.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
```

## Example 1: Basic Normalized Count Plot

The most common use case - comparing cell counts across conditions relative to a control.

```python
# Basic normalized count plot
fig, ax = count_plot(
    df=df,
    norm_control="DMSO",  # Control condition for normalization
    conditions=["DMSO", "Nutlin", "Etop", "Noc"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Cell Viability - MCF10A",
    save=True,
    path=output_dir,
    file_format="pdf"
)

plt.show()
```

**Use Case**: Standard drug screening to assess cell viability or proliferation relative to vehicle control.

## Example 2: Absolute Count Plot

When you want to see actual cell numbers rather than relative changes.

```python
# Absolute count plot
fig, ax = count_plot(
    df=df,
    norm_control="DMSO",  # Still required for processing
    conditions=["DMSO", "Nutlin", "Etop", "Noc"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    plot_type=PlotType.ABSOLUTE,
    title="Absolute Cell Counts - MCF10A",
    fig_size=(8, 6),
    save=True,
    path=output_dir
)

plt.show()
```

**Use Case**: When absolute numbers are important for experimental planning or comparing across different experiments with different seeding densities.

## Example 3: Multi-Cell Line Comparison

Compare counts across different cell lines using subplot layout.

```python
# Compare multiple cell lines
cell_lines = ["MCF10A", "RPE1", "U2OS"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cell_line in enumerate(cell_lines):
    count_plot(
        df=df,
        norm_control="DMSO",
        conditions=["DMSO", "Nutlin", "Etop", "Noc"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val=cell_line,
        axes=axes[i],
        title=f"{cell_line} Cell Counts",
        x_label=(i == 1)  # Only show x-labels on middle plot
    )

plt.tight_layout()
plt.savefig(output_dir / "multi_cell_line_comparison.pdf", dpi=300)
plt.show()
```

**Use Case**: Comparing drug sensitivity across different cell lines in the same figure.

## Example 4: Grouped Layout for Many Conditions

When you have many conditions, grouping improves readability.

```python
# Many conditions with grouping
many_conditions = [
    "DMSO", "Drug_A_1uM", "Drug_A_10uM",
    "Drug_B_1uM", "Drug_B_10uM",
    "Combo_1", "Combo_2"
]

fig, ax = count_plot(
    df=df,
    norm_control="DMSO",
    conditions=many_conditions,
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    group_size=3,  # Group in sets of 3
    within_group_spacing=0.15,
    between_group_gap=0.8,
    title="Drug Combination Screen - Grouped Layout",
    fig_size=(12, 6),
    save=True,
    path=output_dir
)

plt.show()
```

**Use Case**: Drug combination screens or dose-response studies with many conditions.

## Example 5: Time-Course Analysis

Analyzing cell counts over time points.

```python
# Time course analysis
time_points = ["0h", "6h", "12h", "24h", "48h"]

fig, ax = count_plot(
    df=df,
    norm_control="0h",
    conditions=time_points,
    condition_col="time_point",
    selector_col="treatment",
    selector_val="Drug_Treatment",
    title="Cell Growth Over Time - Drug Treatment",
    fig_size=(8, 6),
    save=True,
    path=output_dir
)

plt.show()
```

**Use Case**: Time-course experiments tracking cell proliferation or death over time.

## Example 6: Custom Styling and Colors

Customizing the appearance for publication-ready figures.

```python
# Custom styling
from omero_screen_plots.colors import COLOR

fig, ax = count_plot(
    df=df,
    norm_control="Control",
    conditions=["Control", "Treatment_1", "Treatment_2", "Treatment_3"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Custom Styled Count Plot",
    colors=[COLOR.BLUE.value, COLOR.RED.value, COLOR.GREEN.value, COLOR.ORANGE.value],
    fig_size=(7, 5),
    dpi=600,  # High resolution for publication
    tight_layout=True,
    save=True,
    path=output_dir,
    file_format="svg"  # Vector format
)

# Additional customization after plotting
ax.set_ylabel("Relative Cell Count", fontsize=12)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.grid(True, alpha=0.3)

plt.show()
```

**Use Case**: Creating publication-ready figures with specific styling requirements.

## Example 7: Error Handling and Data Validation

Demonstrating proper error handling when data is missing or malformed.

```python
# Example with error handling
try:
    fig, ax = count_plot(
        df=df,
        norm_control="NonExistentControl",  # This will raise an error
        conditions=["DMSO", "Treatment"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A"
    )
except ValueError as e:
    print(f"Error: {e}")

    # Correct approach after checking available data
    print("Available conditions:", df['condition'].unique())
    print("Available cell lines:", df['cell_line'].unique())

    # Use correct control condition
    fig, ax = count_plot(
        df=df,
        norm_control="DMSO",  # Correct control
        conditions=["DMSO", "Nutlin", "Etop"],
        condition_col="condition",
        selector_col="cell_line",
        selector_val="MCF10A",
        title="Corrected Count Plot"
    )
```

**Use Case**: Robust analysis pipelines that handle data inconsistencies gracefully.

## Example 8: Batch Analysis

Processing multiple datasets or conditions in a loop.

```python
# Batch analysis for multiple experiments
experiments = ["Exp_1", "Exp_2", "Exp_3"]
treatments = ["DMSO", "Drug_A", "Drug_B", "Drug_C"]

for exp in experiments:
    # Filter data for this experiment
    exp_data = df[df['experiment'] == exp]

    if not exp_data.empty:
        fig, ax = count_plot(
            df=exp_data,
            norm_control="DMSO",
            conditions=treatments,
            condition_col="condition",
            selector_col="cell_line",
            selector_val="MCF10A",
            title=f"Cell Counts - {exp}",
            save=True,
            path=output_dir / exp,
            file_format="pdf"
        )
        plt.close()  # Close to free memory
```

**Use Case**: High-throughput analysis of multiple experimental replicates or conditions.

## Example 9: Integration with Statistical Analysis

Combining count plots with additional statistical analysis.

```python
# Count plot with additional statistical analysis
fig, ax = count_plot(
    df=df,
    norm_control="DMSO",
    conditions=["DMSO", "Treatment_1", "Treatment_2"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="MCF10A",
    title="Count Analysis with Statistics"
)

# The plot automatically includes statistical significance markers
# when ≥3 plates are present in the data

# Extract the processed data for additional analysis
from omero_screen_plots.countplot_factory import CountPlot, CountPlotConfig

config = CountPlotConfig()
plot_obj = CountPlot(config)

# Get summary statistics
print("Summary statistics for each condition:")
summary = df.groupby(['condition', 'plate_id']).size().reset_index(name='count')
summary_stats = summary.groupby('condition')['count'].agg(['mean', 'std', 'sem'])
print(summary_stats)

plt.show()
```

**Use Case**: When you need both visualization and quantitative statistical results.

## Example 10: Memory-Efficient Large Dataset Processing

Handling large datasets efficiently.

```python
# Memory-efficient processing of large datasets
def process_large_dataset(df, chunk_size=10000):
    """Process large datasets in chunks to avoid memory issues."""

    # Get unique combinations to process
    combinations = df[['cell_line', 'experiment']].drop_duplicates()

    results = []

    for _, combo in combinations.iterrows():
        # Filter data for this combination
        subset = df[
            (df['cell_line'] == combo['cell_line']) &
            (df['experiment'] == combo['experiment'])
        ]

        if len(subset) > 0:
            try:
                fig, ax = count_plot(
                    df=subset,
                    norm_control="DMSO",
                    conditions=["DMSO", "Treatment"],
                    condition_col="condition",
                    selector_col="cell_line",
                    selector_val=combo['cell_line'],
                    title=f"{combo['cell_line']} - {combo['experiment']}",
                    save=True,
                    path=output_dir / "large_dataset_analysis"
                )

                results.append({
                    'cell_line': combo['cell_line'],
                    'experiment': combo['experiment'],
                    'status': 'success'
                })

                plt.close()  # Important: close to free memory

            except Exception as e:
                results.append({
                    'cell_line': combo['cell_line'],
                    'experiment': combo['experiment'],
                    'status': f'error: {e}'
                })

    return results

# Process large dataset
# results = process_large_dataset(very_large_df)
```

**Use Case**: Processing very large datasets from high-throughput screens with memory constraints.

## Key Points for Effective Usage

### Data Requirements
- Ensure your DataFrame has `plate_id`, `well`, and `experiment` columns
- The condition column should contain your experimental conditions
- For filtering, make sure selector columns and values exist in your data

### Performance Tips
- Use specific condition lists rather than processing all conditions
- Close plots with `plt.close()` when generating many figures
- Consider using absolute paths for file saving
- Use vector formats (SVG, PDF) for publication-quality figures

### Statistical Considerations
- Statistical analysis is automatically performed when ≥3 plates are present
- The normalization control should be included in your conditions list
- Consider the biological meaning of your normalization strategy

### Troubleshooting Common Issues
- **"Missing required columns"**: Ensure `plate_id`, `well`, `experiment` columns exist
- **"Control condition not found"**: Check that `norm_control` is in your conditions list
- **"No data after filtering"**: Verify your selector column and values exist in the data
- **Empty plots**: Check that your conditions exist in the specified condition column

These examples should cover most common use cases for the `count_plot` function in high-content screening analysis.
