# Migration Guide: omero-screen-plots v0.1.2

This guide explains how to migrate from the function-based interface to the new class-based architecture.

## Overview of Changes

### Before (Function-based)
```python
from omero_screen_plots import cellcycle_plot

# Function call with many parameters
cellcycle_plot(
    df=data,
    conditions=["Control", "Treatment"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="RPE-1",
    title="My Plot",
    save=True,
    path=output_path
)
```

### After (Class-based)
```python
from omero_screen_plots import CellCyclePlot

# Create plot instance
plot = CellCyclePlot(
    data=data,
    conditions=["Control", "Treatment"],
    condition_col="condition",
    selector_col="cell_line",
    selector_val="RPE-1",
    title="My Plot"
)

# Generate and save
fig = plot.generate()
plot.save(output_path)
plot.close()
```

## Key Benefits

1. **Reduced Code Duplication**: Common functionality is inherited from base class
2. **Consistent API**: All plot types have the same interface
3. **Better Integration**: Can integrate with existing matplotlib figures
4. **Composability**: Easy to create multi-panel figures
5. **Extensibility**: Easy to add new plot types

## Migration Examples

### 1. Basic Cell Cycle Plot

**Old way:**
```python
cellcycle_plot(
    df=data,
    conditions=conditions,
    selector_col="cell_line",
    selector_val="RPE-1",
    title="Cell Cycle Analysis",
    save=True,
    path=Path("output")
)
```

**New way:**
```python
plot = CellCyclePlot(
    data=data,
    conditions=conditions,
    selector_col="cell_line",
    selector_val="RPE-1",
    title="Cell Cycle Analysis"
)
fig = plot.generate()
plot.save("output/cellcycle.pdf")
```

### 2. Stacked Bar Plot

**Old way:**
```python
stacked_barplot(
    ax=None,
    df=data,
    conditions=conditions,
    # ... many parameters
)
```

**New way:**
```python
plot = CellCycleStackedPlot(
    data=data,
    conditions=conditions,
    selector_col="cell_line",
    selector_val="RPE-1"
)
fig = plot.generate()
```

### 3. Integration with Existing Figures

**New capability:**
```python
# Create your own figure layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Use plots on specific axes
plot1 = CellCycleStackedPlot(data=data, conditions=conditions, ax=axes[0,0])
plot1.generate()

plot2 = CellCycleStackedPlot(data=data2, conditions=conditions, ax=axes[0,1])
plot2.generate()
```

### 4. Composite Plots

**New capability:**
```python
# Create multiple plots
plots = [
    CellCycleStackedPlot(data=data, conditions=conditions, selector_val="RPE-1"),
    CellCycleStackedPlot(data=data, conditions=conditions, selector_val="HeLa")
]

# Combine into grid
combo = OmeroCombPlots(
    plots=plots,
    layout=(1, 2),
    titles=["RPE-1", "HeLa"]
)
fig = combo.generate_grid()
combo.save_composite("composite_plot.pdf")
```

## Backward Compatibility

The old function-based interface is still available:

```python
# This still works
from omero_screen_plots import cellcycle_plot, feature_plot

cellcycle_plot(df=data, conditions=conditions, ...)
```

## Available Plot Classes

### Current (v0.1.2)
- `CellCyclePlot`: 2x2 subplot grid for each phase
- `CellCycleStackedPlot`: Stacked bar plot of all phases
- `CellCycleGroupedPlot`: Grouped stacked bars with replicates

### Base Classes
- `OmeroPlots`: Base class for all plot types
- `OmeroCombPlots`: Composite figure manager

## Common Patterns

### 1. Simple Plot Generation
```python
plot = PlotType(data=data, conditions=conditions, **kwargs)
fig = plot.generate()
plot.save("output.pdf")
plot.close()  # Good practice
```

### 2. Custom Styling
```python
plot = PlotType(
    data=data,
    conditions=conditions,
    colors=["#FF0000", "#00FF00", "#0000FF"],
    figsize=(10, 6),
    title="Custom Plot"
)
```

### 3. Integration with Subplots
```python
fig, ax = plt.subplots()
plot = PlotType(data=data, conditions=conditions, ax=ax)
plot.generate()
# Add additional customizations to ax
```

## Best Practices

1. **Always call `plot.close()`** when you own the figure
2. **Use try/finally** for resource cleanup:
   ```python
   plot = CellCyclePlot(...)
   try:
       fig = plot.generate()
       plot.save("output.pdf")
   finally:
       plot.close()
   ```
3. **Use context managers** if you create them:
   ```python
   # Future enhancement - context manager support
   with CellCyclePlot(...) as plot:
       fig = plot.generate()
       plot.save("output.pdf")
   ```

## Configuration

The new architecture centralizes configuration:

```python
from omero_screen_plots.config import CONFIG

# Access global settings
print(CONFIG.DEFAULT_DPI)
print(CONFIG.colors)

# Modify if needed (affects all plots)
CONFIG.DEFAULT_DPI = 600
```

## Next Steps

1. Try the new classes alongside existing code
2. Gradually migrate critical plotting code
3. Take advantage of new features like composite plots
4. Provide feedback for future enhancements
