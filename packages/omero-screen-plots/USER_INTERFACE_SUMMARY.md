# User Interface Summary: cellcycle_standard_plot

## Overview

The `cellcycle_standard_plot` function provides a clean, comprehensive user interface that combines **all possible arguments** from the base classes (`OmeroPlots` and `BaseCellCyclePlot`) and the `CellCyclePlot` class into a single, intuitive function.

## Function Signature

```python
def cellcycle_standard_plot(
    data: pd.DataFrame,
    conditions: List[str],
    # Base class arguments
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    # CellCyclePlot specific arguments
    phases: Optional[List[str]] = None,
    show_significance: bool = True,
    show_points: bool = True,
    # Output arguments
    save: bool = False,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,
    # Additional matplotlib/save arguments
    dpi: int = 300,
    format: str = "pdf",
    tight_layout: bool = True,
    **kwargs
) -> Figure
```

## Argument Categories

### 🔧 **Core Data Arguments** (Required)
- `data`: DataFrame with cell cycle data
- `conditions`: List of experimental conditions to plot

### 📊 **Data Filtering Arguments**
- `condition_col`: Column name for conditions (default: "condition")
- `selector_col`: Column for data selection (default: "cell_line")
- `selector_val`: Value to filter by (e.g., "RPE-1")

### 🎨 **Appearance Arguments**
- `title`: Plot title (auto-generated if None)
- `colors`: Custom color palette (uses defaults if None)
- `figsize`: Figure size in inches (uses defaults if None)
- `phases`: Cell cycle phases to plot (max 4 for 2x2 grid)

### 📈 **Analysis Features**
- `show_significance`: Enable statistical significance markers
- `show_points`: Show individual replicate data points

### 💾 **Output Control**
- `save`: Whether to save the plot to file
- `output_path`: Directory for saving (required if save=True)
- `filename`: Custom filename (auto-generated if None)

### 🖼️ **Quality Settings**
- `dpi`: Resolution for saved figures
- `format`: File format ("pdf", "png", "svg", etc.)
- `tight_layout`: Apply tight layout before saving

## Usage Examples

### Minimal Usage (Just 3 Arguments!)
```python
fig = cellcycle_standard_plot(
    data=cell_data,
    conditions=["Control", "Treatment"],
    selector_val="RPE-1"
)
```

### Save to File
```python
fig = cellcycle_standard_plot(
    data=cell_data,
    conditions=["Control", "CDK4i", "CDK6i"],
    selector_val="RPE-1",
    save=True,
    output_path="figures/"
)
```

### Publication Ready
```python
fig = cellcycle_standard_plot(
    data=cell_data,
    conditions=["Control", "CDK4i", "CDK6i"],
    selector_val="RPE-1",
    title="Cell Cycle Analysis - RPE-1",
    colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    figsize=(12, 9),
    save=True,
    output_path="figures/",
    filename="figure_2_cellcycle.pdf",
    dpi=600
)
```

## Intelligent Defaults

### Auto-Generated Elements
- **Title**: "Cell Cycle Analysis - {selector_val}" if not provided
- **Filename**: "cellcycle_standard_{selector_val}.{format}" if not provided
- **Colors**: Uses package default color palette
- **Figure Size**: Optimized for cell cycle plots
- **Phases**: ["G1", "S", "G2/M", "Polyploid"] if not specified

### Smart Behaviors
- **Error Handling**: Comprehensive validation with helpful messages
- **Path Management**: Automatically creates output directories
- **Format Handling**: Ensures correct file extensions
- **Resource Cleanup**: Proper figure management and cleanup

## Complete Argument Integration

The function combines arguments from:

### From `OmeroPlots` Base Class:
- `data`, `conditions`, `condition_col`
- `selector_col`, `selector_val`
- `title`, `colors`, `figsize`

### From `BaseCellCyclePlot`:
- `phases` (cell cycle specific)
- Data validation and processing
- Statistical analysis capabilities

### From `CellCyclePlot`:
- `show_significance`, `show_points`
- 2x2 subplot grid generation
- Phase-specific plotting

### Additional User Conveniences:
- `save`, `output_path`, `filename`
- `dpi`, `format`, `tight_layout`
- Automatic file management

## Benefits for Users

### 🚀 **Ease of Use**
- Single function call for complete functionality
- Minimal required arguments (just 3!)
- Intelligent defaults for everything else

### 🔧 **Full Customization**
- Every aspect can be customized when needed
- Progressive disclosure: simple by default, powerful when needed
- All base class functionality accessible

### 📁 **File Management**
- Automatic filename generation
- Multiple format support
- Directory creation and validation

### 🛡️ **Robustness**
- Comprehensive error checking
- Helpful error messages
- Resource cleanup and management

### 📖 **Documentation**
- Extensive docstring with examples
- Clear argument categorization
- Usage patterns demonstrated

## Implementation Benefits

### For Developers:
- **Maintainable**: Single location for user interface
- **Extensible**: Easy to add new arguments
- **Testable**: Clear interface for comprehensive testing
- **Consistent**: Uniform behavior across all functionality

### For Users:
- **Discoverable**: All options in one place
- **Predictable**: Consistent argument patterns
- **Flexible**: Works for simple and complex use cases
- **Reliable**: Robust error handling and validation

## Future Extensions

The pattern established here can be extended to other plot types:

```python
# Future functions following the same pattern
featureplot_standard_plot(...)
countplot_standard_plot(...)
stackedplot_standard_plot(...)
```

Each would collect all relevant arguments from their respective base classes while maintaining the same user-friendly interface pattern.

## Summary

The `cellcycle_standard_plot` function successfully:

✅ **Collects all possible arguments** from base classes
✅ **Provides intuitive user interface** with intelligent defaults
✅ **Maintains full customization** capabilities
✅ **Handles all output scenarios** with flexible file management
✅ **Ensures robustness** with comprehensive error handling
✅ **Demonstrates best practices** for user-facing API design

This approach creates the perfect balance between **simplicity for basic use** and **power for advanced customization**, exactly as requested.
