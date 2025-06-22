# cellcycle_stacked_plot Function Summary

## Complete Parameter Interface

The `cellcycle_stacked_plot` function collects **all possible arguments** from the base classes and provides a comprehensive user interface for creating stacked cell cycle plots.

## Function Signature with All Parameters

```python
def cellcycle_stacked_plot(
    # REQUIRED PARAMETERS
    data: pd.DataFrame,
    conditions: List[str],

    # BASE CLASS ARGUMENTS (from OmeroPlots and BaseCellCyclePlot)
    condition_col: str = "condition",
    selector_col: Optional[str] = "cell_line",
    selector_val: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,

    # CELLCYCLE STACKED PLOT SPECIFIC ARGUMENTS
    phases: Optional[List[str]] = None,
    reverse_stack: bool = False,
    show_legend: bool = True,
    legend_position: str = "right",

    # INTEGRATION ARGUMENTS
    ax: Optional[Axes] = None,

    # OUTPUT ARGUMENTS
    save: bool = False,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,

    # SAVE QUALITY ARGUMENTS
    dpi: int = 300,
    format: str = "pdf",
    tight_layout: bool = True,

    **kwargs
) -> Figure
```

## Complete Parameter Reference

### 🔧 **Required Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Cell cycle data with 'cell_cycle', 'plate_id', 'experiment' columns |
| `conditions` | `List[str]` | Experimental conditions to plot |

### 📊 **Base Class Parameters** (from OmeroPlots/BaseCellCyclePlot)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition_col` | `str` | `"condition"` | Column name containing experimental conditions |
| `selector_col` | `Optional[str]` | `"cell_line"` | Column name for data filtering |
| `selector_val` | `Optional[str]` | `None` | Value to filter by (e.g., "RPE-1") |
| `title` | `Optional[str]` | `None` | Plot title (auto-generated if None) |
| `colors` | `Optional[List[str]]` | `None` | Custom color palette (uses config default if None) |
| `figsize` | `Optional[tuple]` | `None` | Figure size in inches (uses config default if None) |

### 📈 **Stacked Plot Specific Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phases` | `Optional[List[str]]` | `None` | Cell cycle phases to plot (default: ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]) |
| `reverse_stack` | `bool` | `False` | Reverse the stacking order of phases |
| `show_legend` | `bool` | `True` | Whether to show the phase legend |
| `legend_position` | `str` | `"right"` | Legend position ("right", "bottom", "top", "left") |

### 🔗 **Integration Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `Optional[Axes]` | `None` | Matplotlib axes to plot on (enables subplot integration) |

### 💾 **Output Control Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | `bool` | `False` | Whether to save the figure to file |
| `output_path` | `Optional[str]` | `None` | Directory for saving (required if save=True) |
| `filename` | `Optional[str]` | `None` | Custom filename (auto-generated if None) |

### 🖼️ **Save Quality Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | `int` | `300` | Resolution for saved figures (dots per inch) |
| `format` | `str` | `"pdf"` | File format ("pdf", "png", "svg", "eps", "tiff") |
| `tight_layout` | `bool` | `True` | Apply tight layout before saving |

### ⚙️ **Additional Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `**kwargs` | `dict` | `{}` | Additional arguments passed to base class |

## Usage Examples with All Parameters

### Minimal Usage (Just Required Parameters)
```python
fig = cellcycle_stacked_plot(
    data=cell_data,
    conditions=["Control", "Treatment"],
    selector_val="RPE-1"
)
```

### Complete Parameter Specification
```python
fig = cellcycle_stacked_plot(
    # Required parameters
    data=cell_data,
    conditions=["Control", "CDK4i", "CDK6i", "Combination"],

    # Base class parameters (explicitly showing defaults)
    condition_col="condition",                    # Default
    selector_col="cell_line",                     # Default
    selector_val="RPE-1",                         # Required for meaningful results
    title="Cell Cycle Distribution - RPE-1",     # Auto-generated if None
    colors=None,                                  # Uses config default
    figsize=None,                                 # Uses config default

    # Stacked plot specific parameters (showing defaults)
    phases=None,                                  # Uses ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
    reverse_stack=False,                          # Default stacking order
    show_legend=True,                             # Show legend
    legend_position="right",                      # Default position

    # Integration parameters
    ax=None,                                      # Create own figure

    # Output parameters (showing defaults)
    save=False,                                   # Don't save by default
    output_path=None,                             # No default path
    filename=None,                                # Auto-generated if saving

    # Quality parameters (showing defaults)
    dpi=300,                                      # Standard resolution
    format="pdf",                                 # Default format
    tight_layout=True                             # Apply tight layout
)
```

### Custom Styling Example
```python
fig = cellcycle_stacked_plot(
    data=cell_data,
    conditions=["DMSO", "CDK4i_5μM", "CDK6i_10μM", "Combination"],
    selector_val="HeLa",

    # Custom appearance
    title="CDK4/6 Inhibitor Effects in HeLa Cells",
    colors=["#8B0000", "#FF4500", "#FFD700", "#32CD32", "#4169E1"],
    figsize=(12, 8),
    phases=["Sub-G1", "G1", "S", "G2/M", "Polyploid"],

    # Custom stacking
    reverse_stack=True,
    show_legend=True,
    legend_position="bottom",

    # High-quality output
    save=True,
    output_path="figures/",
    filename="hela_cellcycle_analysis.pdf",
    dpi=600,
    format="pdf"
)
```

### Integration with Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Use function with provided axis
cellcycle_stacked_plot(
    data=cell_data,
    conditions=["Control", "Treatment"],
    selector_val="RPE-1",
    ax=axes[0, 0],                               # Key: provide axis
    show_legend=False,                           # Manage legend manually
    save=False                                   # Don't save individual plots
)

# Save the composite figure manually
fig.savefig("composite_analysis.pdf")
```

## Key Features

### ✨ **Intelligent Defaults**
- **Auto-generated titles**: "Cell Cycle Distribution - {selector_val}"
- **Auto-generated filenames**: "cellcycle_stacked_{selector_val}_{options}.{format}"
- **Default phase order**: Optimized for stacked visualization
- **Smart error handling**: Helpful validation messages

### 🔧 **Complete Customization**
- **All base class functionality**: Every parameter accessible
- **Stacked-specific features**: Reverse stacking, legend positioning
- **Output flexibility**: Multiple formats, quality settings
- **Integration support**: Works with existing matplotlib figures

### 🛡️ **Robust Implementation**
- **Comprehensive validation**: All inputs checked
- **Resource management**: Proper figure cleanup
- **Error handling**: Clear, actionable error messages
- **Documentation**: Extensive docstring with examples

### 📊 **Output Options**
- **Formats**: PDF, PNG, SVG, EPS, TIFF
- **Quality**: Configurable DPI and layout options
- **Integration**: Subplot support with axis parameter
- **Automation**: Auto-generated filenames and paths

## Default Values Summary

```python
# Data filtering
condition_col = "condition"
selector_col = "cell_line"
selector_val = None  # Must be provided for filtering

# Appearance
title = None  # Auto-generated: "Cell Cycle Distribution - {selector_val}"
colors = None  # Uses package default color palette
figsize = None  # Uses config default size

# Stacked plot behavior
phases = None  # Uses ["Polyploid", "G2/M", "S", "G1", "Sub-G1"]
reverse_stack = False
show_legend = True
legend_position = "right"

# Integration
ax = None  # Creates own figure

# Output
save = False
output_path = None
filename = None  # Auto-generated: "cellcycle_stacked_{selector_val}.{format}"

# Quality
dpi = 300
format = "pdf"
tight_layout = True
```

This comprehensive interface ensures that users have access to all functionality while maintaining simplicity for basic use cases.
