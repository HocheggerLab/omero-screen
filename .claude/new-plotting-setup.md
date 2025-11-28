# OMERO-Screen Plots: New Plotting Architecture Guide

## Table of Contents
1. [Architecture Evolution](#architecture-evolution)
2. [Component Overview](#component-overview)
3. [Data Flow Walkthrough](#data-flow-walkthrough)
4. [Code Examples](#code-examples)
5. [Architecture Comparison](#architecture-comparison)
6. [Benefits and Trade-offs](#benefits-and-trade-offs)
7. [Developer Usage Guide](#developer-usage-guide)

## Architecture Evolution

The omero-screen-plots package has evolved through three architectural iterations:

### Version 1: Monolithic Functions
- Single large functions handling all aspects of plotting
- Limited reusability and difficult to extend
- Tightly coupled data processing and visualization

### Version 2: Base Class Architecture
- Introduced abstract base classes for separation of concerns
- Used factory pattern for plot type registration
- Better modularity but increased complexity

### Version 3: Simplified Single-Class Architecture
- Consolidated functionality into focused classes
- Maintained separation of concerns within single class
- Added simple registry pattern for extensibility

## Component Overview

### 1. Base Classes (base.py)

The foundation provides three abstract base classes:

```python
@dataclass
class BasePlotConfig:
    """Base configuration for all plots."""
    # Common figure settings
    fig_size: tuple[float, float] = (7, 7)
    size_units: str = "cm"
    dpi: int = 300

    # Common save settings
    save: bool = False
    file_format: str = "pdf"
    tight_layout: bool = False
    path: Path | None = None

    # Common display settings
    title: str | None = None
    colors: list[str] = field(default_factory=list)
```

**BasePlotConfig**: Provides common configuration options that all plot types share. Uses dataclass for clean initialization and automatic `__repr__`.

```python
class BaseDataProcessor(ABC):
    """Base class for data processing."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validate_dataframe()

    @abstractmethod
    def validate_dataframe(self) -> None:
        """Validate required columns exist."""
        pass

    def filter_data(self, condition_col: str, conditions: list[str],
                   selector_col: str | None = None,
                   selector_val: str | None = None) -> pd.DataFrame:
        """Common filtering logic with validation."""
        # Provides reusable filtering with proper error messages
```

**BaseDataProcessor**: Handles data validation and filtering logic that's common across plot types. Each plot type implements specific validation and processing.

```python
class BasePlotBuilder(ABC):
    """Base class for plot builders."""

    def create_figure(self, axes: Axes | None = None) -> 'BasePlotBuilder':
        """Create or use existing figure."""

    @abstractmethod
    def build_plot(self, data: pd.DataFrame, **kwargs) -> 'BasePlotBuilder':
        """Build the specific plot type."""

    def finalize_plot(self, default_title: str | None = None) -> 'BasePlotBuilder':
        """Finalize plot with title and store filename."""

    def save_figure(self, filename: str | None = None) -> 'BasePlotBuilder':
        """Save figure if configured."""
```

**BasePlotBuilder**: Implements builder pattern for constructing plots step-by-step. Handles figure creation, finalization, and saving.

### 2. Factory Pattern (plot_factory.py)

The factory centralizes plot creation:

```python
class PlotFactory:
    """Factory for creating different plot types."""

    _registry: dict[str, tuple[type[BasePlotConfig],
                               type[BaseDataProcessor],
                               type[BasePlotBuilder]]] = {}

    @classmethod
    def register(cls, plot_type: str,
                config_class: type[BasePlotConfig],
                processor_class: type[BaseDataProcessor],
                builder_class: type[BasePlotBuilder]) -> None:
        """Register a new plot type."""
        cls._registry[plot_type] = (config_class, processor_class, builder_class)
```

The factory:
- Maintains a registry of plot types
- Coordinates the creation process
- Handles parameter routing to appropriate components
- Provides consistent interface across all plot types

### 3. Simple Registry (plot_registry.py)

Alternative lightweight registry pattern:

```python
class PlotRegistry:
    """Simple registry for plot creation functions."""

    _plots: Dict[str, Callable] = {}

    @classmethod
    def register(cls, plot_type: str, plot_func: Callable) -> None:
        """Register a plot creation function."""
        cls._plots[plot_type] = plot_func

# Decorator for easy registration
@register_plot('count')
def create_count_plot(df, norm_control, conditions, **kwargs):
    # plot creation logic
    return fig, ax
```

The simple registry:
- Provides minimal overhead for registration
- Uses decorator pattern for convenience
- Better suited for simpler plot functions

### 4. Simplified Count Plot (countplot_v3.py)

The latest iteration combines all functionality into a single class:

```python
class CountPlot:
    """Simplified count plot implementation combining config, processing, and plotting."""

    def __init__(self, config: CountPlotConfig | None = None):
        """Initialize with configuration."""
        self.config = config or CountPlotConfig()

    def create_plot(
        self,
        df: pd.DataFrame,
        norm_control: str,
        conditions: list[str],
        # ... other parameters
    ) -> tuple[Figure, Axes]:
        """Create complete count plot."""
        # Validate inputs
        self._validate_inputs(df, norm_control, conditions, ...)

        # Filter and process data
        processed_data = self._process_data(df, norm_control, conditions, ...)

        # Create figure
        self._create_figure(axes)

        # Build plot
        self._build_plot(processed_data, conditions, condition_col)

        # Finalize
        self._finalize_plot(selector_val)

        # Save if configured
        self._save_figure()

        return self.fig, self.ax
```

This approach:
- Encapsulates all related functionality in one place
- Uses private methods for organization
- Maintains clear separation of concerns
- Easier to understand and debug

## Data Flow Walkthrough

Let's trace how a count plot is created step-by-step:

### Step 1: User Call
```python
fig, ax = count_plot(
    df=df,
    norm_control="control",
    conditions=['control', 'treatment1', 'treatment2'],
    selector_col="cell_line",
    selector_val="MCF10A",
    plot_type=PlotType.NORMALISED
)
```

### Step 2: Configuration Creation
The `count_plot` function (in countplot_api.py) creates a configuration object:
```python
config = CountPlotConfig(
    fig_size=fig_size,
    size_units=size_units,
    plot_type=plot_type,
    # ... other settings
)
```

### Step 3: Plot Instance Creation
```python
plot = CountPlot(config)
```

### Step 4: Data Validation
The `_validate_inputs` method checks:
- Required columns exist
- Conditions are present in data
- Selector values are valid
- Control condition is in conditions list

### Step 5: Data Processing
The `_process_data` method:
1. Filters data by conditions and selector
```python
mask = df[condition_col].isin(conditions)
if selector_col and selector_val:
    mask &= (df[selector_col] == selector_val)
filtered_df = df.loc[mask]
```

2. Counts experiments per well
```python
well_counts = (
    filtered_df.groupby(["plate_id", condition_col, "well"])
    .size()
    .reset_index(name="well_count")
)
```

3. Calculates mean counts per condition
```python
grouped = (
    well_counts.groupby(["plate_id", condition_col], as_index=False)
    ["well_count"]
    .mean()
    .to_frame("count")
)
```

4. Creates normalized values
```python
pivot_df = grouped.pivot(index="plate_id", columns=condition_col, values="count")
normalized_df = pivot_df.div(pivot_df[norm_control], axis=0)
```

### Step 6: Figure Creation
```python
if axes:
    self.fig = axes.figure
    self.ax = axes
else:
    fig_inches = convert_size_to_inches(self.config.fig_size, self.config.size_units)
    self.fig, self.ax = plt.subplots(figsize=fig_inches)
```

### Step 7: Plot Building
The `_build_plot` method:
1. Determines which column to plot (normalized or absolute)
2. Calculates x positions for grouping
3. Creates bars using seaborn or manual placement
4. Adds individual data points
5. Adds statistical significance markers
6. Formats axes

### Step 8: Finalization
- Adds title (from config or generated)
- Applies consistent formatting
- Generates filename for saving

### Step 9: Saving (if configured)
```python
if self.config.save and self.config.path:
    save_fig(
        self.fig,
        self.config.path,
        self._filename,
        tight_layout=self.config.tight_layout,
        fig_extension=self.config.file_format,
        resolution=self.config.dpi
    )
```

## Code Examples

### Example 1: Using the Simplified Architecture (v3)

```python
from omero_screen_plots import CountPlot, CountPlotConfig, PlotType

# Create configuration
config = CountPlotConfig(
    plot_type=PlotType.NORMALISED,
    fig_size=(10, 8),
    save=True,
    path=Path("./output"),
    group_size=2,  # Group conditions in pairs
    within_group_spacing=0.3,
    between_group_gap=0.8
)

# Create plot instance
plot = CountPlot(config)

# Generate plot
fig, ax = plot.create_plot(
    df=data,
    norm_control="DMSO",
    conditions=["DMSO", "Drug1", "Drug2", "Drug3"],
    condition_col="treatment",
    selector_col="cell_line",
    selector_val="HeLa"
)
```

### Example 2: Using the Factory Pattern (v2)

```python
from omero_screen_plots.plot_factory import PlotFactory
from omero_screen_plots.countplot_v2 import CountPlotConfig, CountDataProcessor, CountPlotBuilder

# Register the count plot type
PlotFactory.register(
    'count',
    CountPlotConfig,
    CountDataProcessor,
    CountPlotBuilder
)

# Create plot using factory
fig, ax = PlotFactory.create_plot(
    'count',
    df=data,
    config={'plot_type': PlotType.NORMALISED, 'save': True},
    norm_control="DMSO",
    conditions=["DMSO", "Drug1", "Drug2", "Drug3"]
)
```

### Example 3: Using the Simple Registry

```python
from omero_screen_plots.plot_registry import PlotRegistry, register_plot

# Register a custom plot function
@register_plot('custom_count')
def create_custom_count_plot(df, **kwargs):
    # Custom implementation
    return fig, ax

# Use the registry
fig, ax = PlotRegistry.create_plot(
    'custom_count',
    df=data,
    norm_control="control",
    conditions=conditions
)
```

### Example 4: Extending with New Plot Types

To add a new plot type using the simplified architecture:

```python
from dataclasses import dataclass
from omero_screen_plots.base import BasePlotConfig

@dataclass
class HeatmapConfig:
    """Configuration for heatmap plots."""
    # Inherit common settings
    fig_size: tuple[float, float] = (12, 8)

    # Heatmap specific settings
    cmap: str = "RdBu_r"
    show_values: bool = True
    cluster_rows: bool = True
    cluster_cols: bool = True

class HeatmapPlot:
    def __init__(self, config: HeatmapConfig | None = None):
        self.config = config or HeatmapConfig()

    def create_plot(self, df: pd.DataFrame, **kwargs) -> tuple[Figure, Axes]:
        # Implementation
        pass
```

## Architecture Comparison

### Version 2 (Base Class Architecture)

**Advantages:**
- Clear separation of concerns with distinct classes
- Highly extensible through inheritance
- Factory pattern provides centralized registration
- Each component has single responsibility

**Disadvantages:**
- More complex with multiple classes to understand
- Requires coordination between three classes
- Potential for inheritance hierarchy complexity
- More boilerplate code for simple plots

**Best for:**
- Large projects with many plot types
- Teams with multiple developers
- When maximum flexibility is needed
- Complex plotting requirements

### Version 3 (Simplified Architecture)

**Advantages:**
- Single class contains all related functionality
- Easier to understand and debug
- Less boilerplate code
- Self-contained implementation
- Clearer data flow

**Disadvantages:**
- Less separation between concerns
- Potentially larger classes
- May lead to code duplication across plot types
- Less flexible for complex inheritance scenarios

**Best for:**
- Smaller projects or packages
- Rapid development and prototyping
- When simplicity is valued over flexibility
- Single developer or small team

## Benefits and Trade-offs

### Configuration Management

**Dataclass Approach:**
```python
@dataclass
class CountPlotConfig:
    fig_size: tuple[float, float] = (7, 7)
    plot_type: PlotType = PlotType.NORMALISED
```

Benefits:
- Type hints for all parameters
- Automatic `__init__` and `__repr__`
- Default values clearly visible
- Easy to extend with new parameters

Trade-offs:
- Requires Python 3.7+
- Less flexible than dictionary approach
- All parameters must be predefined

### Error Handling

The new architecture emphasizes detailed error messages:

```python
def _validate_inputs(self, df, norm_control, conditions, ...):
    if missing_conditions := set(conditions) - available_conditions:
        raise ValueError(
            f"Conditions not found in data: {missing_conditions}. "
            f"Available conditions: {sorted(available_conditions)}"
        )
```

Benefits:
- Users get actionable error messages
- Easier debugging
- Prevents silent failures

Trade-offs:
- More validation code
- Slightly slower execution
- Larger codebase

### Backward Compatibility

The `countplot_api.py` maintains compatibility:

```python
def count_plot(df, norm_control, conditions, **kwargs):
    """Backward-compatible wrapper."""
    config = CountPlotConfig(**kwargs)
    plot = CountPlot(config)
    return plot.create_plot(df, norm_control, conditions, ...)
```

This allows existing code to work unchanged while providing access to new features.

## Developer Usage Guide

### Creating a New Plot Type

1. **Define Configuration:**
```python
@dataclass
class MyPlotConfig:
    # Common settings (could inherit from BasePlotConfig)
    fig_size: tuple[float, float] = (8, 6)

    # Plot-specific settings
    my_param: str = "default"
```

2. **Implement Plot Class:**
```python
class MyPlot:
    def __init__(self, config: MyPlotConfig | None = None):
        self.config = config or MyPlotConfig()

    def create_plot(self, df: pd.DataFrame, **kwargs):
        # Validate
        self._validate_inputs(df, **kwargs)

        # Process
        data = self._process_data(df, **kwargs)

        # Create figure
        self._create_figure()

        # Build plot
        self._build_plot(data)

        # Finalize and save
        self._finalize_plot()

        return self.fig, self.ax
```

3. **Register with Registry:**
```python
@register_plot('myplot')
def create_my_plot(df, **kwargs):
    plot = MyPlot()
    return plot.create_plot(df, **kwargs)
```

### Best Practices

1. **Configuration First**: Always define configuration as a dataclass
2. **Validate Early**: Check inputs before processing
3. **Private Methods**: Use underscore prefix for internal methods
4. **Type Hints**: Add type hints for all public methods
5. **Error Messages**: Provide helpful, actionable error messages
6. **Documentation**: Document all public methods and parameters
7. **Testing**: Write tests for each major method

### Testing Your Plot

```python
import pytest
from omero_screen_plots import MyPlot, MyPlotConfig

def test_my_plot_creation():
    # Setup
    df = create_test_data()
    config = MyPlotConfig(fig_size=(10, 8))

    # Execute
    plot = MyPlot(config)
    fig, ax = plot.create_plot(df, param1="value1")

    # Assert
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Expected Title"

def test_my_plot_validation():
    df = pd.DataFrame()  # Empty dataframe
    plot = MyPlot()

    with pytest.raises(ValueError, match="Input dataframe is empty"):
        plot.create_plot(df)
```

### Integration with Existing Code

The architecture is designed to integrate smoothly:

```python
# In a Jupyter notebook
from omero_screen_plots import count_plot, PlotType

# Simple usage
fig, ax = count_plot(df, "control", ["control", "treatment"])

# Advanced usage with configuration
from omero_screen_plots import CountPlot, CountPlotConfig

config = CountPlotConfig(
    plot_type=PlotType.ABSOLUTE,
    save=True,
    path=Path("./figures")
)

plot = CountPlot(config)
fig, ax = plot.create_plot(df, "control", conditions)

# Using in subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, condition_set in enumerate(condition_sets):
    plot.create_plot(df, "control", condition_set, axes=axes.flat[i])
```

## Conclusion

The new plotting architecture in omero-screen-plots represents a thoughtful evolution toward simplicity and usability. The version 3 simplified architecture strikes a balance between modularity and ease of use, making it ideal for the package's target use case of scientific data visualization.

Key takeaways:
- The simplified single-class approach reduces complexity while maintaining functionality
- Configuration management through dataclasses provides type safety and clarity
- The registry pattern enables extensibility without heavyweight abstractions
- Backward compatibility ensures smooth migration
- Clear error messages and validation improve the developer experience

This architecture serves as a solid foundation for building publication-quality scientific plots while remaining approachable for users at all skill levels.
EOF < /dev/null
