# OMERO-Screen-Plots: Current Project State

*Last Updated: 06.08.2025*

## Project Overview

OMERO-Screen-Plots is a comprehensive Python package for visualizing high-content screening data from the OMERO-Screen pipeline. The package provides standardized, publication-ready plots for feature analysis, cell cycle analysis, and statistical comparisons.

## Recent Major Refactoring (January 2025)

### Architecture Overhaul
We completed a comprehensive refactoring that eliminated code duplication and improved maintainability:

- **Unified Base Architecture**: All plot classes now inherit from `BasePlotBuilder` and use `BasePlotConfig`
- **Eliminated Code Duplication**: Removed duplicate implementations of save, create figure, and finalize methods
- **Template Method Pattern**: Consistent plot creation pipeline across all plot types
- **Improved Type Safety**: Enhanced mypy compliance and proper type annotations

### New Class Hierarchy
```
BasePlotConfig
├── FeaturePlotConfig
├── CountPlotConfig
└── CellCyclePlotConfig

BasePlotBuilder (shared functionality)
├── BaseFeaturePlot (template for feature-based plots)
│   ├── StandardFeaturePlot (box/violin plots with scatter)
│   └── NormFeaturePlot (threshold-based stacked bars)
├── CountPlot (cell counting visualizations)
└── BaseCellCyclePlot (template for cell cycle plots)
    └── StandardCellCyclePlot (stacked bars with flexible layouts)
```

## Core Plot Types

### 1. Feature Plots (`featureplot_api.py`)
**Purpose**: Visualize quantitative features (intensity, area, shape metrics) across experimental conditions.

**Two Main Functions**:
- `feature_plot()` - Standard box/violin plots with scatter overlay
- `feature_norm_plot()` - Normalized threshold analysis with stacked bars

**Key Features**:
- Support for multiple cell lines and plate replicates
- Statistical significance testing (3+ replicates)
- Flexible grouping and spacing options
- Custom color schemes (green, blue, purple for norm plots)
- Optional violin plots instead of box plots
- Scatter point overlay for raw data visualization

**Implementation**: Uses factory pattern with `StandardFeaturePlot` and `NormFeaturePlot` classes

### 2. Count Plots (`countplot_api.py`)
**Purpose**: Visualize cell counts per condition, with normalization options.

**Features**:
- Absolute vs normalized count display
- Flexible grouping layouts (pairs, triplets, etc.)
- Repeat point visualization with different markers per plate
- Statistical significance marks for multi-plate experiments

**Implementation**: Single `CountPlot` class with configurable plot types

### 3. Cell Cycle Plots (`cellcycleplot_api.py`)
**Purpose**: Analyze cell cycle distributions and phase-specific markers with comprehensive phase analysis.

**Key Functions**:
- `cellcycle_plot()` - Main API function for creating cell cycle visualizations

**Advanced Features**:
- **Flexible Phase Support**: Automatic detection of 4-6 cell cycle phases (G1, S, G2, M, sub-phases)
- **Adaptive Layouts**: Dynamic subplot arrangement (2x2 for ≤4 phases, 2x3 for 5-6 phases)
- **Automatic M Phase Detection**: Intelligently identifies M phase from data when present
- **Flexible Terminology**: Support for "cell cycle" vs "DNA content" naming conventions
- **Plate Identification**: Different marker shapes per plate with comprehensive legends
- **Multi-Condition Analysis**: Side-by-side comparison across experimental conditions
- **EdU/BrdU Integration**: Support for incorporation markers and phase-specific analysis
- **Phase-Specific Markers**: Expression analysis for cycle-dependent proteins (p21, cyclin, etc.)

**Implementation**: Uses factory pattern with `BaseCellCyclePlot` and `StandardCellCyclePlot` classes

**Technical Capabilities**:
- Variable subplot layouts based on detected phases
- Automatic legend positioning and formatting
- Robust phase detection from data columns
- Support for complex experimental designs with multiple variables
- Comprehensive error handling and validation

### 4. Combined Plots (`combplot.py`)
**Purpose**: Create multi-panel figures combining different plot types.

**Features**:
- Flexible subplot arrangements
- Consistent styling across panels
- Shared legends and axes where appropriate

## Configuration System

### Unified Configuration Classes
All plots use a hierarchical configuration system:

- **`BasePlotConfig`**: Common settings (figure size, DPI, save options, colors)
- **`FeaturePlotConfig`**: Feature-specific settings (scaling, thresholds, violin mode)
- **`CountPlotConfig`**: Count-specific settings (normalization type, grouping)
- **`CellCyclePlotConfig`**: Cell cycle-specific settings (terminology, layout options, phase detection)

### Key Configuration Options
```python
# Common to all plots
fig_size: tuple[float, float] = (7, 7)
size_units: str = "cm"
dpi: int = 300
save: bool = False
colors: list[str] = field(default_factory=list)

# Feature plot specific
scale: bool = False  # Scale data before plotting
violin: bool = False  # Use violin instead of box plots
threshold: float = 1.5  # For normalized plots
show_scatter: bool = True  # Overlay scatter points
normalize_by_plate: bool = True  # Plate-wise normalization

# Cell cycle plot specific
terminology: str = "cell_cycle"  # "cell_cycle" or "dna_content"
auto_detect_phases: bool = True  # Automatically detect available phases
force_layout: Optional[str] = None  # Force specific subplot layout
```

## Data Processing Pipeline

### Standard Feature Plots
1. **Input Validation**: Check required columns (plate_id, condition, feature)
2. **Data Filtering**: Apply selector column filters if specified
3. **Scaling** (optional): Use `prepare_plot_data()` with percentile clipping
4. **Plot Creation**: Box/violin plots with statistical overlays
5. **Statistical Analysis**: Repeat points and significance marks (3+ plates)

### Normalized Feature Plots
1. **Input Validation**: Same as standard plots
2. **Data Filtering**: Apply selectors
3. **Normalization**: Mode-based normalization (peak = 1.0) per plate
4. **Threshold Analysis**: Calculate % cells above/below threshold
5. **Plot Creation**: Stacked bars showing proportions
6. **Optional QC**: Save before/after normalization plots

### Count Plots
1. **Cell Counting**: Group by well, count cells per condition
2. **Normalization** (optional): Normalize to control condition
3. **Aggregation**: Calculate mean ± std per condition across plates
4. **Plot Creation**: Bar plots with repeat points overlay

### Cell Cycle Plots
1. **Phase Detection**: Automatically identify available cell cycle phases from data columns
2. **Layout Determination**: Choose optimal subplot arrangement based on phase count
3. **Data Aggregation**: Calculate phase proportions per condition and plate
4. **M Phase Handling**: Special detection and processing for M phase data
5. **Plot Creation**: Stacked bar charts with adaptive layouts
6. **Legend Generation**: Comprehensive legends for plates and phases
7. **Statistical Overlays**: Repeat points with different markers per plate

## Styling and Visualization

### Color System (`colors.py`)
Centralized color definitions using enum:
```python
class COLOR(Enum):
    BLUE = "#1f77b4"
    YELLOW = "#ff7f0e"
    PINK = "#e377c2"
    OLIVE = "#8c564b"
    LIGHT_GREEN = "#98df8a"
    # ... more colors
```

### Plot Aesthetics
- **Consistent styling**: All plots use same font, line weights, spacing
- **Publication ready**: High DPI, vector formats (PDF/SVG)
- **Flexible sizing**: Support for cm/inch units, custom DPI
- **Color schemes**: Predefined schemes for different plot types

### Statistical Overlays
- **Repeat Points**: Different shapes (square, circle, triangle) per plate
- **Significance Marks**: Adaptive positioning based on group layouts
- **Error Bars**: Standard error or confidence intervals
- **Sample Size**: Automatic handling of different replicate numbers

## API Design Philosophy

### Backward-Compatible Wrappers
Public API functions maintain backward compatibility:
```python
# Simple function call
fig, ax = feature_plot(
    df=data,
    feature="area_nucleus",
    conditions=["control", "treatment"]
)

# Cell cycle analysis
fig, ax = cellcycle_plot(
    df=data,
    conditions=["control", "treatment"],
    terminology="cell_cycle"
)

# Advanced configuration
config = FeaturePlotConfig(
    violin=True,
    show_scatter=False,
    colors=["red", "blue"]
)
plot = StandardFeaturePlot(config)
fig, ax = plot.create_plot(df, feature, conditions)
```

### Template Method Pattern
All plot classes follow the same creation pipeline:
1. `create_plot()` - Main entry point
2. `_validate_inputs()` - Check data integrity
3. `_process_data()` - Filter and transform data
4. `_setup_figure()` - Create matplotlib objects
5. `build_plot()` - **Abstract method** - Create visualization
6. `_add_statistical_elements()` - Overlays and annotations
7. `_format_axes()` - Labels, ticks, limits
8. `_finalize_plot()` - Title and cleanup
9. `_save_plot()` - Export if configured

## Testing Strategy

### Comprehensive Test Coverage
- **201 tests** covering all major functionality (increased from 143 with cellcycle plot additions)
- **Unit tests**: Individual plot components
- **Integration tests**: Full plot creation pipeline
- **Edge cases**: Empty data, missing conditions, single plates
- **Backward compatibility**: API wrapper functions

### Test Categories
1. **Basic Functionality**: Standard use cases with synthetic data
2. **Parameter Validation**: Input checking and error messages
3. **Edge Cases**: Unusual data configurations
4. **Customization**: Color schemes, sizing, grouping options
5. **Statistical Analysis**: Significance testing, repeat points
6. **Cell Cycle Specific**: Phase detection, layout adaptation, terminology handling

### Cell Cycle Plot Testing (58 comprehensive tests)
- **Phase Detection**: Automatic identification of 4-6 phases
- **Layout Adaptation**: 2x2 and 2x3 subplot arrangements
- **M Phase Handling**: Special cases for M phase detection
- **Terminology Flexibility**: Both "cell_cycle" and "dna_content" naming
- **Multi-Plate Analysis**: Different marker shapes and legend handling
- **Error Conditions**: Missing phases, invalid data, configuration errors

### Synthetic Test Data
Realistic test datasets simulating:
- Multiple cell lines (MCF10A, HeLa)
- Multiple experimental conditions
- Multiple plate replicates
- Various feature measurements (area, intensity, shape)
- Cell cycle phase distributions (G1, S, G2, M, sub-phases)

## Documentation

### Sphinx Documentation (`docs/`)
- **API Reference**: Auto-generated from docstrings
- **User Guides**: Step-by-step tutorials
- **Examples**: Jupyter notebooks with real use cases
- **Plot Gallery**: Visual examples of all plot types

### Example Notebooks (`examples/`)
- **featureplot.ipynb**: Feature plot tutorial
- **countplot.ipynb**: Count plot examples
- **feature_norm_plot.ipynb**: Normalization analysis
- **cellcycleplot.ipynb**: Comprehensive cell cycle analysis tutorial (7 detailed examples)
- **Real data**: Sample CSV files for testing

### Cell Cycle Documentation Highlights
- **Complete API documentation**: Detailed function signatures and parameters
- **Seven comprehensive examples**: From basic to advanced use cases
- **Phase detection guidance**: How to structure data for automatic detection
- **Layout customization**: When and how to override automatic layouts
- **Terminology guidelines**: Choosing appropriate naming conventions

## Dependencies and Requirements

### Core Dependencies
```toml
python = "^3.12"
pandas = "^2.2.2"
matplotlib = "^3.8.0"
numpy = "^1.26.0"
seaborn = "^0.13.2"
scipy = "^1.13.1"
```

### Development Tools
```toml
pytest = "^8.3.2"
mypy = "^1.11.1"
ruff = "^0.5.7"  # Linting and formatting
sphinx = "^7.4.7"  # Documentation
```

## Code Quality Metrics

### Recent Improvements (Including Cell Cycle Refactoring)
- **Mypy compliance**: Achieved full type safety across all cell cycle plot modules
- **Code duplication**: Eliminated through inheritance refactoring and factory pattern
- **Test coverage**: All 201 tests passing (58 new tests for cell cycle plots)
- **Ruff compliance**: Clean code formatting and linting
- **Documentation coverage**: Complete API documentation with examples

### Architecture Quality
- **SOLID principles**: Single responsibility, open/closed, etc.
- **Template method pattern**: Consistent plot creation pipeline
- **Factory pattern**: Configurable plot type creation for all plot types
- **Composition over inheritance**: Flexible component assembly
- **Robust error handling**: Comprehensive validation and meaningful error messages

## Future Development Roadmap

### Planned Features
1. **Interactive Plots**: Plotly/Bokeh support for web applications
2. **Additional Plot Types**: Heatmaps, correlation matrices, dose-response curves
3. **Enhanced Statistics**: More sophisticated statistical tests, effect sizes
4. **Performance**: Optimization for large datasets (>1M cells)
5. **Export Options**: PowerPoint, publication templates

### Technical Debt
1. **Configuration validation**: More robust parameter checking across all plot types
2. **Memory management**: Better handling of large matplotlib figure collections
3. **Performance optimization**: Caching and lazy loading for large datasets

## Integration with OMERO-Screen Pipeline

### Data Flow
```
OMERO Images → Segmentation → Feature Extraction → CSV Export
                                                      ↓
                              omero-screen-plots ← CSV Import
                                                      ↓
                              Publication Figures → Analysis Reports
```

### Compatibility
- **Input format**: Standard CSV with required columns (plate_id, condition, features, cell cycle phases)
- **Metadata support**: Cell line, timepoint, experimental annotations
- **Output formats**: PDF, PNG, SVG for different use cases

### Usage in Analysis Pipeline
1. **Quality Control**: Count plots to verify cell numbers per condition
2. **Feature Analysis**: Feature plots for quantitative measurements
3. **Cell Cycle Analysis**: Cell cycle plots for phase distribution analysis
4. **Normalization**: Norm plots to handle batch effects
5. **Statistical Reporting**: Significance testing and effect visualization

## Conclusion

OMERO-Screen-Plots has evolved into a mature, well-architected visualization package with:

- **Clean codebase** following established patterns across all plot types
- **Comprehensive test coverage** ensuring reliability (201 tests)
- **Flexible configuration** supporting diverse use cases including complex cell cycle analysis
- **Publication-quality output** for scientific communication
- **Strong documentation** with extensive examples and tutorials

The recent cell cycle plot refactoring has significantly enhanced the package's capabilities for cell cycle analysis while maintaining the unified architecture established in the earlier refactoring. The addition of automatic phase detection, flexible layouts, and comprehensive testing demonstrates the maturity and robustness of the codebase. The package now provides complete coverage for the most common high-content screening visualization needs, from basic feature analysis to sophisticated cell cycle dynamics.
EOF < /dev/null
