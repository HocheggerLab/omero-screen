various# OMERO-Screen-Plots: Current Project State

*Last Updated: 07.08.2025*

## Project Overview

OMERO-Screen-Plots is a comprehensive Python package for visualizing high-content screening data from the OMERO-Screen pipeline. The package provides standardized, publication-ready plots for feature analysis, cell cycle analysis, and statistical comparisons.

## Latest Major Development (August 2025)

### Cell Cycle Stacked Plots Unification
We completed a comprehensive refactoring of the cell cycle plotting system, unifying scattered functionality into a cohesive, factory-based architecture:

- **Unified API**: Replaced separate `cellcycle_stacked` and `cellcycle_grouped` functions with a single, configurable `cellcycle_stacked()` function
- **Factory Pattern Implementation**: Created `StackedCellCyclePlot` class following the established factory pattern used throughout the package
- **Comprehensive Configuration**: Added `StackedCellCyclePlotConfig` with extensive customization options
- **Visual Consistency**: Coordinated legend positioning, bar widths, and spacing to match feature plots exactly
- **Enhanced Documentation**: Created complete API documentation with 10 comprehensive examples

### Key Architectural Improvements
```
BasePlotConfig
├── FeaturePlotConfig
├── CountPlotConfig
└── CellCyclePlotConfig
    └── StackedCellCyclePlotConfig  # NEW

BasePlotBuilder (shared functionality)
├── BaseFeaturePlot (template for feature-based plots)
│   ├── StandardFeaturePlot (box/violin plots with scatter)
│   └── NormFeaturePlot (threshold-based stacked bars)
├── CountPlot (cell counting visualizations)
└── BaseCellCyclePlot (template for cell cycle plots)
    ├── StandardCellCyclePlot (variable grid layouts)
    └── StackedCellCyclePlot (stacked bars with unified API)  # NEW
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

### 3. Cell Cycle Plots (`cellcycleplot_api.py`) - **ENHANCED**
**Purpose**: Analyze cell cycle distributions and phase-specific markers with comprehensive analysis capabilities.

**Key Functions**:
- `cellcycle_plot()` - Variable grid layout showing each phase separately (2x2, 2x3 subplots)
- `cellcycle_stacked()` - **NEW**: Unified stacked bar plots with flexible display modes

#### Cell Cycle Stacked Plots - **Major New Feature**
**Unified API Features**:
- **Single Function**: Replaces separate `cellcycle_stacked` and `cellcycle_grouped` functions
- **Dual Modes**: Summary mode (aggregated bars with error bars) or triplicates mode (individual bars per plate)
- **Flexible Grouping**: Group conditions in pairs, triplets, or custom arrangements
- **Visual Consistency**: Coordinated with feature plots for multi-panel figures
- **Comprehensive Configuration**: 15+ customization parameters

**Display Options**:
- **Summary Mode** (default): Aggregated percentages with error bars (mean ± SEM)
- **Triplicate Mode**: Individual bars for each replicate with optional box outlines
- **Mixed Grouping**: Combine conditions into visual groups with custom spacing
- **Legend Control**: Toggle legend display and positioning
- **Phase Terminology**: Cell cycle phases (G1, S, G2/M) or DNA content (<2N, 2N, S, 4N, >4N)
- **Custom Colors**: Specify custom color schemes for phases

**Implementation**: Uses factory pattern with `StackedCellCyclePlot` class and `StackedCellCyclePlotConfig`

#### Standard Cell Cycle Plots
**Advanced Features**:
- **Flexible Phase Support**: Automatic detection of 4-6 cell cycle phases
- **Adaptive Layouts**: Dynamic subplot arrangement (2x2 for ≤4 phases, 2x3 for 5-6 phases)
- **Automatic M Phase Detection**: Intelligently identifies M phase from data
- **Flexible Terminology**: Support for "cell cycle" vs "DNA content" naming
- **Multi-Condition Analysis**: Side-by-side comparison across conditions

**Implementation**: Uses factory pattern with `BaseCellCyclePlot` and `StandardCellCyclePlot` classes

### 4. Combined Plots (`combplot.py`)
**Purpose**: Create multi-panel figures combining different plot types.

**Features**:
- Flexible subplot arrangements
- Consistent styling across panels
- Shared legends and axes where appropriate

## Configuration System

### Enhanced Configuration Classes
All plots use a hierarchical configuration system, now including comprehensive stacked cell cycle options:

- **`BasePlotConfig`**: Common settings (figure size, DPI, save options, colors)
- **`FeaturePlotConfig`**: Feature-specific settings (scaling, thresholds, violin mode)
- **`CountPlotConfig`**: Count-specific settings (normalization type, grouping)
- **`CellCyclePlotConfig`**: Cell cycle-specific settings (terminology, layout options)
- **`StackedCellCyclePlotConfig`**: **NEW** - Stacked plot specific settings with 15+ parameters

### New Stacked Cell Cycle Configuration Options
```python
# Stacked cell cycle specific (in addition to base options)
show_triplicates: bool = False          # Show individual vs summary bars
group_size: int = 1                     # Group conditions (1=no grouping)
within_group_spacing: float = 0.2       # Space between bars within group
between_group_gap: float = 0.5          # Gap between groups
bar_width: float = 0.5                  # Width of bars
repeat_offset: float = 0.18             # Spacing for triplicate bars
max_repeats: int = 3                    # Maximum repeats to display
show_boxes: bool = True                 # Boxes around triplicates
show_legend: bool = True                # Phase color legend
legend_position: tuple[float, float] = (1.05, 1)  # Legend placement
cc_phases: bool = True                  # Use cell cycle vs DNA terminology
phase_order: list[str] | None = None    # Custom phase order
show_error_bars: bool = True            # Error bars in summary mode
```

## Data Processing Pipeline

### Enhanced Cell Cycle Processing
The cell cycle processing pipeline now supports both standard and stacked visualizations:

#### Standard Cell Cycle Plots
1. **Phase Detection**: Automatically identify available cell cycle phases
2. **Layout Determination**: Choose optimal subplot arrangement
3. **Data Aggregation**: Calculate phase proportions per condition and plate
4. **Plot Creation**: Variable grid layouts with adaptive subplot arrangement

#### Stacked Cell Cycle Plots - **NEW**
1. **Input Validation**: Check required columns (cell_cycle, plate_id, condition)
2. **Data Processing**: Calculate phase percentages per condition and plate
3. **Mode Selection**: Choose summary (aggregated) or triplicates (individual) display
4. **Statistical Calculation**: Compute means and standard errors for summary mode
5. **Plot Creation**: Stacked bars with configurable grouping and spacing
6. **Visual Elements**: Add error bars, boxes, legends based on configuration
7. **Axis Formatting**: Coordinate with feature plots for consistent appearance

### Other Processing Pipelines
Standard feature plots, normalized feature plots, and count plots maintain their established processing pipelines as documented in the previous version.

## Bug Fixes and Improvements (August 2025)

### Major Bug Fixes
1. **Subplot Title Placement**: Fixed issue where cell cycle plots would show misplaced titles when used with external axes (subplots)
2. **Visual Consistency**: Coordinated legend positioning and bar widths between cell cycle and feature plots for multi-panel figures
3. **Color Handling**: Fixed custom color functionality and preserved default color schemes
4. **Triplicate Display**: Fixed bar positioning and box drawing to match original implementations

### Technical Improvements
1. **Unified Architecture**: Brought cell cycle stacked plots in line with the factory pattern used throughout the package
2. **Proper Inheritance**: Ensured `StackedCellCyclePlot` properly inherits from `BaseCellCyclePlot`
3. **Configuration Validation**: Enhanced parameter validation and error handling
4. **Backward Compatibility**: Maintained deprecated `cellcycle_grouped()` function with proper deprecation warnings

## Testing Strategy

### Comprehensive Test Coverage
- **220+ tests** covering all major functionality (increased from 201 with stacked cellcycle additions)
- **Unit tests**: Individual plot components and configurations
- **Integration tests**: Full plot creation pipeline
- **Edge cases**: Empty data, missing conditions, single plates
- **Backward compatibility**: API wrapper functions and deprecated methods

### New Test Categories (Stacked Cell Cycle)
1. **API Function Tests**: All parameter variations for `cellcycle_stacked()`
2. **Factory Class Tests**: `StackedCellCyclePlot` configuration and plot building
3. **Display Mode Tests**: Summary vs triplicates mode functionality
4. **Configuration Tests**: `StackedCellCyclePlotConfig` validation and defaults
5. **Backward Compatibility**: Deprecated `cellcycle_grouped()` function testing
6. **Integration Tests**: Multi-panel figure creation with other plot types
7. **Visual Consistency**: Coordinate spacing and positioning with feature plots

### Test File Structure
- **test_cellcycleplot.py**: Standard cell cycle plots (58 tests)
- **test_cellcycle_stacked.py**: **NEW** - Stacked cell cycle plots (19+ comprehensive tests)
- **Other test files**: Feature plots, count plots, utilities, etc.

## Documentation

### Enhanced Documentation (`docs/`)
- **API Reference**: Auto-generated from docstrings with new stacked cell cycle documentation
- **User Guides**: Step-by-step tutorials including stacked cell cycle usage
- **Examples**: Comprehensive Jupyter notebooks with real use cases
- **Plot Gallery**: Visual examples of all plot types including 10 stacked cell cycle variants

### New Documentation Features
- **cellcyclestacked.rst**: **NEW** - Complete API documentation for stacked cell cycle plots
- **10 Comprehensive Examples**: Basic plots, triplicates, custom colors, DNA terminology, grouping, etc.
- **Integration Examples**: Multi-panel figures combining stacked cell cycle with feature plots
- **Configuration Guide**: Detailed explanation of all 15+ stacked cell cycle parameters

### Updated Example Notebooks (`examples/`)
- **cellcycle_stacked.ipynb**: **ENHANCED** - Comprehensive tutorial with 10 detailed examples
- **Other notebooks**: Updated to demonstrate integration with new stacked cell cycle functionality

### Documentation Generation
- **generate_example_plots.py**: **ENHANCED** - Added `cellcycle_stacked_examples()` function generating 10 SVG examples for documentation
- **Automated plot generation**: All documentation images generated programmatically

## Code Quality Metrics

### Recent Quality Improvements (August 2025)
- **Enhanced Architecture**: Unified cell cycle plotting under consistent factory pattern
- **Type Safety**: Proper type annotations for new stacked cell cycle components (note: some typing issues remain to be addressed)
- **Code Consistency**: Eliminated duplicate code by leveraging existing base classes and utilities
- **Error Handling**: Comprehensive validation and meaningful error messages for new functionality
- **Documentation Coverage**: Complete API documentation with visual examples

### Backward Compatibility
- **API Preservation**: All existing functions maintain their interfaces
- **Deprecation Warnings**: Proper warnings for deprecated `cellcycle_grouped()` function
- **Migration Path**: Clear documentation for transitioning to new unified API

## Integration with OMERO-Screen Pipeline

### Enhanced Data Flow
```
OMERO Images → Segmentation → Feature Extraction → CSV Export
                                                      ↓
                              omero-screen-plots ← CSV Import
                                      ↓
                   ┌─ Standard Cell Cycle Analysis (subplot grids)
                   ├─ Stacked Cell Cycle Analysis (unified bars)  # NEW
                   ├─ Feature Analysis (box/violin plots)
                   ├─ Normalized Analysis (threshold plots)
                   └─ Count Analysis (cell number plots)
                                      ↓
                              Publication Figures → Analysis Reports
```

### Enhanced Usage in Analysis Pipeline
1. **Quality Control**: Count plots to verify cell numbers per condition
2. **Feature Analysis**: Feature plots for quantitative measurements
3. **Cell Cycle Analysis**:
   - **Standard plots**: Individual phase analysis with subplot grids
   - **Stacked plots**: **NEW** - Comparative phase distribution analysis
4. **Multi-Panel Analysis**: **NEW** - Combined stacked cell cycle + feature analysis figures
5. **Normalization**: Norm plots to handle batch effects
6. **Statistical Reporting**: Significance testing and effect visualization

## Future Development Roadmap

### Immediate Priorities
1. **Type Annotations**: Complete the typing system for new stacked cell cycle components
2. **Performance Optimization**: Optimize stacked plot rendering for large datasets
3. **Configuration Validation**: Enhanced parameter checking for new stacked cell cycle options

### Planned Features
1. **Interactive Plots**: Plotly/Bokeh support for web applications
2. **Additional Plot Types**: Heatmaps, correlation matrices, dose-response curves
3. **Enhanced Statistics**: More sophisticated statistical tests, effect sizes
4. **Export Options**: PowerPoint, publication templates
5. **Cell Cycle Extensions**: Time-course analysis, transition rate calculations

### Technical Debt
1. **Memory Management**: Better handling of large matplotlib figure collections
2. **Configuration System**: Unified validation across all plot type configurations
3. **Performance**: Caching and lazy loading for complex multi-panel figures

## Conclusion

OMERO-Screen-Plots has significantly evolved with the addition of unified cell cycle stacked plotting capabilities:

- **Enhanced Architecture**: Consistent factory pattern now extends to all cell cycle functionality
- **Unified API**: Single `cellcycle_stacked()` function replaces multiple scattered implementations
- **Comprehensive Configuration**: 15+ parameters enabling extensive customization
- **Visual Consistency**: Perfect coordination with feature plots for publication-quality multi-panel figures
- **Complete Documentation**: 10 comprehensive examples with programmatically generated illustrations
- **Robust Testing**: 19+ new tests ensuring reliability and preventing regressions
- **Backward Compatibility**: Seamless migration path while preserving existing functionality

The package now provides complete coverage for cell cycle analysis workflows, from detailed phase-specific analysis (standard plots) to comparative distribution analysis (stacked plots). The unified architecture ensures consistency across all plot types while maintaining the flexibility needed for diverse high-content screening applications.

**Test Coverage**: 220+ comprehensive tests
**Documentation**: Complete API reference with visual examples
**Architecture**: Mature, consistent factory pattern implementation
**Quality**: Publication-ready output with extensive customization options
