# Code Quality Review: omero-screen-plots

## Executive Summary

The omero-screen-plots package provides a comprehensive plotting library for OMERO screen data analysis. While the codebase demonstrates functional implementation with good domain-specific capabilities, it requires significant improvements in code organization, type safety, error handling, and testing infrastructure. The most critical issues include the complete absence of unit tests, inconsistent error handling, and potential security vulnerabilities in data scaling operations.

## Critical Issues 🚨

### 1. Complete Absence of Unit Tests
**Location**: No test files found in the entire package
**Issue**: The package has zero test coverage, making it vulnerable to regressions and difficult to maintain.
**Fix**: Implement comprehensive test suite covering all modules.

```python
# Example test structure for tests/test_featureplot.py
import pytest
import pandas as pd
import numpy as np
from omero_screen_plots.featureplot import feature_plot, prepare_plot_data

class TestFeaturePlot:
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'condition': ['control', 'control', 'treated', 'treated'] * 10,
            'feature': np.random.randn(40),
            'plate_id': ['plate1'] * 20 + ['plate2'] * 20,
            'cell_line': ['MCF7'] * 40
        })

    def test_feature_plot_basic(self, sample_data):
        """Test basic feature plot functionality."""
        fig, ax = feature_plot(
            sample_data,
            feature='feature',
            conditions=['control', 'treated'],
            save=False
        )
        assert fig is not None
        assert ax is not None
        assert len(ax.patches) > 0  # Check that plot elements exist
```

### 2. SQL Injection-like Vulnerability in Data Filtering
**Location**: `utils.py:63-79` (selector_val_filter function)
**Issue**: Direct DataFrame filtering without validation could be exploited with malicious column names.
**Fix**: Add input validation and sanitization.

```python
def selector_val_filter(
    df: pd.DataFrame,
    selector_col: Optional[str],
    selector_val: Optional[str],
    condition_col: Optional[str],
    conditions: Optional[list[str]],
) -> Optional[pd.DataFrame]:
    """Check if selector_val is provided for selector_col and filter df."""
    # Validate column names exist in dataframe
    if condition_col and condition_col not in df.columns:
        raise ValueError(f"Column '{condition_col}' not found in dataframe")
    if selector_col and selector_col not in df.columns:
        raise ValueError(f"Column '{selector_col}' not found in dataframe")

    # Validate conditions are valid values
    if condition_col and conditions:
        valid_conditions = df[condition_col].unique()
        invalid = set(conditions) - set(valid_conditions)
        if invalid:
            raise ValueError(f"Invalid conditions: {invalid}")
        df = df[df[condition_col].isin(conditions)].copy()

    if selector_col and selector_val:
        if selector_val not in df[selector_col].unique():
            raise ValueError(f"'{selector_val}' not found in column '{selector_col}'")
        return df[df[selector_col] == selector_val].copy()
    elif selector_col:
        raise ValueError(f"selector_val for {selector_col} must be provided")
    else:
        return df.copy()
```

### 3. Division by Zero Risk in Statistical Calculations
**Location**: `stats.py:38` (t-test calculation)
**Issue**: While some variance checks exist, edge cases could still cause issues.
**Fix**: Comprehensive error handling already partially implemented but needs improvement.

```python
def calculate_pvalues(
    df: pd.DataFrame, conditions: list[str], condition_col: str, column: str
) -> list[float]:
    """Calculate p-values with robust error handling."""
    df2 = df[df[condition_col].isin(conditions)]

    # Validate data exists
    if df2.empty:
        logger.warning("No data found for specified conditions")
        return [1.0] * (len(conditions) - 1)

    count_list = [
        df2[df2[condition_col] == condition][column].tolist()
        for condition in conditions
    ]

    # Ensure reference group has data
    if not count_list[0]:
        logger.warning("Reference condition has no data")
        return [1.0] * (len(conditions) - 1)

    # ... rest of implementation
```

## Performance Optimizations ⚡

### 1. Inefficient Data Sampling in select_datapoints
**Location**: `utils.py:182-195`
**Issue**: Multiple DataFrame concatenations in a loop cause O(n²) performance.
**Fix**: Collect samples first, then concatenate once.

```python
def select_datapoints(
    df: pd.DataFrame, conditions: list[str], condition_col: str, n: int = 30
) -> pd.DataFrame:
    """Select n random datapoints per category and plate-id efficiently."""
    sampled_dfs = []

    for condition in conditions:
        for plate_id in df.plate_id.unique():
            df_sub = df[
                (df[condition_col] == condition) & (df.plate_id == plate_id)
            ]
            if len(df_sub) > n:
                sampled_dfs.append(df_sub.sample(n=n, random_state=1))
            else:
                sampled_dfs.append(df_sub)

    return pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()
```

### 2. Repeated Style Path Resolution
**Location**: Multiple files load the same matplotlib style
**Issue**: Path resolution happens multiple times unnecessarily.
**Fix**: Create a centralized configuration module.

```python
# config.py
from pathlib import Path
import matplotlib.pyplot as plt

_STYLE_LOADED = False

def load_plot_style():
    """Load the custom matplotlib style once."""
    global _STYLE_LOADED
    if not _STYLE_LOADED:
        current_dir = Path(__file__).parent
        style_path = (current_dir / "../../hhlab_style01.mplstyle").resolve()
        if style_path.exists():
            plt.style.use(style_path)
            _STYLE_LOADED = True
        else:
            raise FileNotFoundError(f"Style file not found: {style_path}")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]

# Usage in other modules
from omero_screen_plots.config import load_plot_style
COLORS = load_plot_style()
```

## Code Quality Improvements 📝

### 1. Inconsistent Type Annotations
**Location**: Throughout the codebase
**Issue**: Mix of modern (list[str]) and old-style (Optional[str]) type hints.
**Fix**: Standardize on Python 3.12+ syntax.

```python
# Before
def feature_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    axes: Optional[Axes] = None,
    x_label: bool = True,
    ymax: float | tuple[float, float] | None = None,
    ...
) -> tuple[Figure, Axes]:

# After (consistent modern syntax)
def feature_plot(
    df: pd.DataFrame,
    feature: str,
    conditions: list[str],
    axes: Axes | None = None,
    x_label: bool = True,
    ymax: float | tuple[float, float] | None = None,
    ...
) -> tuple[Figure, Axes]:
```

### 2. Magic Numbers Without Constants
**Location**: Multiple locations (e.g., utils.py:208, normalise.py:273)
**Issue**: Hard-coded values like 65535, 0.5, 99.5 without explanation.
**Fix**: Define named constants with documentation.

```python
# constants.py
# Data scaling constants
SCALE_MIN_PERCENTILE = 1.0  # Remove bottom 1% outliers
SCALE_MAX_PERCENTILE = 99.0  # Remove top 1% outliers
UINT16_MAX = 65535  # Maximum value for 16-bit unsigned integer

# Plot styling constants
DEFAULT_ALPHA = 0.75
DEFAULT_LINEWIDTH = 0.5
DEFAULT_MARKER_SIZE = 3
DEFAULT_JITTER_WIDTH = 0.07

# Statistical thresholds
MIN_SAMPLE_SIZE = 2  # Minimum samples for t-test
SIGNIFICANCE_ALPHA = 0.05
```

### 3. Duplicate Code in Plot Functions
**Location**: `featureplot.py` - Multiple similar plotting functions
**Issue**: Code duplication between feature_plot and feature_plot_simple.
**Fix**: Extract common functionality.

```python
class PlotBuilder:
    """Builder class for creating standardized plots."""

    def __init__(self, df: pd.DataFrame, feature: str, conditions: list[str]):
        self.df = df
        self.feature = feature
        self.conditions = conditions
        self.fig = None
        self.ax = None

    def prepare_data(self, condition_col: str, selector_col: str | None,
                    selector_val: str | None, scale: bool = False):
        """Prepare and filter data for plotting."""
        self.df_filtered = prepare_plot_data(
            self.df, self.feature, self.conditions,
            condition_col, selector_col, selector_val, scale
        )
        return self

    def create_figure(self, axes: Axes | None, fig_size: tuple[float, float],
                     size_units: str = "cm"):
        """Setup figure and axes."""
        self.fig, self.ax = setup_figure(axes, fig_size, size_units)
        self.axes_provided = axes is not None
        return self

    def add_boxplots(self, x_positions: list[float], colors: list[str]):
        """Add boxplots to the figure."""
        # Implementation here
        return self

    def build(self) -> tuple[Figure, Axes]:
        """Return the completed figure and axes."""
        return self.fig, self.ax
```

## Architectural Recommendations 🏗️

### 1. Separate Plotting Logic from Data Processing
**Current**: Functions mix data preparation, statistical analysis, and plotting.
**Proposed**: Create distinct layers for better separation of concerns.

```python
# data_processing.py
class PlotDataProcessor:
    """Handles all data preparation for plots."""

    def prepare_feature_data(self, df: pd.DataFrame, feature: str,
                           conditions: list[str], **kwargs) -> pd.DataFrame:
        """Prepare data for feature plots."""
        # Data filtering and scaling logic

    def calculate_statistics(self, df: pd.DataFrame,
                           groupby_cols: list[str]) -> pd.DataFrame:
        """Calculate summary statistics for plots."""
        # Statistical calculations

# plotting.py
class PlotRenderer:
    """Handles the actual plot rendering."""

    def __init__(self, style_config: dict):
        self.style = style_config

    def render_boxplot(self, ax: Axes, data: pd.DataFrame, **kwargs):
        """Render a boxplot with consistent styling."""
        # Plotting logic only
```

### 2. Configuration Management
**Current**: Configuration scattered across modules.
**Proposed**: Centralized configuration system.

```python
# config.py
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior."""
    figure_size: tuple[float, float] = (6, 6)
    size_units: str = "cm"
    dpi: int = 300
    file_format: str = "pdf"
    colors: list[str] = None
    style_path: Path = None

    @classmethod
    def from_file(cls, config_path: Path) -> 'PlotConfig':
        """Load configuration from JSON file."""
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_inches(self) -> tuple[float, float]:
        """Convert figure size to inches if needed."""
        if self.size_units == "cm":
            return (self.figure_size[0] / 2.54, self.figure_size[1] / 2.54)
        return self.figure_size
```

### 3. Plugin Architecture for Plot Types
**Current**: Each plot type is a separate function with similar structure.
**Proposed**: Plugin-based architecture for extensibility.

```python
# base.py
from abc import ABC, abstractmethod

class PlotPlugin(ABC):
    """Base class for all plot types."""

    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate that data has required columns."""
        pass

    @abstractmethod
    def process_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process data for this plot type."""
        pass

    @abstractmethod
    def create_plot(self, ax: Axes, data: pd.DataFrame, **kwargs) -> None:
        """Create the actual plot."""
        pass

    def plot(self, df: pd.DataFrame, **kwargs) -> tuple[Figure, Axes]:
        """Main plotting method."""
        self.validate_data(df)
        processed_data = self.process_data(df, **kwargs)
        fig, ax = self.setup_figure(**kwargs)
        self.create_plot(ax, processed_data, **kwargs)
        return fig, ax
```

## Prioritized Action Items ✅

### High Priority
1. **Create comprehensive test suite** - Zero test coverage is the most critical issue
2. **Fix security vulnerabilities** - Add input validation to prevent data injection
3. **Implement proper error handling** - Add try-except blocks and logging throughout
4. **Add data validation** - Validate DataFrame contents before processing

### Medium Priority
1. **Refactor duplicate code** - Extract common functionality into base classes
2. **Standardize type annotations** - Use consistent Python 3.12+ syntax
3. **Optimize performance bottlenecks** - Fix inefficient pandas operations
4. **Create configuration system** - Centralize all configuration options

### Low Priority
1. **Improve documentation** - Add docstrings to all public functions
2. **Implement plugin architecture** - Make plot types more extensible
3. **Add more plot customization options** - Expose more matplotlib parameters
4. **Create example gallery** - Add visual examples for each plot type

## Additional Recommendations

### Documentation
- Add comprehensive API documentation using Sphinx
- Create user guide with examples for each plot type
- Document all statistical methods and assumptions
- Add type stubs for better IDE support

### Development Practices
- Set up pre-commit hooks for code formatting
- Add GitHub Actions for automated testing
- Implement code coverage reporting (aim for >80%)
- Add performance benchmarks for large datasets

### Dependencies
- Consider using `pydantic` for data validation
- Add `pytest-benchmark` for performance testing
- Consider `hypothesis` for property-based testing
- Pin all dependencies with specific versions

### Code Examples

Well-written sections worth preserving:
- Statistical significance marking in `stats.py`
- Grouped x-position calculations in `utils.py`
- Color enum implementation in `colors.py`

## Proposed Scalable Architecture 🏗️

### Overview

After analyzing all plot modules (`countplot.py`, `featureplot.py`, `cellcycleplot.py`, `combplot.py`), a scalable architecture has been designed that addresses the current issues while providing a foundation for future plot types.

### Current Issues Across All Plot Types

1. **Similar Function Signatures**: All plots have 15-20+ parameters
2. **Repeated Code**: Style loading, data filtering, figure setup in each module
3. **Inconsistencies**: Different approaches to grouping, validation, and error handling
4. **Monolithic Functions**: Single functions handle data processing, validation, and plotting

### Proposed Base Architecture

```python
# base.py - Foundation for all plot types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    path: Optional[Path] = None

    # Common display settings
    title: Optional[str] = None
    colors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for kwargs."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

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
                   selector_col: Optional[str] = None,
                   selector_val: Optional[str] = None) -> pd.DataFrame:
        """Common filtering logic with validation."""
        # Validation with proper error messages
        if condition_col not in self.df.columns:
            raise ValueError(f"Column '{condition_col}' not found")

        # Filter by conditions
        filtered = self.df[self.df[condition_col].isin(conditions)].copy()

        # Apply selector filter
        if selector_col and selector_val:
            if selector_col not in filtered.columns:
                raise ValueError(f"Column '{selector_col}' not found")
            if selector_val not in filtered[selector_col].unique():
                raise ValueError(f"Value '{selector_val}' not found in '{selector_col}'")
            filtered = filtered[filtered[selector_col] == selector_val]

        if filtered.empty:
            raise ValueError("No data after filtering")

        return filtered

    @abstractmethod
    def process_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process data for specific plot type."""
        pass

class BasePlotBuilder(ABC):
    """Base class for plot builders."""

    def __init__(self, config: BasePlotConfig):
        self.config = config
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

    def create_figure(self, axes: Optional[Axes] = None) -> 'BasePlotBuilder':
        """Create or use existing figure."""
        if axes:
            self.fig = axes.figure
            self.ax = axes
        else:
            from omero_screen_plots.utils import convert_size_to_inches
            fig_inches = convert_size_to_inches(
                self.config.fig_size,
                self.config.size_units
            )
            self.fig, self.ax = plt.subplots(figsize=fig_inches)
        return self

    @abstractmethod
    def build_plot(self, data: pd.DataFrame, **kwargs) -> 'BasePlotBuilder':
        """Build the specific plot type."""
        pass

    def set_title(self, title: Optional[str] = None) -> 'BasePlotBuilder':
        """Set plot title."""
        if title or self.config.title:
            self.ax.set_title(title or self.config.title)
        return self

    def save_figure(self, filename: str) -> 'BasePlotBuilder':
        """Save figure if configured."""
        if self.config.save and self.config.path:
            from omero_screen_plots.utils import save_fig
            save_fig(
                self.fig,
                self.config.path,
                filename,
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi
            )
        return self

    def build(self) -> tuple[Figure, Axes]:
        """Return completed figure and axes."""
        return self.fig, self.ax
```

### Specific Plot Type Implementations

#### Count Plot Architecture

```python
# countplot.py - Using base classes
@dataclass
class CountPlotConfig(BasePlotConfig):
    """Configuration specific to count plots."""
    plot_type: str = "normalised"  # or "absolute"
    group_size: int = 1
    within_group_spacing: float = 0.2
    between_group_gap: float = 0.5
    show_x_labels: bool = True
    rotation: int = 45

class CountDataProcessor(BaseDataProcessor):
    """Count plot specific data processing."""

    def validate_dataframe(self) -> None:
        required = ['plate_id', 'well', 'experiment']
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def process_data(self, df: pd.DataFrame, norm_control: str,
                    condition_col: str = "condition") -> pd.DataFrame:
        """Process data for count plots."""
        counts = self._calculate_counts(df, condition_col)
        return self._normalize_counts(counts, norm_control, condition_col)

class CountPlotBuilder(BasePlotBuilder):
    """Build count plots."""

    def build_plot(self, data: pd.DataFrame, conditions: list[str],
                  condition_col: str, value_col: str) -> 'CountPlotBuilder':
        """Build count plot with bars and points."""
        # Count-specific plotting logic
        return self
```

#### Feature Plot Architecture

```python
# featureplot.py - Using base classes
@dataclass
class FeaturePlotConfig(BasePlotConfig):
    """Configuration for feature plots."""
    plot_style: str = "box"  # or "violin"
    scale: bool = False
    show_points: bool = True
    group_size: int = 1

class FeatureDataProcessor(BaseDataProcessor):
    """Feature plot data processing."""

    def validate_dataframe(self) -> None:
        # Feature plot specific validation
        pass

    def process_data(self, df: pd.DataFrame, feature: str,
                    scale: bool = False) -> pd.DataFrame:
        """Process data for feature plots."""
        if scale:
            from omero_screen_plots.normalise import scale_data
            df[f"{feature}_scaled"] = scale_data(df[feature])
        return df
```

### Factory Pattern for Plot Creation

```python
# plot_factory.py
from typing import Type, Dict
from .base import BasePlotConfig, BaseDataProcessor, BasePlotBuilder

class PlotFactory:
    """Factory for creating different plot types."""

    _registry: Dict[str, tuple[Type[BasePlotConfig],
                              Type[BaseDataProcessor],
                              Type[BasePlotBuilder]]] = {}

    @classmethod
    def register(cls, plot_type: str,
                config_class: Type[BasePlotConfig],
                processor_class: Type[BaseDataProcessor],
                builder_class: Type[BasePlotBuilder]) -> None:
        """Register a new plot type."""
        cls._registry[plot_type] = (config_class, processor_class, builder_class)

    @classmethod
    def create_plot(cls, plot_type: str, df: pd.DataFrame,
                   config: Optional[dict] = None, **kwargs) -> tuple[Figure, Axes]:
        """Create a plot of the specified type."""
        if plot_type not in cls._registry:
            raise ValueError(f"Unknown plot type: {plot_type}")

        config_cls, processor_cls, builder_cls = cls._registry[plot_type]

        # Create config
        plot_config = config_cls(**(config or {}))

        # Process data
        processor = processor_cls(df)
        processed_data = processor.filter_data(**kwargs)
        final_data = processor.process_data(processed_data, **kwargs)

        # Build plot
        builder = builder_cls(plot_config)
        fig, ax = (builder
            .create_figure(kwargs.get('axes'))
            .build_plot(final_data, **kwargs)
            .set_title()
            .save_figure(f"{plot_type}_plot")
            .build()
        )

        return fig, ax

# Register plot types
PlotFactory.register('count', CountPlotConfig, CountDataProcessor, CountPlotBuilder)
PlotFactory.register('feature', FeaturePlotConfig, FeatureDataProcessor, FeaturePlotBuilder)
PlotFactory.register('cellcycle', CellCyclePlotConfig, CellCycleDataProcessor, CellCyclePlotBuilder)
```

### Simplified Public API

```python
# High-level API maintaining backward compatibility
def count_plot(df: pd.DataFrame, **kwargs) -> tuple[Figure, Axes]:
    """Create count plot with simplified API."""
    return PlotFactory.create_plot('count', df, **kwargs)

def feature_plot(df: pd.DataFrame, **kwargs) -> tuple[Figure, Axes]:
    """Create feature plot with simplified API."""
    return PlotFactory.create_plot('feature', df, **kwargs)
```

### Benefits of This Architecture

1. **Reusability**: Common functionality in base classes eliminates duplication
2. **Extensibility**: Easy to add new plot types by implementing base classes
3. **Consistency**: Enforced patterns across all plot types
4. **Maintainability**: Clear separation of concerns (data, config, plotting)
5. **Type Safety**: Strong typing throughout with dataclasses
6. **Backward Compatibility**: Existing APIs preserved through wrapper functions
7. **Testability**: Each component can be unit tested independently
8. **Configuration**: Centralized, type-safe configuration management

### Migration Strategy

1. **Phase 1**: Implement base classes and refactor `countplot.py`
2. **Phase 2**: Migrate `featureplot.py` to new architecture
3. **Phase 3**: Migrate `cellcycleplot.py` and `combplot.py`
4. **Phase 4**: Add factory pattern and unified API
5. **Phase 5**: Deprecate old function signatures (with warnings)

This architecture addresses all major code quality issues while providing a solid foundation for future development.

## Conclusion

The omero-screen-plots package provides valuable functionality for OMERO screen data visualization but requires significant improvements in testing, error handling, and code organization. The highest priority should be establishing a test suite and fixing security vulnerabilities. The proposed scalable architecture provides a clear path forward for modernizing the codebase while maintaining backward compatibility. With these improvements, the package would be more maintainable, reliable, and suitable for production use.
