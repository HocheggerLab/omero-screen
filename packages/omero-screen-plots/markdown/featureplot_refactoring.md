⏺ Update Todos
  ⎿  ☐ Create base configuration class (FeaturePlotConfig) extending BasePlotConfig
     ☐ Implement BaseFeaturePlot abstract class with common functionality
     ☐ Extract and refactor common data processing methods
     ☐ Implement StandardFeaturePlot class (boxenplot + swarmplot)
     ☐ Implement SimpleFeaturePlot class (violin/box plots)
     ☐ Implement ThresholdFeaturePlot class (stacked bar plots)
     ☐ Create FeaturePlotFactory for plot type selection
     ☐ Implement backward-compatible wrapper functions in featureplot_api.py
     ☐ Update imports and module structure
     ☐ Add comprehensive unit tests for new classes
     ☐ Update documentation and examples

⏺ Detailed Step-by-Step Refactoring Plan for Feature Plots

  Based on my analysis, here's a comprehensive plan for refactoring the feature plots to use the same class-based structure as count plots:

  Step 1: Create the Configuration Class

  First, create FeaturePlotConfig that extends BasePlotConfig:

  # In featureplot_config.py
  @dataclass
  class FeaturePlotConfig(BasePlotConfig):
      """Configuration for all feature plot types."""
      # Common feature plot settings
      scale: bool = False
      group_size: int = 1
      within_group_spacing: float = 0.2
      between_group_gap: float = 0.5
      show_x_labels: bool = True
      ymax: float | tuple[float, float] | None = None
      rotation: float = 45

      # Plot type selection
      plot_style: str = "standard"  # "standard", "simple", "threshold"

      # Style-specific settings
      violin: bool = False  # for simple plots
      threshold: float = 0.0  # for threshold plots
      legend: tuple[str, list[str]] | None = None  # for simple plots
      show_significance: bool = True
      show_repeat_points: bool = True

  Step 2: Create the Base Feature Plot Class

  Implement the base class with common functionality:

  # In featureplot_factory.py
  class BaseFeaturePlot(ABC):
      """Base class for all feature plot implementations."""

      def __init__(self, config: FeaturePlotConfig):
          self.config = config
          self.fig: Figure | None = None
          self.ax: Axes | None = None
          self._axes_provided = False
          self._filename: str | None = None

      def create_plot(
          self,
          df: pd.DataFrame,
          feature: str,
          conditions: list[str],
          condition_col: str = "condition",
          selector_col: str | None = None,
          selector_val: str | None = None,
          axes: Axes | None = None,
      ) -> tuple[Figure, Axes]:
          """Create the feature plot."""
          # Step 1: Validation
          self._validate_inputs(df, feature, conditions, condition_col)

          # Step 2: Data processing
          filtered_data = self._filter_data(
              df, conditions, condition_col, selector_col, selector_val
          )
          processed_data = self._process_data(
              filtered_data, feature, conditions, condition_col
          )

          # Step 3: Figure setup
          self._create_figure(axes)

          # Step 4: Build plot (delegated to subclass)
          self._build_plot(processed_data, feature, conditions, condition_col)

          # Step 5: Add statistics (if applicable)
          if self.config.group_size == 1:
              self._add_statistics(processed_data, feature, conditions, condition_col)

          # Step 6: Format and finalize
          self._format_axes(feature, conditions)
          self._finalize_plot(feature, selector_val)

          # Step 7: Save if configured
          self._save_figure()

          return self.fig, self.ax

  Step 3: Implement Plot-Specific Classes

  StandardFeaturePlot (replaces feature_plot):

  class StandardFeaturePlot(BaseFeaturePlot):
      """Standard feature plot with boxenplot and swarmplot overlay."""

      def _build_plot(self, data, feature, conditions, condition_col):
          if self.config.group_size > 1:
              self._build_grouped_plot(data, feature, conditions, condition_col)
          else:
              self._build_standard_plot(data, feature, conditions, condition_col)

  SimpleFeaturePlot (replaces feature_plot_simple):

  class SimpleFeaturePlot(BaseFeaturePlot):
      """Simple feature plot with violin or box plots."""

      def _build_plot(self, data, feature, conditions, condition_col):
          x_positions = self._get_x_positions(len(conditions))

          for idx, condition in enumerate(conditions):
              cond_data = data[data[condition_col] == condition]
              if not cond_data.empty:
                  if self.config.violin:
                      create_standard_violin(
                          self.ax, cond_data[feature].values,
                          x_positions[idx], color=self.config.colors[-1]
                      )
                  else:
                      create_standard_boxplot(
                          self.ax, cond_data[feature].values,
                          x_positions[idx], color=self.config.colors[-1]
                      )

  ThresholdFeaturePlot (replaces feature_threshold_plot):

  class ThresholdFeaturePlot(BaseFeaturePlot):
      """Threshold feature plot with stacked percentage bars."""

      def _build_plot(self, data, feature, conditions, condition_col):
          # Apply threshold
          data["threshold"] = np.where(
              data[feature] > self.config.threshold, "pos", "neg"
          )
          # Build stacked bar plot
          self._build_stacked_bars(data, conditions, condition_col)

  Step 4: Implement the Factory

  class FeaturePlotFactory:
      """Factory for creating feature plots."""

      _registry = {
          "standard": StandardFeaturePlot,
          "simple": SimpleFeaturePlot,
          "threshold": ThresholdFeaturePlot,
      }

      @classmethod
      def create_plot(
          cls,
          plot_style: str,
          config: FeaturePlotConfig | None = None,
      ) -> BaseFeaturePlot:
          """Create a feature plot of the specified style."""
          if plot_style not in cls._registry:
              raise ValueError(
                  f"Unknown plot style: {plot_style}. "
                  f"Available styles: {list(cls._registry.keys())}"
              )

          if config is None:
              config = FeaturePlotConfig(plot_style=plot_style)
          else:
              config.plot_style = plot_style

          return cls._registry[plot_style](config)

  Step 5: Create Backward-Compatible API

  Create featureplot_api.py with wrapper functions:

  def feature_plot(df, feature, conditions, **kwargs):
      """Backward-compatible wrapper for standard feature plot."""
      # Extract config parameters
      config_params = {
          k: v for k, v in kwargs.items()
          if k in FeaturePlotConfig.__annotations__
      }
      config = FeaturePlotConfig(plot_style="standard", **config_params)

      # Extract plot creation parameters
      plot_params = {
          'condition_col': kwargs.get('condition_col', 'condition'),
          'selector_col': kwargs.get('selector_col'),
          'selector_val': kwargs.get('selector_val'),
          'axes': kwargs.get('axes'),
      }

      plot = FeaturePlotFactory.create_plot("standard", config)
      return plot.create_plot(df, feature, conditions, **plot_params)

  Step 6: Migration Strategy

  1. Phase 1: Create new files alongside existing ones
    - featureplot_config.py
    - featureplot_factory.py
    - featureplot_api.py
  2. Phase 2: Gradually migrate functionality
    - Move common utilities to base class
    - Implement each plot type class
    - Test extensively
  3. Phase 3: Update imports
    - Change imports in notebooks/scripts to use new API
    - Keep old functions available but mark as deprecated

  Key Benefits of This Approach

  1. Consistency: Same architecture as count plots
  2. Maintainability: Common code in one place
  3. Extensibility: Easy to add new plot types
  4. Type Safety: Configuration validation
  5. Backward Compatibility: Existing code continues to work

  Implementation Order

  1. Start with configuration and base class
  2. Implement StandardFeaturePlot first (most complex)
  3. Add SimpleFeaturePlot and ThresholdFeaturePlot
  4. Create factory and API wrappers
  5. Add comprehensive tests
  6. Update documentation

  This refactoring will significantly improve the codebase while maintaining all existing functionality and allowing for future extensions.
