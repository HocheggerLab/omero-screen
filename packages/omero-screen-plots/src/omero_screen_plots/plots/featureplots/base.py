"""Base class for feature plots.

This module provides the base functionality shared across all feature plot types,
including data processing, feature scaling, and common validation.
"""

from abc import abstractmethod
from typing import Any, Optional, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...base import OmeroPlots
from ...utils import scale_data, selector_val_filter


class BaseFeaturePlot(OmeroPlots):
    """Base class for all feature plot types.

    Provides common functionality for feature data processing including:
    - Data validation for required columns
    - Feature data filtering and scaling
    - Common plotting utilities for feature data
    - Statistical analysis support
    """

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        feature: str,
        scale: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize base feature plot.

        Args:
            data: DataFrame containing feature data
            conditions: List of conditions to plot
            feature: Column name of the feature to plot
            scale: Whether to scale the feature data
            **kwargs: Additional arguments passed to base OmeroPlots class
        """
        super().__init__(data, conditions, **kwargs)

        self.feature = feature
        self.scale = scale

        # Validate required columns
        self._validate_feature_data()

        # Process the data if scaling is requested
        if self.scale:
            self.data = self._scale_feature_data()

    def _validate_feature_data(self) -> None:
        """Validate that data contains required columns for feature analysis."""
        if self.feature not in self.data.columns:
            raise ValueError(
                f"Feature column '{self.feature}' not found in data"
            )

        # Check if we have any feature data
        if self.data[self.feature].isna().all():
            raise ValueError(
                f"No data found for feature '{self.feature}' (all values are NaN)"
            )

    def _scale_feature_data(self) -> pd.DataFrame:
        """Scale feature data if requested.

        Returns:
            DataFrame with scaled feature data
        """
        return scale_data(
            self.data,
            self.feature,
        )

    def get_feature_data(self, condition: str) -> pd.DataFrame:
        """Get feature data for a specific condition.

        Args:
            condition: Condition to filter for

        Returns:
            DataFrame containing only data for the specified condition
        """
        condition_data = self.data[
            self.data[self.condition_col] == condition
        ].copy()

        # Apply selector filter if specified
        if self.selector_col and self.selector_val:
            filtered_data = selector_val_filter(
                condition_data,
                self.selector_col,
                self.selector_val,
                self.condition_col,
                [condition],
            )
            if filtered_data is not None:
                condition_data = filtered_data

        return condition_data

    def get_feature_range(self) -> tuple[float, float]:
        """Calculate appropriate range for feature axis.

        Returns:
            Tuple of (min, max) values for feature axis
        """
        feature_min = self.data[self.feature].quantile(0.01)
        feature_max = self.data[self.feature].quantile(0.99)

        # Add some padding
        padding = (feature_max - feature_min) * 0.1
        return feature_min - padding, feature_max + padding

    def format_feature_label(self) -> str:
        """Format feature name for axis label.

        Returns:
            Formatted feature label
        """
        # Handle common feature naming patterns
        label = self.feature.replace("_", " ")

        # Check for normalized features
        if "norm" in self.feature:
            label = label.replace(" norm", " (normalized)")

        # Check for intensity features
        if "intensity" in self.feature.lower():
            label = label.replace("intensity", "int.")

        # Check for integrated features
        if "integrated" in self.feature.lower():
            label = label.replace("integrated", "integ.")

        # Capitalize appropriately
        return label.title()

    def setup_feature_axis(
        self,
        ax: Axes,
        ylabel: Optional[str] = None,
        ylim: Optional[Union[float, tuple[float, float]]] = None,
    ) -> None:
        """Setup common feature axis configuration.

        Args:
            ax: Matplotlib axis to configure
            ylabel: Y-axis label. If None, uses formatted feature name
            ylim: Y-axis limits. Can be a single value (max) or tuple (min, max)
        """
        # Set y-label
        if ylabel is None:
            ylabel = self.format_feature_label()
        ax.set_ylabel(ylabel, fontsize=6)

        # Set y-limits
        if ylim is None:
            # Auto-calculate limits
            y_min, y_max = self.get_feature_range()
            ax.set_ylim(y_min, y_max)

        elif isinstance(ylim, int | float):
            ax.set_ylim(0, ylim)
        else:
            ax.set_ylim(ylim)
        # General styling
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.grid(False)

    @property
    def plot_type(self) -> str:
        """Return base plot type. Should be overridden by subclasses."""
        return "feature_base"

    @abstractmethod
    def generate(self) -> Figure:
        """Generate the plot. Must be implemented by subclasses."""

    def get_grouped_positions(
        self,
        group_size: int = 2,
        within_group_spacing: float = 0.5,
        between_group_gap: float = 1.0,
    ) -> list[float]:
        """Calculate x-positions for grouped plotting.

        This helper simply forwards to
        :pyfunc:`omero_screen_plots.utils.grouped_x_positions`, exposing the
        extra *spacing* arguments so that subclasses (e.g. grouped plots) can
        fine-tune their layout while maintaining sensible defaults.

        Args:
            group_size: Number of conditions per visual group.
            within_group_spacing: Horizontal spacing between neighbouring
                conditions inside a group.
            between_group_gap: Extra spacing that separates two consecutive
                groups.

        Returns:
            A list with the resolved x-axis positions for every condition in
            ``self.conditions``.
        """
        from ...utils import grouped_x_positions

        return grouped_x_positions(
            len(self.conditions),
            group_size=group_size,
            within_group_spacing=within_group_spacing,
            between_group_gap=between_group_gap,
        )

    def setup_grouped_x_axis(
        self,
        ax: Axes,
        x_positions: list[float],
        show_labels: bool = True,
    ) -> None:
        """Setup x-axis for grouped plots.

        Args:
            ax: Matplotlib axis
            x_positions: X-positions for each condition
            show_labels: Whether to show x-axis labels
        """
        ax.set_xticks(x_positions)

        if show_labels:
            # Rotate labels if they're long
            max_label_length = max(len(str(cond)) for cond in self.conditions)
            rotation = 45 if max_label_length > 6 else 0

            ax.set_xticklabels(
                self.conditions,
                rotation=rotation,
                ha="right" if rotation else "center",
            )
        else:
            ax.set_xticklabels([])

        # Set x-limits with some padding
        x_padding = 0.5
        ax.set_xlim(min(x_positions) - x_padding, max(x_positions) + x_padding)

    def calculate_threshold_percentages(
        self, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Calculate percentages above/below threshold for each condition.

        Args:
            threshold: Threshold value for categorization

        Returns:
            DataFrame with columns: condition, category (pos/neg), percentage
        """
        results = []

        for condition in self.conditions:
            condition_data = self.get_feature_data(condition)

            if not condition_data.empty:
                total_cells = len(condition_data)
                above_threshold = (
                    condition_data[self.feature] > threshold
                ).sum()
                below_threshold = total_cells - above_threshold

                # Calculate percentages
                results.append(
                    {
                        self.condition_col: condition,
                        "category": "pos",
                        "percentage": (above_threshold / total_cells) * 100,
                        "count": above_threshold,
                    }
                )
                results.append(
                    {
                        self.condition_col: condition,
                        "category": "neg",
                        "percentage": (below_threshold / total_cells) * 100,
                        "count": below_threshold,
                    }
                )

        return pd.DataFrame(results)

    def get_threshold_label(self, threshold: float = 0.0) -> str:
        """Get a formatted label describing the threshold.

        Args:
            threshold: Threshold value for categorization

        Returns:
            Formatted threshold description
        """
        feature_label = self.format_feature_label()
        if threshold == 0:
            return f"{feature_label} > 0"
        else:
            return f"{feature_label} > {threshold:.2f}"
