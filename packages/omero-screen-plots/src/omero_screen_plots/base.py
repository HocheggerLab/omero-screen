"""Base classes for omero-screen-plots scalable architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omero_screen_plots.utils import convert_size_to_inches, save_fig


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for kwargs."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseDataProcessor(ABC):
    """Base class for data processing."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the data processor."""
        self.df = df
        self.validate_dataframe()

    @abstractmethod
    def validate_dataframe(self) -> None:
        """Validate required columns exist."""

    def filter_data(
        self,
        condition_col: str,
        conditions: list[str],
        selector_col: str | None = None,
        selector_val: str | None = None,
    ) -> pd.DataFrame:
        """Common filtering logic with validation."""
        # Validation with proper error messages
        if condition_col not in self.df.columns:
            raise ValueError(
                f"Column '{condition_col}' not found in dataframe"
            )

        # Filter by conditions
        filtered = self.df[self.df[condition_col].isin(conditions)].copy()

        # Apply selector filter if provided
        if selector_col and selector_val:
            if selector_col not in filtered.columns:
                raise ValueError(
                    f"Column '{selector_col}' not found in dataframe"
                )
            if selector_val not in filtered[selector_col].unique():
                raise ValueError(
                    f"Value '{selector_val}' not found in column '{selector_col}'"
                )
            filtered = filtered[filtered[selector_col] == selector_val]
        elif selector_col:
            raise ValueError(
                f"selector_val for {selector_col} must be provided"
            )

        if filtered.empty:
            raise ValueError("No data remaining after filtering")

        return filtered

    @abstractmethod
    def process_data(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Process data for specific plot type."""


class BasePlotBuilder(ABC):
    """Base class for plot builders."""

    def __init__(self, config: BasePlotConfig):
        """Initialize the plot builder."""
        self.config = config
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.axes_provided: bool = False
        self._filename: str | None = None

    def create_figure(self, axes: Axes | None = None) -> "BasePlotBuilder":
        """Create or use existing figure."""
        if axes:
            self.fig = cast(
                Figure, axes.figure
            )  # Cast SubFigure to Figure for our use case
            self.ax = axes
            self.axes_provided = True
        else:
            fig_inches = convert_size_to_inches(
                self.config.fig_size, self.config.size_units
            )
            self.fig, self.ax = plt.subplots(figsize=fig_inches)
            self.axes_provided = False
        return self

    @abstractmethod
    def build_plot(
        self, data: pd.DataFrame, **kwargs: Any
    ) -> "BasePlotBuilder":
        """Build the specific plot type."""

    def finalize_plot(
        self, default_title: str | None = None
    ) -> "BasePlotBuilder":
        """Finalize plot with title and store filename.

        Args:
            default_title: Default title to use if none provided
        """
        # Use provided title, config title, or default
        title = self.config.title or default_title

        # Use finalize_plot_with_title utility for consistent formatting
        from omero_screen_plots.utils import finalize_plot_with_title

        if self.fig is not None:
            self._filename = finalize_plot_with_title(
                self.fig,
                title,
                default_title or "plot",  # fallback feature name
                self.axes_provided,
            )
        return self

    def save_figure(self, filename: str | None = None) -> "BasePlotBuilder":
        """Save figure if configured."""
        if self.config.save and self.config.path and self.fig is not None:
            # Use filename from finalize_plot if not provided
            final_filename = filename or self._filename or "plot"
            save_fig(
                self.fig,
                self.config.path,
                final_filename,
                tight_layout=self.config.tight_layout,
                fig_extension=self.config.file_format,
                resolution=self.config.dpi,
            )
        return self

    def build(self) -> tuple[Figure, Axes]:
        """Return completed figure and axes."""
        if self.fig is None or self.ax is None:
            raise RuntimeError(
                "Figure and axes must be created before calling build()"
            )
        return self.fig, self.ax
