"""Base class for cell cycle plots.

This module provides the base functionality shared across all cell cycle plot types,
including data processing, percentage calculations, and common validation.
"""

import warnings
from abc import abstractmethod
from typing import Any, Optional

import pandas as pd
from matplotlib.figure import Figure

from ...base import OmeroPlots


class BaseCellCyclePlot(OmeroPlots):
    """Base class for all cell cycle plot types.

    Provides common functionality for cell cycle data processing including:
    - Data validation for required columns
    - Cell cycle percentage calculations
    - Phase ordering and filtering
    - Statistical analysis for cell cycle data
    """

    def __init__(
        self,
        data: pd.DataFrame,
        conditions: list[str],
        phases: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base cell cycle plot.

        Args:
            data: DataFrame containing cell cycle data with required columns:
                  - 'cell_cycle': Cell cycle phase for each cell
                  - 'plate_id': Plate/replicate identifier
                  - 'experiment': Unique cell identifier
            conditions: List of conditions to plot
            phases: List of cell cycle phases to include. If None, uses default phases
            **kwargs: Additional arguments passed to base OmeroPlots class
        """
        super().__init__(data, conditions, **kwargs)

        # Validate required columns
        self._validate_cellcycle_data()

        # Set default phases if not provided
        self.phases = phases or self._get_default_phases()

        # Process the data to calculate percentages
        self.processed_data = self._calculate_cell_cycle_percentages()

    def _validate_cellcycle_data(self) -> None:
        """Validate that data contains required columns for cell cycle analysis."""
        required_columns = ["cell_cycle", "plate_id", "experiment"]
        if missing_columns := [
            col for col in required_columns if col not in self.data.columns
        ]:
            raise ValueError(
                f"Data missing required columns for cell cycle analysis: {missing_columns}"
            )

        # Check if we have any cell cycle data
        if self.data["cell_cycle"].isna().all():
            raise ValueError("No cell cycle data found (all values are NaN)")

    def _get_default_phases(self) -> list[str]:
        """Get default cell cycle phases.

        Can be overridden by subclasses for different phase sets.
        """
        return ["G1", "S", "G2/M", "Polyploid"]

    def _calculate_cell_cycle_percentages(self) -> pd.DataFrame:
        """Calculate the percentage of cells in each cell cycle phase.

        Groups by plate_id, cell_line (if present), condition, and cell_cycle phase
        to calculate what percentage of cells are in each phase for each experimental condition.

        Returns:
            DataFrame with columns: plate_id, cell_line (if present), condition,
                                   cell_cycle, percent
        """
        # Define grouping columns based on what's available in the data
        grouping_cols = ["plate_id", self.condition_col, "cell_cycle"]
        if "cell_line" in self.data.columns:
            grouping_cols.insert(-1, "cell_line")  # Insert before cell_cycle

        # Calculate total cells per experimental unit (excluding cell_cycle from denominator)
        denominator_cols = [
            col for col in grouping_cols if col != "cell_cycle"
        ]

        # Count cells in each phase
        phase_counts = (
            self.data.groupby(grouping_cols)["experiment"]
            .count()
            .reset_index()
            .rename(columns={"experiment": "phase_count"})
        )

        # Count total cells per experimental unit
        total_counts = (
            self.data.groupby(denominator_cols)["experiment"]
            .count()
            .reset_index()
            .rename(columns={"experiment": "total_count"})
        )

        # Merge and calculate percentages
        merged = phase_counts.merge(total_counts, on=denominator_cols)
        merged["percent"] = (
            merged["phase_count"] / merged["total_count"]
        ) * 100

        return merged.drop(columns=["phase_count", "total_count"])

    def get_phase_data(self, phase: str) -> pd.DataFrame:
        """Get data for a specific cell cycle phase.

        Args:
            phase: Cell cycle phase to filter for

        Returns:
            DataFrame containing only data for the specified phase
        """
        return self.processed_data[
            (self.processed_data.cell_cycle == phase)
            & (self.processed_data[self.condition_col].isin(self.conditions))
        ].copy()

    def get_mean_percentages(self) -> pd.DataFrame:
        """Calculate mean percentages across replicates for each condition and phase.

        Returns:
            DataFrame with conditions as index and phases as columns
        """
        # Group by condition and phase, calculate mean
        mean_data = (
            self.processed_data[
                self.processed_data[self.condition_col].isin(self.conditions)
            ]
            .groupby([self.condition_col, "cell_cycle"])["percent"]
            .mean()
            .reset_index()
            .pivot_table(
                columns=["cell_cycle"],
                index=[self.condition_col],
                values="percent",
                fill_value=0,
            )
        )

        return self._extracted_from_get_std_percentages_24(mean_data)

    def get_std_percentages(self) -> pd.DataFrame:
        """Calculate standard deviation of percentages across replicates.

        Returns:
            DataFrame with conditions as index and phases as columns
        """
        if self.processed_data.plate_id.nunique() <= 1:
            # If only one replicate, return zeros
            mean_data = self.get_mean_percentages()
            return pd.DataFrame(
                0, index=mean_data.index, columns=mean_data.columns
            )

        std_data = (
            self.processed_data[
                self.processed_data[self.condition_col].isin(self.conditions)
            ]
            .groupby([self.condition_col, "cell_cycle"])["percent"]
            .std()
            .reset_index()
            .pivot_table(
                columns=["cell_cycle"],
                index=[self.condition_col],
                values="percent",
                fill_value=0,
            )
        )

        return self._extracted_from_get_std_percentages_24(std_data)

    # TODO Rename this here and in `get_mean_percentages` and `get_std_percentages`
    def _extracted_from_get_std_percentages_24(
        self, arg0: pd.DataFrame
    ) -> pd.DataFrame:
        for phase in self.phases:
            if phase not in arg0.columns:
                arg0[phase] = 0
        available_phases = [p for p in self.phases if p in arg0.columns]
        arg0 = arg0[available_phases]
        return arg0

    def has_sufficient_replicates(self, min_replicates: int = 3) -> bool:
        """Check if there are sufficient replicates for statistical analysis.

        Args:
            min_replicates: Minimum number of replicates required

        Returns:
            True if sufficient replicates are available
        """
        return int(self.processed_data.plate_id.nunique()) >= min_replicates

    def add_significance_markers_to_axis(
        self, ax: Any, phase_data: pd.DataFrame, column: str = "percent"
    ) -> None:
        """Add significance markers to a specific axis for cell cycle data.

        Args:
            ax: Matplotlib axis to add markers to
            phase_data: DataFrame containing data for a specific phase
            column: Column name to use for statistical testing
        """
        try:
            if not self.has_sufficient_replicates():
                return

            # Calculate statistics using the phase-specific data
            from ...stats import calculate_pvalues, get_significance_marker

            pvalues = calculate_pvalues(
                phase_data, self.conditions, self.condition_col, column
            )
            y_max = ax.get_ylim()[1]

            for i, _condition in enumerate(self.conditions[1:], start=1):
                if i - 1 < len(pvalues):
                    p_value = pvalues[i - 1]
                    significance = get_significance_marker(p_value)

                    # Position the significance marker
                    x_pos = i
                    y_pos = y_max * 1.05

                    ax.text(
                        x_pos,
                        y_pos,
                        significance,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        except (ValueError, KeyError, IndexError) as e:
            # Skip significance markers if calculation fails
            warnings.warn(
                f"Could not calculate significance markers: {e}", stacklevel=2
            )

    @property
    def plot_type(self) -> str:
        """Return base plot type. Should be overridden by subclasses."""
        return "cellcycle_base"

    @abstractmethod
    def generate(self) -> Figure:
        """Generate the plot. Must be implemented by subclasses."""
