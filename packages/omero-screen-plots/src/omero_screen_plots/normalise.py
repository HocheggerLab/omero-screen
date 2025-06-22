"""Concise mode normalization for immunofluorescence data.

Simple, production-ready functions for normalizing IF intensity data
by setting the peak (mode) to 1.0.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def find_intensity_mode(data: pd.Series[float], n_bins: int = 5000) -> float:
    """Find the mode (peak) of intensity data using histogram analysis.

    Args:
        data: Intensity measurements
        n_bins: Number of histogram bins

    Returns:
        Mode value (intensity at peak)
    """
    # Clean data - remove NaN, infinite, and non-positive values
    clean_data = data.dropna()
    clean_data = clean_data[np.isfinite(clean_data) & (clean_data > 0)]

    if len(clean_data) < 100:
        raise ValueError(f"Insufficient data points: {len(clean_data)} < 100")

    # Create histogram using robust range (excludes extreme outliers)
    data_min, data_max = np.percentile(clean_data, [0.5, 99.5])
    counts, bin_edges = np.histogram(
        clean_data, bins=n_bins, range=(data_min, data_max)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth to reduce noise and find peak
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=1.0)
    max_idx = np.argmax(smoothed_counts)
    mode_value = bin_centers[max_idx]

    return float(mode_value)


def normalize_by_mode(
    df: pd.DataFrame,
    intensity_column: str,
    group_column: Optional[str] = None,
    suffix: str = "_norm",
) -> pd.DataFrame:
    """Normalize intensity column by setting the mode (peak) to 1.0.

    Args:
        df: Input dataframe
        intensity_column: Column name to normalize
        group_column: Optional column to group by (e.g., 'plate_id', 'cell_line')
        suffix: Suffix for new normalized column

    Returns:
        DataFrame with additional normalized column

    Example:
        # Normalize p21 intensity by plate
        df_norm = normalize_by_mode(df, 'intensity_mean_p21_nucleus', 'plate_id')

        # Normalize without grouping
        df_norm = normalize_by_mode(df, 'intensity_mean_p21_nucleus')
    """
    df_result = df.copy()
    normalized_column = f"{intensity_column}{suffix}"

    # Initialize normalized column
    df_result[normalized_column] = np.nan

    if group_column is not None:
        # Normalize within each group
        for group_value in df[group_column].unique():
            group_mask = df[group_column] == group_value
            group_data = df.loc[group_mask, intensity_column]

            try:
                mode_value = find_intensity_mode(group_data)  # type: ignore[arg-type]

                # Normalize: divide by mode so peak becomes 1.0
                valid_mask = (
                    group_mask
                    & np.isfinite(df[intensity_column])
                    & (df[intensity_column] > 0)
                )
                df_result.loc[valid_mask, normalized_column] = (
                    df.loc[valid_mask, intensity_column] / mode_value  # type: ignore[operator]
                )

                print(f"Group {group_value}: mode = {mode_value:.0f}")

            except ValueError as e:
                warnings.warn(
                    f"Normalization failed for {group_column}={group_value}: {e}",
                    stacklevel=2,
                )
                # Use raw values as fallback
                df_result.loc[group_mask, normalized_column] = df.loc[
                    group_mask, intensity_column
                ]

    else:
        # Normalize entire dataset
        try:
            mode_value = find_intensity_mode(df[intensity_column])  # type: ignore[arg-type]

            # Normalize: divide by mode so peak becomes 1.0
            valid_mask = np.isfinite(df[intensity_column]) & (
                df[intensity_column] > 0
            )
            df_result.loc[valid_mask, normalized_column] = (
                df.loc[valid_mask, intensity_column] / mode_value  # type: ignore[operator]
            )

            print(f"Dataset mode = {mode_value:.0f}")

        except ValueError as e:
            warnings.warn(f"Normalization failed: {e}", stacklevel=2)
            # Use raw values as fallback
            df_result[normalized_column] = df[intensity_column]

    return df_result


def plot_normalization_result(
    df: pd.DataFrame,
    original_column: str,
    normalized_column: str,
    group_column: Optional[str] = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Plot before and after normalization distributions.

    Args:
        df: DataFrame with both original and normalized columns
        original_column: Name of original intensity column
        normalized_column: Name of normalized intensity column
        group_column: Optional grouping column for separate plots
        figsize: Figure size tuple

    Example:
        plot_normalization_result(df, 'intensity_mean_p21_nucleus',
                                 'intensity_mean_p21_nucleus_norm', 'plate_id')
    """
    if group_column is not None:
        # Plot by group
        groups = df[group_column].unique()
        n_groups = len(groups)

        fig, axes = plt.subplots(2, n_groups, figsize=(5 * n_groups, 8))
        if n_groups == 1:
            axes = axes.reshape(2, 1)

        colors = ["blue", "red", "green", "orange", "purple"]

        for i, group_value in enumerate(groups):
            group_data = df[df[group_column] == group_value]
            color = colors[i % len(colors)]

            # Original distribution
            clean_original = group_data[original_column].dropna()
            clean_original = clean_original[clean_original > 0]
            if len(clean_original) > 0:
                hist_range = np.percentile(clean_original, [1, 99])
                axes[0, i].hist(
                    clean_original,
                    bins=50,
                    alpha=0.7,
                    color=color,
                    density=True,
                    range=hist_range,
                )
                axes[0, i].set_title(f"Original - {group_value}")
                axes[0, i].set_xlabel("Raw Intensity")
                axes[0, i].set_ylabel("Density")

            # Normalized distribution
            clean_normalized = group_data[normalized_column].dropna()
            clean_normalized = clean_normalized[clean_normalized > 0]
            if len(clean_normalized) > 0:
                axes[1, i].hist(
                    clean_normalized,
                    bins=50,
                    alpha=0.7,
                    color=color,
                    density=True,
                    range=(0, 4),
                )
                axes[1, i].axvline(
                    1.0,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label="Mode = 1.0",
                )
                axes[1, i].axvline(
                    1.5,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label="Threshold?",
                )
                axes[1, i].set_title(f"Normalized - {group_value}")
                axes[1, i].set_xlabel("Normalized Intensity")
                axes[1, i].set_ylabel("Density")
                if i == 0:  # Only show legend on first plot
                    axes[1, i].legend()

                # Add statistics
                mean_norm = clean_normalized.mean()
                median_norm = clean_normalized.median()
                axes[1, i].text(
                    0.02,
                    0.98,
                    f"Mean: {mean_norm:.2f}\\nMedian: {median_norm:.2f}",
                    transform=axes[1, i].transAxes,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "lightyellow",
                        "alpha": 0.8,
                    },
                )

    else:
        # Single plot for entire dataset
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Original distribution
        clean_original = df[original_column].dropna()
        clean_original = clean_original[clean_original > 0]
        if len(clean_original) > 0:
            hist_range = np.percentile(clean_original, [1, 99])
            axes[0].hist(
                clean_original,
                bins=50,
                alpha=0.7,
                color="skyblue",
                density=True,
                range=hist_range,
            )
            axes[0].set_title("Original Distribution")
            axes[0].set_xlabel("Raw Intensity")
            axes[0].set_ylabel("Density")

        # Normalized distribution
        clean_normalized = df[normalized_column].dropna()
        clean_normalized = clean_normalized[clean_normalized > 0]
        if len(clean_normalized) > 0:
            axes[1].hist(
                clean_normalized,
                bins=50,
                alpha=0.7,
                color="lightcoral",
                density=True,
                range=(0, 4),
            )
            axes[1].axvline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Mode = 1.0",
            )
            axes[1].axvline(
                1.5,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Suggested Threshold",
            )
            axes[1].set_title("Normalized Distribution")
            axes[1].set_xlabel("Normalized Intensity (Mode = 1.0)")
            axes[1].set_ylabel("Density")
            axes[1].legend()

            # Add statistics
            mean_norm = clean_normalized.mean()
            median_norm = clean_normalized.median()
            pos_pct = np.mean(clean_normalized > 1.5) * 100
            axes[1].text(
                0.02,
                0.98,
                f"Mean: {mean_norm:.2f}\\nMedian: {median_norm:.2f}\\n>1.5: {pos_pct:.1f}%",
                transform=axes[1].transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "lightyellow",
                    "alpha": 0.8,
                },
            )

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\\n=== NORMALIZATION SUMMARY ===")

    if group_column is not None:
        for group_value in df[group_column].unique():
            group_data = df[df[group_column] == group_value]
            clean_norm = group_data[normalized_column].dropna()
            clean_norm = clean_norm[clean_norm > 0]
            if len(clean_norm) > 0:
                pos_pct = np.mean(clean_norm > 1.5) * 100
                print(f"{group_value}: {pos_pct:.1f}% cells above 1.5x mode")
    else:
        clean_norm = df[normalized_column].dropna()
        clean_norm = clean_norm[clean_norm > 0]
        if len(clean_norm) > 0:
            pos_pct = np.mean(clean_norm > 1.5) * 100
            print(f"Overall: {pos_pct:.1f}% cells above 1.5x mode")

    print("Suggested threshold: 1.5 (50% above background)")


# Convenience function for complete workflow
def normalize_and_plot(
    df: pd.DataFrame,
    intensity_column: str,
    group_column: Optional[str] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """Complete workflow: normalize by mode and optionally plot results.

    Args:
        df: Input dataframe
        intensity_column: Column to normalize
        group_column: Optional grouping column
        plot: Whether to generate plots

    Returns:
        DataFrame with normalized column added

    Example:
        # Complete workflow in one line
        df_norm = normalize_and_plot(df, 'intensity_mean_p21_nucleus', 'plate_id')
    """
    # Normalize
    df_normalized = normalize_by_mode(df, intensity_column, group_column)
    normalized_column = f"{intensity_column}_norm"

    # Plot if requested
    if plot:
        plot_normalization_result(
            df_normalized, intensity_column, normalized_column, group_column
        )

    print(f"\\n✅ Added column: '{normalized_column}'")
    print("🎯 Mode is now 1.0, try threshold = 1.5 for positive cells")

    return df_normalized


def set_threshold_categories(
    df: pd.DataFrame, norm_feat: str, threshold: float = 1.5
) -> pd.DataFrame:
    """Set threshold categories for a normalized feature.

    Args:
        df: Input dataframe
        norm_feat: Name of normalized feature
        threshold: Threshold value

    Returns:
        DataFrame with threshold categories added
    """
    feature = norm_feat.split("_")[2]
    df[f"{feature} %"] = np.where(
        df[norm_feat] > threshold, f"{feature}+", f"{feature}-"
    )
    return df
