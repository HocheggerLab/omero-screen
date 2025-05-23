"""This module provides functions for visualizing quality control data.

It includes utilities to plot the median intensity values for each image position and channel,
with error bars representing the standard deviation, to help assess the consistency and quality of image data.

Typical usage example:
    import pandas as pd
    from omero_screen.quality_control import quality_control_fig

    # df should be a pandas DataFrame with columns: 'position', 'channel', 'intensity_median'
    fig = quality_control_fig(df)
    fig.show()
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def quality_control_fig(df: pd.DataFrame) -> Figure:
    """Plot the quality control data for each image position and channel.

    Args:
        df: Quality control data

    Returns:
        Quality control figure
    """
    df["position"] = df["position"].astype("category")
    medians = (
        df.groupby(["position", "channel"], observed=False)["intensity_median"]
        .mean()
        .reset_index()
    )
    std = (
        df.groupby(["position", "channel"], observed=False)["intensity_median"]
        .std()
        .reset_index()
    )
    channel_num = len(df.channel.unique())
    well_num = len(df.position.unique())
    # Plotting the results
    fig, ax = plt.subplots(nrows=channel_num, figsize=(well_num, channel_num))
    # Ensure ax is always a list of Axes, even when there's only one subplot
    if channel_num == 1:
        ax = [ax]
    for i, channel in enumerate(df.channel.unique()):
        channel_df = medians[medians["channel"] == channel]
        channel_std = std.loc[std["channel"] == channel, "intensity_median"]
        # std is nan if there is 1 image per (position, channel) group
        channel_std[pd.isna(channel_std)] = 0
        print(channel_std)
        y_min = (channel_df["intensity_median"] - channel_std).min()
        y_max = (channel_df["intensity_median"] + channel_std).max()
        padding = max(
            (y_max - y_min) * 0.1, 1
        )  # 10% padding; avoid zero padding
        ax[i].errorbar(
            channel_df["position"],
            channel_df["intensity_median"],
            yerr=channel_std,
            fmt="o",
        )
        ax[i].set_title(channel)
        ax[i].set_xticks(range(len(channel_df["position"])))
        ax[i].set_xticklabels(channel_df["position"])
        ax[i].set_xlim(-0.5, len(channel_df["position"]) - 0.5)
        ax[i].set_ylim(y_min - padding, y_max + padding)
    plt.tight_layout()
    return fig
