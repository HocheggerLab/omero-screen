"""Per-well QC plot composer for the OMERO-Screen pipeline.

``well_qc_plot`` produces a compact figure with a DNA-content histogram, a
DNA vs EdU scatter coloured by cell-cycle phase, and a stacked cell-cycle
barplot. It is the post-refactor replacement for the local ``combplot``
that used to live in ``omero_screen.cellcycle_analysis`` and is attached
to each well in OMERO by the pipeline.

Use ``combplot_cellcycle`` instead when you want a multi-condition
comparison figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from omero_screen_plots.cellcycleplot_api import cellcycle_stacked
from omero_screen_plots.histogramplot_api import histogram_plot
from omero_screen_plots.scatterplot_api import scatter_plot
from omero_screen_plots.utils import find_dna_norm_column, save_fig


def well_qc_plot(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    dna_norm_col: str | None = None,
    fig_size: tuple[float, float] = (12, 7),
    size_units: str = "cm",
    save: bool = False,
    path: Path | None = None,
    file_format: str = "png",
    dpi: int = 300,
) -> Figure:
    """Build a compact per-well QC figure (histogram + scatter + cell-cycle bar).

    Layout — 2×2 ``GridSpec`` matching the legacy per-well QC plot:

    - Top-left: DNA-content histogram (log₂ x-axis).
    - Bottom-left: DNA vs EdU scatter coloured by cell-cycle phase, with
      KDE overlay and threshold lines.
    - Right (spans both rows): stacked cell-cycle phase barplot.

    The caller is expected to pass a DataFrame that contains a single well's
    cells already filtered down (or trust the resolver / the underlying
    primitives to filter via the ``well`` column). The figure does not
    sample cells — the histogram and scatter use all rows passed in.

    Args:
        df: Single-cell DataFrame with cell-cycle columns
            (``cell_cycle``, ``intensity_mean_EdU_nucleus_norm``, and the
            normalised DNA-content column).
        title: Figure-level suptitle. If ``None``, no title is drawn.
        dna_norm_col: Normalised DNA-content column to plot on the x-axis.
            Auto-resolved via :func:`find_dna_norm_column` when ``None``.
        fig_size: Figure size in ``size_units`` (default: cm).
        size_units: ``"cm"`` (default) or ``"inches"``.
        save: Save the figure to ``path``.
        path: Directory to save the figure into when ``save=True``.
        file_format: Save format.
        dpi: Save resolution.

    Returns:
        The composed matplotlib Figure.
    """
    if dna_norm_col is None:
        dna_norm_col = find_dna_norm_column(df)

    if size_units == "cm":
        fig_size_in = (fig_size[0] / 2.54, fig_size[1] / 2.54)
    else:
        fig_size_in = fig_size

    fig = plt.figure(figsize=fig_size_in)
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1.2])

    # All cells in df share the same well (caller's responsibility). The
    # primitives still want a condition_col + conditions pair though — use a
    # synthetic ``_well_qc`` constant so the filter is a no-op.
    df = df.copy()
    df["_well_qc"] = "all"

    ax_hist = fig.add_subplot(gs[0, 0])
    histogram_plot(
        df=df,
        feature=dna_norm_col,
        conditions="all",
        condition_col="_well_qc",
        axes=ax_hist,
        log_scale=True,
        log_base=2,
        x_limits=(1, 16),
    )
    ax_hist.set_xlabel("")
    ax_hist.xaxis.set_visible(False)

    ax_scatter = fig.add_subplot(gs[1, 0])
    scatter_plot(
        df=df,
        conditions="all",
        condition_col="_well_qc",
        x_feature=dna_norm_col,
        y_feature="intensity_mean_EdU_nucleus_norm",
        hue="cell_cycle",
        x_scale="log",
        y_scale="log",
        x_limits=(1, 16),
        kde_overlay=True,
        vline=3,
        hline=3,
        axes=ax_scatter,
        save=False,
    )
    ax_scatter.set_xlabel("norm. DNA content")
    ax_scatter.set_ylabel("norm. EdU intensity")

    ax_bar = fig.add_subplot(gs[:, 1])
    cellcycle_stacked(
        df=df,
        conditions=["all"],
        condition_col="_well_qc",
        selector_col=None,
        show_error_bars=False,
        axes=ax_bar,
        bar_width=0.5,
        x_label=False,
    )
    # Strip the "all" tick label — it's noise for a single-well plot.
    ax_bar.set_xticklabels([""])

    if title:
        fig.suptitle(
            title, fontsize=8, weight="bold", x=0.02, y=0.99, ha="left"
        )
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    if save and path is not None and title:
        save_fig(
            fig,
            path,
            title.replace(" ", "_"),
            tight_layout=False,
            fig_extension=file_format,
        )

    return fig
