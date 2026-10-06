"""Render a followed cell as a filmstrip with its reporter trace.

Top: one tile per frame, centred on the cell (:mod:`.follow`), as a colour
composite of the chosen channels. The cell's own nucleus is outlined; other
nuclei are drawn faintly; a frame where the cell has no mask gets a cross at
its expected position. Bottom: PIP, geminin and area over the cell's whole
path, with PIP-FUCCI phase bands, gap frames, curated events and the frames
shown above marked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from omero_screen_napari.review.follow import FollowedCell  # noqa: E402

#: Default display colours by channel-name prefix (lower case).
CHANNEL_COLOURS = {
    # DNA is dimmed so the reporter colours dominate the nucleus.
    "spydna": (0.35, 0.35, 0.35),
    "dapi": (0.35, 0.35, 0.35),
    "hoechst": (0.35, 0.35, 0.35),
    "pip": (0.2, 0.9, 0.2),
    "geminin": (0.95, 0.2, 0.6),
    "bf": (0.6, 0.6, 0.6),
}
PHASE_COLOURS = {"G1": "#cfe8cf", "S": "#f6d6e6", "G2": "#e2d6f6"}


def _colour(name: str) -> tuple[float, float, float]:
    low = name.lower()
    for key, col in CHANNEL_COLOURS.items():
        if low.startswith(key):
            return col
    return (1.0, 1.0, 1.0)


def composite(cell: FollowedCell, i: int) -> np.ndarray:
    """RGB composite of tile ``i`` with the strip's fixed limits."""
    rgb = np.zeros((*cell.images.shape[-2:], 3), dtype=np.float32)
    for c, name in enumerate(cell.channels):
        lo, hi = cell.limits[c]
        norm = np.clip((cell.images[i, c] - lo) / (hi - lo), 0, 1)
        rgb += norm[..., None] * np.array(_colour(name), dtype=np.float32)
    return np.clip(rgb, 0, 1)


def _pick_frames(frames: list[int], max_tiles: int) -> list[int]:
    if len(frames) <= max_tiles:
        return list(range(len(frames)))
    return sorted(
        {int(round(x)) for x in np.linspace(0, len(frames) - 1, max_tiles)}
    )


def render_filmstrip(
    cell: FollowedCell,
    title: str = "",
    max_tiles: int = 24,
    columns: int = 12,
    interval_minutes: float = 20.0,
    events: list[dict[str, Any]] | None = None,
    trace_channels: tuple[str, ...] = ("pip", "geminin"),
) -> Figure:
    """Draw the filmstrip and trace for a followed cell.

    Args:
        cell: Crops from :func:`~omero_screen_napari.review.follow.follow`.
        title: Figure title (e.g. the review item id).
        max_tiles: At most this many frames are shown, evenly spread; the
            trace always covers the whole path.
        columns: Tiles per row.
        interval_minutes: Frame interval, for the time axis.
        events: Curated events ``{"kind", "frame"}`` to mark on the trace.
        trace_channels: Path columns to plot (normalised to their maximum).

    Returns:
        The matplotlib figure.
    """
    shown = _pick_frames(cell.frames, max_tiles)
    rows = max(1, int(np.ceil(len(shown) / columns)))
    fig = plt.figure(figsize=(1.3 * columns, 1.45 * rows + 2.3), dpi=110)
    grid = fig.add_gridspec(
        rows + 1,
        columns,
        height_ratios=[1.0] * rows + [1.5],
        hspace=0.25,
        wspace=0.04,
    )
    phases = (
        cell.path.set_index("timepoint")["phase"]
        if "phase" in cell.path
        else None
    )

    for k, i in enumerate(shown):
        ax = fig.add_subplot(grid[k // columns, k % columns])
        ax.imshow(composite(cell, i), interpolation="nearest")
        labels = cell.labels[i]
        others = find_boundaries(labels, mode="inner") & (
            labels != cell.cell_labels[i]
        )
        overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
        overlay[others] = (1, 1, 1, 0.25)
        if cell.cell_labels[i]:
            own = find_boundaries(labels == cell.cell_labels[i], mode="inner")
            overlay[own] = (0.0, 1.0, 1.0, 1.0)
        ax.imshow(overlay, interpolation="nearest")
        if not cell.cell_labels[i]:
            mid = labels.shape[0] / 2
            ax.plot(mid, mid, marker="x", color="magenta", markersize=9, mew=2)
        t = cell.frames[i]
        phase = f" {phases.get(t, '')}" if phases is not None else ""
        ax.set_title(f"t{t}{phase}", fontsize=7, pad=2)
        ax.set_axis_off()

    ax = fig.add_subplot(grid[rows, :])
    path = cell.path
    hours = path["timepoint"] * interval_minutes / 60
    if phases is not None:
        for (t, ph), nxt in zip(
            phases.items(),
            list(phases.index[1:]) + [phases.index[-1] + 1],
            strict=True,
        ):
            if ph in PHASE_COLOURS:
                ax.axvspan(
                    t * interval_minutes / 60,
                    nxt * interval_minutes / 60,
                    color=PHASE_COLOURS[ph],
                    lw=0,
                )
    for gap_t in path.loc[path["gap"], "timepoint"]:
        h = gap_t * interval_minutes / 60
        ax.axvspan(
            h, h + interval_minutes / 60, color="#bbbbbb", alpha=0.6, lw=0
        )
    for ch in trace_channels:
        if ch in path:
            vals = (
                path[ch] / np.nanmax(path[ch])
                if np.nanmax(path[ch]) > 0
                else path[ch]
            )
            ax.plot(hours, vals, color=_colour(ch), lw=1.4, label=ch)
    if "area" in path:
        ax.plot(
            hours,
            path["area"] / np.nanmax(path["area"]),
            color="0.35",
            lw=0.9,
            ls="--",
            label="area",
        )
    for t in (cell.frames[i] for i in shown):
        ax.axvline(t * interval_minutes / 60, color="0.6", lw=0.4, ymax=0.06)
    for ev in events or []:
        h = ev["frame"] * interval_minutes / 60
        ax.axvline(h, color="black", lw=1.0)
        ax.text(h, 1.02, ev["kind"], fontsize=7, ha="center", va="bottom")
    ax.set_xlim(hours.min(), hours.max() + interval_minutes / 60)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("time (h)", fontsize=8)
    ax.set_ylabel("rel. level", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", ncol=4, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    return fig


def save_filmstrip(fig: Figure, out: Path) -> Path:
    """Write the figure (PNG or PDF by suffix) and close it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
