"""Follow one tracked cell through time in the zarr cache.

Given a cell's per-frame path (:func:`cellview.tracks.cells.cell_frames`), cut
a fixed-size window centred on the cell in **every** frame, so the cell stays
in view however far it moves between frames. Image and nucleus-label crops are
read straight from the cached OME-Zarr; no viewer and no Qt are involved, so
this works headless, in batch, and behind an agent tool.

Contrast is fixed per channel for a whole strip (taken from the crops of that
strip), so brightness changes between frames are real, not rescaling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass
class FollowedCell:
    """Crops of one cell over time.

    Attributes:
        frames: Timepoints, one per crop.
        images: ``(T, C, H, W)`` image crops for ``channels``.
        labels: ``(T, H, W)`` raw nucleus-label crops.
        cell_labels: The cell's raw label per frame (0 = no mask: a gap).
        centres: ``(T, 2)`` crop centres ``(y, x)`` in canvas pixels.
        channels: Channel names, as in the zarr cache.
        limits: Per-channel ``(low, high)`` display limits.
        path: The per-frame table the crops were cut from.
    """

    frames: list[int]
    images: np.ndarray
    labels: np.ndarray
    cell_labels: list[int]
    centres: np.ndarray
    channels: list[str]
    limits: list[tuple[float, float]]
    path: pd.DataFrame


def match_channels(
    available: list[str], wanted: list[str] | None
) -> list[int]:
    """Indices of ``wanted`` channels in ``available``, matched case-insensitively by prefix.

    ``"spyDNA"`` matches ``"spyDNA_nucleus"``. ``None`` selects every
    fluorescence channel, i.e. all but brightfield (``bf*``).

    Raises:
        ValueError: If a wanted channel matches nothing.
    """
    if not wanted:
        return [
            i
            for i, a in enumerate(available)
            if not a.lower().startswith("bf")
        ]
    lower = [a.lower() for a in available]
    out = []
    for w in wanted:
        hits = [
            i
            for i, a in enumerate(lower)
            if a == w.lower() or a.startswith(w.lower())
        ]
        if not hits:
            raise ValueError(
                f"No channel matches {w!r}; available: {', '.join(available)}"
            )
        out.append(hits[0])
    return out


def _window(
    arr: Any, t: int, cy: int, cx: int, half: int, channel: int | None = None
) -> np.ndarray:
    """``(2*half, 2*half)`` crop centred on ``(cy, cx)``, zero-padded at edges."""
    height, width = arr.shape[-2], arr.shape[-1]
    y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
    sy0, sy1, sx0, sx1 = (
        max(y0, 0),
        min(y1, height),
        max(x0, 0),
        min(x1, width),
    )
    if channel is None:
        block = (
            np.asarray(arr[t, sy0:sy1, sx0:sx1])
            if arr.ndim == 3
            else np.asarray(arr[t, 0, sy0:sy1, sx0:sx1])
        )
    else:
        block = np.asarray(arr[t, channel, sy0:sy1, sx0:sx1])
    out = np.zeros((2 * half, 2 * half), dtype=block.dtype)
    out[
        sy0 - y0 : sy0 - y0 + block.shape[0],
        sx0 - x0 : sx0 - x0 + block.shape[1],
    ] = block
    return out


def follow(
    image: Any,
    nuclei: Any,
    channel_names: list[str],
    path: pd.DataFrame,
    channels: list[str] | None = None,
    size: int = 160,
    frames: list[int] | None = None,
    percentiles: tuple[float, float] = (1.0, 99.95),
    patch_fn: Any = None,
) -> FollowedCell:
    """Cut crops centred on the cell in each frame.

    Args:
        image: Level-0 image array ``(T, C, Y, X)`` (zarr or numpy).
        nuclei: Level-0 nucleus labels ``(T, Y, X)``.
        channel_names: Names of the image channels.
        path: Per-frame table with ``timepoint, label, y, x`` (gaps have
            ``label == 0`` and an interpolated position).
        channels: Channels to crop (prefix match); default all.
        size: Crop edge in pixels (even).
        frames: Subset of timepoints; default every row of ``path``.
        percentiles: Display limits per channel over all crops of the strip.
        patch_fn: Optional ``(frame, y0, x0, label_crop) -> label_crop`` that
            applies reviewer-made masks (see :func:`.masks.paint_patches`).

    Returns:
        A :class:`FollowedCell`.
    """
    idx = match_channels(channel_names, channels)
    rows = path.dropna(subset=["y", "x"])
    if frames is not None:
        rows = rows[rows["timepoint"].isin(frames)]
    half = size // 2
    imgs, labs, cents, cell_labels, ts = [], [], [], [], []
    # itertuples rows carry dynamic per-column attributes.
    for row in cast(Iterable[Any], rows.itertuples()):
        t, cy, cx = int(row.timepoint), int(round(row.y)), int(round(row.x))
        imgs.append(
            np.stack([_window(image, t, cy, cx, half, c) for c in idx])
        )
        lab = _window(nuclei, t, cy, cx, half)
        if patch_fn is not None:
            lab = patch_fn(t, cy - half, cx - half, lab)
        labs.append(lab)
        cents.append((cy, cx))
        cell_labels.append(int(row.label))
        ts.append(t)
    images = (
        np.stack(imgs).astype(np.float32)
        if imgs
        else np.zeros((0, len(idx), size, size), np.float32)
    )
    labels = np.stack(labs) if labs else np.zeros((0, size, size), np.uint32)
    limits = []
    for c in range(images.shape[1]):
        vals = images[:, c][images[:, c] > 0]
        lo, hi = np.percentile(vals, percentiles) if vals.size else (0.0, 1.0)
        limits.append((float(lo), float(max(hi, lo + 1))))
    return FollowedCell(
        frames=ts,
        images=images,
        labels=labels,
        cell_labels=cell_labels,
        centres=np.array(cents, dtype=float).reshape(-1, 2),
        channels=[channel_names[i] for i in idx],
        limits=limits,
        path=path,
    )


def follow_from_cache(
    plate_id: int, well: str, path: pd.DataFrame, **kwargs: Any
) -> FollowedCell:
    """:func:`follow` on the cached well of ``plate_id``.

    Raises:
        FileNotFoundError: If the well is not in the zarr cache.
    """
    from omero_screen_napari.zarr_cache.reader import read_well

    data = read_well(plate_id, well)
    if not data["image"] or not data["nuclei"]:
        raise FileNotFoundError(
            f"Plate {plate_id} well {well} is not in the zarr cache with nuclei labels."
        )
    return follow(
        data["image"][0],
        data["nuclei"][0],
        data["channel_names"],
        path,
        **kwargs,
    )
