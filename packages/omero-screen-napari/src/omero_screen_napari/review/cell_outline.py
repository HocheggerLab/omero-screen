"""Small per-frame crops of the reviewed cell, and when Follow should pan.

Track Review outlines the reviewed cell's nucleus in each frame. It used to
show the outline as a lazy mask the size of the whole well: every time step
relabelled every tile of the frame, and napari traced contours over the full
canvas (~320 ms per step on a 3200² well, #76). The outline only needs the
pixels around the cell, so the widget now shows one small crop per frame,
read with :func:`cell_crop` and placed on the canvas with ``translate``.

Plain numpy, no Qt: these functions are tested without a viewer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Crop half-size (pixels) tried first, and the largest the crop may grow to.
START_HALF = 64
MAX_HALF = 512


def cell_crop(
    nuclei: Any,
    frame: int,
    label: int,
    y: float,
    x: float,
    half: int = START_HALF,
    max_half: int = MAX_HALF,
) -> tuple[np.ndarray, int, int] | None:
    """The cell's nucleus in one frame, as a small uint8 mask.

    Reads a window around ``(y, x)`` from ``nuclei`` (``(T, Y, X)``: numpy,
    zarr or dask) and keeps pixels equal to ``label``. If the nucleus touches
    an edge of the window that is not the canvas edge, the window doubles,
    up to ``max_half``.

    Args:
        nuclei: Label stack, level 0 of the pyramid.
        frame: Time index.
        label: The cell's label in that frame.
        y: Centroid row (canvas pixels).
        x: Centroid column (canvas pixels).
        half: Initial window half-size.
        max_half: Largest window half-size.

    Returns:
        ``(mask, y0, x0)`` with ``mask`` 1 on the nucleus and ``(y0, x0)`` its
        top-left canvas pixel, or ``None`` when the label is not in the window.
    """
    height, width = nuclei.shape[-2:]
    cy, cx = int(round(y)), int(round(x))
    while True:
        y0, y1 = max(cy - half, 0), min(cy + half, height)
        x0, x1 = max(cx - half, 0), min(cx + half, width)
        mask = np.asarray(nuclei[frame, y0:y1, x0:x1]) == label
        if not mask.any():
            return None
        clipped = (
            (mask[0].any() and y0 > 0)
            or (mask[-1].any() and y1 < height)
            or (mask[:, 0].any() and x0 > 0)
            or (mask[:, -1].any() and x1 < width)
        )
        if not clipped or half >= max_half:
            return mask.astype(np.uint8), y0, x0
        half *= 2


def patch_crop(
    patch: tuple[int, int, np.ndarray],
) -> tuple[np.ndarray, int, int]:
    """A reviewer-made mask ``(y0, x0, mask)`` as ``(uint8 mask, y0, x0)``."""
    y0, x0, mask = patch
    return np.asarray(mask, dtype=bool).astype(np.uint8), int(y0), int(x0)


def needs_recentre(
    cell: tuple[float, float],
    centre: tuple[float, float],
    view: tuple[float, float],
    margin: float = 0.6,
) -> bool:
    """Whether the cell has left the central part of the view.

    Follow mode used to move the camera on every time step, which made every
    layer slice the frame twice (+145 ms per step, #76). It now pans only when
    the cell leaves the central ``margin`` of the view.

    Args:
        cell: Cell position ``(y, x)``, world units.
        centre: Camera centre ``(y, x)``, world units.
        view: Visible extent ``(height, width)``, world units.
        margin: Fraction of the view, centred, the cell may move in.

    Returns:
        True when the camera should be recentred on the cell.
    """
    if view[0] <= 0 or view[1] <= 0:
        return True
    dy, dx = abs(cell[0] - centre[0]), abs(cell[1] - centre[1])
    return dy > margin * view[0] / 2 or dx > margin * view[1] / 2
