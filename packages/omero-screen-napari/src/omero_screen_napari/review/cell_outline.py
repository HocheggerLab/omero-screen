"""Small per-frame crops of the reviewed cell, and when Follow should pan.

Track Review outlines the reviewed cell's nucleus in each frame. It used to
show the outline as a lazy mask the size of the whole well: every time step
relabelled every tile of the frame, and napari traced contours over the full
canvas (~320 ms per step on a 3200² well, #76). The outline only needs the
pixels around the cell: :func:`cell_crop` reads one small crop per frame.

The crops were then shown in a labels layer, swapped in per frame. Labels
layers slice asynchronously, so the outline appeared one load after the
nuclei, behind the cell's marker. The widget now traces each crop into
polygons (:func:`crop_outline`) in a background thread, nearest frames first
(:func:`frames_from`), and shows them in one in-memory shapes layer that
follows the drawn frame exactly like the marker does.

Plain numpy, no Qt: these functions are tested without a viewer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import find_contours

#: Crop half-size (pixels) tried first, and the largest the crop may grow to.
START_HALF = 64
MAX_HALF = 512


def read_window(
    nuclei: Any, frame: int, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """``nuclei[frame, y0:y1, x0:x1]`` as numpy, safe off the GUI thread.

    A dask compute with the default ``callbacks=None`` swaps dask's global
    callbacks out and back in, and napari's dask cache is one of them: two
    threads computing at once (the outline worker and napari's slicing)
    corrupt the cache and fail with a ``KeyError``. An explicit empty list
    leaves the global callbacks alone and computes in this thread.
    """
    block = nuclei[frame, y0:y1, x0:x1]
    if hasattr(block, "dask"):
        return np.asarray(block.compute(scheduler="synchronous", callbacks=[]))
    return np.asarray(block)


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
        mask = read_window(nuclei, frame, y0, y1, x0, x1) == label
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


def crop_outline(
    crop: tuple[np.ndarray, int, int], frame: int
) -> list[np.ndarray]:
    """The outline of a crop as ``(t, y, x)`` polygons on the canvas.

    Args:
        crop: ``(mask, y0, x0)`` from :func:`cell_crop` or :func:`patch_crop`.
        frame: Time index written into the polygons' first column.

    Returns:
        One ``(N, 3)`` float array per closed contour (several when the mask
        has holes or separate pieces); empty when the mask is empty.
    """
    mask, y0, x0 = crop
    # Padding closes contours that touch the crop's border.
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    polygons = []
    for contour in find_contours(padded, 0.5):
        if len(contour) < 3:
            continue
        polygon = np.empty((len(contour), 3))
        polygon[:, 0] = frame
        polygon[:, 1] = contour[:, 0] + y0 - 1
        polygon[:, 2] = contour[:, 1] + x0 - 1
        polygons.append(polygon)
    return polygons


def frames_from(frames: list[int], centre: int) -> list[int]:
    """``frames`` ordered by distance from ``centre`` (earlier frame first on ties)."""
    return sorted(frames, key=lambda t: (abs(t - centre), t))


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
