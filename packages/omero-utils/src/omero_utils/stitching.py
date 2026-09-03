"""Tile stitching for plate-scale imaging.

Combines two responsibilities:

* **Composition** — ``compose_tiles_from_offsets`` blends image tiles into a single
  canvas with optional overlap and edge blending; ``compose_labels_from_offsets``
  does the same for label masks, remapping IDs of objects that span
  adjacent tiles to a shared ID.

* **Position-based placement** — ``positions_to_offsets`` creates
  canvas offsets from stage coordinates for use with the
  composition functions.

* **Canvas-wide segmentation round-trip** — ``split_stitched_from_offsets``
  and ``stitch_from_offsets`` are a paired split/recompose for masks
  produced by canvas-wide segmentation (Phase-1 stitched analysis). They
  preserve original label IDs and boundary-cell pixels losslessly, and
  bypass ``merge_labels`` entirely — required because label IDs are
  globally unique by construction.

Used by both the analysis pipeline (segment-on-stitched-canvas) and
the napari widget (display stitched wells); kept in omero-utils to
avoid a circular dependency between those packages.
"""

import json
import math
import os
from typing import Any, cast

import numpy as np
import scipy.ndimage
from loguru import logger
from numpy.typing import NDArray
from omero_screen.config import is_level_enabled
from skimage.util import map_array

# Stitching calibration constants. These are microscope-level
# values (not per-plate or per-well) and have been stable for the lab's
# Operetta acquisitions. Used by both the analysis pipeline (stitched
# segmentation) and the napari widget; the stitch widget
# allows interactive override but defaults to these.
# Override using a path to a JSON file in the
# environment variable OMERO_SCREEN_STITCH_CONFIG.
STITCH_DEFAULTS: dict[str, int] = {
    "overlap_x": 7,
    "overlap_y": 7,
    "translate_x": -3,
    "translate_y": 3,
}


def load_stitching_config(path: str) -> None:
    """Load stitching configuration from a JSON config file.

    Args:
        path: Path to JSON config file.

    Raises:
        FileNotFoundError: If the file is missing.
        json.decoder.JSONDecodeError: If the config is not a valid JSON file.
        ValueError: If there are unrecognised or missing keys, or the key values
            are not integers.
    """
    try:
        with open(path) as f:
            data = json.load(f)
            # Require all keys for stitching
            if data.keys() != STITCH_DEFAULTS.keys():
                raise ValueError(
                    "Unknown/missing keys in stitch configuration: " + path
                )
            for k, v in data.items():
                STITCH_DEFAULTS[k] = int(v)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load stitch configuration '{path}': {e}")
        raise e


# Load stitch configuration from file if available
path = os.getenv("OMERO_SCREEN_STITCH_CONFIG")
if path is not None and os.path.exists(path):
    load_stitching_config(path)


# --------------------------------------------------------------------------
# Position → grid layout
# --------------------------------------------------------------------------


def positions_to_layout(
    positions: list[tuple[float, float] | None] | list[tuple[float, float]],
    angle_tolerance: float = 5,
) -> list[tuple[int, int]]:
    """Convert stage positions to a tile grid layout.

    Positions are sorted by x and assigned into columns if the angle to the
    next position is 90 degrees, otherwise a new column is started.
    This is repeated after sorting by y for rows and 0 degrees.

    All positions in the same row or column must be within the angle tolerance
    of 0 or 90 degrees respectively.

    Note: This method works for a non-sparse grid layout. An entire missing
    column or row will not be detected and non-adjacent columns/rows will
    be placed adjacent.

    Missing positions are returned as (-1, -1). If positions cannot be mapped
    to a unique grid layout (duplicate cells) all entries are returns as (-1, -1).
    The caller must count the number of entries with a positive x (or y) index
    to determine the number of valid positions that can be mapped to a layout.

    This method will return an empty array if the input positions is empty.

    Args:
        positions: List of (pos_x, pos_y) for each image.
        angle_tolerance: The angle tolerance for a row or column

    Returns:
        List of (col, row) for each image.
    """
    # Get data but ignore invalid positions
    data = []
    for i, p in enumerate(positions):
        if p is not None and p[0] is not None and p[1] is not None:
            data.append((p[0], p[1], i))

    n_pos = len(data)
    if n_pos == 0:
        # No valid positions
        return [(-1, -1)] * len(positions)

    # We require:
    # angle < tolerance
    # arctan(y / x) < tolerance
    # y / x < tan(tolerance)
    tol = math.tan(math.radians(math.fabs(angle_tolerance)))

    # Initialised as unassigned
    row = np.full(len(positions), -1, dtype=np.int_)
    col = row.copy()

    # Sort by x, then y (index does not matter)
    data = sorted(data)
    # Compute angles between consective positions: atan(dx/dy) -> 0 for a column
    current = 0
    col[data[0][2]] = 0
    for i in range(1, n_pos):
        dx = data[i][0] - data[i - 1][0]
        dy = data[i][1] - data[i - 1][1]
        # handle divide by zero
        if dy:
            angle = dx / dy
        elif dx:
            angle = math.inf
        else:
            logger.debug("Duplicate positions in layout")
            return [(-1, -1)] * len(positions)
        if math.fabs(angle) >= tol:
            # new column
            current += 1
        col[data[i][2]] = current

    # Sort by y, then x
    data = sorted(data, key=lambda x: (x[1], x[0]))
    # Compute angles between consective positions: atan(dy/dx) -> 0 for a row
    current = 0
    row[data[0][2]] = 0
    for i in range(1, n_pos):
        dx = data[i][0] - data[i - 1][0]
        dy = data[i][1] - data[i - 1][1]
        # handle divide by zero
        # Duplicate (x,y) positions previously filtered,
        # if dx is zero then dy must be non-zero
        angle = dy / dx if dx else math.inf
        if math.fabs(angle) >= tol:
            # new row
            current += 1
        row[data[i][2]] = current

    # Store the output layout
    out = [(int(x), int(y)) for x, y in zip(col, row, strict=True)]

    # Note: This is a safety net check.
    # If tolerance < 45 degrees then it is not possible for two positions
    # to be in the same row and column since either one or the other will be true
    # for the same vector.
    out_set = set(out)
    out_set.discard((-1, -1))
    n_cells = len(out_set)
    if n_cells < len(data):
        logger.debug(
            f"Duplicate cells in layout: cells {n_cells:d} < valid positions {n_pos:d}"
        )
        return [(-1, -1)] * len(positions)

    maxx = int(col.max()) + 1
    maxy = int(row.max()) + 1
    logger.info(
        f"Position grid: {maxx:d} cols x {maxy:d} rows (valid positions {n_pos:d} / {len(positions):d})"
    )

    if is_level_enabled("DEBUG"):
        # Print information for stitching.
        # Output the grid using -1 for a missing position in the column/row:
        # [-1, 3, -1]
        # [1, 0, 2]
        # [-1, 4, -1]
        grid = np.full((maxy, maxx), -1)
        for i, (x, y) in enumerate(out):
            # Only require checking 1 of x or y
            if x >= 0:
                grid[y, x] = i
        # Output the mean spacing between rows + columns
        rx, ry = [], []
        cx, cy = [], []
        for y in range(maxy):
            for x in range(1, maxx):
                i = grid[y, x - 1]
                j = grid[y, x]
                if i >= 0 and j >= 0:
                    rx.append(positions[j][0] - positions[i][0])  # type: ignore[index]
                    ry.append(positions[j][1] - positions[i][1])  # type: ignore[index]
        for x in range(maxx):
            for y in range(1, maxy):
                i = grid[y - 1, x]
                j = grid[y, x]
                if i >= 0 and j >= 0:
                    cx.append(positions[j][0] - positions[i][0])  # type: ignore[index]
                    cy.append(positions[j][1] - positions[i][1])  # type: ignore[index]

        logger.debug(positions)
        # Avoid numpy warning for empty lists
        rmx, rmy, rsx, rsy = 0.0, 0.0, 0.0, 0.0
        if rx:
            rmx = np.mean(rx)
            rsx = np.std(rx)
            rmy = np.mean(ry)
            rsy = np.std(ry)
        cmx, cmy, csx, csy = 0.0, 0.0, 0.0, 0.0
        if cx:
            cmx = np.mean(cx)
            csx = np.std(cx)
            cmy = np.mean(cy)
            csy = np.std(cy)
        logger.debug(
            f"Position grid: {grid.tolist()}; row {rmx:.3},{rmy:.3} +/- {rsx:.3},{rsy:.3}, col {cmx:.3},{cmy:.3} +/- {csx:.3},{csy:.3} (raw units)"
        )

    return out


# --------------------------------------------------------------------------
# Tile composition
# --------------------------------------------------------------------------


def positions_to_offsets(
    positions: list[tuple[float, float] | None],
    tile_w: int,
    tile_h: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[np.int_]:
    """Convert tile stage positions to canvas offsets.

    Any invalid position is returned as (-1, -1) to mark
    a missing canvas offset. If duplicate positions are detected
    the entire returned array is -1. Valid indices can be obtained
    by checking either x or y is positive, e.g. ``offsets[:, 0] >= 0``.

    This is a compositions of ``positions_to_layout`` and ``layout_to_offsets``.
    See those method for further details of validation.

    Args:
        positions: List of (pos_x, pos_y) for each image.
        tile_w: Tile size in x.
        tile_h: Tile size in y.
        overlap_x: Overlap in x-dimension.
        overlap_y: Overlap in y-dimension.
        translate_x: Row translation in x.
        translate_y: Column translation in y.

    Returns:
        array of [N, (ox, oy)]
    """
    layout = positions_to_layout(positions)
    return layout_to_offsets(
        layout,
        tile_w,
        tile_h,
        overlap_x,
        overlap_y,
        translate_x,
        translate_y,
    )


def compose_tiles_from_offsets(
    tiles: NDArray[Any],
    offsets: NDArray[np.int_],
    edge: int = 0,
) -> NDArray[Any]:
    """Compose tiles into a single image (YXC, all tiles same shape).

    Tiles are composed using the provided offsets. Overrlapping regions
    are blended using a weighted average. The ``edge`` parameter creates
    a linear ramp weighting over the specified size to the image border.

    Args:
        tiles: Array of shape (N, Y, X, C).
        offsets: Array of [N, (ox, oy)] (must be positive).
        edge: Edge size for blending overlaps.

    Returns:
        The composed image (YXC).
    """
    _validate_offsets(offsets)

    dtype = tiles.dtype
    # Take the tile dims straight from `tiles` rather than round-tripping
    # them through `m.shape`, which numpy's stubs cannot narrow to a pair.
    tile_h, tile_w = int(tiles.shape[1]), int(tiles.shape[2])
    m = np.ones((tile_h, tile_w), dtype=int)

    if edge:
        # Distance transform does not use out-of-bounds as background.
        # Pad with 1 pixel and crop.
        d = scipy.ndimage.distance_transform_edt(np.pad(m, 1))
        d = d[1:-1, 1:-1]
        d = np.clip(d, a_min=0, a_max=edge + 1)
        m = d / (edge + 1)

    max_pos = offsets.max(axis=0)

    channels = tiles.shape[3]
    out = np.zeros(
        (
            # Note: Offset max is (x, y) not (y, x)
            max_pos[1] + tile_h,
            max_pos[0] + tile_w,
            channels,
        )
    )
    sum_arr = np.zeros(out.shape[0:2])

    for im, pos in zip(tiles, offsets, strict=True):
        xp, yp = pos
        for c in range(channels):
            out[yp : yp + tile_h, xp : xp + tile_w, c] += m * im[..., c]
        sum_arr[yp : yp + tile_h, xp : xp + tile_w] += m

    indices = sum_arr != 0
    for c in range(channels):
        out[..., c] = np.divide(
            out[..., c], sum_arr, where=indices, out=np.zeros(sum_arr.shape)
        )
    return _as_dtype(dtype, out)


def _as_dtype(
    dtype: Any, array: np.ndarray[Any, np.dtype[Any]]
) -> np.ndarray[Any, np.dtype[Any]]:
    """Clip in-place to the dtype range and return the dtype-cast array."""
    if issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
        return np.clip(
            array, a_min=info.min, a_max=info.max, out=array
        ).astype(dtype)
    if issubclass(dtype.type, np.floating):
        f_info = np.finfo(dtype)
        return np.clip(
            array, a_min=f_info.min, a_max=f_info.max, out=array
        ).astype(dtype)
    return array


def compose_labels_from_offsets(
    tiles: NDArray[Any],
    offsets: NDArray[np.int_],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Compose label tiles into a single image (YXC, all tiles same shape).

    Unique label IDs are remapped. Overlapping labels on adjacent tiles
    are mapped to the same ID.

    Args:
        tiles: Array of shape (N, Y, X, C).
        offsets: Array of [N, (ox, oy)] (must be positive).

    Returns:
        The composed labels (YXC).
    """
    _validate_offsets(offsets)

    dtype = tiles.dtype
    tile_h, tile_w = tiles.shape[1:3]

    max_pos = offsets.max(axis=0)

    channels = tiles.shape[3]
    out = [
        np.zeros(
            (
                # Note: Offset max is (x, y) not (y, x)
                max_pos[1] + tile_h,
                max_pos[0] + tile_w,
            ),
            dtype=dtype,
        )
        for i in range(channels)
    ]

    border = get_overlap(offsets, tile_h, tile_w)

    for im, pos in zip(tiles, offsets, strict=True):
        xp, yp = pos
        for c in range(channels):
            out[c] = merge_labels(
                out[c], im[..., c], xp=xp, yp=yp, border=border
            )

    return np.dstack(out)


def merge_labels(
    im1: np.ndarray[Any, np.dtype[Any]],
    im2: np.ndarray[Any, np.dtype[Any]],
    xp: int = 0,
    yp: int = 0,
    border: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Merge the labels in image 2 into image 1.

    Image 2 may be smaller than image 1.  Scans pixels in the border
    against the current labels. Any overlapping labels in the new image
    adopt the ID of the overlapping label.

    Args:
        im1: Current labels.
        im2: New labels.
        xp: Offset in x.
        yp: Offset in y.
        border: Border width.

    Returns:
        updated (np.array): The updated labels.
    """
    im2 = im2.copy()
    s = im2.shape
    if not (border and im1.any()):
        return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    im1a = im1[yp : yp + s[0], xp : xp + s[1]]
    overlap = (im1a != 0) & (im2 != 0)
    if not overlap.any():
        return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    # Pixels in the overlap region.
    # Later set to zero for ignored overlaps.
    oi1 = im1a[overlap]
    oi2 = im2[overlap]

    h1o = np.bincount(oi1)
    h2o = np.bincount(oi2)
    # Require a new -> old ID overlap histogram. Assume new IDs are
    # sequential from 1.  Remap old IDs that are in the overlap from 1
    # to save memory.
    map_arr = np.zeros(len(h1o), dtype=np.uint16)
    rmap = np.zeros(len(h1o), dtype=np.uint16)
    id_counter = 0
    for i, c in enumerate(h1o):
        if c:
            map_arr[i] = id_counter
            rmap[id_counter] = i
            id_counter += 1
    h = np.zeros((np.nonzero(h2o)[0][-1] + 1, id_counter), dtype=np.uint16)
    for a, b in zip(im2.reshape(-1), im1a.reshape(-1), strict=False):
        if a and b:
            h[a][map_arr[b]] += 1

    h1 = np.bincount(im1.reshape(-1))
    h2 = np.bincount(im2.reshape(-1))
    overlaps = []
    for j, a in enumerate(h):
        for i, c in enumerate(a):
            if c:
                i = rmap[i]
                f = c / max(h1[i], h2[j])
                overlaps.append((i, j, c, f))
    overlaps.sort(reverse=True, key=lambda x: x[-1])

    omap1 = np.arange(len(h1))
    omap2 = np.arange(len(h2))
    map1 = np.zeros(len(h1), dtype=np.uint16)
    map2 = np.zeros(len(h2), dtype=np.uint16)
    # Offset for image 2 labels
    m1 = len(h1) - 1

    # Remap labels to use the ID from the object they overlap.
    # Greedy: largest overlap wins; subsequent overlaps remove pixels.
    for i, j, c, _ in overlaps:
        # Check if either object is mapped.
        if map1[i] or map2[j]:
            # Remove overlap pixels from largest label
            # by setting to zero.
            mask = (oi1 == i) & (oi2 == j)
            assert c == mask.sum()
            if h2[j] > h1[i]:
                oi2[mask] = 0
            else:
                oi1[mask] = 0
        else:
            # None are mapped: assign the mapping to the largest label.
            if h2[j] > h1[i]:
                map2[j] = j + m1
                map1[i] = map2[j]
            else:
                map1[i] = i
                map2[j] = map1[i]

    # Remove ignored overlapping pixels
    im1a[overlap] = oi1
    im2[overlap] = oi2

    im1[yp : yp + s[0], xp : xp + s[1]] = im1a

    map1 = cast(
        np.ndarray[Any, np.dtype[np.uint16]],
        np.where(map1 == 0, omap1, map1),
    )
    map2 = cast(
        np.ndarray[Any, np.dtype[np.uint16]],
        np.where(map2 == 0, omap2 + m1, map2),
    )
    map2[0] = 0

    # Compress map1 and map2 non-zero IDs to ascending from 1.
    m = np.arange(np.max([map1.max(), map2.max()]) + 1, dtype=np.uint16)
    for x in map1:
        m[x] = x
    for x in map2:
        m[x] = x
    # non-zeros remap to ascending
    non_zero = m != 0
    m[non_zero] = np.arange(1, non_zero.sum() + 1)

    map1[:] = m[map1]
    map2[:] = m[map2]

    map_array(im1, omap1, map1, out=im1)  # type: ignore
    map_array(im2, omap2, map2, out=im2)  # type: ignore

    im1[yp : yp + s[0], xp : xp + s[1]] |= im2

    return im1


def _merge_nonoverlapping_labels(
    im1: np.ndarray[Any, np.dtype[Any]],
    im2: np.ndarray[Any, np.dtype[Any]],
    xp: int = 0,
    yp: int = 0,
    m1: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Merge labels in image 2 into image 1 assuming no overlap.

    Both inputs are assumed to have ascending IDs from 1.
    """
    s = im2.shape
    if not m1:
        m1 = np.max(im1)
    np.add(im2, m1, where=im2 != 0, out=im2)
    im1[yp : yp + s[0], xp : xp + s[1]] += im2

    return im1


# --------------------------------------------------------------------------
# Position-based stitching (public entry points)
# --------------------------------------------------------------------------


def stitch_from_offsets(
    images: NDArray[Any],
    offsets: NDArray[np.int_],
    edge: int = -1,
) -> NDArray[Any]:
    """Stitch images using their canvas offsets.

    Args:
        images: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        offsets: Array of [N, (ox, oy)] (must be positive).
        edge: Edge blending width in pixels (set to negative to auto-detect).

    Returns:
        Stitched array of shape (Y, X, C) or (T, Y, X, C).
    """
    ndim = images.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"
    assert len(images) == len(offsets), "Expected each image to have an offset"
    _validate_offsets(offsets)

    # Auto-edge
    if edge < 0:
        tile_h, tile_w = images.shape[-3:-1]
        edge = get_overlap(offsets, tile_h, tile_w)

    if ndim == 5:
        # (N, T, Y, X, C) → stitch per timepoint, then stack
        n_timepoints = images.shape[1]
        layers = [
            compose_tiles_from_offsets(
                images[:, t],
                offsets,
                edge=edge,
            )
            for t in range(n_timepoints)
        ]
        return np.stack(layers)
    else:
        return compose_tiles_from_offsets(
            images,
            offsets,
            edge=edge,
        )


def stitch_into_canvas(
    images: NDArray[Any],
    offsets: NDArray[np.int_],
    canvas_hw: tuple[int, int],
    edge: int = -1,
) -> NDArray[Any]:
    """Stitch images onto a canvas of a fixed size, allowing negative offsets.

    :func:`stitch_from_offsets` rejects negative offsets and *derives* its canvas
    size from ``offsets.max(axis=0)``. Neither suits placing a cyclic-IF restain
    round into the master round's frame: the alignment shift is subtracted from
    the master's canvas offsets, which can push tiles left of or above the origin,
    and the result must land on a canvas of exactly the master's dimensions so the
    rounds stack channel-wise.

    This wrapper pads the offsets non-negative, stitches, then crops back to
    ``canvas_hw``. Tiles (or parts of tiles) falling outside the canvas are
    dropped; areas no tile covers are left at zero.

    The returned array is **always** exactly ``canvas_hw`` in its spatial
    dimensions, whatever the offsets. Callers building lazy dask arrays rely on
    that: the block shape is declared from a probe of the first block, so a
    stitch that silently returned a different size would corrupt the write rather
    than raise.

    Args:
        images: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        offsets: Array of [N, (ox, oy)]. May be negative.
        canvas_hw: Target canvas as (height, width).
        edge: Edge blending width in pixels (negative to auto-detect).
            **Pass this explicitly when stitching rounds that must match.**
            Auto-detection derives the width from the offsets via
            :func:`get_overlap`, so per-field shifts -- which change the relative
            tile geometry -- would blend differently between rounds, giving the
            same stain different pixel values.

    Returns:
        Stitched array of shape (H, W, C) or (T, H, W, C).
    """
    ndim = images.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"
    offsets = np.asarray(offsets)
    height, width = int(canvas_hw[0]), int(canvas_hw[1])

    # Shift every tile by a constant so the minimum offset is zero. A constant
    # does not change the relative geometry, so blending is unaffected.
    pad = np.maximum(0, -offsets.min(axis=0)) if len(offsets) else np.zeros(2)
    pad_x, pad_y = int(pad[0]), int(pad[1])
    stitched = stitch_from_offsets(images, offsets + (pad_x, pad_y), edge=edge)

    # The master canvas origin sits at (pad_y, pad_x) in the padded stitch.
    out = np.zeros(
        stitched.shape[:-3] + (height, width, stitched.shape[-1]),
        dtype=stitched.dtype,
    )
    copy_h = min(height, int(stitched.shape[-3]) - pad_y)
    copy_w = min(width, int(stitched.shape[-2]) - pad_x)
    if copy_h > 0 and copy_w > 0:
        out[..., :copy_h, :copy_w, :] = stitched[
            ..., pad_y : pad_y + copy_h, pad_x : pad_x + copy_w, :
        ]
    return out


def _validate_offsets(offsets: NDArray[np.int_]) -> None:
    """Validates the offsets are all positive."""
    if offsets.min() < 0:
        raise ValueError(f"Offsets must be positive: {offsets}")


def get_overlap(
    offsets: NDArray[np.int_],
    tile_h: int,
    tile_w: int,
) -> int:
    """Get the maximum overlap between canvas tiles.

    The overlap is computed using all tile intersections. For each
    intersection the smaller of the width or height is the distance
    inside the tile from the tile edge. The overlap is the maximum
    of these edge distances across all intersections.
    It describes the maximum distance inside the tile covered by
    any intersection.

    Args:
        offsets: Array of [N, (ox, oy)].
        tile_h: Original per-field height in pixels.
        tile_w: Original per-field width in pixels.

    Returns:
        maximum tile overlap
    """
    # Note: No requirement to validate the offsets are positive
    overlap = 0
    # Compute all-vs-all rectangle intersections
    for i in range(len(offsets)):
        x1, y1 = offsets[i]
        for j in range(i + 1, len(offsets)):
            x2, y2 = offsets[j]
            # lower-left corner
            ix1, iy1 = max(x1, x2), max(y1, y2)
            # upper-right corner
            ix2, iy2 = min(x1, x2) + tile_w, min(y1, y2) + tile_h
            if ix1 < ix2 and iy1 < iy2:
                # Intersection
                ox, oy = ix2 - ix1, iy2 - iy1
                # Get the closest distance to the edge of the tile
                overlap = max(overlap, min(ox, oy))
    return overlap


def split_stitched_from_offsets(
    stitched: NDArray[Any],
    offsets: NDArray[np.int_],
    tile_h: int,
    tile_w: int,
) -> list[NDArray[Any]]:
    """Pseudo-inverse of ``stitch_from_offsets`` for per-image storage round-trip of derived images.

    This method can be used seperate a single channel image derived from the stitched image into
    tiles corresponding to the original stitched tiles, for example splitting a
    label mask.

    Note that pixels within tile overlap regions are duplicated in the neighbouring tiles.

    The result can be reassembled using ``recompose_tiles``.

    Args:
        stitched: Stitched canvas of shape (T, Y, X).
        offsets: Array of [N, (ox, oy)] (must be positive).
        tile_h: Original tile height in pixels.
        tile_w: Original tile width in pixels.

    Returns:
        List of (T, tile_h, tile_w) mask arrays, in the same order as ``offsets``.
    """
    _validate_offsets(offsets)
    if stitched.ndim != 3:
        raise ValueError(f"stitched must be (T, Y, X), got {stitched.shape}")

    out: list[NDArray[Any]] = [
        stitched[:, yp : yp + tile_h, xp : xp + tile_w].copy()
        for (xp, yp) in offsets
    ]

    return out


def missing_field_boxes(
    offsets: NDArray[np.int_],
    tile_h: int,
    tile_w: int,
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of interior canvas regions no field covers.

    A field whose acquisition failed has no stage position, so
    :func:`positions_to_offsets` returns ``(-1, -1)`` for it and the
    stitch leaves a blank tile-sized gap. Its canvas slot cannot be
    recovered from the offsets — there is no position to place it from,
    and the acquisition pattern itself leaves grid cells empty (a 21-field
    Operetta run sits in a 5x5 grid), so an unoccupied lattice point is
    ambiguous.

    The gap is therefore found geometrically rather than inferred: take
    the area no valid tile covers, and keep the connected regions that

    * do not touch the canvas border — an unimaged grid corner and the
      shear wedge along each edge both run to the boundary, whereas a
      dropped interior field is enclosed by its neighbours; and
    * are at least half a tile in area — this discards the thin wedges
      left between rows by the rotation between stage and camera frames.

    A dropped field on the outer ring is indistinguishable from the
    acquisition pattern's own shape and is deliberately not reported;
    visually it is not a hole.

    Args:
        offsets: ``(N, 2)`` canvas offsets, ``(-1, -1)`` where invalid.
        tile_h: Field height in pixels.
        tile_w: Field width in pixels.

    Returns:
        ``(y0, x0, y1, x1)`` boxes in canvas pixel coordinates, ordered
        top-left to bottom-right. Empty when the canvas has no interior
        gap.
    """
    valid = (offsets[:, 0] >= 0) & (offsets[:, 1] >= 0)
    valid_offsets = offsets[valid]
    if not len(valid_offsets):
        return []

    height = int(valid_offsets[:, 1].max()) + tile_h
    width = int(valid_offsets[:, 0].max()) + tile_w
    covered = np.zeros((height, width), dtype=bool)
    for ox, oy in valid_offsets:
        covered[oy : oy + tile_h, ox : ox + tile_w] = True

    labelled, _ = scipy.ndimage.label(~covered)
    min_area = (tile_h * tile_w) // 2
    boxes: list[tuple[int, int, int, int]] = []
    for i, slices in enumerate(scipy.ndimage.find_objects(labelled), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        if (
            y_slice.start == 0
            or x_slice.start == 0
            or y_slice.stop >= height
            or x_slice.stop >= width
        ):
            continue
        if int((labelled[slices] == i).sum()) < min_area:
            continue
        boxes.append(
            (y_slice.start, x_slice.start, y_slice.stop, x_slice.stop)
        )
    return sorted(boxes)


def layout_to_offsets(
    layout: list[tuple[int, int]],
    tile_w: int,
    tile_h: int,
    overlap_x: int,
    overlap_y: int,
    translate_x: int,
    translate_y: int,
) -> NDArray[np.int_]:
    """Compute per-field (xp, yp) canvas positions from stage grid layout.

    Any grid layout with negative x or y is returned as (-1, -1) to mark
    a missing grid position.

    Params:
        layout: List of (col, row) for each image.
        tile_w: Tile size in x.
        tile_h: Tile size in y.
        overlap_x: Overlap in x-dimension.
        overlap_y: Overlap in y-dimension.
        translate_x: Row translation in x.
        translate_y: Column translation in y.

    Returns:
        array of [N, (ox, oy)]
    """
    ox = -overlap_x
    oy = -overlap_y
    tx = translate_x
    ty = translate_y

    offsets = np.full((len(layout), 2), -1, dtype=np.int_)
    valid = np.ones(len(layout), dtype=np.bool_)
    for i, (x, y) in enumerate(layout):
        if x >= 0 and y >= 0:
            offsets[i] = (
                x * (tile_w + ox) + y * tx,
                y * (tile_h + oy) + x * ty,
            )
        else:
            valid[i] = False

    if np.any(valid):
        offsets[valid] = offsets[valid] - offsets[valid].min(axis=0)
    return offsets


def assign_tile_by_centroid(
    centroids_yx: NDArray[np.floating[Any]],
    offsets: NDArray[np.int_],
    tile_h: int,
    tile_w: int,
) -> NDArray[np.intp]:
    """Assign each centroid to the tile who owns it.

    Used by canvas-wide stitched segmentation to tag each measurement
    row with the OMERO image id of the field that owns the cell's
    centroid.

    For centroids that fall inside the overlap region of multiple
    tiles, the tile whose centre is nearest (Euclidean) is chosen --
    deterministic and intuitive. Centroids outside every tile rect
    (defensive: should not occur for cells discovered inside the
    canvas) fall back to the globally nearest tile centre.

    Args:
        centroids_yx: ``(N, 2)`` array of (y, x) centroid coordinates
            in canvas pixel space (matches regionprops ``centroid-0``,
            ``centroid-1``).
        offsets: Array of [K, (ox, oy)] (ignores negative offsets).
        tile_h: Per-field tile height in pixels.
        tile_w: Per-field tile width in pixels.

    Returns:
        ``(N,)`` array of field indices into ``offsets``.
    """
    centroids = np.asarray(centroids_yx, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids_yx must be (N, 2), got {centroids.shape}")
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError(f"offsets must be (K, 2), got {offsets.shape}")
    valid = (offsets[:, 0] >= 0) & (offsets[:, 1] >= 0)
    if not np.any(valid):
        raise ValueError(f"No valid positive offsets: {offsets}")

    xps = offsets[valid, 0:1].T  # (1, K)
    yps = offsets[valid, 1:2].T

    cy = centroids[:, 0:1]  # (N, 1)
    cx = centroids[:, 1:2]
    # Tile centres for distance tie-break.in
    centre_y = yps + tile_h / 2.0  # (1, K)
    centre_x = xps + tile_w / 2.0
    dy = cy - centre_y  # (N, K)
    dx = cx - centre_x
    dist2 = dy * dy + dx * dx

    # Inside-rect mask: centroid in [xp, xp+tile_w) × [yp, yp+tile_h).
    inside = (
        (cx >= xps)
        & (cx < (xps + tile_w))
        & (cy >= yps)
        & (cy < (yps + tile_h))
    )

    # Among containing tiles, pick the nearest centre. Centroids inside
    # no rect (defensive) fall back to the globally nearest centre.
    masked = np.where(inside, dist2, np.inf)
    any_inside = inside.any(axis=1)
    chosen = np.where(
        any_inside,
        np.argmin(masked, axis=1),
        np.argmin(dist2, axis=1),
    ).astype(np.intp)
    # map chosen back to the original offset index
    remap = np.arange(len(offsets))[valid]
    return remap[chosen]


def recompose_tiles(
    per_field_tiles: NDArray[Any] | list[NDArray[Any]],
    offsets: NDArray[np.int_],
) -> NDArray[Any]:
    """Inverse of ``split_stitched_from_offsets``.

    Reassembles per-field tiles into a single stitched canvas.

    **Label invariant required**: the per-field tiles must come from a single
    canvas-wide segmentation, so label IDs are globally unique. Where the
    same label appears in two adjacent tiles (a cell straddling a tile
    boundary), the pixels are co-located by construction — a simple
    copy reassembles the canvas without renumbering or overlap
    logic.

    This function is **not** a general-purpose label merger; for
    independently-segmented tiles with name collisions, use
    ``stitch_labels_from_offsets`` (which goes through ``merge_labels``).

    Accepts two input shapes:

    * ``list[NDArray]`` of ``(T, tile_h, tile_w)`` — the direct output of
      ``split_stitched_from_offsets``. Returns ``(T, Y, X)``.
    * ``NDArray`` of ``(N, tile_h, tile_w, C)`` or ``(N, T, tile_h, tile_w, C)``
      — matches the napari label-stack shape. Channels and timepoints are
      handled internally. Returns ``(Y, X, C)`` or ``(T, Y, X, C)``.

    Args:
        per_field_tiles: Per-field tiles (see input shapes above).
        offsets: Array of [N, (ox, oy)] (must be positive).

    Returns:
        Stitched canvas. Shape depends on input — see above.
    """
    _validate_offsets(offsets)

    # Normalise to a (N, T, tile_h, tile_w, C) array internally; track which
    # dims were synthetic so we can squeeze them back out for the caller.
    if isinstance(per_field_tiles, list):
        if not per_field_tiles:
            raise ValueError("per_field_tiles must not be empty")
        first = per_field_tiles[0]
        if first.ndim != 3:
            raise ValueError(
                f"list tiles must be (T, tile_h, tile_w), got {first.shape}"
            )
        # Stack as (N, T, H, W) then add C=1
        stacked = np.stack(per_field_tiles, axis=0)[..., np.newaxis]
        squeeze_c = True
        squeeze_t = False
    else:
        arr = per_field_tiles
        if arr.ndim == 4:
            # (N, H, W, C) → add T axis
            stacked = arr[:, np.newaxis, ...]
            squeeze_c = False
            squeeze_t = True
        elif arr.ndim == 5:
            # (N, T, H, W, C)
            stacked = arr
            squeeze_c = False
            squeeze_t = False
        else:
            raise ValueError(
                f"array tiles must be (N,H,W,C) or (N,T,H,W,C), got {arr.shape}"
            )

    if stacked.shape[0] != len(offsets):
        raise ValueError(
            f"tile count ({stacked.shape[0]}) and offsets "
            f"({len(offsets)}) must match"
        )

    n_t = stacked.shape[1]
    tile_h = stacked.shape[2]
    tile_w = stacked.shape[3]
    n_c = stacked.shape[4]
    dtype = stacked.dtype

    # Canvas extent = furthest tile's far corner.
    max_pos = offsets.max(axis=0)

    canvas = np.zeros(
        (
            n_t,
            # Note: Offset max is (x, y) not (y, x)
            max_pos[1] + tile_h,
            max_pos[0] + tile_w,
            n_c,
        ),
        dtype=dtype,
    )

    for im, pos in zip(stacked, offsets, strict=True):
        xp, yp = pos
        canvas[:, yp : yp + tile_h, xp : xp + tile_w, :] = im

    # Squeeze synthetic axes back out to match caller's input shape.
    if squeeze_c:
        canvas = canvas[..., 0]  # drop C → (T, Y, X)
    if squeeze_t:
        canvas = canvas[0]  # drop T → (Y, X, C) or (Y, X)
    return canvas


def stitch_labels_from_offsets(
    labels: NDArray[Any],
    offsets: NDArray[np.int_],
) -> NDArray[Any]:
    """Stitch label masks using their canvas offsets.

    Args:
        labels: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        offsets: Array of [N, (ox, oy)] (must be positive).

    Returns:
        Stitched labels of shape (Y, X, C) or (T, Y, X, C).
    """
    ndim = labels.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"
    assert len(labels) == len(offsets), "Expected each label to have an offset"
    _validate_offsets(offsets)

    # TODO...
    if ndim == 5:
        # (N, T, Y, X, C) → stitch per timepoint, then stack
        n_timepoints = labels.shape[1]
        layers = [
            compose_labels_from_offsets(
                labels[:, t],
                offsets,
            )
            for t in range(n_timepoints)
        ]
        return np.stack(layers)
    else:
        return compose_labels_from_offsets(
            labels,
            offsets,
        )
