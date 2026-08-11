"""Tile stitching for plate-scale imaging.

Combines two responsibilities:

* **Composition** — ``compose_tiles`` blends image tiles into a single
  canvas with optional overlap and edge blending; ``compose_labels``
  does the same for label masks, remapping IDs of objects that span
  adjacent tiles to a shared ID.

* **Position-based placement** — ``stitch_from_positions`` and
  ``stitch_labels_from_positions`` convert absolute stage coordinates
  into a tile grid (``positions_to_grid``) and delegate to the
  composition functions.

* **Canvas-wide segmentation round-trip** — ``split_stitched_mask_to_fields``
  and ``recompose_split_labels`` are a paired split/recompose for masks
  produced by canvas-wide segmentation (Phase-1 stitched analysis). They
  preserve original label IDs and boundary-cell pixels losslessly, and
  bypass ``merge_labels`` entirely — required because label IDs are
  globally unique by construction.

The clustering tolerance used to derive the grid is computed
adaptively from the gap distribution so positions in µm, mm, or
reference-frame units all work.

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
        logger.error(f"Failed to load configuration '{path}': {e}")
        raise e


# Load stitch configuration from file if available
path = os.getenv("OMERO_SCREEN_STITCH_CONFIG")
if path is not None and os.path.exists(path):
    load_stitching_config(path)


# --------------------------------------------------------------------------
# Position → grid
# --------------------------------------------------------------------------


def has_valid_positions(
    positions: list[tuple[float, float] | None],
) -> bool:
    """Return True if positions can be used for stitching.

    Requires at least 2 positions, all non-None, and positions that
    span more than one grid cell (i.e. not all identical).
    """
    valid = [p for p in positions if p is not None]
    if len(valid) < max(2, len(positions)):
        if is_level_enabled("DEBUG"):
            if len(valid) < len(positions):
                logger.debug(
                    f"Missing positions: {len(valid):d} < {len(positions):d}"
                )
            else:
                logger.debug(f"Not enough positions: {len(positions)}")
        return False

    xs = [p[0] for p in valid if p[0] is not None]
    ys = [p[1] for p in valid if p[1] is not None]
    if min(len(xs), len(ys)) < len(positions):
        logger.debug(
            f"Missing X/Y positions: {len(xs):d},{len(ys):d} < {len(positions):d}"
        )
        return False

    tol_x = _adaptive_tolerance(xs)
    tol_y = _adaptive_tolerance(ys)
    x_clusters = _cluster_values(xs, tol_x)
    y_clusters = _cluster_values(ys, tol_y)

    def _nearest_cluster(value: float, clusters: list[float]) -> int:
        return int(np.argmin([abs(value - c) for c in clusters]))

    location = set()
    for px, py in valid:
        col = _nearest_cluster(px, x_clusters)
        row = _nearest_cluster(py, y_clusters)
        location.add((col, row))

    if len(location) < len(valid):
        logger.warning(
            f"Stage positions form {len(location):d} grid cells for {len(valid):d} images — cannot stitch without losing data. Positions (first 5): {valid[:5]}"
        )
        return False
    return True


def _adaptive_tolerance(values: list[float]) -> float:
    """Compute a clustering tolerance from the gap distribution.

    Positions may be in µm, mm, or reference-frame units, so a
    hardcoded tolerance doesn't work.  Instead we look at the gaps
    between sorted values:

    * If all gaps are similar (uniform grid, no noise) we use 25% of
      the minimum gap — small enough to keep each position separate.
    * If gaps vary widely (noisy positions within columns/rows plus
      larger inter-column/row jumps) we use the geometric mean of the
      smallest and largest gap, which sits between the two populations.
    """
    if len(values) < 2:
        return 0.0
    sorted_v = np.sort(values)

    gaps = sorted_v[1:] - sorted_v[:-1]
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0.0

    min_gap = gaps.min()
    max_gap = gaps.max()

    if max_gap / min_gap < 10:
        return float(min_gap * 0.25)
    return float(np.sqrt(min_gap * max_gap))


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    """Group sorted values within *tolerance* and return cluster centroids."""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if abs(v - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def positions_to_grid(
    positions: list[tuple[float, float]],
) -> dict[int, dict[int, int]]:
    """Convert stage positions to a tile grid.

    The clustering tolerance is determined automatically from the data.

    Args:
        positions: List of (pos_x, pos_y) for each image.

    Returns:
        dict[col][row] = image_index
    """
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    tol_x = _adaptive_tolerance(xs)
    tol_y = _adaptive_tolerance(ys)
    x_clusters = _cluster_values(xs, tol_x)
    y_clusters = _cluster_values(ys, tol_y)

    def _nearest_cluster(value: float, clusters: list[float]) -> int:
        return int(np.argmin([abs(value - c) for c in clusters]))

    grid_map: dict[int, dict[int, int]] = {}
    for idx, (px, py) in enumerate(positions):
        col = _nearest_cluster(px, x_clusters)
        row = _nearest_cluster(py, y_clusters)
        if col not in grid_map:
            grid_map[col] = {}
        grid_map[col][row] = idx

    n_cells = sum(len(rows) for rows in grid_map.values())
    logger.info(
        f"Position grid: {len(x_clusters):d} cols x {len(y_clusters):d} rows ({n_cells:d} cells for {len(positions):d} images)"
    )

    if n_cells and is_level_enabled("DEBUG"):
        # Print information for stitching.
        # Rows/columns assumed orthogonal and aligned to x/y axes.
        maxx = np.max(list(grid_map.keys()))
        maxy = 0
        for x_dict in grid_map.values():
            maxy = np.max(list(x_dict.keys()), initial=maxy)
        grid = []
        rx, ry = [], []
        cx, cy = [], []
        empty: dict[int, int] = {}
        for y in range(maxy + 1):
            grid_row = []
            for x in range(maxx + 1):
                grid_row.append(grid_map.get(x, empty).get(y, -1))
            grid.append(grid_row)
            for x in range(maxx):
                i = grid_row[x]
                j = grid_row[x + 1]
                if i >= 0 and j >= 0:
                    rx.append(positions[j][0] - positions[i][0])
                    ry.append(positions[j][1] - positions[i][1])
        for x in range(maxx + 1):
            grid_col = []
            for y in range(maxy + 1):
                grid_col.append(grid_map.get(x, empty).get(y, -1))
            for y in range(maxy):
                i = grid_col[y]
                j = grid_col[y + 1]
                if i >= 0 and j >= 0:
                    cx.append(positions[j][0] - positions[i][0])
                    cy.append(positions[j][1] - positions[i][1])

        logger.debug(positions)
        logger.debug(
            f"Position grid: {grid}; row {np.mean(rx):.3f},{np.mean(ry):.3f} +/- {np.std(rx):.3f},{np.std(ry):.3f}, col {np.mean(cx):.3f},{np.mean(cy):.3f} +/- {np.std(cx):.3f},{np.std(cy):.3f} (raw units)"
        )

    return grid_map


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


def compose_tiles(
    tiles: dict[int, dict[int, np.ndarray[Any, np.dtype[Any]]]],
    ox: int = 0,
    oy: int = 0,
    tx: int = 0,
    ty: int = 0,
    edge: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Compose tiles into a single image (YXC, all tiles same shape).

    Tiles are composed on the grid by default with edge-to-edge contact. If the tiles
    should overlap/gap then use a tile offset (ox, oy). If successive rows or columns
    are translated relative to the previous row/column then use tx or ty.
    The position of each tile is:

    xp = x * (size_x + ox) + y * tx
    yp = y * (size_y + oy) + x * ty

    The tile positions are used to generate the output image bounds, and then adjusted
    so the tiles are composed within the output image bounds.

    Args:
        tiles: Dictionary of dictionaries of np.array tiles, keyed by [x][y].
        ox: Tile offset in x (use negative for overlap).
        oy: Tile offset in y (use negative for overlap).
        tx: Row translation in x.
        ty: Column translation in y.
        edge: Edge size for blending overlaps.

    Returns:
        composed (np.array): The composed image (YXC).
    """
    maxx = np.max(list(tiles.keys()))
    maxy = 0
    for x_dict in tiles.values():
        maxy = np.max(list(x_dict.keys()), initial=maxy)

    y = next(iter(tiles[maxx]))
    im = tiles[maxx][y]
    os_ = im.shape
    dtype = im.dtype
    m = np.ones(os_[0:2], dtype=int)

    if edge:
        # Distance transform does not use out-of-bounds as background.
        # Pad with 1 pixel and crop.
        d = scipy.ndimage.distance_transform_edt(np.pad(m, 1))
        d = d[1:-1, 1:-1]
        d = np.clip(d, a_min=0, a_max=edge)
        m = d / edge

    pos = np.zeros((maxx + 1, maxy + 1, 2), dtype=int)
    valid = np.zeros((maxx + 1, maxy + 1), dtype=int)
    for x, d in tiles.items():
        for y in d:
            pos[x, y] = (
                x * (os_[1] + ox) + y * tx,
                y * (os_[0] + oy) + x * ty,
            )
            valid[x, y] = 1
    min_pos = pos[valid == 1].min(axis=0)
    pos[:, :] -= min_pos
    max_pos = pos[valid == 1].max(axis=0)

    channels = os_[2]
    out = np.zeros(
        (
            max_pos[1] + os_[0],
            max_pos[0] + os_[1],
            channels,
        )
    )
    sum_arr = np.zeros(out.shape[0:2])

    for x, d in tiles.items():
        for y, im in d.items():
            xp, yp = pos[x, y]
            for c in range(channels):
                out[yp : yp + os_[0], xp : xp + os_[1], c] += m * im[..., c]
            sum_arr[yp : yp + os_[0], xp : xp + os_[1]] += m

    indices = sum_arr != 0
    for c in range(channels):
        out[..., c] = np.divide(
            out[..., c], sum_arr, where=indices, out=np.zeros(sum_arr.shape)
        )
    return _as_dtype(dtype, out)


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


def compose_labels(
    tiles: dict[int, dict[int, np.ndarray[Any, np.dtype[Any]]]],
    ox: int = 0,
    oy: int = 0,
    tx: int = 0,
    ty: int = 0,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Compose label tiles into a single image (YXC, all tiles same shape).

    Unique label IDs are remapped. Overlapping labels on adjacent tiles
    are mapped to the same ID.

    See ``compose_tiles`` for details of the offset and translation
    parameters.

    Args:
        tiles: Dictionary of dictionaries of np.array tiles, keyed by [x][y].
        ox: Tile offset in x (use negative for overlap).
        oy: Tile offset in y (use negative for overlap).
        tx: Row translation in x.
        ty: Column translation in y.

    Returns:
        composed (np.array): The composed labels (YXC).
    """
    maxx = np.max(list(tiles.keys()))
    maxy = 0
    for x_dict in tiles.values():
        maxy = np.max(list(x_dict.keys()), initial=maxy)

    y = next(iter(tiles[maxx]))
    im = tiles[maxx][y]
    os_ = im.shape
    dtype = im.dtype

    pos = np.zeros((maxx + 1, maxy + 1, 2), dtype=int)
    valid = np.zeros((maxx + 1, maxy + 1), dtype=int)
    for x, d in tiles.items():
        for y in d:
            pos[x, y] = (
                x * (os_[1] + ox) + y * tx,
                y * (os_[0] + oy) + x * ty,
            )
            valid[x, y] = 1
    min_pos = pos[valid == 1].min(axis=0)
    pos[:, :] -= min_pos
    max_pos = pos[valid == 1].max(axis=0)

    channels = os_[2]
    out = [
        np.zeros(
            (
                max_pos[1] + os_[0],
                max_pos[0] + os_[1],
            ),
            dtype=dtype,
        )
        for i in range(channels)
    ]

    border = 0
    if ox < 0:
        border = -ox
    if oy < 0:
        border = max(border, -oy)
    if tx * ty > 0:
        border = max(border, abs(tx), abs(ty))

    for x, d in tiles.items():
        for y, im in d.items():
            xp, yp = pos[x, y]
            for c in range(channels):
                out[c] = merge_labels(
                    out[c], im[..., c], xp=xp, yp=yp, border=border
                )

    return np.dstack(out)


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


def stitch_from_positions(
    images: NDArray[Any],
    positions: list[tuple[float, float]],
    edge: int = 0,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[Any]:
    """Stitch images using their absolute stage positions.

    Args:
        images: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        positions: Stage positions per image, length N.
        edge: Edge blending width in pixels.
        overlap_x: Overlap in x-dimension.
        overlap_y: Overlap in y-dimension.
        translate_x: Row translation in x.
        translate_y: Column translation in y.

    Returns:
        Stitched array of shape (Y, X, C) or (T, Y, X, C).
    """
    ndim = images.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"

    grid_map = positions_to_grid(positions)

    def _build_tiles(
        source: NDArray[Any],
    ) -> dict[int, dict[int, NDArray[Any]]]:
        tiles: dict[int, dict[int, NDArray[Any]]] = {}
        for col, row_map in grid_map.items():
            tiles[col] = {}
            for row, idx in row_map.items():
                tiles[col][row] = source[idx]
        return tiles

    if ndim == 5:
        # (N, T, Y, X, C) → stitch per timepoint, then stack
        n_timepoints = images.shape[1]
        layers: list[NDArray[Any]] = []
        for t in range(n_timepoints):
            tiles = _build_tiles(images[:, t])
            layers.append(
                compose_tiles(
                    tiles,
                    ox=-overlap_x,
                    oy=-overlap_y,
                    tx=translate_x,
                    ty=translate_y,
                    edge=edge,
                )
            )
        return np.stack(layers)
    else:
        tiles = _build_tiles(images)
        return compose_tiles(
            tiles,
            ox=-overlap_x,
            oy=-overlap_y,
            tx=translate_x,
            ty=translate_y,
            edge=edge,
        )


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


def split_stitched_mask_to_fields(
    stitched_mask: NDArray[Any],
    positions: list[tuple[float, float]],
    tile_h: int,
    tile_w: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> list[NDArray[Any]]:
    """Inverse of ``recompose_split_labels`` for storage round-trip.

    Slices the stitched mask back into per-field tiles at each field's
    placement position. Labels in the overlap zone between adjacent tiles
    are shared with their neighbours by canvas-original ID (because the
    canvas-wide segmentation assigns globally unique IDs). ``recompose_split_labels``
    reassembles them losslessly via a non-zero copy, preserving the
    original label IDs and boundary-cell pixels.

    Args:
        stitched_mask: Stitched label canvas of shape (T, Y, X).
        positions: Stage positions per field, length N — same order as
            the original input to ``stitch_from_positions``.
        tile_h: Original per-field height in pixels.
        tile_w: Original per-field width in pixels.
        overlap_x: Overlap in x-dimension (matches stitching params).
        overlap_y: Overlap in y-dimension (matches stitching params).
        translate_x: Row translation in x (matches stitching params).
        translate_y: Column translation in y (matches stitching params).

    Returns:
        List of (T, tile_h, tile_w) mask arrays, one per field, in the
        same order as ``positions``.
    """
    if stitched_mask.ndim != 3:
        raise ValueError(
            f"stitched_mask must be (T, Y, X), got {stitched_mask.shape}"
        )

    n_fields = len(positions)
    grid_map = positions_to_grid(positions)
    pos, _ = _field_placements(
        grid_map,
        tile_w,
        tile_h,
        overlap_x,
        overlap_y,
        translate_x,
        translate_y,
    )

    n_t = stitched_mask.shape[0]
    # Pre-allocate list with a default image
    a = np.empty((0,))
    out: list[NDArray[Any]] = [a for _ in range(n_fields)]

    for col, row_map in grid_map.items():
        for row, idx in row_map.items():
            xp, yp = pos[col, row]
            out[idx] = stitched_mask[
                :, yp : yp + tile_h, xp : xp + tile_w
            ].copy()

    # Handle the possibility of a sparse grid causing a missing tile
    for idx, tile in enumerate(out):
        if len(tile) == 0:
            out[idx] = np.zeros(
                (n_t, tile_h, tile_w), dtype=stitched_mask.dtype
            )

    return out


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


def _field_placements(
    grid_map: dict[int, dict[int, int]],
    tile_w: int,
    tile_h: int,
    overlap_x: int,
    overlap_y: int,
    translate_x: int,
    translate_y: int,
) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    """Compute per-field (xp, yp) canvas positions from stage positions.

    Shared helper for ``split_stitched_mask_to_fields`` and
    ``recompose_split_labels`` — both need the same grid-to-canvas math.

    Returns:
        pos: (maxx+1, maxy+1, 2) array of (xp, yp) canvas positions.
        valid: (maxx+1, maxy+1) bool mask of which grid cells are occupied.
    """
    ox = -overlap_x
    oy = -overlap_y
    tx = translate_x
    ty = translate_y

    maxx = max(grid_map.keys())
    maxy = 0
    for d in grid_map.values():
        maxy = max(maxy, max(d.keys()))
    pos = np.zeros((maxx + 1, maxy + 1, 2), dtype=int)
    valid = np.zeros((maxx + 1, maxy + 1), dtype=bool)
    for x, d in grid_map.items():
        for y in d:
            pos[x, y] = (
                x * (tile_w + ox) + y * tx,
                y * (tile_h + oy) + x * ty,
            )
            valid[x, y] = True
    min_pos = pos[valid].min(axis=0)
    pos -= min_pos
    return pos, valid


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


def assign_field_by_centroid(
    centroids_yx: NDArray[np.floating[Any]],
    positions: list[tuple[float, float]],
    tile_h: int,
    tile_w: int,
    *,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[np.intp]:
    """Assign each centroid to the field whose tile owns it.

    Used by canvas-wide stitched segmentation to tag each measurement
    row with the OMERO image id of the field that owns the cell's
    centroid. Mirrors the (xp, yp) placements computed by
    ``_field_placements`` / ``split_stitched_mask_to_fields`` so that
    measurements and per-field masks agree on field ownership.

    For centroids that fall inside the overlap region of multiple
    tiles, the tile whose centre is nearest (Euclidean) is chosen --
    deterministic and intuitive. Centroids outside every tile rect
    (defensive: should not occur for cells discovered inside the
    canvas) fall back to the globally nearest tile centre.

    Args:
        centroids_yx: ``(N, 2)`` array of (y, x) centroid coordinates
            in canvas pixel space (matches regionprops ``centroid-0``,
            ``centroid-1``).
        positions: Stage positions per field, same ordering as the
            input to ``stitch_from_positions``.
        tile_h: Per-field tile height in pixels.
        tile_w: Per-field tile width in pixels.
        overlap_x: Stitch overlap in x (must match stitching params).
        overlap_y: Stitch overlap in y (must match stitching params).
        translate_x: Row translation in x (must match stitching params).
        translate_y: Column translation in y (must match stitching
            params).

    Returns:
        ``(N,)`` array of field indices into ``positions``.
    """
    centroids = np.asarray(centroids_yx, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids_yx must be (N, 2), got {centroids.shape}")

    grid_map = positions_to_grid(positions)
    pos, valid = _field_placements(
        grid_map,
        tile_w,
        tile_h,
        overlap_x,
        overlap_y,
        translate_x,
        translate_y,
    )

    # Flatten valid grid cells into per-field (idx, xp, yp) records, in
    # the same ordering as ``positions``.
    n_fields = len(positions)
    xps = np.zeros(n_fields, dtype=int)
    yps = np.zeros(n_fields, dtype=int)
    for col, row_map in grid_map.items():
        for row, idx in row_map.items():
            xp, yp = pos[col, row]
            xps[idx] = xp
            yps[idx] = yp

    cy = centroids[:, 0:1]  # (N, 1)
    cx = centroids[:, 1:2]
    # Tile centres for distance tie-break.
    centre_y = yps + tile_h / 2.0  # (K,)
    centre_x = xps + tile_w / 2.0
    dy = cy - centre_y[np.newaxis, :]  # (N, K)
    dx = cx - centre_x[np.newaxis, :]
    dist2 = dy * dy + dx * dx

    # Inside-rect mask: centroid in [xp, xp+tile_w) × [yp, yp+tile_h).
    inside = (
        (cx >= xps[np.newaxis, :])
        & (cx < (xps + tile_w)[np.newaxis, :])
        & (cy >= yps[np.newaxis, :])
        & (cy < (yps + tile_h)[np.newaxis, :])
    )

    # Among containing tiles, pick the nearest centre. Centroids inside
    # no rect (defensive) fall back to the globally nearest centre.
    masked = np.where(inside, dist2, np.inf)
    any_inside = inside.any(axis=1)
    chosen = np.where(
        any_inside,
        np.argmin(masked, axis=1),
        np.argmin(dist2, axis=1),
    )
    return chosen.astype(np.intp)


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


def recompose_split_labels(
    per_field_tiles: NDArray[Any] | list[NDArray[Any]],
    positions: list[tuple[float, float]],
    tile_h: int,
    tile_w: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[Any]:
    """Inverse of ``split_stitched_mask_to_fields``.

    Reassembles per-field label tiles into a single stitched canvas.

    **Invariant required**: the per-field tiles must come from a single
    canvas-wide segmentation, so label IDs are globally unique. Where the
    same label appears in two adjacent tiles (a cell straddling a tile
    boundary), the pixels are co-located by construction — a simple
    non-zero copy reassembles the canvas without renumbering or overlap
    logic. This function is **not** a general-purpose label merger; for
    independently-segmented tiles with name collisions, use
    ``stitch_labels_from_positions`` (which goes through ``merge_labels``).

    Accepts two input shapes:

    * ``list[NDArray]`` of ``(T, tile_h, tile_w)`` — the direct output of
      ``split_stitched_mask_to_fields``. Returns ``(T, Y, X)``.
    * ``NDArray`` of ``(N, tile_h, tile_w, C)`` or ``(N, T, tile_h, tile_w, C)``
      — matches the napari label-stack shape. Channels and timepoints are
      handled internally. Returns ``(Y, X, C)`` or ``(T, Y, X, C)``.

    Args:
        per_field_tiles: Per-field tiles (see input shapes above).
        positions: Stage positions per field, length N.
        tile_h: Per-field height in pixels (matches split inputs).
        tile_w: Per-field width in pixels (matches split inputs).
        overlap_x: Overlap in x-dimension (matches stitching params).
        overlap_y: Overlap in y-dimension (matches stitching params).
        translate_x: Row translation in x (matches stitching params).
        translate_y: Column translation in y (matches stitching params).

    Returns:
        Stitched canvas. Shape depends on input — see above.
    """
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

    if stacked.shape[0] != len(positions):
        raise ValueError(
            f"tile count ({stacked.shape[0]}) and positions "
            f"({len(positions)}) must match"
        )

    n_t = stacked.shape[1]
    n_c = stacked.shape[4]
    dtype = stacked.dtype

    grid_map = positions_to_grid(positions)
    pos, _ = _field_placements(
        grid_map,
        tile_w,
        tile_h,
        overlap_x,
        overlap_y,
        translate_x,
        translate_y,
    )

    # Canvas extent = furthest tile's far corner.
    max_xp = 0
    max_yp = 0
    for col, row_map in grid_map.items():
        for row in row_map:
            xp, yp = pos[col, row]
            max_xp = max(max_xp, xp + tile_w)
            max_yp = max(max_yp, yp + tile_h)

    canvas = np.zeros((n_t, max_yp, max_xp, n_c), dtype=dtype)

    for col, row_map in grid_map.items():
        for row, idx in row_map.items():
            xp, yp = pos[col, row]
            tile = stacked[idx]  # (T, H, W, C)
            for t in range(n_t):
                for c in range(n_c):
                    target = canvas[t, yp : yp + tile_h, xp : xp + tile_w, c]
                    src = tile[t, :, :, c]
                    np.copyto(target, src, where=src != 0)

    # Squeeze synthetic axes back out to match caller's input shape.
    if squeeze_c:
        canvas = canvas[..., 0]  # drop C → (T, Y, X)
    if squeeze_t:
        canvas = canvas[0]  # drop T → (Y, X, C) or (Y, X)
    return canvas


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
    ``stitch_labels_from_positions`` (which goes through ``merge_labels``).

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


def stitch_labels_from_positions(
    labels: NDArray[Any],
    positions: list[tuple[float, float]],
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[Any]:
    """Stitch label masks using their absolute stage positions.

    Args:
        labels: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        positions: Stage positions per image, length N.
        overlap_x: Overlap in x-dimension.
        overlap_y: Overlap in y-dimension.
        translate_x: Row translation in x.
        translate_y: Column translation in y.

    Returns:
        Stitched labels of shape (Y, X, C) or (T, Y, X, C).
    """
    ndim = labels.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"

    grid_map = positions_to_grid(positions)

    def _build_tiles(
        source: NDArray[Any],
    ) -> dict[int, dict[int, NDArray[Any]]]:
        tiles: dict[int, dict[int, NDArray[Any]]] = {}
        for col, row_map in grid_map.items():
            tiles[col] = {}
            for row, idx in row_map.items():
                tiles[col][row] = source[idx]
        return tiles

    if ndim == 5:
        # (N, T, Y, X, C) → stitch per timepoint, then stack
        n_timepoints = labels.shape[1]
        layers: list[NDArray[Any]] = []
        for t in range(n_timepoints):
            tiles = _build_tiles(labels[:, t])
            layers.append(
                compose_labels(
                    tiles,
                    ox=-overlap_x,
                    oy=-overlap_y,
                    tx=translate_x,
                    ty=translate_y,
                )
            )
        return np.stack(layers)
    else:
        tiles = _build_tiles(labels)
        return compose_labels(
            tiles,
            ox=-overlap_x,
            oy=-overlap_y,
            tx=translate_x,
            ty=translate_y,
        )


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
