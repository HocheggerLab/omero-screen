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

from typing import Any, cast

import numpy as np
import scipy.ndimage
from loguru import logger
from numpy.typing import NDArray
from omero_screen.config import is_level_enabled
from skimage.util import map_array

# Operetta stitching calibration constants. These are microscope-level
# values (not per-plate or per-well) and have been stable for the lab's
# Operetta acquisitions. Used by both the analysis pipeline (stitched
# segmentation) and the napari widget; the widget's _STITCH_DEFAULTS
# allows interactive override but defaults to these.
OPERETTA_STITCH_DEFAULTS: dict[str, int] = {
    "overlap_x": 7,
    "overlap_y": 7,
    "translate_x": -3,
    "translate_y": 3,
}


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
    ys = [p[1] for p in valid if p[0] is not None]
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


def _compute_overlap(
    clusters: list[float],
    tile_size_px: int,
    pixel_size: float,
    axis_label: str,
    fallback: int = 0,
) -> int:
    """Compute tile overlap in pixels from cluster spacing.

    If positions are in µm the spacing divided by pixel_size gives the
    step in pixels, and overlap = tile_size - step.  When the computed
    step is not plausible (outside 50 %–150 % of tile_size) the
    positions are assumed to be in a different unit and *fallback* is
    returned instead.
    """
    if len(clusters) < 2:
        return 0

    spacing = clusters[1] - clusters[0]
    step_px = spacing / pixel_size

    if 0.5 * tile_size_px < step_px < 1.5 * tile_size_px:
        overlap = max(0, int(round(tile_size_px - step_px)))
        return overlap

    logger.info(
        f"{axis_label} spacing {spacing:.6g} / {pixel_size:.6g} gives {step_px:.0f} px step (tile={tile_size_px:d} px) — positions likely not in µm, using fallback overlap {fallback:d}"
    )
    return fallback


# --------------------------------------------------------------------------
# Tile composition
# --------------------------------------------------------------------------


def positions_to_offsets(
    positions: list[tuple[float, float]],
    tile_w: int,
    tile_h: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
    translate_x: int = 0,
    translate_y: int = 0,
) -> NDArray[np.int_]:
    """Convert tile stage positions to canvas offsets.

    The clustering tolerance is determined automatically from the data.

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
    out = np.zeros((len(positions), 2), dtype=np.int_)
    for x, d in grid_map.items():
        for y, idx in d.items():
            out[idx] = pos[x, y]
    return out


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
    m = np.ones(tiles.shape[1:3], dtype=int)
    tile_h, tile_w = m.shape

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

    h1o = np.bincount(im1a.reshape(-1), weights=overlap.reshape(-1))
    h2o = np.bincount(im2.reshape(-1), weights=overlap.reshape(-1))
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
    m1 = len(h1)

    remove1 = []
    remove2 = []

    # Remap labels to use the ID from the object they overlap.
    # Greedy: largest overlap wins; subsequent overlaps remove pixels.
    for i, j, c, _ in overlaps:
        f1 = c / h1[i]
        f2 = c / h2[j]
        if f1 > f2:
            if map1[i]:
                remove1.append(i)
                continue
            if map2[j]:
                remove1.append(i)
                continue
            map2[j] = j + m1
            map1[i] = map2[j]
        else:
            if map2[j]:
                remove2.append(j)
                continue
            if map1[i]:
                remove2.append(j)
                continue
            map1[i] = i
            map2[j] = map1[i]

    if remove2:
        for v in remove2:
            im2[(im2 == v) & overlap] = 0
    if remove1:
        for v in remove1:
            im1a[(im1a == v) & overlap] = 0
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

    # Compress IDs to ascending from 1
    u_ints = {int(x) for x in map1}
    u_ints.update(int(x) for x in map2)
    u_ints.add(0)
    m = np.zeros(max(u_ints) + 1, dtype=np.uint16)
    for i, v in enumerate(sorted(u_ints)):
        m[v] = np.uint16(i)

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
        offsets: Array of [K, (ox, oy)] (must be positive).
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
    _validate_offsets(offsets)

    xps = offsets[:, 0:1].T  # (1, K)
    yps = offsets[:, 1:2].T

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
    )
    return chosen.astype(np.intp)


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
