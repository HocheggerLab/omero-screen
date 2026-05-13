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

The clustering tolerance used to derive the grid is computed
adaptively from the gap distribution so positions in µm, mm, or
reference-frame units all work.

Used by both the analysis pipeline (segment-on-stitched-canvas) and
the napari widget (display stitched wells); kept in omero-utils to
avoid a circular dependency between those packages.
"""

import logging
from typing import Any, cast

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
from omero_screen.config import get_logger
from skimage.util import map_array

logger = get_logger(__name__)


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
        if logger.isEnabledFor(logging.DEBUG):
            if len(valid) < len(positions):
                logger.debug(
                    "Missing positions: %d < %d", len(valid), len(positions)
                )
            else:
                logger.debug("Not enough positions: %d", len(positions))
        return False

    xs = [p[0] for p in valid if p[0] is not None]
    ys = [p[1] for p in valid if p[0] is not None]
    if min(len(xs), len(ys)) < len(positions):
        logger.debug(
            "Missing X/Y positions: %d,%d < %d",
            len(xs),
            len(ys),
            len(positions),
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
            "Stage positions form %d grid cells for %d images — "
            "cannot stitch without losing data. Positions (first 5): %s",
            len(location),
            len(valid),
            valid[:5],
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
        "Position grid: %d cols x %d rows (%d cells for %d images)",
        len(x_clusters),
        len(y_clusters),
        n_cells,
        len(positions),
    )

    if n_cells and logger.isEnabledFor(logging.DEBUG):
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

        logger.debug(
            "Position grid: %s; row %.3f,%.3f +/- %.3f,%.3f, col %.3f,%.3f +/- %.3f,%.3f (raw units)",
            grid,
            np.mean(rx),
            np.mean(ry),
            np.std(rx),
            np.std(ry),
            np.mean(cx),
            np.mean(cy),
            np.std(cx),
            np.std(cy),
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
        "%s spacing %.6g / %.6g gives %.0f px step (tile=%d px) — "
        "positions likely not in µm, using fallback overlap %d",
        axis_label,
        spacing,
        pixel_size,
        step_px,
        tile_size_px,
        fallback,
    )
    return fallback


# --------------------------------------------------------------------------
# Tile composition
# --------------------------------------------------------------------------


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
