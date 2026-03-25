"""Position-based stitching using absolute stage coordinates.

Converts stage positions (in any consistent unit) into a tile grid,
computes overlap in pixels when possible, and delegates to the existing
``compose_tiles`` / ``compose_labels`` blending functions.

Stage positions from OMERO may be in micrometers, millimeters, or
reference-frame units.  The clustering tolerance is computed adaptively
from the gap distribution so it works regardless of unit.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omero_screen.config import get_logger

from omero_screen_napari.welldata_api import compose_labels, compose_tiles

logger = get_logger(__name__)


def has_valid_positions(
    positions: list[tuple[float, float] | None],
) -> bool:
    """Return True if positions can be used for stitching.

    Requires at least 2 positions, all non-None, and positions that
    span more than one grid cell (i.e. not all identical).
    """
    if len(positions) < 2 or not all(p is not None for p in positions):
        return False

    # Check that positions actually form a grid (not all at the same spot)
    valid = [p for p in positions if p is not None]
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
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
    gaps = gaps[gaps > 0]  # drop zero-gaps (duplicates)
    if len(gaps) == 0:
        return 0.0

    min_gap = gaps.min()
    max_gap = gaps.max()

    if max_gap / min_gap < 10:
        # Gaps are similar — clean grid, no noise
        return float(min_gap * 0.25)
    # Clear bimodal distribution: geometric mean separates the two modes
    return float(np.sqrt(min_gap * max_gap))


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    """Group sorted values within *tolerance* and return cluster centroids.

    Args:
        values: Coordinate values to cluster.
        tolerance: Maximum distance between consecutive values in the
            same cluster.

    Returns:
        Sorted list of cluster centroid values.
    """
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

    # Map each image to (col, row)
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
        # The grid computation works on the assumption the
        # rows and columns are othogonal and aligned to the x/y axes.
        maxx = np.max(list(grid_map.keys()))
        maxy = 0
        for x_dict in grid_map.values():
            maxy = np.max(list(x_dict.keys()), initial=maxy)
        grid = []
        # row/column translations between adjacent images
        rx, ry = [], []
        cx, cy = [], []
        empty: dict[int, int] = {}
        # rows
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
        # columns
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

    # Sanity check: step should be roughly one tile width
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
        pixel_size: (pixel_size_x, pixel_size_y) in µm/pixel.
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
        pixel_size: (pixel_size_x, pixel_size_y) in µm/pixel.
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
