"""Position-based stitching using absolute stage coordinates.

Converts stage positions (in any consistent unit) into a tile grid,
computes overlap in pixels when possible, and delegates to the existing
``compose_tiles`` / ``compose_labels`` blending functions.

Stage positions from OMERO may be in micrometers, millimeters, or
reference-frame units.  The clustering tolerance is computed adaptively
from the gap distribution so it works regardless of unit.
"""

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
    n_cells = len(x_clusters) * len(y_clusters)
    if n_cells < len(valid):
        logger.warning(
            "Stage positions form %d grid cells for %d images — "
            "cannot stitch without losing data. Positions (first 5): %s",
            n_cells,
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
    sorted_v = sorted(values)
    if len(sorted_v) < 2:
        return 0.0

    gaps = [sorted_v[i + 1] - sorted_v[i] for i in range(len(sorted_v) - 1)]
    gaps = [g for g in gaps if g > 0]  # drop zero-gaps (duplicates)
    if not gaps:
        return 0.0

    min_gap = min(gaps)
    max_gap = max(gaps)

    if max_gap / min_gap < 10:
        # Gaps are similar — clean grid, no noise
        return min_gap * 0.25
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
    tile_shape_yx: tuple[int, int],
    pixel_size: tuple[float, float],
    fallback_overlap: tuple[int, int] = (0, 0),
) -> tuple[dict[int, dict[int, int]], int, int]:
    """Convert stage positions to a tile grid with overlap values.

    The clustering tolerance is determined automatically from the data.
    Overlap is computed by interpreting position spacing as µm; when
    the result is implausible (spacing not close to tile size) the
    *fallback_overlap* is used instead.

    Args:
        positions: List of (pos_x, pos_y) for each image.
        tile_shape_yx: (height, width) of a single tile in pixels.
        pixel_size: (pixel_size_x, pixel_size_y) in µm/pixel.
        fallback_overlap: (overlap_x, overlap_y) in pixels, used when
            positions are not in µm and overlap cannot be computed.

    Returns:
        Tuple of (grid_map, overlap_x, overlap_y) where grid_map is
        ``dict[col][row] = image_index`` and overlaps are in pixels.
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

    # Compute overlap from spacing
    tile_h, tile_w = tile_shape_yx
    px_size_x, px_size_y = pixel_size

    fb_x, fb_y = fallback_overlap
    overlap_x = _compute_overlap(x_clusters, tile_w, px_size_x, "X", fb_x)
    overlap_y = _compute_overlap(y_clusters, tile_h, px_size_y, "Y", fb_y)

    n_cells = sum(len(rows) for rows in grid_map.values())
    logger.info(
        "Position grid: %d cols x %d rows (%d cells for %d images), "
        "overlap=(%d, %d) px, tol=(%.6g, %.6g)",
        len(x_clusters),
        len(y_clusters),
        n_cells,
        len(positions),
        overlap_x,
        overlap_y,
        tol_x,
        tol_y,
    )
    return grid_map, overlap_x, overlap_y


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
        "%s spacing %.6g gives %.0f px step (tile=%d px) — "
        "positions likely not in µm, using fallback overlap %d",
        axis_label,
        spacing,
        step_px,
        tile_size_px,
        fallback,
    )
    return fallback


def stitch_from_positions(
    images: NDArray[Any],
    positions: list[tuple[float, float]],
    pixel_size: tuple[float, float],
    rotation: float = 0.0,
    edge: int = 0,
    mode: str = "reflect",
    fallback_overlap: tuple[int, int] = (0, 0),
) -> NDArray[Any]:
    """Stitch images using their absolute stage positions.

    Args:
        images: Array of shape (N, Y, X, C) or (N, T, Y, X, C).
        positions: Stage positions per image, length N.
        pixel_size: (pixel_size_x, pixel_size_y) in µm/pixel.
        rotation: Rotation angle in degrees.
        edge: Edge blending width in pixels.
        mode: Fill mode for rotation.
        fallback_overlap: (overlap_x, overlap_y) in pixels, used when
            positions are not in µm.

    Returns:
        Stitched array of shape (Y, X, C) or (T, Y, X, C).
    """
    ndim = images.ndim
    assert ndim in (4, 5), f"Expected 4D or 5D images, got {ndim}D"

    if ndim == 5:
        tile_shape_yx = (images.shape[2], images.shape[3])
    else:
        tile_shape_yx = (images.shape[1], images.shape[2])

    grid_map, overlap_x, overlap_y = positions_to_grid(
        positions, tile_shape_yx, pixel_size, fallback_overlap=fallback_overlap
    )

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
                    rotation=rotation,
                    ox=-overlap_x,
                    oy=-overlap_y,
                    edge=edge,
                    mode=mode,
                )
            )
        return np.stack(layers)
    else:
        tiles = _build_tiles(images)
        return compose_tiles(
            tiles,
            rotation=rotation,
            ox=-overlap_x,
            oy=-overlap_y,
            edge=edge,
            mode=mode,
        )


def stitch_labels_from_positions(
    labels: NDArray[Any],
    positions: list[tuple[float, float]],
    pixel_size: tuple[float, float],
    rotation: float = 0.0,
    fallback_overlap: tuple[int, int] = (0, 0),
) -> NDArray[Any]:
    """Stitch label masks using their absolute stage positions.

    Args:
        labels: Array of shape (N, Y, X, C).
        positions: Stage positions per image, length N.
        pixel_size: (pixel_size_x, pixel_size_y) in µm/pixel.
        rotation: Rotation angle in degrees.
        fallback_overlap: (overlap_x, overlap_y) in pixels, used when
            positions are not in µm.

    Returns:
        Stitched labels of shape (Y, X, C).
    """
    tile_shape_yx = (labels.shape[1], labels.shape[2])

    grid_map, overlap_x, overlap_y = positions_to_grid(
        positions, tile_shape_yx, pixel_size, fallback_overlap=fallback_overlap
    )

    tiles: dict[int, dict[int, NDArray[Any]]] = {}
    for col, row_map in grid_map.items():
        tiles[col] = {}
        for row, idx in row_map.items():
            tiles[col][row] = labels[idx]

    return compose_labels(
        tiles, rotation=rotation, ox=-overlap_x, oy=-overlap_y
    )
