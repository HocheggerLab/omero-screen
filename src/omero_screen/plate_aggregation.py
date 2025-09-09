"""Aggregates OMERO screen results across plates.

This module provides aggregation of OMERO screen results from repeat experiments on the
same plate using different channel stains. Each plate should contain the same objects.
Each subsequent plate is aligned to the master plate using cross-correlation of the
nucleus channel. The alignment is used to shift the centroids of objects in the
OMERO screen results. Objects are then assigned to the master plate objects and
the screen data combined.
"""

import os
import random
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial
from ezomero import get_image
from matplotlib.collections import LineCollection
from numpy.fft import fft2, ifft2
from omero.gateway import (
    BlitzGateway,
    BlitzObjectWrapper,
    ImageWrapper,
    WellSampleWrapper,
)
from omero.model import Length
from omero_utils.attachments import (
    attach_data,
    attach_figure,
    delete_file_attachment,
    get_file_attachments,
    parse_csv_data,
)
from omero_utils.message import OmeroError, PlateDataError, PlateNotFoundError
from scipy.optimize import linear_sum_assignment

from omero_screen.config import get_logger

from .metadata_parser import MetadataParser
from .plate_dataset import PlateDataset

logger = get_logger(__name__)


def aggregate_plates(
    conn: BlitzGateway,
    plate_id: int,
    alignments: pd.DataFrame,
    threshold: float = 25,
    method: int = 0,
    std_distance: float = 6,
) -> tuple[pd.DataFrame, list[list[str | int]]]:
    """Aggregate multiple OMERO screen plate results.

    The centroids are updated for each repeat plate using the specified alignment shifts
    per well and then mapped to the master plate. Mappings are performed per well sample.
    Any mapping distance more than a factor of the standard deviation above the mean are
    filtered (mean + f * std). A concatenated results data frame is uploaded to the master
    plate.

    The image mapping across plates is returned. Each entry contains the well position
    and the images IDs that are mapped for each well sample, e.g.
    [['A1', 1, 11, 111], ['A1', 2, 22, 222], ['B1', 3, 33, 333], ['B2', 4, 44, 444]].
    The image IDs will correspond to the 'image_id[.n]' column in the data frame. The
    value n corresponds to the index of plate repeat.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the master plate
        alignments: Alignment well shifts (plate, well, x, y) for each repeat plate
        threshold: Distance threshold for alignment mappings
        method: Mapping method (0: linear sum assignment; 1: KD-tree linear sum assignment; 2: Greedy nearest neighbour; 3: Mask overlap)
        std_distance: Number of standard deviations from the mean to exclude mappings

    Returns:
        the aggregated data; and the image mapping

    Raises:
        PlateNotFoundError: if a plate does not exist
        PlateDataError: if plates are missing the OMERO screen results, or if the alignment is missing for a well
    """
    logger.info("Aggregating to master plate: %d", plate_id)
    # Download OMERO screen results
    df1 = _get_results(conn, plate_id)
    map1 = _get_mask_map(conn, plate_id) if method == 3 else {}
    # Get plates
    plate_ids = alignments["plate"].unique()
    # Result centroids are per well sample identified in the results as (well position, image ID)
    images1 = _get_well_images(conn, plate_id)
    image_map: list[list[str | int]] = [list(x) for x in images1]
    for index, plate_other in enumerate(plate_ids):
        logger.info("Mapping plates: %d - %d", plate_id, plate_other)
        df2 = _get_results(conn, plate_other)
        map2 = _get_mask_map(conn, plate_other) if method == 3 else {}
        # Extract centroids: shape = n x 2
        # centroid-0 is first dimension of YX image => Y centre; centroid-1 => X.
        # Update coordinates with alignment shift per well.
        plate_alignments = alignments[alignments["plate"] == plate_other]

        for _, well, x, y in plate_alignments.itertuples(index=False):
            mask = df2["well"] == well
            c2 = df2[mask][["centroid-1", "centroid-0"]].values
            c2 = c2 + (x, y)
            df2.loc[mask, ["centroid-1", "centroid-0"]] = c2
        # Map each result to the original table using minimum Euclidean distance.
        # This must be done per well sample.
        images2 = _get_well_images(conn, plate_other)
        # Append the image ID to the map
        for x, y in zip(image_map, images2, strict=True):
            x.append(y[1])
        all_results = []
        for im1, im2 in zip(images1, images2, strict=True):
            assert im1[0] == im2[0], "Well positions must match"
            df1w = _select_well_sample(df1, im1[0], im1[1])
            df2w = _select_well_sample(df2, im2[0], im2[1])
            # df1 can be the result of concatenation with unmapped objects.
            # Drop NA values from df1 when collecting centroids.
            c1 = df1w[["centroid-1", "centroid-0"]].dropna(axis=0).values
            c2 = df2w[["centroid-1", "centroid-0"]].values

            logger.info(
                "Mapping objects [%s:%d-%d] %d - %d",
                im1[0],
                im1[1],
                im2[1],
                len(c1),
                len(c2),
            )

            # create a mapping:
            # row_ind -> col_ind contains the mapping from df1 to df2
            if len(c1) == 0 or len(c2) == 0:
                row_ind = np.array([], dtype=int)
                col_ind = row_ind.copy()
            elif method == 0:
                row_ind, col_ind = map_full_linear_sum(c1, c2, threshold)
            elif method == 1:
                row_ind, col_ind = map_partial_linear_sum(c1, c2, threshold)
            elif method == 2:
                row_ind, col_ind = map_nearest_neighbour(c1, c2, threshold)
            else:
                a = plate_alignments[plate_alignments["well"] == im1[0]]
                if a.empty:
                    raise OmeroError(
                        f"Alignment missing for plate {plate_other} at well {im1[0]}",
                        logger,
                    )
                x, y = a.iloc[0][-2:]
                label1, label2 = map_masks(
                    _get_mask_from_map(conn, im1[1], map1),
                    _get_mask_from_map(conn, im2[1], map2),
                    (round(y), round(x)),
                )
                # print(label1, len(label1))
                # print(label2)
                # Convert label1 -> label2 map to the row index -> col index
                # Note: df1 int columns are converted to float to handle NAs during concatentation
                row_ind = _map_label_to_index(
                    label1, df1w["label"].dropna(axis=0).values.astype(np.int_)
                )
                col_ind = _map_label_to_index(
                    label2, np.array(df2w["label"].values)
                )
                # print(row_ind, len(row_ind))
                # print(col_ind)
                # Drop unmapped labels
                selected = (row_ind >= 0) & (col_ind >= 0)
                row_ind = row_ind[selected]
                col_ind = col_ind[selected]

            logger.info(
                "Mapped %d / %d",
                len(row_ind),
                min(len(c1), len(c2)),
            )

            if len(row_ind) == 0:
                logger.warning(
                    "No mappings between %s and %s",
                    im1,
                    im2,
                )
            elif std_distance > 0 and method < 3:
                # Remove outliers from the distance mappings
                diff = c1[row_ind] - c2[col_ind]
                diff = diff * diff
                dist = np.sqrt(diff.sum(axis=1))
                mean = np.mean(dist)
                std = np.std(dist)
                if std:
                    selected = dist < (mean + std_distance * std)

                    # debug the worst deviations
                    dist = np.sort(dist)
                    dist = (dist - mean) / std
                    logger.info(
                        "Big deviations: %s (%d)", dist[-10:], len(dist)
                    )
                    q3 = np.quantile(dist, 0.75)
                    iqr = q3 - np.quantile(dist, 0.25)
                    selected2 = dist > (q3 + 1.5 * iqr)
                    logger.info(
                        "Outliers: %s (%d)", dist[selected2], len(dist)
                    )

                    row_ind = row_ind[selected]
                    col_ind = col_ind[selected]
                    logger.info(
                        "Distances within %.1f + %.1f * %.1f: %d / %d",
                        mean,
                        std_distance,
                        std,
                        len(row_ind),
                        min(len(c1), len(c2)),
                    )

            # Join results. Rename columns to preserve all data.
            df2w.rename(
                columns={x: x + f".{index}" for x in df2w.columns},
                inplace=True,
            )
            new_index = np.arange(len(df1w), len(df1w) + len(df2w))
            new_index[col_ind] = row_ind
            df2w.set_index(new_index, inplace=True)
            df1w = pd.concat([df1w, df2w], axis=1, join="outer")
            # Note: df1 may contain rows that are null for the master plate, well and image.
            # These must be preserved in the next iteration.
            df1w[["plate_id", "well", "image_id"]] = plate_id, im1[0], im1[1]
            all_results.append(df1w)
        # end well samples

        # Rebuild the data
        df1 = pd.concat(all_results, axis=0, ignore_index=True)

    # Upload concatenated result
    logger.info("Saving results to OMERO")
    delete_file_attachment(
        conn, conn.getObject("Plate", plate_id), ends_with="agg_data.csv"
    )
    attach_data(conn, df1, conn.getObject("Plate", plate_id), "agg_data")

    return df1, image_map


def align_plates(
    conn: BlitzGateway,
    plate_id: int,
    plate_ids: list[int],
    align_ch: str = "DAPI",
    number_of_alignments: int = 5,
    threshold: float = 100,
    tolerance: float = 5,
    seed: int | None = None,
    output_alignments: bool = False,
) -> tuple[pd.DataFrame, list[list[npt.NDArray[Any]]] | None]:
    """Align plates in 2D using the specified channel.

    A random subset of well samples are used to create an average alignment between plates.

    Each alignment must be with the given distance threshold. All alignments must be within
    the given distance tolerance of the centroid. An alignment results data frame is uploaded
    to the master plate.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the master plate
        plate_ids: IDs of the plate repeat experiments
        align_ch: Channel to use for alignment
        number_of_alignments: Number of alignments used to create the average per well position
        threshold: Distance threshold for alignments
        tolerance: Distance threshold for all alignments to the centroid
        seed: Seed for random selection of well samples
        output_alignments: Set to True to return the alignment images

    Returns:
        DataFrame of alignment shifts (X,Y) required to align each plate well to the master plate,
        and optionally the alignments of each plate to the master plate (NYXC).

    Raises:
        PlateNotFoundError: if a plate does not exist
        PlateDataError: if plates do not have compatible dimensions, have multiple Z or T dimensions,
            are missing the alignment channel, or are missing the OMERO screen results
        OmeroError: if plates do not align below the threshold, or if the alignments are not within
            the tolerance to the centroid
    """
    logger.info("Aligning to master plate: %d", plate_id)
    # Check plates are compatible
    plate_dim = _plate_dimensions(conn, plate_id)
    # Currently only support T=Z=1
    for x in plate_dim:
        if x[-4] != 1 and x[-3] != 1:
            raise PlateDataError(
                f"Plate {plate_id} must have T=Z=1: TZYX={x[-4:]}", logger
            )
    for plate_other in plate_ids:
        plate_dim2 = _plate_dimensions(conn, plate_other)
        if plate_dim != plate_dim2:
            # Show first error
            msg = "NA"
            if len(plate_dim) != len(plate_dim2):
                msg = f"Well count mismatch: {len(plate_dim)} != {len(plate_dim2)}"
            else:
                for a, b in zip(plate_dim, plate_dim2, strict=False):
                    if a != b:
                        msg = f"Well mismatch: {a} != {b}"
                        break
            raise PlateDataError(
                f"Plate {plate_id} and plate {plate_other} dimension mismatch: {msg}",
                logger,
            )

    # Check all plates have alignment channel (use MetadataParser)
    metadata = {x: MetadataParser(conn, x) for x in plate_ids}
    metadata[plate_id] = MetadataParser(conn, plate_id)
    for meta in metadata.values():
        meta.manage_metadata()
        if align_ch not in meta.channel_data:
            raise PlateDataError(
                f"Plate {meta.plate_id} is missing alignment channel: {align_ch}",
                logger,
            )

    # Select random well sample images for alignment
    well_samples1 = _get_well_samples(conn, plate_id)
    if seed is None:
        seed = int.from_bytes(os.urandom(8))
    logger.info("Selecting well samples using seed: %d", seed)
    random.seed(a=seed)
    # Shuffle the indices and use the top n for samples.
    # We skip frames if the image is blank so we need the entire list.
    selected_indices = list(range(len(next(iter(well_samples1.values())))))
    random.shuffle(selected_indices)
    # list: plate,well,x,y
    alignments = []
    examples = []
    ch1 = int(metadata[plate_id].channel_data[align_ch])
    # XYZCT order
    start_coords = [0, 0, 0, ch1, 0]
    axis_lengths = [plate_dim[0][-1], plate_dim[0][-2], 1, 1, 1]
    for plate_other in plate_ids:
        logger.info("Aligning plates: %d - %d", plate_id, plate_other)
        well_samples2 = _get_well_samples(conn, plate_other)
        ch2 = int(metadata[plate_other].channel_data[align_ch])
        start_coords2 = start_coords.copy()
        start_coords2[-2] = ch2
        shifted = []
        # Compute the shift for each well
        for well, samples1 in well_samples1.items():
            shifts = []
            samples2 = well_samples2[well]
            # Perform alignment on selected channel
            for idx in selected_indices:
                _, im1 = get_image(
                    conn,
                    samples1[idx].getImage().getId(),
                    start_coords=start_coords,
                    axis_lengths=axis_lengths,
                )
                _, im2 = get_image(
                    conn,
                    samples2[idx].getImage().getId(),
                    start_coords=start_coords2,
                    axis_lengths=axis_lengths,
                )
                # Skip empty frames
                b1 = (im1 == 0).all()
                if b1 or (im2 == 0).all():
                    logger.warning(
                        "Skipping empty frame alignment %s [%s] from plate %d",
                        well,
                        idx,
                        plate_id if b1 else plate_other,
                    )
                    continue
                # Convert TZYXC to YX before alignment
                # Q. Would a Gaussian blur improve alignment?
                trans = _translation(im1.squeeze(), im2.squeeze())
                logger.info("Sample alignment %s [%s] %s", well, idx, trans)
                if output_alignments:
                    shifted.append(
                        _translate(im1.squeeze(), im2.squeeze(), trans)
                    )
                shifts.append(trans)
                if len(shifts) >= number_of_alignments:
                    break
            if output_alignments:
                examples.append(shifted)
            # Validate alignment. If we skipped all frames use no translation.
            if not shifts:
                shifts.append((0, 0))
            distances = np.array(
                [np.sqrt((np.array(x) ** 2).sum()) for x in shifts]
            )
            logger.info("Alignment distances: %s", distances)
            if np.any(distances >= threshold):
                raise OmeroError(
                    f"Plate alignment {plate_id} to {plate_other} [{well}] above distance threshold {threshold}: {distances[distances >= threshold]}",
                    logger,
                )
            # Compute mean. Reverse YX to XY shifts.
            a = np.array(shifts).mean(axis=0)
            b = np.array(shifts).std(axis=0)
            shift = (a[1], a[0])
            logger.info(
                "Alignment: (%.1f, %.1f) +/- (%.1f, %.1f)",
                shift[0],
                shift[1],
                b[1],
                b[0],
            )
            # Validate all alignments are within a tolerance to the mean
            distances = np.array(
                [np.sqrt(((np.array(x) - a) ** 2).sum()) for x in shifts]
            )
            logger.info("Alignment distances to centroid: %s", distances)
            if np.any(distances >= tolerance):
                raise OmeroError(
                    f"Plate alignment {plate_id} to {plate_other} [{well}] above distance tolerance {tolerance} to centroid: {distances[distances >= tolerance]}",
                    logger,
                )
            alignments.append((plate_other, well) + shift)

    df = pd.DataFrame(alignments, columns=["plate", "well", "x", "y"])

    # Upload result
    logger.info("Saving alignment results to OMERO")
    delete_file_attachment(
        conn, conn.getObject("Plate", plate_id), ends_with="alignment.csv"
    )
    attach_data(conn, df, conn.getObject("Plate", plate_id), "alignment")

    return df, examples if examples else None


def _plate_dimensions(
    conn: BlitzGateway, plate_id: int
) -> list[tuple[int, int, int, int, int, int, int]]:
    """Create a sorted list of the dimensions of the plate wells.

    The dimensions are: row, column, well index, image TZYX.
    The list allows comparison for compatibility between plates for each well position
    by comparing the well samples returned in order from the well.

    Note: Channels are ignored as these are not required for compatibilty.

    Args:
        conn: The BlitzGateway connection
        plate_id: The plate ID

    Returns:
        sorted list of dimensions
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise PlateNotFoundError("Plate:{plate_id}", logger)
    # Add the well index. Well samples must be returned in the same order for the plates to be compatible.
    samples = []
    for well in plate.listChildren():
        samples.extend(
            [
                (
                    well.row,
                    well.column,
                    i,
                    # Well positions are based on the physical placement of the plate in the reader.
                    # Repeat experiments will have small differences.
                    # TODO: Can the WellSample provide other information?
                    # Can we assign well samples to a grid and return the grid position?
                    # _get_length_in_m(ws, ws.getPosX()),
                    # _get_length_in_m(ws, ws.getPosY()),
                )
                + _get_dimensions(ws.getImage())
                for i, ws in enumerate(well.listChildren())
            ]
        )
    if not samples:
        raise PlateNotFoundError("Plate:{plate_id} has no wells", logger)
    samples.sort()
    return samples


def _get_length_in_m(obj: BlitzObjectWrapper, position: Length) -> float:
    """Get the length in metres using the Blitz object to convert the units.

    The input length must have units of omero.model.enums.UnitsLength
    """
    if position is None:
        return 0
    return float(obj._unwrapunits(position, units="METER").getValue())


def _get_dimensions(image: ImageWrapper) -> tuple[int, int, int, int]:
    """Get the image dimensions as a TZYX tuple."""
    return (
        image.getSizeT(),
        image.getSizeZ(),
        image.getSizeY(),
        image.getSizeX(),
    )


def _get_results(conn: BlitzGateway, plate_id: int) -> pd.DataFrame:
    """Get the OMERO screen results for the plate sorted by well position."""
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise PlateNotFoundError("Plate:{plate_id}", logger)
    filename = "final_data.csv"
    att = get_file_attachments(plate, filename)
    df = None
    if att:
        df = parse_csv_data(att[0])
    if df is None:
        raise PlateDataError(
            f"Plate {plate_id} is missing OMERO screen result data: {filename}",
            logger,
        )
    # Sort by well position
    df.sort_values(
        ["well_id", "image_id", "label"], inplace=True, ignore_index=True
    )
    return df


def _select_well_sample(
    df: pd.DataFrame, well: str, image_id: int
) -> pd.DataFrame:
    """Get a dataframe for the specified well sample."""
    mask1 = df["well"].values == well
    mask2 = df["image_id"].values == image_id
    df1 = df[mask1 & mask2].copy()
    df1.reset_index(drop=True, inplace=True)
    # mypy identifies this as Series[Any] and not a DataFrame
    return df1  # type: ignore[return-value]


def _get_well_samples(
    conn: BlitzGateway, plate_id: int
) -> dict[str, list[WellSampleWrapper]]:
    """Create a list of well samples for each well position in the plate, ordered sample index.

    Args:
        conn: The BlitzGateway connection
        plate_id: The plate ID

    Returns:
        dictionary of well position to list of well sample IDs
    """
    plate = conn.getObject("Plate", plate_id)
    assert plate is not None
    wells = []
    for well in plate.listChildren():
        wells.append(
            (
                well.row,
                well.column,
                well.getWellPos(),
                list(well.listChildren()),
            )
        )
    wells.sort()
    return {x[-2]: x[-1] for x in wells}


def _get_well_images(
    conn: BlitzGateway, plate_id: int
) -> list[tuple[str, int]]:
    """Create a list of well sample images for the plate, ordered by row, column then sample index.

    Args:
        conn: The BlitzGateway connection
        plate_id: The plate ID

    Returns:
        list of well sample positions and image Id, e.g. ('A1', 123)
    """
    plate = conn.getObject("Plate", plate_id)
    assert plate is not None
    wells = []
    for well in plate.listChildren():
        wells.extend(
            [
                (
                    well.row,
                    well.column,
                    i,
                    well.getWellPos(),
                    ws.getImage().getId(),
                )
                for i, ws in enumerate(well.listChildren())
            ]
        )
    wells.sort()
    return [x[-2:] for x in wells]


def _translation(
    im1: npt.NDArray[Any], im2: npt.NDArray[Any]
) -> tuple[int, int]:
    """Compute the alignment between 2D images using phase correlation.

    Args:
        im1: First image
        im2: Second image

    Returns:
        [x, y] translation to shift im2 onto im1
    """
    shape = im1.shape
    f0 = fft2(im1)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return int(t0), int(t1)


def _translate(
    im1: npt.NDArray[Any],
    im2: npt.NDArray[Any],
    trans: tuple[int, int],
    stacked: bool = True,
) -> npt.NDArray[Any]:
    """Translate image 2 onto image 1, returning as CYX.

    Args:
        im1: First image
        im2: Second image
        trans: Translation (YX)
        stacked: Set to True to return a CYX stack, otherwise return the shifted second image

    Returns:
        im2 shifted onto im1 as a CYX stack, or a separate image
    """
    shape = im1.shape
    # Extract two rectangles and obtain the intersection at image 1
    a = np.array([[0, 0], [shape[0], shape[1]]])
    b = np.array([[0, 0], [shape[0], shape[1]]]) + trans
    i1 = np.array(
        [
            np.max(np.array([a[0], b[0]]), axis=0),  # max of the minimums
            np.min(np.array([a[1], b[1]]), axis=0),  # min of the maximums
        ]
    )
    # create source crop
    i2 = i1 - trans
    # translate image 2
    im3 = np.zeros_like(im2)
    logger.debug(
        "translate %s + %s : %d:%d, %d:%d = %d:%d, %d:%d",
        shape,
        trans,
        i1[0][0],
        i1[1][0],
        i1[0][1],
        i1[1][1],
        i2[0][0],
        i2[1][0],
        i2[0][1],
        i2[1][1],
    )
    im3[i1[0][0] : i1[1][0], i1[0][1] : i1[1][1]] = im2[
        i2[0][0] : i2[1][0], i2[0][1] : i2[1][1]
    ]
    if stacked:
        # CYX format
        return np.stack([im1, im3])
    return im3


def map_full_linear_sum(
    c1: npt.NDArray[Any], c2: npt.NDArray[Any], threshold: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Minimum weight bipartite graph matching using full cost matrix of Euclidean distance.

    Args:
        c1: Coordinates 1
        c2: Coordinates 1
        threshold: Distance threshold for alignment mappings

    Returns:
        Mapping from row -> column
    """
    # Dense matrix
    cost = np.zeros((len(c1), len(c2)))
    for i, v1 in enumerate(c1):
        for j, v2 in enumerate(c2):
            d = v1 - v2
            cost[i][j] = np.sqrt((d * d).sum())
    logger.info("Computed distance matrix")
    row_ind, col_ind = linear_sum_assignment(cost)

    # Ignore large distance mappings
    for i, (r, c) in enumerate(zip(row_ind, col_ind, strict=True)):
        if cost[r][c] > threshold:
            row_ind[i], col_ind[i] = -1, -1
    selected = row_ind >= 0
    row_ind = row_ind[selected]
    col_ind = col_ind[selected]
    return row_ind, col_ind


def map_partial_linear_sum(
    c1: npt.NDArray[Any], c2: npt.NDArray[Any], threshold: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Minimum weight bipartite graph matching using partial cost matrix of Euclidean distance.

    Args:
        c1: Coordinates 1
        c2: Coordinates 1
        threshold: Distance threshold for alignment mappings

    Returns:
        Mapping from row -> column
    """
    # Dense matrix built using KD-Trees with some false edges.
    tree1 = scipy.spatial.KDTree(c1)
    tree2 = scipy.spatial.KDTree(c2)
    logger.info("Created KD-Trees")
    # Ensure a full matching exists by setting false edges between all vertices.
    # Use a distance that cannot be chosen over an actual edge.
    cost = np.full((len(c1), len(c2)), len(c1) * threshold)
    indexes = tree1.query_ball_tree(tree2, r=threshold)
    count = 0
    for i, v1 in enumerate(c1):
        cm = cost[i]
        count += len(indexes[i])
        # Note: If there are no indexes then the threshold is too low.
        # We could: (a) Increase the threshold until there some edges for
        # all vertices; (b) choose n random points, find their closest
        # neighbours and use to estimate the threshold. Currently the
        # vertex should join a false edge and be removed later.
        for j in indexes[i]:
            v2 = c2[j]
            d = v1 - v2
            cm[j] = np.sqrt((d * d).sum())
    logger.info(
        "Computed distance matrix with %d distances under threshold %.1f",
        count,
        threshold,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    # Ignore large distance mappings. Can occur due to false edges.
    for i, (r, c) in enumerate(zip(row_ind, col_ind, strict=True)):
        if cost[r][c] > threshold:
            row_ind[i], col_ind[i] = -1, -1
    selected = row_ind >= 0
    row_ind = row_ind[selected]
    col_ind = col_ind[selected]
    return row_ind, col_ind


def map_nearest_neighbour(
    c1: npt.NDArray[Any], c2: npt.NDArray[Any], threshold: float
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Gready nearest neighbour matching.

    Args:
        c1: Coordinates 1
        c2: Coordinates 1
        threshold: Distance threshold for alignment mappings

    Returns:
        Mapping from row -> column
    """
    #
    tree1 = scipy.spatial.KDTree(c1)
    tree2 = scipy.spatial.KDTree(c2)
    logger.info("Created KD-Trees")
    indexes = tree1.query_ball_tree(tree2, r=threshold)
    pairs = []
    for i, v1 in enumerate(c1):
        for j in indexes[i]:
            v2 = c2[j]
            d = v1 - v2
            pairs.append(((d * d).sum(), i, j))
    logger.info(
        "Computed %d distances under threshold %.1f",
        len(pairs),
        threshold,
    )

    # Sort list and build assignments
    pairs.sort()
    col_free = np.full(len(c2), True)
    row_ind: npt.NDArray[np.int_] = np.full(len(c1), -1)
    col_ind = row_ind.copy()
    remaining = min(len(c1), len(c2))
    for _, i, j in pairs:
        if col_free[j] and row_ind[i] == -1:
            # mark pair (i, j) as assigned
            col_free[j] = False
            row_ind[i] = i
            col_ind[i] = j
            remaining -= 1
            if remaining == 0:
                break
        # Extra greedy mode:
        # Only allow objects to pair if both are nearest neighbours.
        # Mark the column as observed.
        col_free[j] = False
        # Mark the row as observed if it is not assigned.
        if row_ind[i] == -1:
            row_ind[i] = -2

    selected = row_ind >= 0
    row_ind = row_ind[selected]
    col_ind = col_ind[selected]
    return row_ind, col_ind


def map_masks(
    m1: npt.NDArray[Any],
    m2: npt.NDArray[Any],
    trans: tuple[int, int],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Matching using mask overlap.

    Args:
        m1: first image
        m2: second image
        trans: Translation (YX) of second image

    Returns:
        Mapping from label1 -> label2
    """
    m2 = _translate(m1, m2, trans, stacked=False)
    # Compute all-vs-all histogram
    logger.debug("Computing histogram")
    h = _histogram(m1, m2)
    n, m = len(h), len(h[0])
    logger.debug("Computing sums")
    # Remove unused background-background count
    h[0, 0] = 0
    # Size of objects
    s1 = np.sum(h, axis=1)
    s2 = np.sum(h, axis=0)
    # Compute overlap
    logger.debug("Computing overlaps")
    pairs = []
    for i in range(1, n):
        hh = h[i]
        for j in range(1, m):
            # Require 50% overlap of the smallest object
            if hh[j] and hh[j] / min(s1[i], s2[j]) > 0.5:
                # Order by pixel overlap, then fraction of largest object.
                overlap1 = hh[j] / max(s1[i], s2[j])
                pairs.append((hh[j], overlap1, i, j))
    # Assign using a greedy algorithm (largest overlap first)
    pairs.sort(reverse=True)

    # Note: From here the code copies the nearest neighbour mapping
    col_free = np.full(m, True)
    row_ind: npt.NDArray[np.int_] = np.full(n, -1)
    col_ind = row_ind.copy()
    remaining = min(n, m)
    for _, _, i, j in pairs:
        if col_free[j] and row_ind[i] == -1:
            # mark pair (i, j) as assigned
            col_free[j] = False
            row_ind[i] = i
            col_ind[i] = j
            remaining -= 1
            if remaining == 0:
                break
        # # Extra greedy mode:
        # # Only allow objects to pair if both are the first overlap observed for the label.
        # # Mark the column as observed.
        # col_free[j] = False
        # # Mark the row as observed if it is not assigned.
        # if row_ind[i] == -1:
        #     row_ind[i] = -2

    selected = row_ind >= 0
    row_ind = row_ind[selected]
    col_ind = col_ind[selected]
    logger.info("Selected %d of %d overlaps", len(row_ind), len(pairs))
    return row_ind, col_ind


def _histogram(
    m1: npt.NDArray[Any],
    m2: npt.NDArray[Any],
) -> npt.NDArray[np.int_]:
    """Compute all-vs-all histogram on two input arrays.

    Finds the maximum value in each input array (n, m) and returns an overlap
    histogram h[n+1][m+1]. Inputs must not contain negative values.

    Args:
        m1: Array 1
        m2: Array 2

    Returns:
        Overlap histogram.
    """
    # Equivelent of
    # n = np.max(m1) + 1
    # m = np.max(m2) + 1
    # h = np.zeros((n, m), dtype=np.uint32)
    # for i, j in zip(m1.reshape(-1), m2.reshape(-1), strict=False):
    #    h[i, j] += 1

    # Use the index [i, j] = i*m + j
    # Multiplication must not overflow so change to int
    n = np.max(m1) + 1
    m = np.max(m2) + 1
    count = np.bincount(
        m1.ravel().astype(np.int_) * m + m2.ravel(), minlength=n * m
    )
    return count.reshape((n, m))


def _get_mask_map(
    conn: BlitzGateway, plate_id: int
) -> dict[int, ImageWrapper]:
    """Get a map from the original image ID to the segmentation mask image object."""
    dataset_id = PlateDataset(conn, plate_id).dataset_id
    dataset = conn.getObject("Dataset", dataset_id)
    d = {}
    for image in dataset.listChildren():
        s = image.getName().removesuffix("_segmentation")
        if len(s) < len(image.getName()):
            d[int(s)] = image
    return d


def _get_mask_from_map(
    conn: BlitzGateway, image_id: int, image_map: dict[int, ImageWrapper]
) -> npt.NDArray[Any]:
    """Get the nuclei segmentation mask from the image map."""
    image = image_map.get(image_id)
    if image is not None:
        logger.info("Segmentation masks found for image %d", image_id)
        # Only download first channel: XYZCT order
        axis_lengths = [image.getSizeX(), image.getSizeY(), 1, 1, 1]
        _, masks = get_image(conn, image.getId(), axis_lengths=axis_lengths)
        # masks is TZYXC: convert to a YX image
        return masks[0][0][..., 0]  # type: ignore[no-any-return]
    raise OmeroError(
        f"Segmentation not found in for image {image_id}",
        logger,
    )


def _map_label_to_index(
    labels: npt.NDArray[np.int_], label_index: npt.NDArray[Any]
) -> npt.NDArray[np.int_]:
    """Maps the labels array to indices into the provided labels index array.

    Any unmapped label is returned in the output as -1.
    """
    # The map will be incomplete if the label is not in the labels index.
    # Return -1 for unmapped labels
    mapping = np.full(
        max(label_index.max(initial=0), labels.max(initial=0)) + 1, -1
    )
    for i, x in enumerate(label_index):
        mapping[x] = i
    return mapping[labels]


def get_plate_alignments(conn: BlitzGateway, plate_id: int) -> pd.DataFrame:
    """Get the alignments for the plate.

    The alignments are: plate, well, x, y.
    The alignment data is created and uploaded to OMERO by align_plates.

    Args:
        conn: The BlitzGateway connection
        plate_id: The plate ID

    Returns:
        DataFrame of plate alignments

    Raises:
        PlateNotFoundError: if a plate does not exist
        PlateDataError: if plates are missing the OMERO screen results
    """
    # Download the alignments
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise PlateNotFoundError("Plate:{plate_id}", logger)
    filename = "alignment.csv"
    att = get_file_attachments(plate, filename)
    df = None
    if att:
        df = parse_csv_data(att[0])
    if df is None:
        raise PlateDataError(
            f"Plate {plate_id} is missing alignment result data: {filename}",
            logger,
        )
    return df


def mapping_gallery(
    conn: BlitzGateway,
    plate_id: int,
    df: pd.DataFrame,
    image_map: list[list[str | int]],
    grid_size: int = 4,
    seed: int | None = None,
) -> None:
    """Create a gallery of the mappings between object centroids.

    The input data is the output generated by aggregate_plates.
    The gallery is saved to the master plate.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the master plate
        df: the aggregated data
        image_map: the image map
        grid_size: the size of the gallery grid
        seed: Seed for random selection of well samples

    Raises:
        PlateNotFoundError: if the plate does not exist
    """
    logger.info("Generating alignment gallery for master plate: %d", plate_id)
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise PlateNotFoundError("Plate:{plate_id}", logger)
    # Create the samples
    if seed is None:
        seed = int.from_bytes(os.urandom(8))
    logger.info("Selecting mapping samples using seed: %d", seed)
    random.seed(a=seed)
    selected_indices = random.sample(
        range(len(image_map)), min(grid_size * grid_size, len(image_map))
    )
    selected_indices.sort()
    grid_size = int(np.ceil(np.sqrt(len(selected_indices))))

    # Create the grid
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axs = axs.reshape(grid_size, grid_size)  # Ensure axs is a 2D grid

    # Create the column names for the centroids.
    # n_plates is the number of repeat plates (excluding the master plate).
    n_plates = len(image_map[0]) - 2
    col_names = ["centroid-1", "centroid-0"]
    for i in range(n_plates):
        col_names.extend([f"centroid-1.{i}", f"centroid-0.{i}"])

    # Create colours for each plate
    cmap = matplotlib.colormaps["plasma"]
    colors = cmap(np.linspace(0, 1, n_plates + 1))

    for idx, ax in enumerate(axs.flat):
        if idx < len(selected_indices):
            images = image_map[selected_indices[idx]]
            ax.set_title(f"{images[0]}: {images[1]}")
            ax.invert_yaxis()
            # Select all rows that match the images. The well is redundant during selection.
            mask = df["image_id"] == images[1]
            for i in range(n_plates):
                mask = mask | (df[f"image_id.{i}"] == images[i + 2])
            df1 = df[mask][col_names]

            lines = []
            line_colors = []
            for data in df1.itertuples(index=False, name=None):
                x, y = data[:2]
                marker = "v"  # unmapped
                if np.isnan(x):
                    # plot unmapped centroids
                    for i in range(1, 1 + n_plates):
                        x1, y1 = data[2 * i : 2 * (i + 1)]
                        if not np.isnan(x1):
                            ax.plot(x1, y1, color=colors[i], marker=marker)
                            # There is only one centroid per row if the master centroid is missing
                            break
                else:
                    # draw lines to each other centroid
                    for i in range(1, n_plates + 1):
                        x1, y1 = data[2 * i : 2 * (i + 1)]
                        if not np.isnan(x1):
                            marker = "."  # mapped
                            ax.plot(x1, y1, color=colors[i], marker=marker)
                            lines.append([(x, y), (x1, y1)])
                            line_colors.append(colors[i])
                    # plot master centroid
                    ax.plot(x, y, color=colors[0], marker=marker)
            if lines:
                ax.add_collection(
                    LineCollection(lines, color=line_colors, lw=1)
                )

    name = "mapping_gallery"
    logger.info("Saving gallery: %s", name)
    delete_file_attachment(conn, plate, ends_with=name + ".png")
    attach_figure(conn, fig, plate, name)
