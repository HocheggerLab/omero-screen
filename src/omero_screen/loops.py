"""Processes and analyzes wells in an OMERO plate, including segmentation, feature extraction, cell cycle analysis, and result attachment.

This module provides the main workflow for high-content screening data analysis using OMERO. It orchestrates the following steps:

- Metadata parsing and management for the plate and its wells.
- Flatfield correction mask generation and application.
- Iterative processing of all wells and images in the plate, including segmentation (using Cellpose), feature extraction, and quality control.
- Optional cell cycle analysis if appropriate channels are present.
- Aggregation of results into pandas DataFrames.
- Attachment of results (data tables and figures) back to OMERO as file and image attachments.

Typical usage involves calling `plate_loop`, which coordinates the entire process for a given plate ID and OMERO connection.

Functions:
    plate_loop(conn, plate_id):
        Main entry point for processing a plate. Returns final data, cell cycle data (if available), quality control data, and inference galleries.
    process_wells(...):
        Processes all wells in the plate, performing segmentation and feature extraction.
    _well_loop(...):
        Processes all images in a single well.
    _add_welldata(...):
        Attaches well-level results and figures to OMERO.
    _save_results(...):
        Attaches summary results and figures to OMERO.

Args:
    conn (BlitzGateway): OMERO connection object.
    plate_id (int): OMERO plate identifier.

Returns:
    tuple: DataFrames and figures summarizing the analysis, attached to OMERO.
"""

import os

os.environ.setdefault(
    "TQDM_DISABLE", "1"
)  # suppress Cellpose tile-level progress bars

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ezomero import get_image
from matplotlib.figure import Figure
from omero.gateway import BlitzGateway, WellWrapper
from omero_utils.attachments import (
    attach_data,
    attach_figure,
    delete_file_attachment,
    get_file_attachments,
    parse_csv_data,
)
from omero_utils.images import upload_masks
from omero_utils.map_anns import parse_annotations
from omero_utils.message import WellAnnotationError
from omero_utils.stitching import (
    OPERETTA_STITCH_DEFAULTS,
    split_stitched_mask_to_fields,
    stitch_from_positions,
)

from omero_screen import default_config
from omero_screen.cellcycle_analysis import cellcycle_analysis
from omero_screen.config import get_logger
from omero_screen.constants import OmeroScreenNS
from omero_screen.gallery_figure import create_gallery
from omero_screen.general_functions import filter_segmentation, scale_img
from omero_screen.image_analysis import (
    Image,
    ImageProperties,
    StitchedWellImage,
    get_cell_model,
)
from omero_screen.image_classifier import ImageClassifier
from omero_screen.metadata_parser import strip_role_suffix
from omero_screen.quality_control import quality_control_fig
from omero_screen.segmentation import SegmentationModel

from .benchmarking import get_benchmark
from .flatfield_corr import flatfieldcorr
from .metadata_parser import MetadataParser
from .plate_dataset import PlateDataset
from .progress import ScreenProgress

logger = get_logger(__name__)


@contextmanager
def _nullctx() -> Generator[None, None, None]:
    """No-op context manager used when no ScreenProgress is available."""
    yield


def plate_loop(
    conn: BlitzGateway,
    plate_id: int,
    segmentation_mode: bool = False,
    stitch_mode: bool = False,
) -> tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, Figure] | None
]:
    """Main loop to process a plate.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the plate
        segmentation_mode: Only perform image segmentation
        stitch_mode: Run stitched-well segmentation instead of per-field
    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, Dict]: Three DataFrames containing the final data and quality control data;
        dictionary of matplotlib figures of the inference gallery keyed by class (can be None)
    """
    logger.info("Processing plate %s", plate_id)
    bench = get_benchmark()

    with bench.stage("metadata_parsing"):
        metadata = MetadataParser(conn, plate_id)
        metadata.manage_metadata()
    logger.debug("Channel Metadata: %s", str(metadata.channel_data))

    # Validate cell line required for segmentation model
    for cell_line in set(metadata.well_data["cell_line"]):
        if get_cell_model(str(cell_line)) is None:
            raise WellAnnotationError(
                f"Unrecognised cell line: {cell_line}", logger
            )

    dataset_id = PlateDataset(conn, plate_id).dataset_id

    with bench.stage("flatfield_correction"):
        flatfield_dict = flatfieldcorr(conn, metadata, dataset_id)

    _print_device_info()

    df_final, df_quality_control, dict_gallery = process_wells(
        conn,
        metadata,
        dataset_id,
        flatfield_dict,
        segmentation_mode,
        stitch_mode=stitch_mode,
    )
    if segmentation_mode:
        logger.info("Segmentation complete")
        # Data frames should be empty
        assert df_final.empty and df_quality_control.empty, (
            "Segmentation mode should create empty results"
        )
        return df_final, None, df_quality_control, None

    logger.debug("Final data sample: %s", df_final.head())
    logger.debug("Final data columns: %s", df_final.columns)

    # check conditions for cell cycle analysis
    logger.info("Performing cell cycle analysis")
    keys = metadata.channel_data.keys()

    with bench.stage("cell_cycle_analysis"):
        if "EdU" in keys:
            try:
                H3 = "H3P" in keys
                cyto = "cell" in metadata.channel_roles
                # Token used in the feature column names for the segmented nucleus
                # channel; matches the rule applied by image_analysis when building
                # ``integrated_int_{token}`` / ``intensity_mean_{token}_nucleus``.
                nucleus_channel = strip_role_suffix(
                    metadata.channel_roles["nucleus"]
                )

                if H3 and cyto:
                    df_final_cc = cellcycle_analysis(
                        df_final,
                        H3=True,
                        cyto=True,
                        nucleus_channel=nucleus_channel,
                    )
                elif H3:
                    df_final_cc = cellcycle_analysis(
                        df_final, H3=True, nucleus_channel=nucleus_channel
                    )
                elif not cyto:
                    df_final_cc = cellcycle_analysis(
                        df_final, cyto=False, nucleus_channel=nucleus_channel
                    )
                else:
                    df_final_cc = cellcycle_analysis(
                        df_final, nucleus_channel=nucleus_channel
                    )
                wells = list(
                    conn.getObject("Plate", metadata.plate_id).listChildren()
                )
                _add_welldata(conn, wells, df_final_cc)
            except KeyError as e:
                logger.error(
                    "Cell cycle analysis failed — missing column: %s. "
                    "This usually means a required channel (EdU, H3P, or DAPI) "
                    "is missing or misspelled in the metadata.",
                    e,
                )
                df_final_cc = None
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Cell cycle analysis failed with unexpected error: %s. "
                    "Check the log file for details.",
                    e,
                )
                logger.debug("Cell cycle analysis traceback:", exc_info=True)
                df_final_cc = None
        else:
            df_final_cc = None

    with bench.stage("save_results"):
        _save_results(
            conn,
            df_final,
            df_final_cc,
            df_quality_control,
            dict_gallery,
            metadata,
        )
    _remove_intermediate_well_results(
        conn, list(conn.getObject("Plate", metadata.plate_id).listChildren())
    )
    return df_final, df_final_cc, df_quality_control, dict_gallery


def _print_device_info() -> None:
    """Print whether the code is using Cellpose with GPU or CPU.

    This function checks if a GPU is available and prints a message to the logger.
    """
    import omero_screen.torch

    logger.info("Using Cellpose with %s", str(omero_screen.torch.get_device()))


def process_wells(
    conn: BlitzGateway,
    metadata: MetadataParser,
    dataset_id: int,
    flatfield_dict: dict[str, npt.NDArray[Any]],
    segmentation_mode: bool = False,
    stitch_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Figure] | None]:
    """Process the wells of the plate.

    Args:
        conn: Connection to OMERO
        metadata: Metadata associated with the plate
        dataset_id: Dataset associated with the plate
        flatfield_dict: Dictionary containing flatfield correction data
        segmentation_mode: Only perform image segmentation
        stitch_mode: Run stitched-well segmentation instead of per-field
    Returns:
        Two DataFrames containing the final data and quality control data; dictionary of
        matplotlib figures of the inference gallery keyed by class (can be None)
    """
    df_final = pd.DataFrame()
    df_quality_control = pd.DataFrame()
    image_classifier = None
    inference_model_names = os.getenv("OMERO_SCREEN_INFERENCE_MODEL")
    gallery_width = int(
        os.getenv("OMERO_SCREEN_INFERENCE_GALLERY_WIDTH", "10")
    )
    batch_size = int(os.getenv("OMERO_SCREEN_INFERENCE_BATCH_SIZE", "100"))
    if inference_model_names and not segmentation_mode:
        image_classifier = [
            _create_classifier(conn, x, gallery_width, batch_size)
            for x in inference_model_names.split(":")
        ]
    border = int(os.getenv("OMERO_SCREEN_CLEAR_BORDER", "5"))
    wells = list(conn.getObject("Plate", metadata.plate_id).listChildren())
    get_benchmark().set_well_count(len(wells))

    # Pre-filter to non-empty wells so the progress bar total is accurate
    non_empty_wells = []
    for well in wells:
        ann_lower = {
            k.lower(): v
            for k, v in parse_annotations(
                well, ns=OmeroScreenNS.METADATA
            ).items()
        }
        cell_line = ann_lower.get("cell_line")
        if cell_line is None:
            raise WellAnnotationError(
                f"Well {well.getWellPos()} is missing a 'cell_line' annotation. "
                f"Available annotations: {list(ann_lower.keys())}. "
                f"Check your metadata — each well needs a 'cell_line' entry.",
                logger,
            )
        if cell_line != "Empty":
            non_empty_wells.append(well)

    with ScreenProgress(metadata.plate_id, len(non_empty_wells)) as prog:
        for count, well in enumerate(non_empty_wells):
            n_images = len(list(well.listChildren()))
            well_pos = well.getWellPos()
            well_data, well_quality = (
                [None, None]
                if segmentation_mode
                else _download_well_results(conn, well)
            )
            if well_data is not None:
                logger.info(
                    "Loaded well results %s (%d/%d).",
                    well_pos,
                    count + 1,
                    len(non_empty_wells),
                )
                # Still tick through the well context so the bar advances
                with prog.well(well_pos, n_images):
                    prog.set_stage("loaded from cache")
                    for _ in range(n_images):
                        prog.image_done()
            else:
                logger.info(
                    "Analysing well %s (%d/%d).",
                    well_pos,
                    count + 1,
                    len(non_empty_wells),
                )
                well_data, well_quality = _well_loop(
                    conn,
                    well,
                    metadata,
                    dataset_id,
                    flatfield_dict,
                    image_classifier=image_classifier,
                    segmentation_mode=segmentation_mode,
                    border=border,
                    prog=prog,
                    stitch_mode=stitch_mode,
                )
                if not segmentation_mode:
                    _save_well_results(conn, well, well_data, well_quality)
            df_final = pd.concat([df_final, well_data])
            df_quality_control = pd.concat([df_quality_control, well_quality])

    # Create and save galleries after the loop
    dict_gallery = None
    if image_classifier is not None and gallery_width:
        logger.info("Generating gallery images")
        dict_gallery = {}
        for cls in image_classifier:
            prefix = cls.class_name + "_"
            for predicted_class, data in cls.gallery_dict.items():
                selected_images, total = data
                if selected_images:
                    dict_gallery[prefix + predicted_class] = create_gallery(
                        selected_images, gallery_width
                    )
                    logger.info(
                        "Gallery created for '%s/%s': %d/%d",
                        cls.class_name,
                        predicted_class,
                        len(selected_images),
                        total,
                    )

    return df_final, df_quality_control, dict_gallery


def _create_classifier(
    conn: BlitzGateway, model_name: str, gallery_width: int, batch_size: int
) -> ImageClassifier:
    image_classifier = ImageClassifier(
        conn, model_name, class_name=f"classifier_{model_name}"
    )
    image_classifier.gallery_size = gallery_width**2
    image_classifier.batch_size = batch_size
    return image_classifier


def _download_well_results(
    conn: BlitzGateway,
    well: WellWrapper,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Downloads the previous well results from OMERO.

    Args:
        conn: Connection to OMERO
        well: WellWrapper object
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the final data and quality control data
    """
    ann1 = get_file_attachments(well, "data.csv")
    ann2 = get_file_attachments(well, "quality.csv")
    if ann1 is not None and ann2 is not None:
        df = parse_csv_data(ann1[0])
        df_quality = parse_csv_data(ann2[0])
        return df, df_quality
    return None, None


def _save_well_results(
    conn: BlitzGateway,
    well: WellWrapper,
    df: pd.DataFrame,
    df_quality: pd.DataFrame,
) -> None:
    """Saves the well results to OMERO.

    Args:
        conn: Connection to OMERO
        well: WellWrapper object
        df: Analysis results
        df_quality: Quality control results
    """
    attach_data(conn, df, well, "data", cols=_columns(df))
    attach_data(conn, df_quality, well, "quality", cols=_columns(df_quality))


def _remove_intermediate_well_results(
    conn: BlitzGateway, wells: list[WellWrapper]
) -> None:
    for well in wells:
        delete_file_attachment(conn, well, ends_with="data.csv")
        delete_file_attachment(conn, well, ends_with="quality.csv")


def _load_well_fields(
    conn: BlitzGateway,
    well: WellWrapper,
    metadata: MetadataParser,
    dataset_id: int,
    flatfield_dict: dict[str, npt.NDArray[Any]],
) -> tuple[
    dict[str, npt.NDArray[Any]],
    list[tuple[float, float]],
    list[int],
]:
    """Fetch all fields of a well, flatfield-correct, and return per-channel stacks.

    Returns:
        per_channel: dict mapping channel name to an array of shape (N, T, Y, X)
            where N is the number of fields in the well.
        positions: list of (pos_x, pos_y) per field, in the same order as N.
        image_ids: OMERO image IDs per field, in the same order as N.
    """
    channels = metadata.channel_data
    n_fields = len(list(well.listChildren()))

    # Collect raw per-field arrays per channel, plus stage positions and image ids
    per_channel: dict[str, list[npt.NDArray[Any]]] = {
        ch: [] for ch in channels
    }
    positions: list[tuple[float, float]] = []
    image_ids: list[int] = []

    for n in range(n_fields):
        ws = well.getWellSample(n)
        image_obj = ws.getImage()
        image_ids.append(image_obj.getId())
        # Stage position via WellSample (microscope reference frame)
        px = ws.getPosX()
        py = ws.getPosY()
        positions.append(
            (
                px.getValue() if px is not None else 0.0,
                py.getValue() if py is not None else 0.0,
            )
        )

        _, array = get_image(conn, image_obj.getId())
        for ch, idx in channels.items():
            ch_idx = int(idx)
            if ch not in flatfield_dict:
                raise KeyError(
                    f"Channel '{ch}' not found in flatfield correction masks. "
                    f"Available channels: {list(flatfield_dict.keys())}."
                )
            img = array[..., ch_idx] / flatfield_dict[ch]
            # Reduce (tzyx) → (tyx)
            img = np.squeeze(img, axis=1)
            per_channel[ch].append(img)

    # Stack each channel's fields → (N, T, Y, X)
    stacked: dict[str, npt.NDArray[Any]] = {
        ch: np.stack(arrs) for ch, arrs in per_channel.items()
    }
    return stacked, positions, image_ids


def _nuc_diameter_for_cell_line(cell_line: str) -> int:
    """Match the diameter heuristic used by Image._n_segmentation."""
    if "40X" in cell_line.upper():
        return 100
    if "20X" in cell_line.upper():
        return 25
    return 10


def _segment_stitched_nuclei(
    stitched_img: npt.NDArray[Any],
    nucleus_channel_index: int,
    cell_line: str,
    border: int,
) -> npt.NDArray[Any]:
    """Segment the nucleus channel of a stitched (T, Y, X, C) canvas.

    Cellpose's internal tiling handles the large canvas. ``border``
    applies only at the *outer* edge of the canvas; internal field
    seams have been stitched away and no longer exist as boundary
    pixels, so no objects are excluded at seams. This is the bias fix.

    Args:
        stitched_img: Stitched canvas of shape (T, Y, X, C).
        nucleus_channel_index: Channel-axis index of the nucleus channel.
        cell_line: Cell line name (used for diameter heuristic).
        border: Width of the outer-edge border (negative to disable).

    Returns:
        Mask array of shape (T, Y, X) with uint16 labels (max 65535 cells/well —
        well above realistic crowding for a 1080×1080 × N grid).
    """
    model_name = default_config.MODEL_DICT.get("nuclei")
    if model_name is None:
        raise RuntimeError(
            "No nuclei segmentation model configured. "
            "Add a 'nuclei' entry to MODEL_DICT in your config."
        )
    segmentation_model = SegmentationModel(model_name)

    if segmentation_model.get_type() == "cellpose3":
        diameter: int | None = _nuc_diameter_for_cell_line(cell_line)
        logger.info("Segmenting stitched nuclei with diameter %s", diameter)
    else:
        diameter = None  # cellpose 4 is scale-independent

    n_t = stitched_img.shape[0]
    masks = np.zeros(stitched_img.shape[:3], dtype=np.uint16)
    for t in range(n_t):
        # (Y, X) → cellpose-ready (1, Y, X) single-channel stack
        img_t = stitched_img[t, ..., nucleus_channel_index]
        scaled = np.stack([scale_img(img_t)])
        try:
            mask = segmentation_model.eval(
                scaled,
                diameter=diameter,
                normalize=False,
            )
        except IndexError:
            logger.warning(
                "Stitched nucleus segmentation failed (t=%d) — "
                "returning empty mask.",
                t,
            )
            mask = np.zeros(img_t.shape, dtype=np.uint16)
        # Outer-edge border filter only. clear_border treats the array
        # edge as the boundary; since this is the full stitched canvas,
        # only true outer-edge objects are removed.
        masks[t] = filter_segmentation(mask, border=border)

    return masks


def _segment_stitched_cells(
    stitched_img: npt.NDArray[Any],
    cell_channel_index: int,
    nucleus_channel_index: int,
    cell_line: str,
    border: int,
) -> npt.NDArray[Any]:
    """Segment cells on the stitched canvas using the cell-line cellpose model.

    Mirrors ``Image._c_segmentation``: the cell channel and nucleus channel
    are scaled and stacked as a 2-channel cellpose input. Outer-border
    objects are removed; internal seams have been stitched away.

    Args:
        stitched_img: Stitched canvas of shape (T, Y, X, C).
        cell_channel_index: Channel-axis index of the cell channel.
        nucleus_channel_index: Channel-axis index of the nucleus channel.
        cell_line: Cell line name (used to select the cellpose model).
        border: Width of the outer-edge border (negative to disable).

    Returns:
        Cell mask array of shape (T, Y, X) with uint16 labels.
    """
    model_name = get_cell_model(cell_line)
    if model_name is None:
        raise RuntimeError(
            f"Unknown cell-segmentation model for cell line: {cell_line}"
        )
    segmentation_model = SegmentationModel(model_name)
    logger.info("Segmenting stitched cells with model %s", model_name)

    n_t = stitched_img.shape[0]
    masks = np.zeros(stitched_img.shape[:3], dtype=np.uint16)
    for t in range(n_t):
        cell_t = stitched_img[t, ..., cell_channel_index]
        nuc_t = stitched_img[t, ..., nucleus_channel_index]
        comb = np.stack([scale_img(cell_t), scale_img(nuc_t)])
        try:
            mask = segmentation_model.eval(
                comb,
                normalize=False,
            )
        except IndexError:
            logger.warning(
                "Stitched cell segmentation failed (t=%d) — "
                "returning empty mask.",
                t,
            )
            mask = np.zeros(cell_t.shape, dtype=np.uint16)
        masks[t] = filter_segmentation(mask, border=border)
    return masks


def _stitched_cyto(
    n_mask: npt.NDArray[Any], c_mask: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Derive the cytoplasm mask = cell - nucleus, mirroring Image._get_cyto."""
    overlap = (c_mask != 0) * (n_mask != 0)
    cyto_binary = (c_mask != 0) * (overlap == 0)
    result: npt.NDArray[Any] = (c_mask * cyto_binary).astype(c_mask.dtype)
    return result


def _stitch_well(
    per_channel: dict[str, npt.NDArray[Any]],
    positions: list[tuple[float, float]],
) -> npt.NDArray[Any]:
    """Stitch per-channel field stacks into a single (T, Y, X, C) canvas.

    Channels are stitched independently and stacked along the channel
    axis so the result matches the layout downstream code expects.
    """
    ch_names = list(per_channel.keys())
    channel_canvases: list[npt.NDArray[Any]] = []
    for ch in ch_names:
        # per_channel[ch] is (N, T, Y, X). stitch_from_positions expects
        # (N, T, Y, X, C); we treat each channel as a 1-channel volume.
        stack = per_channel[ch][..., np.newaxis]
        stitched = stitch_from_positions(
            stack,
            positions,
            **OPERETTA_STITCH_DEFAULTS,
        )
        # Result shape (T, Y, X, 1) → squeeze the channel axis
        channel_canvases.append(np.squeeze(stitched, axis=-1))

    # Stack channels along the last axis → (T, Y, X, C)
    return np.stack(channel_canvases, axis=-1)


def _stitched_well_loop(
    conn: BlitzGateway,
    well: WellWrapper,
    metadata: MetadataParser,
    dataset_id: int,
    flatfield_dict: dict[str, npt.NDArray[Any]],
    image_classifier: None | list[ImageClassifier],
    segmentation_mode: bool = False,
    border: int = 5,
    prog: ScreenProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a well as a single stitched canvas.

    Loads all fields, applies flatfield correction, stitches them into one
    (T, Y, X, C) canvas using the Operetta calibration constants, then (in
    later stages) segments the canvas and runs ImageProperties on it.

    For Stage 1 skeleton: only the stitch is implemented; segmentation,
    measurement, and mask upload are stubs handled by subsequent tasks.

    Returns empty DataFrames for now so the surrounding pipeline runs end
    to end without crashing.
    """
    well_pos = well.getWellPos()
    n_fields = len(list(well.listChildren()))
    logger.info(
        "Stitched analysis: well %s with %d fields", well_pos, n_fields
    )

    bench = get_benchmark()
    if prog:
        prog.set_stage("stitching")
    with bench.stage("stitched_download"):
        per_channel, positions, image_ids = _load_well_fields(
            conn, well, metadata, dataset_id, flatfield_dict
        )
    # Preserve channel order — _stitch_well builds the canvas in this order
    channel_order = list(per_channel.keys())
    # Per-field (T, Y, X) shape — needed later to split the stitched
    # mask back into per-field tiles for OMERO upload.
    sample_channel = next(iter(per_channel.values()))
    tile_h = sample_channel.shape[2]
    tile_w = sample_channel.shape[3]
    with bench.stage("stitched_compose"):
        stitched_img = _stitch_well(per_channel, positions)
    # The synthetic per-well image_id is the first OMERO image id in the
    # well — keeps measurement rows joinable with existing OMERO queries.
    synthetic_image_id = image_ids[0]
    logger.info(
        "Stitched canvas for %s: shape %s, dtype %s, synthetic image_id %d",
        well_pos,
        stitched_img.shape,
        stitched_img.dtype,
        synthetic_image_id,
    )

    # Free per-field memory before segmentation — the stitched canvas
    # holds all the pixels we need from here on.
    del per_channel

    nucleus_channel = metadata.channel_roles["nucleus"]
    if nucleus_channel not in channel_order:
        raise KeyError(
            f"Nucleus channel '{nucleus_channel}' missing from channel data; "
            f"available: {channel_order}"
        )
    nucleus_idx = channel_order.index(nucleus_channel)
    cell_channel = metadata.channel_roles.get("cell")
    cell_idx = channel_order.index(cell_channel) if cell_channel else None
    cell_line = metadata.well_conditions(well_pos)["cell_line"]

    if prog:
        prog.set_stage("segmentation")
    with bench.stage("stitched_nucleus_segmentation"):
        stitched_n_mask = _segment_stitched_nuclei(
            stitched_img,
            nucleus_channel_index=nucleus_idx,
            cell_line=cell_line,
            border=border,
        )
    logger.info(
        "Stitched nucleus mask for %s: %d nuclei (border=%d)",
        well_pos,
        int(stitched_n_mask.max()),
        border,
    )

    stitched_c_mask: npt.NDArray[Any] | None = None
    stitched_cyto_mask: npt.NDArray[Any] | None = None
    if cell_channel is not None and cell_idx is not None:
        with bench.stage("stitched_cell_segmentation"):
            stitched_c_mask = _segment_stitched_cells(
                stitched_img,
                cell_channel_index=cell_idx,
                nucleus_channel_index=nucleus_idx,
                cell_line=cell_line,
                border=border,
            )
        logger.info(
            "Stitched cell mask for %s: %d cells (channel=%s)",
            well_pos,
            int(stitched_c_mask.max()),
            cell_channel,
        )
        # Cytoplasm = cell ∖ nucleus (matches Image._get_cyto)
        stitched_cyto_mask = _stitched_cyto(stitched_n_mask, stitched_c_mask)

    # Split the stitched masks back into per-field tiles and upload each
    # as a sibling Image in the dataset, named
    # "<field_id>_stitched_segmentation". This avoids the OMERO pyramid
    # threshold (no individual upload exceeds tile_h × tile_w) and
    # round-trips the bytes through standard per-field segmentation
    # artefacts. Each label belongs to exactly one field by centroid,
    # so the cache layer can restitch with ``compose_labels`` without
    # ID remapping (Stage 2 concern).
    split_kwargs = {
        "positions": positions,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "overlap_x": OPERETTA_STITCH_DEFAULTS["overlap_x"],
        "overlap_y": OPERETTA_STITCH_DEFAULTS["overlap_y"],
        "translate_x": OPERETTA_STITCH_DEFAULTS["translate_x"],
        "translate_y": OPERETTA_STITCH_DEFAULTS["translate_y"],
    }
    with bench.stage("stitched_mask_split"):
        per_field_n_masks = split_stitched_mask_to_fields(
            stitched_n_mask, **split_kwargs
        )
        per_field_c_masks: list[npt.NDArray[Any]] | None = (
            split_stitched_mask_to_fields(stitched_c_mask, **split_kwargs)
            if stitched_c_mask is not None
            else None
        )
    with bench.stage("stitched_mask_upload"):
        for n in range(n_fields):
            field_img = well.getWellSample(n).getImage()
            field_n = per_field_n_masks[n]
            field_c = (
                per_field_c_masks[n] if per_field_c_masks is not None else None
            )
            upload_masks(
                conn,
                dataset_id,
                field_img,
                field_n,
                c_mask=field_c,
                name_suffix="_stitched_segmentation",
                annotation_key="Stitched_Segmentation_Mask",
            )

    if segmentation_mode:
        if prog:
            for _ in range(n_fields):
                prog.image_done()
        return pd.DataFrame(), pd.DataFrame()

    if prog:
        prog.set_stage("feature extraction")
    with bench.stage("stitched_feature_extraction"):
        stitched_image = StitchedWellImage(
            stitched_img=stitched_img,
            stitched_mask=stitched_n_mask,
            channels={ch: idx for idx, ch in enumerate(channel_order)},
            nucleus_channel=nucleus_channel,
            well_pos=well_pos,
            synthetic_image_id=synthetic_image_id,
            c_mask=stitched_c_mask,
            cyto_mask=stitched_cyto_mask,
            cell_channel=cell_channel,
        )
        image_props = ImageProperties(
            well,
            stitched_image,  # type: ignore[arg-type]  # StitchedWellImage duck-types Image
            metadata,
            image_classifier=image_classifier,
        )
    df_well = image_props.image_df
    df_well_quality = image_props.quality_df
    # Mark every measurement row so the cellview importer can populate
    # repeats.stitch_mode on the way in. Constant column — wasteful by
    # row but a single source-of-truth check at import time.
    df_well["stitch_mode"] = True
    logger.info(
        "Stitched features for %s: %d rows, %d columns",
        well_pos,
        len(df_well),
        len(df_well.columns),
    )

    if prog:
        # Tick once per field so the bar advances at the same rate
        # as the per-field path.
        for _ in range(n_fields):
            prog.image_done()

    return df_well, df_well_quality


def _well_loop(
    conn: BlitzGateway,
    well: WellWrapper,
    metadata: MetadataParser,
    dataset_id: int,
    flatfield_dict: dict[str, npt.NDArray[Any]],
    image_classifier: None | list[ImageClassifier],
    segmentation_mode: bool = False,
    border: int = 5,
    prog: ScreenProgress | None = None,
    stitch_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all images in a well.

    Args:
        conn: Connection to OMERO
        well: WellWrapper object
        metadata: MetadataParser object
        dataset_id: Dataset ID
        flatfield_dict: Flatfield dictionary
        image_classifier: Image classifier
        segmentation_mode: Only perform image segmentation
        border: Width of the border examined when filtering segmented objects (negative to disable)
        prog: Live progress display (optional)
        stitch_mode: Run stitched-well segmentation instead of per-field

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the final data and quality control data
    """
    if stitch_mode:
        return _stitched_well_loop(
            conn,
            well,
            metadata,
            dataset_id,
            flatfield_dict,
            image_classifier=image_classifier,
            segmentation_mode=segmentation_mode,
            border=border,
            prog=prog,
        )

    logger.info(
        "Segmenting images"
        if segmentation_mode
        else "Segmenting and analysing images"
    )
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))

    # In segmentation mode skip all previously segmentated images
    seg = set()
    if segmentation_mode:
        dataset = conn.getObject("Dataset", dataset_id)
        for image in dataset.listChildren():
            seg.add(image.getName())

    bench = get_benchmark()
    with prog.well(well.getWellPos(), image_number) if prog else _nullctx():
        for number in range(image_number):
            omero_img = well.getImage(number)
            if f"{omero_img.getId()}_segmentation" in seg:
                if prog:
                    prog.image_done()
                continue
            with bench.image(omero_img.getId(), well=well.getWellPos()):
                if prog:
                    prog.set_stage("segmentation")
                image = Image(
                    conn,
                    well,
                    omero_img,
                    metadata,
                    dataset_id,
                    flatfield_dict,
                    border=border,
                )
                if segmentation_mode:
                    if prog:
                        prog.image_done()
                    continue
                if prog:
                    prog.set_stage("feature extraction")
                with bench.stage("feature_extraction"):
                    image_props = ImageProperties(
                        well,
                        image,
                        metadata,
                        image_classifier=image_classifier,
                    )
                df_well = pd.concat([df_well, image_props.image_df])
                df_well_quality = pd.concat(
                    [df_well_quality, image_props.quality_df]
                )
                if prog:
                    prog.image_done()

    return df_well, df_well_quality


def _add_welldata(
    conn: BlitzGateway, wells: list[WellWrapper], df_final: pd.DataFrame
) -> None:
    """Add well data to OMERO plate.

    Args:
        conn: Connection to OMERO
        wells: Plate wells
        df_final: DataFrame containing the final data
    """
    # Import here to avoid pulling matplotlib/seaborn at module import time
    # when only running segmentation-mode pipelines.
    from omero_screen_plots import well_qc_plot

    logger.debug(
        "Attaching per-well QC figures: %d wells, df has %d rows",
        len(wells),
        len(df_final),
    )
    attached = 0
    for well in wells:
        well_pos = well.getWellPos()
        well_df = df_final[df_final["well"] == well_pos]
        n_cells = len(well_df)
        if n_cells > 100:
            try:
                fig = well_qc_plot(
                    df=well_df,
                    title=_well_qc_title(well_pos, well_df),
                    save=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to render combplot for well %s (%d cells): %s",
                    well_pos,
                    n_cells,
                    exc,
                )
                logger.debug("combplot traceback:", exc_info=True)
                continue
            delete_file_attachment(conn, well, ends_with=f"{well_pos}.png")
            attach_figure(conn, fig, well, well_pos)
            attached += 1
        else:
            logger.warning(
                "Insufficient data for %s (%d cells)", well_pos, n_cells
            )
    logger.debug("Attached %d/%d well QC figures", attached, len(wells))


# Columns that are not biologically interesting for the per-well QC title —
# IDs, measurements, intensities, background corrections, etc.
_WELL_TITLE_EXCLUDE = {
    "label",
    "Cyto_ID",
    "centroid-0",
    "centroid-1",
    "centroid-0_x",
    "centroid-1_x",
    "centroid-0_y",
    "centroid-1_y",
    "timepoint",
    "experiment",
    "plate_id",
    "well",
    "well_id",
    "image_id",
    "cell_cycle",
    "cell_cycle_detailed",
}


def _well_qc_title(well_pos: str, well_df: pd.DataFrame) -> str:
    """Build a descriptive title for the per-well QC combplot.

    Pulls categorical metadata from ``well_df`` (cell_line, drug, siRNA, …)
    and appends the cell count. Numeric and measurement columns are skipped,
    as are columns that vary cell-to-cell within a well.
    """
    bits: list[str] = []
    for col in well_df.columns:
        if col in _WELL_TITLE_EXCLUDE:
            continue
        if col.startswith(("intensity_", "area_", "integrated_int_")):
            continue
        if col.endswith(("_background", "_norm")):
            continue
        # Only include columns that are uniform across the well (true
        # per-well metadata, not per-cell measurements).
        values = well_df[col].dropna().unique()
        if len(values) != 1:
            continue
        bits.append(f"{col}={values[0]}")
    suffix = f" — {len(well_df)} cells"
    if bits:
        return f"{well_pos} ({' | '.join(bits)}){suffix}"
    return f"{well_pos}{suffix}"


def _save_results(
    conn: BlitzGateway,
    df_final: pd.DataFrame,
    df_final_cc: pd.DataFrame | None,
    df_quality_control: pd.DataFrame,
    dict_gallery: dict[str, Figure] | None,
    metadata: MetadataParser,
) -> None:
    """Save the results to OMERO.

    Args:
        conn: Connection to OMERO
        df_final: DataFrame containing the final data
        df_final_cc: DataFrame containing the final cell cycle data
        df_quality_control: DataFrame containing quality control data
        dict_gallery: Dictionary of inference galleries as matplotlib.figure.Figure (or None)
        metadata: Plate metadata
    """
    # Note: Retrieve a new (updated) plate object after all steps that modify the plate

    logger.info("Removing previous results from OMERO")
    # delete pre-existing data
    delete_file_attachment(conn, conn.getObject("Plate", metadata.plate_id))

    logger.info("Saving results to OMERO")
    # load cell cycle data
    attach_data(
        conn,
        df_final,
        conn.getObject("Plate", metadata.plate_id),
        "final_data",
        cols=_columns(df_final),
    )
    if df_final_cc is not None:
        attach_data(
            conn,
            df_final_cc,
            conn.getObject("Plate", metadata.plate_id),
            "final_data_cc",
            cols=_columns(df_final_cc),
        )
    attach_data(
        conn,
        df_quality_control,
        conn.getObject("Plate", metadata.plate_id),
        "quality_ctr",
    )

    # load quality control figure
    quality_fig = quality_control_fig(df_quality_control)
    attach_figure(
        conn,
        quality_fig,
        conn.getObject("Plate", metadata.plate_id),
        "quality_ctr",
    )
    # load inference gallery
    if dict_gallery is not None:
        for cat, fig in dict_gallery.items():
            attach_figure(
                conn,
                fig,
                conn.getObject("Plate", metadata.plate_id),
                f"inference_{cat}",
            )


def _columns(df: pd.DataFrame) -> list[str]:
    """Reorder columns to move 'experiment' to the end.

    This function reorders the columns of a DataFrame to move the 'experiment' column to the end.

    Args:
        df: DataFrame to reorder
    """
    cols: list[str] = df.columns.tolist()
    if "experiment" not in cols:
        logger.warning(
            "'experiment' column not found in DataFrame — returning columns as-is."
        )
        return cols
    i = cols.index("experiment")
    return cols[i:] + cols[:i]
