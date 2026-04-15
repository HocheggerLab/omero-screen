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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from omero.gateway import BlitzGateway, WellWrapper
from omero_utils.attachments import (
    attach_data,
    attach_figure,
    delete_file_attachment,
    get_file_attachments,
    parse_csv_data,
)
from omero_utils.map_anns import parse_annotations
from omero_utils.message import WellAnnotationError

from omero_screen.cellcycle_analysis import cellcycle_analysis, combplot
from omero_screen.config import get_logger
from omero_screen.gallery_figure import create_gallery
from omero_screen.image_analysis import Image, ImageProperties, get_cell_model
from omero_screen.image_classifier import ImageClassifier
from omero_screen.quality_control import quality_control_fig

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
    conn: BlitzGateway, plate_id: int, segmentation_mode: bool = False
) -> tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, Figure] | None
]:
    """Main loop to process a plate.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the plate
        segmentation_mode: Only perform image segmentation
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
        if get_cell_model(cell_line) is None:
            raise WellAnnotationError(
                f"Unrecognised cell line: {cell_line}", logger
            )

    dataset_id = PlateDataset(conn, plate_id).dataset_id

    with bench.stage("flatfield_correction"):
        flatfield_dict = flatfieldcorr(conn, metadata, dataset_id)

    _print_device_info()

    df_final, df_quality_control, dict_gallery = process_wells(
        conn, metadata, dataset_id, flatfield_dict, segmentation_mode
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
                cyto = "Tub" in keys

                if H3 and cyto:
                    df_final_cc = cellcycle_analysis(
                        df_final, H3=True, cyto=True
                    )
                elif H3:
                    df_final_cc = cellcycle_analysis(df_final, H3=True)
                elif not cyto:
                    df_final_cc = cellcycle_analysis(df_final, cyto=False)
                else:
                    df_final_cc = cellcycle_analysis(df_final)
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Figure] | None]:
    """Process the wells of the plate.

    Args:
        conn: Connection to OMERO
        metadata: Metadata associated with the plate
        dataset_id: Dataset associated with the plate
        flatfield_dict: Dictionary containing flatfield correction data
        segmentation_mode: Only perform image segmentation
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
        ann_lower = {k.lower(): v for k, v in parse_annotations(well).items()}
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

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the final data and quality control data
    """
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
    for well in wells:
        well_pos = well.getWellPos()
        if len(df_final[df_final["well"] == well_pos]) > 100:
            fig = combplot(df_final, well_pos)
            delete_file_attachment(conn, well, ends_with=f"{well_pos}.png")
            attach_figure(conn, fig, well, well_pos)
        else:
            logger.warning("Insufficient data for %s", well_pos)


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
