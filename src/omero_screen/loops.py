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
from typing import Any

import numpy.typing as npt
import pandas as pd
import tqdm
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

from omero_screen.cellcycle_analysis import cellcycle_analysis, combplot
from omero_screen.config import get_logger
from omero_screen.gallery_figure import create_gallery
from omero_screen.image_analysis import Image, ImageProperties
from omero_screen.image_classifier import ImageClassifier
from omero_screen.quality_control import quality_control_fig

from .flatfield_corr import flatfieldcorr
from .metadata_parser import MetadataParser
from .plate_dataset import PlateDataset

logger = get_logger(__name__)


def plate_loop(
    conn: BlitzGateway, plate_id: int
) -> tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, Figure] | None
]:
    """Main loop to process a plate.

    Args:
        conn: Connection to OMERO
        plate_id: ID of the plate
    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, Dict]: Three DataFrames containing the final data and quality control data;
        dictionary of matplotlib figures of the inference gallery keyed by class (can be None)
    """
    logger.info("Processing plate %s", plate_id)
    metadata = MetadataParser(conn, plate_id)
    metadata.manage_metadata()
    logger.debug("Channel Metadata: %s", metadata.channel_data)
    dataset_id = PlateDataset(conn, plate_id).dataset_id
    flatfield_dict = flatfieldcorr(conn, metadata, dataset_id)

    # Add plate name to summary file
    # with open(
    #     Defaults["DEFAULT_DEST_DIR"] + "/" + Defaults["DEFAULT_SUMMARY_FILE"], "a"
    # ) as f:
    #     print(plate_name, file=f)

    _print_device_info()

    df_final, df_quality_control, dict_gallery = process_wells(
        conn, metadata, dataset_id, flatfield_dict
    )
    logger.debug("Final data sample: %s", df_final.head())
    logger.debug("Final data columns: %s", df_final.columns)

    # check conditions for cell cycle analysis
    logger.info("Performing cell cycle analysis")
    keys = metadata.channel_data.keys()

    if "EdU" in keys:
        try:
            H3 = "H3P" in keys
            cyto = "Tub" in keys

            if H3 and cyto:
                df_final_cc = cellcycle_analysis(df_final, H3=True, cyto=True)
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
        except Exception as e:  # noqa: BLE001
            logger.exception("Cell cycle analysis failed", e)
            df_final_cc = None
    else:
        df_final_cc = None

    _save_results(
        conn, df_final, df_final_cc, df_quality_control, dict_gallery, metadata
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Figure] | None]:
    """Process the wells of the plate.

    Args:
        conn: Connection to OMERO
        metadata: Metadata associated with the plate
        dataset_id: Dataset associated with the plate
        flatfield_dict: Dictionary containing flatfield correction data
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
    if inference_model_names:
        image_classifier = [
            _create_classifier(conn, x, gallery_width, batch_size)
            for x in inference_model_names.split(":")
        ]
    wells = list(conn.getObject("Plate", metadata.plate_id).listChildren())
    for count, well in enumerate(wells):
        ann = parse_annotations(well)
        try:
            cell_line = ann["cell_line"]
        except KeyError:
            cell_line = ann["Cell_Line"]
        if cell_line == "Empty":
            continue
        well_data, well_quality = _download_well_results(conn, well)
        if well_data is not None:
            logger.info(
                "Loaded well results %s (%d/%d).",
                well.getWellPos(),
                count + 1,
                len(wells),
            )
        else:
            logger.info(
                "Analysing well %s (%d/%d).",
                well.getWellPos(),
                count + 1,
                len(wells),
            )
            well_data, well_quality = _well_loop(
                conn,
                well,
                metadata,
                dataset_id,
                flatfield_dict,
                image_classifier=image_classifier,
            )
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
    image_classifier = ImageClassifier(conn, model_name, class_name=model_name)
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all images in a well.

    Args:
        conn: Connection to OMERO
        well: WellWrapper object
        metadata: MetadataParser object
        dataset_id: Dataset ID
        flatfield_dict: Flatfield dictionary
        image_classifier: Image classifier
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the final data and quality control data
    """
    logger.info("Segmenting and analysing Images")
    df_well = pd.DataFrame()
    df_well_quality = pd.DataFrame()
    image_number = len(list(well.listChildren()))
    for number in tqdm.tqdm(range(image_number)):
        omero_img = well.getImage(number)
        image = Image(
            conn, well, omero_img, metadata, dataset_id, flatfield_dict
        )
        image_data = ImageProperties(
            well, image, metadata, image_classifier=image_classifier
        )
        df_image, df_image_quality = (
            image_data.image_df,
            image_data.quality_df,
        )
        df_well = pd.concat([df_well, df_image])
        df_well_quality = pd.concat([df_well_quality, df_image_quality])

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
    i = cols.index("experiment")
    return cols[i:] + cols[:i]
