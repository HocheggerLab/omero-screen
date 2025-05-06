"""Module for processing the wells in a plate."""

# import torch
from typing import Any

import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.figure import Figure
from omero.gateway import BlitzGateway
from omero_utils.map_anns import parse_annotations

from omero_screen.config import get_logger

from .flatfield_corr import flatfieldcorr
from .metadata_parser import MetadataParser
from .plate_dataset import PlateDataset

logger = get_logger(__name__)


def plate_loop(
    conn: BlitzGateway, plate_id: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[Any]]]:
    """
    Main loop to process a plate.
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

    print_device_info()

    df_final, df_quality_control, dict_gallery = process_wells(
        conn, metadata, dataset_id, flatfield_dict
    )
    logger.debug("Final data sample: %s", df_final.head())
    logger.debug("Final data columns: %s", df_final.columns)

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def print_device_info() -> None:
    """
    Print whether the code is using Cellpose with GPU or CPU.
    """
    if torch.cuda.is_available():
        logger.info("Using Cellpose with GPU.")
    else:
        logger.info("Using Cellpose with CPU.")


def process_wells(
    conn: BlitzGateway,
    metadata: MetadataParser,
    dataset_id: int,
    flatfield_dict: dict[str, npt.NDArray[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Figure] | None]:
    """
    Process the wells of the plate.
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
    # image_classifier = None
    # TODO: load these
    # inference_model: list[str] | None = None
    # gallery_width = 10
    # if inference_model:
    #     image_classifier = [
    #         _create_classifier(conn, x, gallery_width) for x in inference_model
    #     ]
    wells = list(metadata.plate.listChildren())
    for count, well in enumerate(wells):
        ann = parse_annotations(well)
        try:
            cell_line = ann["cell_line"]
        except KeyError:
            cell_line = ann["Cell_Line"]
        if cell_line != "Empty":
            message = f"Analysing well row:{well.row}/col:{well.column} - {count + 1} of {len(wells)}."
            print(message)
            # well_data, well_quality = well_loop(
            #     conn, well, metadata, project_data, flatfield_dict, image_classifier=image_classifier
            # )
            # df_final = pd.concat([df_final, well_data])
            # df_quality_control = pd.concat([df_quality_control, well_quality])

    # Create and save galleries after the loop
    dict_gallery = None
    # if image_classifier is not None and gallery_width:
    #     logger.info("Generating gallery images")
    #     dict_gallery = {}
    # for cls in image_classifier:
    #     prefix = cls.class_name + '_'
    #     for predicted_class, data in cls.gallery_dict.items():
    #         selected_images, total = data
    #         if selected_images:
    #             dict_gallery[prefix + predicted_class] = create_gallery(selected_images, gallery_width)
    #             logger.info(f"Gallery created for '{cls.class_name}/{predicted_class}': {len(selected_images)}/{total}")

    return df_final, df_quality_control, dict_gallery


# def _create_classifier(
#     conn: BlitzGateway, model_name: str, gallery_width: int
# ) -> None:
#     image_classifier = (
#         None  # ImageClassifier(conn, model_name, class_name=model_name)
#     )
#     # image_classifier.gallery_size = gallery_width**2
#     # image_classifier.batch_size = Defaults["INFERENCE_BATCH_SIZE"]
#     return image_classifier
