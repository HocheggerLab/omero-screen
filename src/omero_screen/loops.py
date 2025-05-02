"""Module for processing the wells in a plate."""

# import torch
from typing import Any

import pandas as pd
from omero.gateway import BlitzGateway

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
    # flatfield_dict =
    flatfieldcorr(conn, metadata, dataset_id)

    # Add plate name to summary file
    # with open(
    #     Defaults["DEFAULT_DEST_DIR"] + "/" + Defaults["DEFAULT_SUMMARY_FILE"], "a"
    # ) as f:
    #     print(plate_name, file=f)

    print_device_info()

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def print_device_info() -> None:
    """
    Print whether the code is using Cellpose with GPU or CPU.
    """
    if True:
        logger.info("Using Cellpose with GPU.")
    else:
        logger.info("Using Cellpose with CPU.")
