"""Samples data from an OMERO plate.

This module provides sampling of images for well sample time points.

Functions:
    segmentation_samples(conn, plate_id):
        Sample images required for segmentation and the current segmentation results (if available).

Args:
    conn (BlitzGateway): OMERO connection object.
    plate_id (int): OMERO plate identifier.
"""

import json
import os
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import omero
from omero.gateway import BlitzGateway, ImageWrapper, MapAnnotationWrapper
from omero.rtypes import unwrap
from tifffile import imwrite

from omero_screen.config import get_logger
from omero_screen.constants import OmeroScreenNS

from .flatfield_corr import flatfieldcorr_name
from .metadata_parser import MetadataParser
from .plate_dataset import PlateDataset

logger = get_logger(__name__)


def segmentation_samples(
    conn: BlitzGateway,
    plate_id: int,
    out_directory: str,
    k: int,
    seed: int | None = None,
    overwrite: bool = False,
    compression: str | None = None,
) -> None:
    """Sample images required for segmentation.

    Saves either the nuclei or nuclei
    and tubulin channels for randomly selected time point samples.

    Note: This saves well metadata to a <plate ID>.csv file.
    The file is appended to prevent overwrite of existing metadata.

    Args:
        conn: Connection to OMERO.
        plate_id: ID of the plate.
        out_directory: Output directory.
        k: number of samples.
        overwrite: True to overwrite existing samples.
        seed: random seed.
        compression: Output TIFF compression.
    """
    logger.info("Processing plate %s", plate_id)
    metadata = MetadataParser(conn, plate_id)
    metadata.manage_metadata()
    logger.debug("Channel Metadata: %s", metadata.channel_data)
    dataset_id = PlateDataset(conn, plate_id).dataset_id

    channels = _get_segmentation_channels(metadata.channel_data)
    logger.info("Segmentation channels: %s", channels)

    # Save flatfield correction
    fn = os.path.join(out_directory, f"{plate_id}.c.tiff")
    if overwrite or not os.path.exists(fn):
        logger.info(
            "Saving flat-field correction image: %s", os.path.basename(fn)
        )
        img = _get_flatfieldcorr(conn, plate_id, dataset_id, channels)
        imwrite(fn, img, compression=compression)

    images = _get_image_ids(conn, plate_id)
    samples = _get_image_samples(images, k, seed)
    logger.info("Sample size: %d", len(samples))

    # Cache the segmentation images from the plate dataset
    dataset = conn.getObject("Dataset", dataset_id)
    seg_images = {x.getName(): x.getId() for x in dataset.listChildren()}

    fn = os.path.join(out_directory, f"{plate_id}.csv")
    with open(fn, "a") as f:
        # For each sample (well_id, image_id, t):
        well_id = -1
        image_id = -1
        for w, i, t in samples:
            # If a new ID get the objects
            if well_id != w:
                well_id = w
                well = conn.getObject("Well", w)
                if well is None:
                    raise Exception(f"No well for image: {i}")
                # Get well metadata. Extract the required cell line field then dump the rest.
                cond_dict = metadata.well_conditions(well.getWellPos())
                cell_line = ""
                for key in cond_dict:
                    if key.lower() == "cell_line":
                        cell_line = cond_dict.pop(key)
                        break
                meta = json.dumps(cond_dict)
            if image_id != i:
                image_id = i
                source_image = conn.getObject("Image", i)
                image = _get_image_to_segment(conn, source_image)
                if image is None:
                    raise Exception(f"No image to segment for image: {i}")
                mask_id = seg_images.get(f"{i}_segmentation", 0)
                mask = conn.getObject("Image", mask_id) if mask_id else None
                if mask is None:
                    logger.debug("No segmentation result for image: %d", i)

            # Record metadata. Extract the required cell line field then dump the rest.
            fn = f"{plate_id}_{image_id}_{t}.i.tiff"
            print(
                f"{fn},{cell_line},{meta}",
                file=f,
            )

            fn = os.path.join(out_directory, fn)
            if overwrite or not os.path.exists(fn):
                imwrite(
                    fn, _get_image(image, t, channels), compression=compression
                )
            if mask is not None:
                fn = os.path.join(
                    out_directory, f"{plate_id}_{image_id}_{t}.m.tiff"
                )
                if overwrite or not os.path.exists(fn):
                    imwrite(fn, _get_image(mask, t), compression=compression)


def _get_image_ids(
    conn: BlitzGateway, plate_id: int
) -> list[tuple[int, int, int]]:
    """Get a list of images from the OMERO object ids.

    Args:
        conn: OMERO connection.
        plate_id: OMERO plate ID.

    Returns:
        list of image details (well_id, image_id, size_t).

    Raises:
        Exception if the plate is not recognised.
    """
    images = []
    # Use the query service for enhanced performance over the OMERO object API
    query_service = conn.getQueryService()
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#plate
    # https://omero.readthedocs.io/en/stable/developers/Model/EveryObject.html#image
    params = omero.sys.ParametersI()
    query = f"""select w.id, i.id, pi.sizeT from Plate as p
        left join p.wells as w
        left join w.wellSamples as ws
        left join ws.image as i
        left join i.pixels as pi
        where p.id = '{plate_id}'
    """
    results = query_service.projection(query, params, None)
    if not len(results):
        raise Exception(f"OMERO plate does not exist: {plate_id}")
    for well_id, image_id, size_t in results:
        images.append((unwrap(well_id), unwrap(image_id), unwrap(size_t)))

    return images


def _get_image_samples(
    images: list[tuple[int, int, int]],
    k: int,
    seed: int | None,
) -> list[tuple[int, int, int]]:
    """Get a list of image time samples from the image list.

    The return list is sorted.

    Args:
        images: list of image details (well_id, image_id, size_t).
        k: number of samples.
        seed: random seed.

    Returns:
        list of image samples (well_id, image_id, t).
    """
    # Expand list to all timepoints
    timepoints = []
    for w, i, size_t in images:
        for t in range(size_t):
            timepoints.append((w, i, t))

    if len(timepoints) > k:
        logger.info("Sample %d / %d using seed %d", k, len(timepoints), seed)
        random.seed(seed)
        timepoints = random.sample(timepoints, k)
        timepoints.sort()

    return timepoints


def _get_segmentation_channels(
    channel_data: dict[str, str],
) -> tuple[int, ...]:
    """Get the channels used for segmentation."""
    nuc_ch = int(channel_data["DAPI"])
    if "Tub" in channel_data:
        return (nuc_ch, int(channel_data["Tub"]))
    return (nuc_ch,)


def _get_flatfieldcorr(
    conn: BlitzGateway,
    plate_id: int,
    dataset_id: int,
    channels: tuple[int, ...],
) -> npt.NDArray[Any] | None:
    """Get the flatfield correction channels used for segmentation."""
    image_name = flatfieldcorr_name(plate_id)
    dataset = conn.getObject("Dataset", dataset_id)
    assert dataset is not None, "Invalid dataset id"
    for image in dataset.listChildren():
        if image.getName() == image_name:
            pixels = image.getPrimaryPixels()
            # Assumes Z=0, T=0
            planes = [pixels.getPlane(0, c, 0) for c in channels]
            return np.array(planes)
    return None


def _get_image_to_segment(
    conn: BlitzGateway,
    image: ImageWrapper,
) -> ImageWrapper | None:
    """Return the image object used for segmentation.

    Typicially this is the original image object; in the case of a z-stack
    it will be the maximum intensity project (MIP) created during image
    analysis if available, or else None.

    Returns:
        Image for segmentation.
    """
    if image.getSizeZ() > 1:
        return conn.getObject("Image", _get_mip_image_id(image))
    return image


def _get_mip_image_id(image: ImageWrapper) -> int:
    """Return the MIP image ID if a map annotation exists.

    Returns:
        The MIP image ID; else 0
    """
    annotations = image.listAnnotations(ns=OmeroScreenNS.METADATA)
    if map_anns := [
        ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
    ]:
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            for k, v in ann_values.items():
                if k == "MIP":
                    return (
                        int(v.split(":")[-1]) if v.find(":") >= 0 else int(v)
                    )
    return 0


def _get_image(
    image: ImageWrapper, t: int, channels: tuple[int, ...] | None = None
) -> npt.NDArray[Any]:
    """Get the time point (CYX) from the image for the specified channels, or all channels."""
    pixels = image.getPrimaryPixels()
    gen = channels if channels else range(image.getSizeC())
    img = [pixels.getPlane(0, c, t) for c in gen]
    return np.array(img)
