"""Module for handling image attachments loaded to the OMERO server."""

from typing import Any

import numpy as np
import numpy.typing as npt
import omero
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    ImageWrapper,
    MapAnnotationWrapper,
)
from omero_screen.config import get_logger
from typing_extensions import Generator

logger = get_logger(__name__)


def upload_masks(
    conn: BlitzGateway,
    dataset_id: int,
    omero_image: ImageWrapper,
    n_mask: npt.NDArray[Any],
    c_mask: npt.NDArray[Any] | None = None,
) -> None:
    """
    Uploads generated images to OMERO server and links them to the specified dataset.
    The id of the mask is stored as an annotation on the original screen image.

    Args:
        conn: OMERO connection
        dataset_id: ID of the dataset to link the masks to
        omero_image: Image object
        n_mask: Nuclei segmentation mask
        c_mask: Cell segmentation mask
    """

    image_name = f"{omero_image.getId()}_segmentation"
    dataset = conn.getObject("Dataset", dataset_id)

    def plane_gen() -> Generator[npt.NDArray[Any]]:
        """Generator that yields each plane in the n_mask and c_mask arrays"""
        for i in range(n_mask.shape[0]):
            yield n_mask[i]
            if c_mask is not None:
                yield c_mask[i]

    # Create the image in the dataset
    num_channels = 2 if c_mask is not None else 1
    mask = conn.createImageFromNumpySeq(
        plane_gen(),
        image_name,
        1,
        num_channels,
        n_mask.shape[0],
        dataset=dataset,
    )

    # Create a map annotation to store the segmentation mask ID
    key_value_data = [["Segmentation_Mask", str(mask.getId())]]

    # Get the existing map annotations of the image
    map_anns = list(
        omero_image.listAnnotations(
            ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
        )
    )
    if map_anns:  # If there are existing map annotations
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            if (
                "Segmentation_Mask" in ann_values
            ):  # If the desired annotation exists
                conn.deleteObject(ann._obj)  # Delete the existing annotation
    # Create a new map annotation
    map_ann = MapAnnotationWrapper(conn)
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(key_value_data)

    map_ann.save()
    omero_image.linkAnnotation(map_ann)


def parse_mip(
    conn: BlitzGateway, image_id: int, dataset_id: int
) -> npt.NDArray[Any]:
    """Get the maximum intensity projection of a z-stack image. The MIP is created
    and saved to OMERO as an annotation if absent; existing map annotations are loaded.
    Args:
        conn: OMERO connection
        image_id: Image ID
        dataset_id: Dataset ID to save/load the MIP.
    Returns:
        MIP image
    """
    image = conn.getObject("Image", image_id)

    mip_id = _check_map_annotation(image)
    if mip_id:
        _, mip_array = get_image(conn, mip_id)
        if isinstance(mip_array, np.ndarray):
            return mip_array
        logger.warning(
            "The image is linked to a missing MIP; this will be regenerated"
        )
    return _load_mip(conn, image, dataset_id)


def _check_map_annotation(image: ImageWrapper) -> int:
    """Check if a MIP map annotation exists.
    Args:
        omero_object: OMERO image object
    Returns:
        The annotation MIP image ID; else 0
    """
    if map_anns := image.listAnnotations(
        ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
    ):
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            for k, v in ann_values.items():
                if k == "MIP":
                    return int(v.split(":")[-1])
    return 0


def _load_mip(
    conn: BlitzGateway, image: ImageWrapper, dataset_id: int
) -> npt.NDArray[Any]:
    """Create a maximum intensity projection of a z-stack image and save to OMERO as an annotation.
    Args:
        conn: OMERO connection
        image: Image object
        dataset_id: Dataset ID to save the MIP.
    Returns:
        MIP image
    """
    dataset = conn.getObject("Dataset", dataset_id)
    mip_array = _process_mip(conn, image.getId())
    channel_num = mip_array.shape[-1]
    mip_name = f"MIP_{image.getId()}"
    img_gen = _image_generator(mip_array)
    new_image = conn.createImageFromNumpySeq(
        # Generator creates size (zct)
        img_gen,
        mip_name,
        1,
        channel_num,
        mip_array.shape[0],
        dataset=dataset,
    )
    _add_mip_annotation(
        conn, image, [("MIP", f"Image ID: {new_image.getId()}")]
    )
    return mip_array


def _process_mip(conn: BlitzGateway, image_id: int) -> npt.NDArray[Any]:
    """Generate maximum intensity projection of an image.
    Args:
        conn: OMERO connection
        image: Image ID
    Returns:
        numpy array of maximum intensity projection (t, 1, y, x, c)
    """
    _, array = get_image(conn, image_id)
    return np.max(array, axis=1, keepdims=True)


def _image_generator(
    image_array: npt.NDArray[Any],
) -> Generator[npt.NDArray[Any]]:
    # Input is TZYXC
    # iterate through T first, then C then Z. Here z=0.
    for c in range(image_array.shape[-1]):
        for t in range(image_array.shape[0]):
            yield image_array[t, 0, ..., c]


def _add_mip_annotation(
    conn: BlitzGateway, image: ImageWrapper, key_value: list[tuple[str, str]]
) -> None:
    """Add a map annotation to an OMERO object.
    Args:
        conn: OMERO connection
        image: OMERO image object
        key_value: List of key-value pairs
    """
    _remove_mip_annotation(conn, image)

    map_ann = omero.gateway.MapAnnotationWrapper(conn)
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(key_value)
    map_ann.save()
    image.linkAnnotation(map_ann)


def _remove_mip_annotation(conn: BlitzGateway, image: ImageWrapper) -> None:
    """Remove the MIP map annotation from an OMERO object.
    Args:
        conn: OMERO connection
        image: OMERO image object
        key_value: List of key-value pairs
    """
    map_anns = list(
        image.listAnnotations(
            ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
        )
    )
    if map_anns:  # If there are existing map annotations
        for ann in map_anns:
            if "MIP" in dict(ann.getValue()):
                conn.deleteObject(ann._obj)  # Delete the annotation


def delete_mip(conn: BlitzGateway, image_id: int) -> None:
    """Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.
    Args:
        conn: OMERO connection
        image_id: OMERO image ID
    """
    image = conn.getObject("Image", image_id)
    mip_id = _check_map_annotation(image)
    if mip_id:
        _remove_mip_annotation(conn, image)
        mip = conn.getObject("Image", mip_id)
        conn.deleteObject(mip._obj)
