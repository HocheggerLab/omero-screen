"""Module for handling image attachments loaded to the OMERO server.

This module provides functions for uploading masks and maximum intensity projections (MIPs) to OMERO datasets.

Available functions:

- upload_masks(conn, dataset_id, image, n_mask, c_mask): Uploads generated images to OMERO server and links them to the specified dataset.
- delete_masks(conn, dataset_id): Removes all segmentation masks from an OMERO dataset.
- parse_mip(conn, image_id, dataset_id): Get the maximum intensity projection of a z-stack image.
- delete_mip(conn, image_id): Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.

"""

from typing import Any

import numpy as np
import numpy.typing as npt
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    ImageWrapper,
    MapAnnotationWrapper,
)
from omero_screen.config import get_logger
from typing_extensions import Generator

from omero_utils.map_anns import add_map_annotations, delete_map_annotation

logger = get_logger(__name__)


def upload_masks(
    conn: BlitzGateway,
    dataset_id: int,
    image: ImageWrapper,
    n_mask: npt.NDArray[Any],
    c_mask: npt.NDArray[Any] | None = None,
) -> None:
    """Uploads generated images to OMERO server and links them to the specified dataset.

    The id of the mask is stored as an annotation on the original screen image.

    Args:
        conn: OMERO connection
        dataset_id: ID of the dataset to link the masks to
        image: Image object
        n_mask: Nuclei segmentation mask (TYX)
        c_mask: Cell segmentation mask (TYX)

    """
    image_name = f"{image.getId()}_segmentation"
    dataset = conn.getObject("Dataset", dataset_id)

    def plane_gen() -> Generator[npt.NDArray[Any]]:
        """Generator that yields each plane in the n_mask and c_mask arrays.

        Yields T first, then C, then Z. Assumes 2d images so no z iteration.
        """
        for i in range(n_mask.shape[0]):
            yield n_mask[i]
        if c_mask is not None:
            for i in range(n_mask.shape[0]):
                yield c_mask[i]

    # Create the image in the dataset
    num_channels = 2 if c_mask is not None else 1
    mask = conn.createImageFromNumpySeq(
        plane_gen(),
        image_name,
        1,  # Z
        num_channels,  # C
        n_mask.shape[0],  # T
        dataset=dataset,
    )

    # Create a map annotation to store the segmentation mask ID
    delete_map_annotation(conn, image, "Segmentation_Mask")
    add_map_annotations(conn, image, {"Segmentation_Mask": mask.getId()})


def delete_masks(conn: BlitzGateway, dataset_id: int) -> None:
    """Removes all segmentation masks from an OMERO dataset.

    Args:
        conn: OMERO connection
        dataset_id: OMERO dataset ID

    """
    dataset = conn.getObject("Dataset", dataset_id)
    for child in dataset.listChildren():
        if child.getName().endswith("_segmentation"):
            image_id = int(child.getName()[: -len("_segmentation")])
            image = conn.getObject("Image", image_id)
            delete_map_annotation(conn, image, "Segmentation_Mask")
            conn.deleteObject(child._obj)


def parse_mip(
    conn: BlitzGateway, image_id: int, dataset_id: int
) -> npt.NDArray[Any]:
    """Get the maximum intensity projection of a z-stack image.

    The MIP is created and saved to OMERO as an annotation if absent;
    existing map annotations are loaded.

    Args:
        conn: OMERO connection
        image_id: Image ID
        dataset_id: Dataset ID to save/load the MIP.

    Returns:
        MIP image

    """
    image = conn.getObject("Image", image_id)

    if mip_id := _check_mip_annotation(image):
        _, mip_array = get_image(conn, mip_id)
        if isinstance(mip_array, np.ndarray):
            return mip_array
        logger.warning(
            "The image is linked to a missing MIP; this will be regenerated"
        )
    return _load_mip(conn, image, dataset_id)


def _check_mip_annotation(image: ImageWrapper) -> int:
    """Check if a MIP map annotation exists.

    Args:
        image: OMERO image object
    Returns:
        The annotation MIP image ID; else 0

    """
    annotations = image.listAnnotations()
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
    delete_map_annotation(conn, image, "MIP")
    add_map_annotations(conn, image, {"MIP": new_image.getId()})
    return mip_array


def _process_mip(conn: BlitzGateway, image_id: int) -> npt.NDArray[Any]:
    """Generate maximum intensity projection of an image.

    Args:
        conn: OMERO connection
        image_id: Image ID
    Returns:
        numpy array of maximum intensity projection (t, 1, y, x, c)

    """
    _, array = get_image(conn, image_id)
    return np.max(array, axis=1, keepdims=True)  # type: ignore


def _image_generator(
    image_array: npt.NDArray[Any],
) -> Generator[npt.NDArray[Any]]:
    # Input is TZYXC
    # iterate through T first, then C then Z. Here z=0.
    for c in range(image_array.shape[-1]):
        for t in range(image_array.shape[0]):
            yield image_array[t, 0, ..., c]


def delete_mip(conn: BlitzGateway, image_id: int) -> None:
    """Removes a maximum intensity projection of a z-stack image saved in OMERO as an annotation.

    Args:
        conn: OMERO connection
        image_id: OMERO image ID

    """
    image = conn.getObject("Image", image_id)
    if mip_id := _check_mip_annotation(image):
        delete_map_annotation(conn, image, "MIP")
        mip = conn.getObject("Image", mip_id)
        conn.deleteObject(mip._obj)
