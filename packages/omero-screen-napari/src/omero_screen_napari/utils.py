from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omero.gateway import BlitzGateway, WellWrapper
from omero_screen.config import get_logger
from skimage import exposure

logger = get_logger(__name__)


def attach_file_to_well(
    well: WellWrapper, file_path: Path, conn: BlitzGateway
) -> None:
    """
    Attach the .npz file to the specified OMERO well.
    """
    namespace = f"Training Data for Well {well.getId()}"
    description = None  # Optionally, add a description
    print(f"\nUploading data to Well ID {well.getId()}")

    # Create the file annotation and link it to the well
    file_ann = conn.createFileAnnfromLocalFile(
        file_path,
        mimetype="application/octet-stream",
        ns=namespace,
        desc=description,
    )
    well.linkAnnotation(file_ann)


def scale_image(
    img: np.ndarray[Any, np.dtype[Any]],
    intensities: dict[int, tuple[int, int]],
) -> np.ndarray[Any, np.dtype[Any]]:
    scaled_channels = []
    for i in range(img.shape[-1]):
        scaled_channel = exposure.rescale_intensity(
            img[..., i],
            in_range=intensities[i],  # type: ignore
        )
        scaled_channels.append(scaled_channel)
    return np.stack(scaled_channels, axis=-1)


def save_fig(
    fig: Any,
    path: Path,
    fig_id: str,
    tight_layout: bool = True,
    fig_extension: str = "pdf",
    resolution: int = 300,
) -> None:
    """
    coherent saving of matplotlib figures as pdfs (default)
    :rtype: object
    :param path: path for saving
    :param fig_id: name of saved figure
    :param tight_layout: option, default True
    :param fig_extension: option, default pdf
    :param resolution: option, default 300dpi
    :return: None, saves Figure in poth
    """

    dest = path / f"{fig_id}.{fig_extension}"
    print("Saving figure", fig_id)
    if tight_layout:
        fig.set_tight_layout(True)
    plt.savefig(dest, format=fig_extension, dpi=resolution, facecolor="white")


path = Path.cwd() / "data"
path.mkdir(exist_ok=True)


def correct_channel_order(
    image_array: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Corrects the channel order in a time series image array where the channels are switched
    for even time points.

    Parameters:
    image_array (numpy array): The input image array with shape (T, Z, Y, X, C)
                               where T is the number of time points, Z is the number of z-planes,
                               Y is the height, X is the width, and C is the number of channels.

    Returns:
    corrected_n_mask (numpy array): The corrected time series for the first channel (nuclei segmentation mask).
    corrected_c_mask (numpy array): The corrected time series for the second channel (cell segmentation mask).
    """
    T, Z, Y, X, C = image_array.shape
    assert C == 2, "The function expects exactly 2 channels."

    corrected_n_mask = np.empty((T, Z, Y, X), dtype=image_array.dtype)
    corrected_c_mask = np.empty((T, Z, Y, X), dtype=image_array.dtype)

    for t in range(T):
        if t % 2 == 0:  # even time points
            corrected_c_mask[t] = image_array[
                t, :, :, :, 1
            ]  # c_mask is actually in the first channel
            corrected_n_mask[t] = image_array[
                t, :, :, :, 0
            ]  # n_mask is actually in the second channel
        else:  # odd time points
            corrected_c_mask[t] = image_array[
                t, :, :, :, 0
            ]  # n_mask is in the first channel
            corrected_n_mask[t] = image_array[
                t, :, :, :, 1
            ]  # c_mask is in the second channel

    return np.stack((corrected_n_mask, corrected_c_mask), axis=-1)
