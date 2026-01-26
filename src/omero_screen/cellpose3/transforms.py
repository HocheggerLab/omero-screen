"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

# Note: This file has been ported from Cellpose v3 and is licensed under the
# BSD 3-Clause "New" or "Revised" License.

import logging
import warnings

import numpy as np
import torch

transforms_logger = logging.getLogger(__name__)

from cellpose.transforms import move_axis


def convert_image(x, channels, channel_axis=None, z_axis=None, do_3D=False, nchan=2):
    """Converts the image to have the z-axis first, channels last.

    Args:
        x (numpy.ndarray or torch.Tensor): The input image.
        channels (list or None): The list of channels to use (ones-based, 0=gray). If None, all channels are kept.
        channel_axis (int or None): The axis of the channels in the input image. If None, the axis is determined automatically.
        z_axis (int or None): The axis of the z-dimension in the input image. If None, the axis is determined automatically.
        do_3D (bool): Whether to process the image in 3D mode. Defaults to False.
        nchan (int): The number of channels to keep if the input image has more than nchan channels.

    Returns:
        numpy.ndarray: The converted image.

    Raises:
        ValueError: If the input image has less than two channels and channels are not specified.
        ValueError: If the input image is 2D and do_3D is True.
        ValueError: If the input image is 4D and do_3D is False.
    """
    # check if image is a torch array instead of numpy array
    # converts torch to numpy
    ndim = x.ndim
    if torch.is_tensor(x):
        transforms_logger.warning("torch array used as input, converting to numpy")
        x = x.cpu().numpy()

    # squeeze image, and if channel_axis or z_axis given, transpose image
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq, s in enumerate(x.shape) if s == 1])
        # remove channel axis if number of channels is 1
        if len(to_squeeze) > 0:
            channel_axis = update_axis(
                channel_axis, to_squeeze,
                x.ndim) if channel_axis is not None else None
            z_axis = update_axis(z_axis, to_squeeze,
                                 x.ndim) if z_axis is not None else None
            x = x.squeeze()

    # put z axis first
    if z_axis is not None and x.ndim > 2 and z_axis != 0:
        x = move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        z_axis = 0
    elif z_axis is None and x.ndim > 2 and channels is not None and min(x.shape) > 5 :
        # if there are > 5 channels and channels!=None, assume first dimension is z
        min_dim = min(x.shape)
        if min_dim != channel_axis:
            z_axis = (x.shape).index(min_dim)
            if z_axis != 0:
                x = move_axis(x, m_axis=z_axis, first=True)
                if channel_axis is not None:
                    channel_axis += 1
            transforms_logger.warning(f"z_axis not specified, assuming it is dim {z_axis}")
            transforms_logger.warning(f"if this is actually the channel_axis, use 'model.eval(channel_axis={z_axis}, ...)'")
            z_axis = 0

    if z_axis is not None:
        if x.ndim == 3:
            x = x[..., np.newaxis]

    # put channel axis last
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == 2:
        x = x[:, :, np.newaxis]

    if do_3D:
        if ndim < 3:
            transforms_logger.critical("ERROR: cannot process 2D images in 3D mode")
            raise ValueError("ERROR: cannot process 2D images in 3D mode")
        elif x.ndim < 4:
            x = x[..., np.newaxis]

    if channel_axis is None:
        x = move_min_dim(x)

    if x.ndim > 3:
        transforms_logger.info(
            "multi-stack tiff read in as having %d planes %d channels" %
            (x.shape[0], x.shape[-1]))

    # convert to float32
    x = x.astype("float32")

    if channels is not None:
        channels = channels[0] if len(channels) == 1 else channels
        if len(channels) < 2:
            transforms_logger.critical("ERROR: two channels not specified")
            raise ValueError("ERROR: two channels not specified")
        x = reshape(x, channels=channels)

    else:
        # code above put channels last
        if nchan is not None and x.shape[-1] > nchan:
            transforms_logger.warning(
                "WARNING: more than %d channels given, use 'channels' input for specifying channels - just using first %d channels to run processing"
                % (nchan, nchan))
            x = x[..., :nchan]

        # if not do_3D and x.ndim > 3:
        #    transforms_logger.critical("ERROR: cannot process 4D images in 2D mode")
        #    raise ValueError("ERROR: cannot process 4D images in 2D mode")

        if nchan is not None and x.shape[-1] < nchan:
            x = np.concatenate((x, np.tile(np.zeros_like(x), (1, 1, nchan - 1))),
                               axis=-1)

    return x


def reshape(data, channels=[0, 0], chan_first=False):
    """Reshape data using channels.

    Args:
        data (numpy.ndarray): The input data. It should have shape (Z x ) Ly x Lx x nchan
            if data.ndim==3 and data.shape[0]<8, it is assumed to be nchan x Ly x Lx.
        channels (list of int, optional): The channels to use for reshaping. The first element
            of the list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). The
            second element of the list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
            For instance, to train on grayscale images, input [0,0]. To train on images with cells
            in green and nuclei in blue, input [2,3]. Defaults to [0, 0].
        chan_first (bool, optional): Whether to return the reshaped data with channel as the first
            dimension. Defaults to False.

    Returns:
        numpy.ndarray: The reshaped data with shape (Z x ) Ly x Lx x nchan (if chan_first==False).
    """
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))

    # use grayscale image
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros(data.shape, "float32")), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros(data.shape, "float32")), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        warnings.warn("'chan to seg' to seg has value range of ZERO")
                    else:
                        warnings.warn(
                            "'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0"
                        )
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros(data.shape, "float32")), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data
