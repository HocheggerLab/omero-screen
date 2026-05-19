"""Segmentation models."""

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from cellpose.models import CellposeModel as CM4

from omero_screen.cellpose3.models import MODEL_DIR as CP3_DIR
from omero_screen.cellpose3.models import CellposeModel as CM3
from omero_screen.torch import get_device


def apply_seg_profile(channel_name: str) -> dict[str, float]:
    """Return the segmentation profile for ``channel_name`` (case-insensitive).

    Looks up ``CHANNEL_SEG_PROFILES`` on the singleton ``default_config``.
    Returns an empty dict when no profile matches — callers should treat
    that as "use current defaults".
    """
    from omero_screen import default_config

    profiles = getattr(default_config, "CHANNEL_SEG_PROFILES", None) or {}
    profile: dict[str, float] = profiles.get(channel_name.lower(), {})
    return profile


def apply_gamma(img: npt.NDArray[Any], gamma: float) -> npt.NDArray[Any]:
    """Apply gamma compression to an image expected in [0, 1].

    ``gamma`` < 1 compresses the bright end and lifts dim values, making
    Cellpose more likely to detect dim objects in a frame with mixed
    expression levels (e.g. live-cell H2B-RFP).
    """
    result: npt.NDArray[Any] = np.power(np.clip(img, 0.0, 1.0), gamma).astype(
        img.dtype, copy=False
    )
    return result


class SegmentationModel:
    """Cellpose based segmentation model."""

    def __init__(self, model_name: str, device: torch.device | None = None):
        """Initialise the model for segmentation.

        Segmentation uses cellpose 3 or cellpose 4 models. The type is identified using
        a model prefix separated from the model name/path using a colon, e.g.
        cp3:cyto3, or cp4:cpsam.

        Names not using a model prefix are automatically detected using the cellpose
        model directory and the corresponding model file. If detection fails
        the model is assumed to be cellpose 3.

        Segmentation using cellpose 3 supports 1 or 2 channel input.

        Cellpose 4 supports up to 3 channel input.

        Parameters:
            model_name: Full path to pretrained model, or model name in models directory.
                Optionally prefix with cellpose version, e.g. 'cp4:'.
            device: Device used for model.
        """
        self.device = (
            device if isinstance(device, torch.device) else get_device()
        )

        if ":" in model_name:
            index = model_name.index(":")
            model_type = model_name[0:index]
            model_name = model_name[index + 1 :]
        else:
            model_type = _guess_model_type(model_name)

        # Dynamic switch between Cellpose 3 and 4
        self._cp3 = model_type == "cp3"
        if self._cp3:
            self.model = CM3(device=self.device, model_type=model_name)  # type: ignore[no-untyped-call]
        else:
            self.model = CM4(device=self.device, pretrained_model=model_name)

    def get_type(self) -> str:
        """Get the type of segmentation model.

        Currently supported types are:

        cellpose3
        cellpose4

        Returns:
            model type
        """
        return "cellpose3" if self._cp3 else "cellpose4"

    def eval(self, img: npt.NDArray[Any], **kwargs) -> npt.NDArray[Any]:  # type: ignore[no-untyped-def]
        """Segment the image.

        Segmentation uses cellpose 3 or cellpose 4 models. Additional arguments may be passed to the
        evaluation method. See the cellpose API reference for supported arguments. Default arguments:

        channel_axis: 0
        normalize: True
        diameter: None
        channels: [[0, 0]] or [[1, 2]] (cellpose 3 models; targets the first channel for segmentation)

        Args:
            img: Image (CYX).
            kwargs: Arguments to cellpose.

        Returns:
            Mask labels
        """
        # Add defaults
        if "channel_axis" not in kwargs:
            kwargs["channel_axis"] = 0
        if "normalize" not in kwargs:
            kwargs["normalize"] = True
        if "diameter" not in kwargs:
            kwargs["diameter"] = None

        if self._cp3 and "channels" not in kwargs:
            nchan = img.shape[kwargs["channel_axis"]]
            kwargs["channels"] = [[0, 0]] if nchan == 1 else [[1, 2]]
        m, _flows, _styles = self.model.eval(img, **kwargs)  # type: ignore[no-untyped-call]
        return m  # type: ignore[no-any-return]


def _guess_model_type(name: str) -> str:
    """Guess the model type based on the model name.

    Returns:
        model type
    """
    # Check if the name is a file, or a file in the cellpose default directory
    fn = None
    if os.path.isfile(name):
        fn = name
    else:
        path = os.path.join(CP3_DIR, name)
        if os.path.isfile(path):
            fn = path
    if fn is not None:
        try:
            # Load the state dictionary memory mapped (to avoid loading the tensors to memory)
            state_dict = torch.load(
                fn, mmap=True, weights_only=True, map_location="cpu"
            )
            # Cellpose 3 and 4 do not share any state dictionary keys.
            # Look for the keys that are specific to each model.
            if "diam_mean" in state_dict:
                return "cp3"
            elif "encoder.blocks.16.norm1.weight" in state_dict:
                return "cp4"
        except Exception as _e:
            # Ignore
            pass
    # Defaults to cellpose3
    return "cp3"
