"""PyTorch utilities.

This module provides utilities for use with PyTorch.

Main Functions:
    - get_device: Get a torch device given the available backends.
"""

import os

import torch


def get_device() -> torch.device:
    """Get a torch device given the available backends.

    Respects the ``OMERO_SCREEN_USE_GPU`` environment variable:
    - ``"1"`` or ``"true"`` (case-insensitive): use GPU if available.
    - ``"0"`` or ``"false"``: force CPU.
    - Unset: auto-detect (default behaviour).
    """
    env = os.getenv("OMERO_SCREEN_USE_GPU", "").lower()
    if env in ("0", "false"):
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
