"""PyTorch utilities.

This module provides utilities for use with PyTorch.

Main Functions:
    - get_device: Get a torch device given the available backends.
"""

import torch


def get_device() -> torch.device:
    """Get a torch device given the available backends."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
