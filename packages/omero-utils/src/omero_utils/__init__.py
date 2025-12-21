"""Module for handling OMERO utilities.

This module provides a shared functionality for handling data
and errors in the omero-utils package.

Available modules:

- omero_connect: Decorator that handles OMERO connection lifecycle management.
- omero_plate: Functions for creating and managing OMERO plates.
- omero_images: Functions for creating and managing OMERO images.
- omero_map_anns: Functions for creating and managing OMERO map annotations.
- omero_message: Functions for handling OMERO messages.

"""

__version__ = "0.2.4"
from omero_screen.config import set_env_vars

set_env_vars()
from .omero_connect import omero_connect  # noqa: E402

__all__ = ["omero_connect"]
