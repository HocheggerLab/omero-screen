"""Module for handling OMERO utilities.

This module provides a shared functionality for handling data
and errors in the omero-utils package.

Available modules:

- omero_connect: Decorator that handles OMERO connection lifecycle management.
- omero_plate: Functions for creating and managing OMERO plates.
- images: Functions for creating and managing OMERO images.
- map_anns: Functions for creating and managing OMERO map annotations.
- message: Functions for handling OMERO messages.
- channels: Channel-name rules (role suffixes, PIP-FUCCI).

Importing the package loads neither ``omero_screen`` nor the OMERO gateway
(#83): ``omero_connect`` imports them when it first connects, and that import
loads the ``.env`` configuration.
"""

__version__ = "0.9.2"

from .omero_connect import omero_connect

__all__ = ["omero_connect"]
