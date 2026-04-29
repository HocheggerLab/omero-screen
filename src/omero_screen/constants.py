"""Module for constants used in OMERO.

This module provides namespace constants for annotations added to OMERO objects.
"""

from enum import StrEnum


class OmeroScreenNS(StrEnum):
    """Namespace constants for OMERO screen annotations."""

    METADATA = "omero-screen/metadata"
    DATASET = "omero-screen/dataset"
