"""Module for managing plate metadata as a singleton instance.

This module provides a singleton instance of PlateMetadata that can be used
across the project to store and access plate metadata in memory.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PlateMetadata:
    """Data class to store plate metadata."""

    channels: dict[str, int]
    well_inputs: dict[str, Any]
    pixel_size: float


class PlateMetadataSingleton:
    """Singleton class to manage a single instance of PlateMetadata."""

    _instance: PlateMetadata | None = None

    @classmethod
    def get_instance(cls) -> PlateMetadata:
        """Get the singleton instance of PlateMetadata.

        Returns:
            PlateMetadata: The singleton instance of PlateMetadata

        Raises:
            RuntimeError: If the instance is accessed before being initialized
        """
        if cls._instance is None:
            raise RuntimeError(
                "PlateMetadata instance not initialized. Call set_instance() first."
            )
        return cls._instance

    @classmethod
    def set_instance(cls, metadata: PlateMetadata) -> None:
        """Set the singleton instance of PlateMetadata.

        Args:
            metadata: The PlateMetadata instance to set

        Raises:
            RuntimeError: If an instance is already set
        """
        if cls._instance is not None:
            raise RuntimeError("PlateMetadata instance already set")
        cls._instance = metadata

    @classmethod
    def clear_instance(cls) -> None:
        """Clear the singleton instance of PlateMetadata."""
        cls._instance = None
