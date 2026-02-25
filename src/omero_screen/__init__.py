"""OMERO Screen: Tools for managing and analyzing high-throughput screening data with OMERO."""

__version__ = "0.3.3"

import json
import os
from dataclasses import dataclass, field

from .config import get_logger, set_env_vars


@dataclass
class DefaultConfig:
    """Default configuration for the OMERO Screen application."""

    MODEL_DICT: dict[str, str] = field(
        default_factory=lambda: {
            "nuclei": "Nuclei_Hoechst",
            "RPE": "RPE-1_Tub_Hoechst",
            "HELA": "HeLa_Tub_Hoechst",
            "U2OS": "U2OS_Tub_Hoechst",
            "HCC1143": "RPE-1_Tub_Hoechst",
            "MM231": "MM231_Tub_Hoechst",
            "PALB": "only_PALB",
        }
    )

    FEATURELIST: list[str] = field(
        default_factory=lambda: [
            "label",
            "area",
            "intensity_max",
            "intensity_min",
            "intensity_mean",
            "centroid",
        ]
    )


# Create a singleton instance of DefaultConfig
default_config = DefaultConfig()

set_env_vars()

# Load configuration from file if available
path = os.getenv("OMERO_SCREEN_CONFIG")
if path is not None and os.path.exists(path):
    try:
        with open(path) as f:
            data = json.load(f)
            models = data.get("MODEL_DICT", None)
            if isinstance(models, dict):
                default_config.MODEL_DICT = models
            features = data.get("FEATURELIST", None)
            if isinstance(features, list):
                default_config.FEATURELIST = features
    except Exception as e:  # noqa: BLE001
        get_logger(__name__).error(
            "Failed to load configuration '%s': %s", path, e
        )
        raise e
