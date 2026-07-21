"""OMERO Screen: Tools for managing and analyzing high-throughput screening data with OMERO."""

__version__ = "0.5.0"

import json
import os
import warnings
from dataclasses import dataclass, field

# Trackastra pulls in chardet>=7, but ``requests`` constrains it to <6 and so
# emits a RequestsDependencyWarning on every import. The libraries function
# fine — silence the noise so napari's notifications panel doesn't double-log
# it at startup. Filter on the message text (regex), not the warning class —
# importing the class would itself trigger the requests check before the
# filter is registered. The wildcard absorbs the versions in the message.
warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version.*",
    module=r"requests(\..*)?",
)

from loguru import logger  # noqa: E402

from .config import set_env_vars  # noqa: E402


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

    # Feature configuration. The structured form states the per-channel /
    # per-mask split explicitly: ``intensity`` features are measured for every
    # channel; ``morphology`` (mask-only geometry) once per segment. ``label``
    # and ``centroid`` are identity columns and added automatically. A legacy
    # flat list is also accepted (see ``image_analysis.normalize_featureset``).
    FEATURELIST: dict[str, list[str]] | list[str] = field(
        default_factory=lambda: {
            "intensity": [
                "intensity_max",
                "intensity_min",
                "intensity_mean",
                "intensity_std",
            ],
            "morphology": ["area"],
        }
    )

    # Channel-name → segmentation-profile overrides.
    # Lookup is case-insensitive. Keys: gamma (dynamic-range compression
    # applied before Cellpose, <1 lifts dim cells), cellprob_threshold and
    # flow_threshold (forwarded to cellpose .eval()).
    CHANNEL_SEG_PROFILES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "h2b_rfp": {"gamma": 0.5, "cellprob_threshold": -2.0},
            "tub_gfp": {
                "gamma": 0.5,
                "cellprob_threshold": -2.0,
                "flow_threshold": 0.6,
            },
        }
    )


# Feature names the downstream pipeline structurally depends on: the
# integrated-intensity column is ``intensity_mean`` × ``area`` of the nucleus,
# so both must be measured. ``label`` and ``centroid`` are identity columns
# added automatically by ``normalize_featureset`` and so are not required here.
REQUIRED_FEATURES: tuple[str, ...] = ("area", "intensity_mean")


def _validate_featurelist(features: dict[str, list[str]] | list[str]) -> None:
    """Validate an override FEATURELIST, raising on a pipeline-breaking omission.

    Accepts either the structured form (``{"intensity": [...],
    "morphology": [...]}``) or a legacy flat list.

    Args:
        features: Candidate feature config loaded from an OMERO_SCREEN_CONFIG file.

    Raises:
        ValueError: If any structurally-required feature is missing.
    """
    if isinstance(features, dict):
        present = set(features.get("intensity", [])) | set(
            features.get("morphology", [])
        )
    else:
        present = set(features)
    missing = [f for f in REQUIRED_FEATURES if f not in present]
    if missing:
        raise ValueError(
            f"FEATURELIST is missing required feature(s) {missing}; "
            f"these are needed by the analysis pipeline. Got: {features}"
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
            if isinstance(features, dict | list):
                _validate_featurelist(features)
                default_config.FEATURELIST = features
            profiles = data.get("CHANNEL_SEG_PROFILES", None)
            if isinstance(profiles, dict):
                merged = {
                    k.lower(): v
                    for k, v in default_config.CHANNEL_SEG_PROFILES.items()
                }
                for name, prof in profiles.items():
                    if isinstance(prof, dict):
                        merged[name.lower()] = prof
                default_config.CHANNEL_SEG_PROFILES = merged
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load configuration '{path}': {e}")
        raise e
