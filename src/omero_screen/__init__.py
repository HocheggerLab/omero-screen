"""OMERO Screen: Tools for managing and analyzing high-throughput screening data with OMERO."""

__version__ = "0.2.4"


from dataclasses import dataclass, field

from .config import set_env_vars


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
