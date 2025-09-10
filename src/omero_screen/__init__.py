"""OMERO Screen: Tools for managing and analyzing high-throughput screening data with OMERO."""

__version__ = "0.2.0"


from dataclasses import dataclass, field

from .config import set_env_vars


@dataclass
class DefaultConfig:
    """Default configuration for the OMERO Screen application."""

    MODEL_DICT: dict[str, str] = field(
        default_factory=lambda: {
            "nuclei": "Nuclei_Hoechst",
            "RPE-1": "RPE-1_Tub_Hoechst",
            "RPE-1_WT": "RPE-1_Tub_Hoechst",
            "RPE-1_P53KO": "RPE-1_Tub_Hoechst",
            "RPE-1_WT_CycE": "RPE-1_Tub_Hoechst",
            "RPE-1_P53KO_CycE": "RPE-1_Tub_Hoechst",
            "HELA": "HeLa_Tub_Hoechst",
            "U2OS": "U2OS_Tub_Hoechst",
            "MM231": "RPE-1_Tub_Hoechst",
            "HCC1143": "RPE-1_Tub_Hoechst",
            "MM231_SCR": "MM231_Tub_Hoechst",
            "MM231_GWL": "MM231_Tub_Hoechst",
            "SCR_MM231": "MM231_Tub_Hoechst",
            "GWL_MM231": "MM231_Tub_Hoechst",
            "RPE1WT": "RPE-1_Tub_Hoechst",
            "RPE1P53KO": "RPE-1_Tub_Hoechst",
            "RPE1wt_PALB": "only_PALB",
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
