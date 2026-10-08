"""Record how a plate was processed: versions, settings and models.

At the start of each ``omero-screen <plate_id>`` run the pipeline writes one
map annotation on the plate (namespace ``omero-screen/provenance``, replaced on
every run) and logs the same block. It answers, for any plate, which release
processed it, with which flags, on which device and with which segmentation
models and stitch calibration: the defaults now depend on the machine
(Cellpose 4 on an NVIDIA GPU, Cellpose 3 elsewhere), so this cannot be
inferred afterwards.
"""

from __future__ import annotations

import os
import platform
import shlex
import socket
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from loguru import logger

from omero_screen.constants import OmeroScreenNS

#: Packages whose versions are recorded.
VERSIONED = ("omero-screen", "cellpose", "torch", "trackastra", "omero-py")

#: Environment variables (pipeline flags are passed through them) recorded.
SETTINGS_ENV = (
    "OMERO_SCREEN_INFERENCE_MODEL",
    "OMERO_SCREEN_TRACKING_MODEL",
    "OMERO_SCREEN_TRACKING_MODE",
    "OMERO_SCREEN_TRACKING_WINDOW",
    "OMERO_SCREEN_USE_GPU",
    "OMERO_SCREEN_CONFIG",
    "OMERO_SCREEN_STITCH_CONFIG",
)


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def run_provenance(
    metadata: Any,
    stitch_mode: bool = False,
    segmentation_mode: bool = False,
    delete_existing: bool = False,
) -> dict[str, str]:
    """The provenance record of the run about to process ``metadata``'s plate.

    Args:
        metadata: The plate's ``MetadataParser`` (after ``manage_metadata``).
        stitch_mode: Stitched-well segmentation.
        segmentation_mode: Segmentation only.
        delete_existing: Masks deleted before the run.
    """
    from omero_screen import default_config, settings
    from omero_screen.image_analysis import get_cell_model, get_nucleus_model
    from omero_screen.torch import get_device

    record: dict[str, str] = {
        f"version {pkg}": _version(pkg) for pkg in VERSIONED
    }
    record["python"] = platform.python_version()
    record["command"] = shlex.join(["omero-screen", *sys.argv[1:]])
    record["run started"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["host"] = socket.gethostname()
    record["omero user"] = os.environ.get("USERNAME", "")
    record["site profile"] = settings.load().site or "none"
    record["device"] = str(get_device())
    record["mode"] = (
        ", ".join(
            name
            for name, on in (
                ("stitched", stitch_mode),
                ("segmentation only", segmentation_mode),
                ("masks deleted first", delete_existing),
            )
            if on
        )
        or "per field"
    )
    record["model override"] = default_config.MODEL_OVERRIDE or "none"
    record["nucleus model"] = get_nucleus_model()
    cell_lines = sorted(
        {str(c) for c in metadata.well_data.get("cell_line", [])}
    )
    for cell_line in cell_lines:
        record[f"cell model {cell_line}"] = str(get_cell_model(cell_line))
    pixel_size = getattr(metadata, "pixel_size", None)
    record["pixel size (um)"] = str(pixel_size)
    if stitch_mode:
        from omero_utils.stitching import resolve_stitch_params

        record["stitch parameters"] = str(resolve_stitch_params(pixel_size))
    for var in SETTINGS_ENV:
        if value := os.environ.get(var):
            record[var] = value
    return record


def record_provenance(
    conn: Any, plate_id: int, record: dict[str, str]
) -> None:
    """Replace the plate's provenance annotation with ``record`` and log it."""
    from omero_utils.map_anns import (
        add_map_annotations,
        delete_map_annotations,
    )

    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        logger.warning(f"Plate {plate_id} not found; provenance not recorded")
        return
    delete_map_annotations(conn, plate, ns=OmeroScreenNS.PROVENANCE)
    add_map_annotations(conn, plate, record, ns=OmeroScreenNS.PROVENANCE)
    lines = "\n".join(f"  {key}: {value}" for key, value in record.items())
    logger.info(f"Provenance of plate {plate_id}:\n{lines}")
