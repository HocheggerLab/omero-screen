"""Recover omero-screen experimental metadata as a re-attachable Excel file.

Harmony's ``Index.idx.xml`` carries channel *names*, so ``MetadataParser`` can
rebuild the plate's channel annotation from the images alone. What it cannot
recover are the **well conditions** (``cell_line``, ``condition``, ...), which
live only in OMERO map annotations. This module writes them back out in the
exact workbook shape ``MetadataParser._load_data_from_excel`` expects:

- exactly two sheets, named ``Sheet1`` then ``Sheet2`` (the parser compares the
  sheet-name list for equality, so both the names and the order matter);
- ``Sheet1``: columns ``Channels`` and ``Index``;
- ``Sheet2``: a ``Well`` column plus ``cell_line`` and any condition columns.

Wells that carry no annotations are written back as ``cell_line = "Empty"``,
which is how the parser marks them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from omero.gateway import BlitzGateway, PlateWrapper
from omero_utils.map_anns import parse_annotations

from omero_screen.constants import OmeroScreenNS

#: Value the metadata parser uses to flag a well with no experimental content.
EMPTY_WELL = "Empty"


def _channel_frame(
    plate: PlateWrapper, fallback_names: list[str]
) -> pd.DataFrame:
    """Build ``Sheet1`` from the plate's channel annotation.

    Args:
        plate: The OMERO plate.
        fallback_names: Channel names read from the images, used when the plate
            has never been through omero-screen and so has no annotation.

    Returns:
        A frame with ``Channels`` and ``Index`` columns.
    """
    annotations = parse_annotations(plate, ns=OmeroScreenNS.METADATA)
    if annotations:
        return pd.DataFrame(
            {
                "Channels": list(annotations.keys()),
                "Index": [str(v) for v in annotations.values()],
            }
        )

    logger.warning(
        f"Plate {plate.getId()} has no omero-screen channel annotation; "
        f"falling back to the image channel names {fallback_names}"
    )
    return pd.DataFrame(
        {
            "Channels": fallback_names,
            "Index": [str(i) for i in range(len(fallback_names))],
        }
    )


def _well_frame(
    plate: PlateWrapper, well_positions: list[str]
) -> pd.DataFrame:
    """Build ``Sheet2`` from per-well map annotations.

    Args:
        plate: The OMERO plate.
        well_positions: Well positions being exported, e.g. ``["A1", "B2"]``.

    Returns:
        A frame with a ``Well`` column plus every annotation key found. Wells
        without annotations get ``cell_line = "Empty"``.
    """
    wanted = {p.upper() for p in well_positions}
    rows: list[dict[str, Any]] = []
    annotated = 0

    for well in plate.listChildren():
        position = well.getWellPos()
        if position.upper() not in wanted:
            continue
        annotation = parse_annotations(well, ns=OmeroScreenNS.METADATA)
        if annotation:
            annotated += 1
            rows.append({"Well": position, **annotation})
        else:
            rows.append({"Well": position, "cell_line": EMPTY_WELL})

    if not annotated:
        logger.warning(
            f"Plate {plate.getId()}: none of the exported wells carry "
            f"omero-screen annotations, so the metadata sheet has no "
            f"experimental conditions. Fill in Sheet2 before re-attaching it."
        )

    frame = pd.DataFrame(rows)
    # Guarantee the parser's required column even on a fully unannotated plate.
    if "cell_line" not in frame.columns:
        frame["cell_line"] = EMPTY_WELL
    columns = ["Well", "cell_line"] + [
        c for c in frame.columns if c not in ("Well", "cell_line")
    ]
    return frame[columns]


def write_metadata_excel(
    conn: BlitzGateway,
    plate_id: int,
    well_positions: list[str],
    fallback_channels: list[str],
    path: Path,
) -> Path:
    """Write the re-attachable metadata workbook for ``plate_id``.

    Args:
        conn: OMERO connection.
        plate_id: Plate whose annotations are being recovered.
        well_positions: Wells included in the export.
        fallback_channels: Channel names from the images, used only when the
            plate has no channel annotation.
        path: Destination ``.xlsx`` path.

    Returns:
        ``path``, for convenience.

    Raises:
        ValueError: If the plate cannot be loaded.
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise ValueError(f"Plate {plate_id} was not found")

    channels = _channel_frame(plate, fallback_channels)
    wells = _well_frame(plate, well_positions)

    # Sheet order is load-bearing: the parser checks the sheet-name list for
    # equality with ["Sheet1", "Sheet2"].
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        channels.to_excel(writer, sheet_name="Sheet1", index=False)
        wells.to_excel(writer, sheet_name="Sheet2", index=False)

    logger.info(
        f"Wrote metadata workbook {path.name}: "
        f"{len(channels)} channel(s), {len(wells)} well(s)"
    )
    return path
