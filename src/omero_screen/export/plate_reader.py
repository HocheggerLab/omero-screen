"""Read an OMERO plate into the plain specs that :mod:`harmony_xml` renders.

All OMERO traversal lives here so the XML builder stays server-free.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from loguru import logger
from omero.gateway import BlitzGateway, ImageWrapper, PlateWrapper, WellWrapper
from omero_utils.message import PlateDataError, PlateNotFoundError

from omero_screen.export.harmony_xml import ImageSpec, PlateSpec

#: Harmony plate geometry inferred from the well positions we actually see.
#: OMERO does not record the physical plate format, so a plate whose wells all
#: fall inside A1-H12 is described as a 96-well; anything larger rounds up.
PLATE_FORMATS: tuple[tuple[int, int, str], ...] = (
    (8, 12, "96 PerkinElmer CellCarrier Ultra"),
    (16, 24, "384 PerkinElmer CellCarrier Ultra"),
)

#: Operetta 10x objective, used when OMERO has no objective settings.
DEFAULT_MAGNIFICATION = 10.0
DEFAULT_NA = 0.3


def _length_in_m(obj: Any, position: Any) -> float:
    """Convert an OMERO ``Length`` to metres, or 0.0 when absent.

    Mirrors ``plate_aggregation._get_length_in_m``: the owning Blitz object
    does the unit conversion, so we never assume the stored symbol.
    """
    if position is None:
        return 0.0
    return float(obj._unwrapunits(position, units="METER").getValue())


def _plate_format(max_row: int, max_col: int) -> tuple[int, int, str]:
    """Pick the smallest known plate format that holds ``max_row``/``max_col``."""
    for rows, cols, name in PLATE_FORMATS:
        if max_row <= rows and max_col <= cols:
            return rows, cols, name
    # Larger than anything we know: describe the plate by its own extent.
    return max_row, max_col, f"{max_row * max_col} custom"


def _resolve_wells(
    plate: PlateWrapper, wells: tuple[str, ...] | None
) -> list[WellWrapper]:
    """Return the wells to export, in plate order.

    Args:
        plate: The OMERO plate.
        wells: Well positions such as ``("A1", "B2")``. ``None`` exports all.

    Returns:
        The matching wells, ordered as OMERO lists them.

    Raises:
        PlateDataError: If a requested well is not on the plate.
    """
    all_wells = list(plate.listChildren())
    if not wells:
        return all_wells

    wanted = {w.strip().upper() for w in wells}
    by_pos = {w.getWellPos().upper(): w for w in all_wells}
    if missing := sorted(wanted - by_pos.keys()):
        raise PlateDataError(
            f"Wells {missing} are not on plate {plate.getId()}. "
            f"Available: {sorted(by_pos)}",
            logger,
        )
    return [w for w in all_wells if w.getWellPos().upper() in wanted]


def _channel_names(image: ImageWrapper) -> list[str]:
    """Channel labels, falling back to ``Channel{n}`` for unnamed channels."""
    return [
        channel.getLabel() or f"Channel{index}"
        for index, channel in enumerate(image.getChannels(), start=1)
    ]


def _to_nm(value: Any) -> float | None:
    """Unwrap an OMERO wavelength to a float, tolerating ``None``.

    ``getExcitationWave``/``getEmissionWave`` return a bare float on some
    servers and a wrapped ``Length`` on others, so handle both.
    """
    if value is None:
        return None
    return float(value.getValue() if hasattr(value, "getValue") else value)


def _wavelengths(
    image: ImageWrapper, index: int
) -> tuple[float | None, float | None]:
    """Excitation/emission in nm for channel ``index``, or ``(None, None)``.

    Both are optional in the Harmony schema, so a plate imported without
    wavelength metadata simply omits the elements rather than inventing values.
    """
    channel = image.getChannels()[index]
    return (
        _to_nm(channel.getExcitationWave()),
        _to_nm(channel.getEmissionWave()),
    )


def _objective(image: ImageWrapper) -> tuple[float, float]:
    """Magnification and NA, falling back to the 10x/0.3 Operetta default."""
    settings = image.getObjectiveSettings()
    objective = None if settings is None else settings.getObjective()
    if objective is None:
        return DEFAULT_MAGNIFICATION, DEFAULT_NA
    mag = objective.getNominalMagnification()
    na = objective.getLensNA()
    return (
        DEFAULT_MAGNIFICATION if mag is None else float(mag),
        DEFAULT_NA if na is None else float(na),
    )


def _acquisition_start(plate: PlateWrapper) -> datetime | None:
    """Start time of the plate's first acquisition, if recorded.

    ``getStartTime`` yields a ``datetime`` on some servers and epoch
    milliseconds on others, so normalise both to a ``datetime``.
    """
    for acquisition in plate.listPlateAcquisitions():
        start = acquisition.getStartTime()
        if isinstance(start, datetime):
            return start
        if start is not None:
            return datetime.fromtimestamp(float(start) / 1000)
    return None


def read_plate(
    conn: BlitzGateway,
    plate_id: int,
    wells: tuple[str, ...] | None = None,
    max_fields: int | None = None,
    name: str | None = None,
) -> PlateSpec:
    """Build a :class:`PlateSpec` describing what to export.

    Args:
        conn: OMERO connection.
        plate_id: Plate to export.
        wells: Well positions to keep (e.g. ``("A1", "B2")``); ``None`` = all.
        max_fields: Keep at most this many fields per well; ``None`` = all.
        name: Override the exported plate name. Defaults to the OMERO name.

    Returns:
        A plate spec with one :class:`ImageSpec` per exported 2D plane.

    Raises:
        PlateNotFoundError: If the plate does not exist.
        PlateDataError: If a requested well is missing or a well has no fields.
    """
    plate = conn.getObject("Plate", plate_id)
    if plate is None:
        raise PlateNotFoundError(f"Plate {plate_id} was not found", logger)

    selected = _resolve_wells(plate, wells)
    if not selected:
        raise PlateDataError(f"Plate {plate_id} has no wells", logger)

    specs: list[ImageSpec] = []
    max_row = max_col = 1

    for well in selected:
        # OMERO rows/columns are 0-based; Harmony is 1-based.
        row, col = well.getRow() + 1, well.getColumn() + 1
        max_row, max_col = max(max_row, row), max(max_col, col)

        samples = list(well.listChildren())
        if max_fields is not None:
            samples = samples[:max_fields]
        if not samples:
            raise PlateDataError(
                f"Well {well.getWellPos()} on plate {plate_id} has no fields",
                logger,
            )

        for field_index, sample in enumerate(samples, start=1):
            image = sample.getImage()
            pixels = image.getPrimaryPixels()
            names = _channel_names(image)
            mag, na = _objective(image)
            # Stage positions are written straight through: the identity
            # OrientationMatrix means they come back unchanged on re-import.
            pos_x = _length_in_m(sample, sample.getPosX())
            pos_y = _length_in_m(sample, sample.getPosY())
            res_x = _length_in_m(image, pixels.getPhysicalSizeX())
            res_y = _length_in_m(image, pixels.getPhysicalSizeY())
            acquired = image.getAcquisitionDate()

            for t in range(image.getSizeT()):
                for z in range(image.getSizeZ()):
                    for c in range(image.getSizeC()):
                        excitation, emission = _wavelengths(image, c)
                        specs.append(
                            ImageSpec(
                                row=row,
                                col=col,
                                field=field_index,
                                plane=z + 1,
                                timepoint=t,
                                channel=c + 1,
                                channel_name=names[c],
                                size_x=int(image.getSizeX()),
                                size_y=int(image.getSizeY()),
                                resolution_x_m=res_x,
                                resolution_y_m=res_y,
                                position_x_m=pos_x,
                                position_y_m=pos_y,
                                abs_time=acquired,
                                excitation_nm=excitation,
                                emission_nm=emission,
                                objective_magnification=mag,
                                objective_na=na,
                                omero_image_id=int(image.getId()),
                            )
                        )

    rows, columns, plate_type = _plate_format(max_row, max_col)
    logger.info(
        f"Plate {plate_id}: {len(selected)} well(s), {len(specs)} plane(s) "
        f"to export as a {plate_type}"
    )

    return PlateSpec(
        name=name or plate.getName(),
        rows=rows,
        columns=columns,
        measurement_id=str(uuid.uuid4()),
        measurement_start=_acquisition_start(plate),
        plate_type_name=plate_type,
        images=specs,
        well_positions=[w.getWellPos() for w in selected],
    )
