"""Build a PerkinElmer Harmony ``Index.idx.xml`` from plain Python specs.

This module is deliberately free of any OMERO import: it turns dataclasses into
the XML string, which makes the fiddly part -- the exact element grammar the
Bio-Formats ``OperettaReader`` expects -- unit-testable without a server.

The grammar was verified against the reference measurements shipped in
``examples/`` (``2D_testdata``, ``3D_testdata``, ``timeseries_testdata``):

- Well id      ``{row:02d}{col:02d}``                e.g. ``0201``
- Image id     ``{well}K{t+1}F{field}P{plane}R{ch}`` e.g. ``0201K2F2P1R3``
- File name    ``r{row:02d}c{col:02d}f{field:02d}p{plane:02d}-ch{ch}sk{t+1}fk1fl1.tiff``

Row, column, field, plane and channel are 1-based; ``TimepointID`` is the only
0-based index, while its appearances in ``K``/``sk`` are 1-based.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

NAMESPACE = "http://www.perkinelmer.com/PEHH/HarmonyV5"

#: Identity affine with no translation.
#:
#: **Do not replace this with a matrix copied from a real acquisition.**
#: ``OperettaReader`` applies ``OrientationMatrix`` as an affine transform to
#: ``PositionX``/``PositionY`` before handing them to OMERO -- a real Harmony
#: matrix carries a Y flip and a metre-scale translation, so copying one in
#: would mirror every stitching offset while leaving the images themselves
#: looking correct. With the identity, the positions we write are the positions
#: OMERO reports back (verified bit-exact on a round-trip import).
IDENTITY_ORIENTATION_MATRIX = "[[1,0,0,0],[0,1,0,0],[0,0,1,0]]"

#: Fields the reader requires but nothing downstream reads. They describe
#: hardware we are not modelling, so they are constants rather than anything
#: recovered from OMERO.
HARMONY_DEFAULTS: dict[str, str] = {
    "State": "Ok",
    "FlimID": "1",
    "ImageType": "Signal",
    "AcquisitionType": "NonConfocal",
    "IlluminationType": "Epifluorescence",
    "ChannelType": "Fluorescence",
    "BinningX": "1",
    "BinningY": "1",
    "CameraType": "AndorZylaCam",
}


def well_id(row: int, col: int) -> str:
    """Return the Harmony well id, e.g. ``(2, 1) -> "0201"``."""
    return f"{row:02d}{col:02d}"


@dataclass(frozen=True)
class ImageSpec:
    """One 2D plane: a single channel/z/timepoint of one field."""

    row: int
    col: int
    field: int
    plane: int
    timepoint: int
    channel: int
    channel_name: str
    size_x: int
    size_y: int
    resolution_x_m: float
    resolution_y_m: float
    position_x_m: float
    position_y_m: float
    position_z_m: float = 0.0
    abs_time: datetime | None = None
    excitation_nm: float | None = None
    emission_nm: float | None = None
    objective_magnification: float = 10.0
    objective_na: float = 0.3
    exposure_time_s: float = 0.01
    max_intensity: int = 65535
    #: OMERO image this plane is read from. Ignored by the XML; the writer
    #: uses it to fetch pixels.
    omero_image_id: int = 0

    @property
    def well(self) -> str:
        """Harmony well id this plane belongs to."""
        return well_id(self.row, self.col)

    @property
    def image_id(self) -> str:
        """Harmony image id, e.g. ``0201K2F2P1R3``."""
        return (
            f"{self.well}K{self.timepoint + 1}"
            f"F{self.field}P{self.plane}R{self.channel}"
        )

    @property
    def url(self) -> str:
        """TIFF file name for this plane."""
        return (
            f"r{self.row:02d}c{self.col:02d}"
            f"f{self.field:02d}p{self.plane:02d}"
            f"-ch{self.channel}sk{self.timepoint + 1}fk1fl1.tiff"
        )


@dataclass
class PlateSpec:
    """Plate-level header of the measurement."""

    name: str
    rows: int
    columns: int
    measurement_id: str
    measurement_start: datetime | None = None
    plate_type_name: str = "96 PerkinElmer CellCarrier Ultra"
    user: str = "omero-screen-export"
    instrument_type: str = "Sonata"
    images: list[ImageSpec] = field(default_factory=list)
    #: OMERO well positions (``A1``) included, in plate order. Not written to
    #: the XML -- carried so the metadata sheet can be built without a second
    #: pass over the plate.
    well_positions: list[str] = field(default_factory=list)


def _fmt_time(value: datetime | None) -> str:
    """Format a datetime the way Harmony writes them (ISO 8601)."""
    return "" if value is None else value.isoformat()


def _sub(parent: ET.Element, tag: str, text: str, **attrib: str) -> ET.Element:
    """Append a leaf element carrying ``text``."""
    el = ET.SubElement(parent, tag, attrib)
    el.text = text
    return el


def _num(value: float) -> str:
    """Render a float without losing precision to ``str()`` rounding."""
    return repr(float(value))


def _append_image(parent: ET.Element, spec: ImageSpec) -> None:
    """Append one fully populated ``<Image>`` element.

    Element order follows a real Harmony index. The reader does not require it,
    but matching makes the output diffable against a reference measurement.
    """
    el = ET.SubElement(parent, "Image", {"Version": "1"})
    d = HARMONY_DEFAULTS
    _sub(el, "id", spec.image_id)
    _sub(el, "State", d["State"])
    _sub(el, "URL", spec.url)
    _sub(el, "Row", str(spec.row))
    _sub(el, "Col", str(spec.col))
    _sub(el, "FieldID", str(spec.field))
    _sub(el, "PlaneID", str(spec.plane))
    _sub(el, "TimepointID", str(spec.timepoint))
    _sub(el, "ChannelID", str(spec.channel))
    _sub(el, "FlimID", d["FlimID"])
    _sub(el, "ChannelName", spec.channel_name)
    _sub(el, "ImageType", d["ImageType"])
    _sub(el, "AcquisitionType", d["AcquisitionType"])
    _sub(el, "IlluminationType", d["IlluminationType"])
    _sub(el, "ChannelType", d["ChannelType"])
    _sub(el, "ImageResolutionX", _num(spec.resolution_x_m), Unit="m")
    _sub(el, "ImageResolutionY", _num(spec.resolution_y_m), Unit="m")
    _sub(el, "ImageSizeX", str(spec.size_x))
    _sub(el, "ImageSizeY", str(spec.size_y))
    _sub(el, "BinningX", d["BinningX"])
    _sub(el, "BinningY", d["BinningY"])
    _sub(el, "MaxIntensity", str(spec.max_intensity))
    _sub(el, "CameraType", d["CameraType"])
    _sub(el, "PositionX", _num(spec.position_x_m), Unit="m")
    _sub(el, "PositionY", _num(spec.position_y_m), Unit="m")
    _sub(el, "PositionZ", _num(spec.position_z_m), Unit="m")
    _sub(el, "AbsPositionZ", _num(spec.position_z_m), Unit="m")
    _sub(el, "MeasurementTimeOffset", "0", Unit="s")
    _sub(el, "AbsTime", _fmt_time(spec.abs_time))
    if spec.excitation_nm is not None:
        _sub(
            el, "MainExcitationWavelength", _num(spec.excitation_nm), Unit="nm"
        )
    if spec.emission_nm is not None:
        _sub(el, "MainEmissionWavelength", _num(spec.emission_nm), Unit="nm")
    _sub(
        el,
        "ObjectiveMagnification",
        _num(spec.objective_magnification),
        Unit="",
    )
    _sub(el, "ObjectiveNA", _num(spec.objective_na), Unit="")
    _sub(el, "ExposureTime", _num(spec.exposure_time_s), Unit="s")
    _sub(el, "OrientationMatrix", IDENTITY_ORIENTATION_MATRIX)


def build_index_xml(plate: PlateSpec) -> bytes:
    """Render ``plate`` as the bytes of an ``Index.idx.xml``.

    Args:
        plate: Plate header plus every plane to be listed.

    Returns:
        UTF-8 bytes **with a BOM**, as Harmony writes them.

    Raises:
        ValueError: If ``plate`` carries no images.
    """
    if not plate.images:
        raise ValueError(f"Plate {plate.name!r} has no images to export")

    ET.register_namespace("", NAMESPACE)
    root = ET.Element(f"{{{NAMESPACE}}}EvaluationInputData", {"Version": "1"})
    _sub(root, "User", plate.user)
    _sub(root, "InstrumentType", plate.instrument_type)

    # Group planes by well, preserving acquisition order.
    wells: dict[str, list[ImageSpec]] = {}
    for spec in plate.images:
        wells.setdefault(spec.well, []).append(spec)

    plates_el = ET.SubElement(root, "Plates")
    plate_el = ET.SubElement(plates_el, "Plate")
    _sub(plate_el, "PlateID", plate.name)
    _sub(plate_el, "MeasurementID", plate.measurement_id)
    _sub(plate_el, "MeasurementStartTime", _fmt_time(plate.measurement_start))
    _sub(plate_el, "Name", plate.name)
    _sub(plate_el, "PlateTypeName", plate.plate_type_name)
    _sub(plate_el, "PlateRows", str(plate.rows))
    _sub(plate_el, "PlateColumns", str(plate.columns))
    for wid in wells:
        ET.SubElement(plate_el, "Well", {"id": wid})

    wells_el = ET.SubElement(root, "Wells")
    for wid, specs in wells.items():
        well_el = ET.SubElement(wells_el, "Well")
        _sub(well_el, "id", wid)
        _sub(well_el, "Row", str(specs[0].row))
        _sub(well_el, "Col", str(specs[0].col))
        for spec in specs:
            ET.SubElement(well_el, "Image", {"id": spec.image_id})

    # No <Maps>: its only entries are FlatfieldProfile blobs, which the reader
    # treats as optional and which omero-screen recomputes anyway.

    images_el = ET.SubElement(root, "Images")
    for spec in plate.images:
        _append_image(images_el, spec)

    ET.indent(root, space="  ")
    body: bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return b"\xef\xbb\xbf" + body
