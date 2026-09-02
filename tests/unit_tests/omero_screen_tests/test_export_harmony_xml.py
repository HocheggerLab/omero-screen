"""Tests for the Harmony ``Index.idx.xml`` builder.

These run without an OMERO server. The grammar assertions are checked against
the real measurements in ``examples/``, so the tests fail if the reference data
and the builder ever disagree.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pytest

from omero_screen.export.harmony_xml import (
    HARMONY_DEFAULTS,
    IDENTITY_ORIENTATION_MATRIX,
    NAMESPACE,
    ImageSpec,
    PlateSpec,
    build_index_xml,
    well_id,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
NS = f"{{{NAMESPACE}}}"


def make_spec(**overrides: object) -> ImageSpec:
    """An ImageSpec with sensible defaults, overridable per test."""
    kwargs: dict[str, object] = {
        "row": 1,
        "col": 1,
        "field": 1,
        "plane": 1,
        "timepoint": 0,
        "channel": 1,
        "channel_name": "DAPI",
        "size_x": 1080,
        "size_y": 1080,
        "resolution_x_m": 1.1959521619135234e-06,
        "resolution_y_m": 1.1959521619135234e-06,
        "position_x_m": 0.0,
        "position_y_m": 0.0,
    }
    kwargs.update(overrides)
    return ImageSpec(**kwargs)  # type: ignore[arg-type]


def make_plate(images: list[ImageSpec]) -> PlateSpec:
    """A minimal PlateSpec wrapping ``images``."""
    return PlateSpec(
        name="TestPlate",
        rows=8,
        columns=12,
        measurement_id="00000000-0000-0000-0000-000000000000",
        measurement_start=datetime(2024, 7, 19, 10, 5, 57),
        images=images,
    )


def parse(plate: PlateSpec) -> ET.Element:
    """Render and re-parse a plate spec."""
    return ET.fromstring(build_index_xml(plate).decode("utf-8-sig"))


# --------------------------------------------------------------------------
# ID and filename grammar
# --------------------------------------------------------------------------


def test_well_id_is_zero_padded_row_then_column() -> None:
    """Well ids are row then column, each zero-padded to two digits."""
    assert well_id(2, 1) == "0201"
    assert well_id(12, 11) == "1211"


@pytest.mark.parametrize(
    ("kwargs", "expected_id", "expected_url"),
    [
        (
            {},
            "0101K1F1P1R1",
            "r01c01f01p01-ch1sk1fk1fl1.tiff",
        ),
        (
            {"row": 2, "col": 1, "field": 2, "timepoint": 1, "channel": 3},
            "0201K2F2P1R3",
            "r02c01f02p01-ch3sk2fk1fl1.tiff",
        ),
        (
            {"row": 2, "col": 1, "field": 2, "plane": 2, "channel": 3},
            "0201K1F2P2R3",
            "r02c01f02p02-ch3sk1fk1fl1.tiff",
        ),
    ],
)
def test_id_and_url_grammar(
    kwargs: dict[str, object], expected_id: str, expected_url: str
) -> None:
    """Cases taken verbatim from examples/{timeseries,3D}_testdata."""
    spec = make_spec(**kwargs)
    assert spec.image_id == expected_id
    assert spec.url == expected_url


def test_timepoint_is_zero_based_but_k_and_sk_are_one_based() -> None:
    """TimepointID stays 0-based while its K/sk appearances are 1-based."""
    spec = make_spec(timepoint=3)
    assert "K4" in spec.image_id
    assert "sk4" in spec.url
    root = parse(make_plate([spec]))
    image = root.find(f"{NS}Images/{NS}Image")
    assert image is not None
    assert image.findtext(f"{NS}TimepointID") == "3"


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_maps_section_is_omitted() -> None:
    """Flatfield profiles are recomputed by omero-screen, never exported."""
    root = parse(make_plate([make_spec()]))
    assert root.find(f"{NS}Maps") is None


def test_orientation_matrix_is_identity() -> None:
    """A real matrix would be applied to PositionX/Y and mirror the offsets."""
    root = parse(make_plate([make_spec()]))
    matrices = [el.text for el in root.iter(f"{NS}OrientationMatrix")]
    assert matrices == [IDENTITY_ORIENTATION_MATRIX]


def test_positions_round_trip_at_full_precision() -> None:
    """Stage positions must not be rounded on the way out."""
    x, y = -26.701287387585275, 40.79740106363608
    root = parse(make_plate([make_spec(position_x_m=x, position_y_m=y)]))
    image = root.find(f"{NS}Images/{NS}Image")
    assert image is not None
    assert float(image.findtext(f"{NS}PositionX", "")) == x
    assert float(image.findtext(f"{NS}PositionY", "")) == y


def test_wells_are_deduplicated_and_list_their_images() -> None:
    """Each well appears once and lists exactly its own planes."""
    specs = [
        make_spec(channel=1),
        make_spec(channel=2),
        make_spec(row=2, col=1, channel=1),
    ]
    root = parse(make_plate(specs))

    plate_wells = [
        el.get("id") for el in root.iter(f"{NS}Well") if el.get("id")
    ]
    assert plate_wells == ["0101", "0201"]

    wells = root.findall(f"{NS}Wells/{NS}Well")
    assert [w.findtext(f"{NS}id") for w in wells] == ["0101", "0201"]
    assert len(wells[0].findall(f"{NS}Image")) == 2
    assert len(wells[1].findall(f"{NS}Image")) == 1


def test_well_row_and_col_are_one_based() -> None:
    """OMERO's 0-based row/column are written out 1-based."""
    root = parse(make_plate([make_spec(row=3, col=5)]))
    well = root.find(f"{NS}Wells/{NS}Well")
    assert well is not None
    assert well.findtext(f"{NS}Row") == "3"
    assert well.findtext(f"{NS}Col") == "5"


def test_output_has_utf8_bom_and_default_namespace() -> None:
    """Harmony writes a BOM and the HarmonyV5 default namespace."""
    raw = build_index_xml(make_plate([make_spec()]))
    assert raw.startswith(b"\xef\xbb\xbf")
    assert f'xmlns="{NAMESPACE}"'.encode() in raw


def test_inert_hardware_fields_are_present() -> None:
    """The reader requires them even though nothing downstream reads them."""
    root = parse(make_plate([make_spec()]))
    image = root.find(f"{NS}Images/{NS}Image")
    assert image is not None
    for tag, value in HARMONY_DEFAULTS.items():
        assert image.findtext(f"{NS}{tag}") == value


def test_optional_wavelengths_are_omitted_when_unknown() -> None:
    """Absent wavelengths are omitted rather than invented."""
    root = parse(make_plate([make_spec()]))
    image = root.find(f"{NS}Images/{NS}Image")
    assert image is not None
    assert image.find(f"{NS}MainExcitationWavelength") is None
    assert image.find(f"{NS}MainEmissionWavelength") is None


def test_wavelengths_are_written_when_known() -> None:
    """Known excitation/emission are written with nm units."""
    root = parse(
        make_plate([make_spec(excitation_nm=365.0, emission_nm=465.0)])
    )
    image = root.find(f"{NS}Images/{NS}Image")
    assert image is not None
    assert float(image.findtext(f"{NS}MainExcitationWavelength", "")) == 365.0
    assert float(image.findtext(f"{NS}MainEmissionWavelength", "")) == 465.0


def test_empty_plate_is_rejected() -> None:
    """Exporting nothing is an error, not an empty index."""
    with pytest.raises(ValueError, match="no images"):
        build_index_xml(make_plate([]))


# --------------------------------------------------------------------------
# Agreement with the reference measurements shipped in examples/
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example", ["2D_testdata", "3D_testdata", "timeseries_testdata"]
)
def test_grammar_matches_reference_measurements(example: str) -> None:
    """Every id/URL pair in the shipped examples is reproduced exactly."""
    index = EXAMPLES / example / "Images" / "Index.idx.xml"
    if not index.exists():
        pytest.skip(f"reference measurement {example} not available")

    root = ET.parse(index).getroot()
    elements = root.findall(f"{NS}Images/{NS}Image")
    assert elements, f"{example} has no <Image> elements"
    for element in elements:
        spec = make_spec(
            row=int(element.findtext(f"{NS}Row", "")),
            col=int(element.findtext(f"{NS}Col", "")),
            field=int(element.findtext(f"{NS}FieldID", "")),
            plane=int(element.findtext(f"{NS}PlaneID", "")),
            timepoint=int(element.findtext(f"{NS}TimepointID", "")),
            channel=int(element.findtext(f"{NS}ChannelID", "")),
        )
        assert spec.image_id == element.findtext(f"{NS}id")
        assert spec.url == element.findtext(f"{NS}URL")
