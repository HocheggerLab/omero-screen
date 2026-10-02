"""Tests for acquisition-timestamp recovery in ``read_plate``.

The Operetta writes a plane even when autofocus fails, but leaves its
acquisition date unset. ``<AbsTime></AbsTime>`` makes the exported bundle
unre-importable, so ``read_plate`` carries the previous field's timestamp
forward. These tests pin that behaviour and its seeding.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from omero_screen.export.plate_reader import read_plate

PLATE_START = datetime(2026, 9, 28, 9, 0, 0)


def make_image(image_id: int, acquired: datetime | None) -> MagicMock:
    """A stand-in ImageWrapper with a single 1×1 plane."""
    image = MagicMock()
    image.getId.return_value = image_id
    image.getAcquisitionDate.return_value = acquired
    image.getSizeT.return_value = 1
    image.getSizeZ.return_value = 1
    image.getSizeC.return_value = 1
    image.getSizeX.return_value = 8
    image.getSizeY.return_value = 8

    channel = MagicMock()
    channel.getLabel.return_value = "DAPI"
    channel.getExcitationWave.return_value = None
    channel.getEmissionWave.return_value = None
    image.getChannels.return_value = [channel]

    pixels = MagicMock()
    pixels.getPhysicalSizeX.return_value = None
    pixels.getPhysicalSizeY.return_value = None
    image.getPrimaryPixels.return_value = pixels

    image.getObjectiveSettings.return_value = None
    return image


def make_sample(image: MagicMock) -> MagicMock:
    """A stand-in WellSampleWrapper wrapping ``image``."""
    sample = MagicMock()
    sample.getImage.return_value = image
    sample.getPosX.return_value = None
    sample.getPosY.return_value = None
    return sample


def make_plate(
    acquisition_dates: list[datetime | None],
    start: datetime | None = PLATE_START,
) -> MagicMock:
    """A one-well plate with one field per entry in ``acquisition_dates``."""
    samples = [
        make_sample(make_image(100 + i, acquired))
        for i, acquired in enumerate(acquisition_dates)
    ]

    well = MagicMock()
    well.getRow.return_value = 0
    well.getColumn.return_value = 0
    well.getWellPos.return_value = "A1"
    well.listChildren.return_value = samples

    acquisition = MagicMock()
    acquisition.getStartTime.return_value = start

    plate = MagicMock()
    plate.getName.return_value = "plate-5054"
    plate.listChildren.return_value = [well]
    plate.listPlateAcquisitions.return_value = [acquisition]
    return plate


@pytest.fixture
def conn() -> MagicMock:
    """A connection whose ``getObject`` returns whatever plate is attached."""
    return MagicMock()


def abs_times(spec) -> list[datetime | None]:  # type: ignore[no-untyped-def]
    """The ``abs_time`` of every exported plane, in order."""
    return [image.abs_time for image in spec.images]


def test_missing_timestamp_reuses_the_previous_field(conn: MagicMock) -> None:
    """A field with no acquisition date inherits the one before it."""
    first = datetime(2026, 9, 28, 10, 0, 0)
    third = datetime(2026, 9, 28, 10, 2, 0)
    conn.getObject.return_value = make_plate([first, None, third])

    spec = read_plate(conn, 5054)

    assert abs_times(spec) == [first, first, third]


def test_consecutive_gaps_all_inherit_the_last_good_value(
    conn: MagicMock,
) -> None:
    """Runs of failed fields carry the same timestamp, not None."""
    first = datetime(2026, 9, 28, 10, 0, 0)
    conn.getObject.return_value = make_plate([first, None, None, None])

    spec = read_plate(conn, 5054)

    assert abs_times(spec) == [first] * 4


def test_leading_gap_falls_back_to_the_plate_start(conn: MagicMock) -> None:
    """A failure on the very first field seeds from the acquisition start."""
    second = datetime(2026, 9, 28, 10, 1, 0)
    conn.getObject.return_value = make_plate([None, second])

    spec = read_plate(conn, 5054)

    assert abs_times(spec) == [PLATE_START, second]


def test_no_timestamps_anywhere_stays_none(conn: MagicMock) -> None:
    """With no plate start either, abs_time is left unset rather than invented."""
    conn.getObject.return_value = make_plate([None, None], start=None)

    spec = read_plate(conn, 5054)

    assert abs_times(spec) == [None, None]


def test_present_timestamps_are_untouched(conn: MagicMock) -> None:
    """The fallback must not perturb a plate that acquired cleanly."""
    dates = [
        datetime(2026, 9, 28, 10, 0, 0),
        datetime(2026, 9, 28, 10, 1, 0),
        datetime(2026, 9, 28, 10, 2, 0),
    ]
    conn.getObject.return_value = make_plate(dates)

    spec = read_plate(conn, 5054)

    assert abs_times(spec) == dates
