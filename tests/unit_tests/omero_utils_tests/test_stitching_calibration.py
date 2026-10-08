"""Stitch calibrations added from a site profile."""

import pytest

from omero_utils import stitching


def test_add_calibrations_from_a_site_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site-profile calibrations join the table and are chosen by pixel size."""
    monkeypatch.setattr(
        stitching, "STITCH_CALIBRATIONS", dict(stitching.STITCH_CALIBRATIONS)
    )
    stitching.add_calibrations(
        {
            "40x": {
                "pixel_size_um": 0.3,
                "overlap_x": 50,
                "overlap_y": 52,
                "translate_x": -10,
                "translate_y": 8,
            }
        }
    )
    assert stitching.resolve_stitch_params(0.3)["overlap_y"] == 52


def test_incomplete_calibration_is_refused() -> None:
    with pytest.raises(ValueError, match="needs exactly the keys"):
        stitching.add_calibrations({"bad": {"pixel_size_um": 0.3}})
