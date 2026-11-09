"""PIP-FUCCI is recognised by the exact channel names ``pip`` and ``geminin`` only."""

import pytest
from omero_utils.channels import PipFucci, resolve_pip_fucci
from omero_utils.message import ChannelAnnotationError


def test_5054_channels() -> None:
    """The channel names of plate 5054 resolve to PIP-FUCCI."""
    names = ["geminin", "pip", "spyDNA_nucleus", "bf_cell"]
    assert resolve_pip_fucci(names, 217) == PipFucci("pip", "geminin")


def test_case_and_role_suffix() -> None:
    """Matching ignores case and a role suffix, and returns names verbatim."""
    assert resolve_pip_fucci(["PIP", "Geminin_nucleus"], 2) == PipFucci(
        "PIP", "Geminin_nucleus"
    )


@pytest.mark.parametrize(
    "names",
    [
        ["geminin", "DAPI"],  # geminin alone, e.g. an IF stain
        ["cdt1", "geminin"],  # classical FUCCI is not PIP-FUCCI
        ["Geminin_488", "pip"],  # a spelled-out reporter is not the name
        ["pip_mcherry", "geminin"],
        ["pipette", "geminin"],
    ],
)
def test_other_names_are_not_fucci(names: list[str]) -> None:
    """Any name other than exactly pip and geminin is not PIP-FUCCI."""
    assert resolve_pip_fucci(names, 217) is None


def test_single_timepoint_is_not_fucci() -> None:
    """A fixed plate is never FUCCI, whatever its channels are called."""
    assert resolve_pip_fucci(["pip", "geminin", "DAPI"], 1) is None


def test_two_channels_for_one_reporter_raise() -> None:
    """Two channels resolving to one reporter are an annotation error."""
    with pytest.raises(ChannelAnnotationError):
        resolve_pip_fucci(["pip", "PIP_nucleus", "geminin"], 10)


def test_metadata_parser_reexports_the_same_objects() -> None:
    """The pipeline's old import path still works and is the same code (#75)."""
    import omero_utils.channels as channels

    from omero_screen import metadata_parser

    for name in (
        "PipFucci",
        "resolve_pip_fucci",
        "strip_role_suffix",
        "ROLE_SUFFIXES",
        "PIP_FUCCI_PIP",
        "PIP_FUCCI_GEMININ",
    ):
        assert getattr(metadata_parser, name) is getattr(channels, name)
