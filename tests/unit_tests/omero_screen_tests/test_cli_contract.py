"""Compatibility contract for the pre-Click ``omero-screen`` CLI."""

from __future__ import annotations

import pytest

from bin.run_omero_screen import get_parser


def test_default_argument_contract() -> None:
    """Freeze the argparse names, types, and defaults used by the runner."""
    args = get_parser().parse_args(["1234", "5678"])

    assert vars(args) == {
        "ID": [1234, 5678],
        "env": None,
        "config": None,
        "inference": None,
        "gallery": 10,
        "batch": 16,
        "segmentation": False,
        "cp4": False,
        "model": None,
        "benchmark": False,
        "stitch": False,
        "stream_stitch": None,
        "track": None,
        "track_mode": "greedy",
        "track_batch_size": 4,
        "track_device": None,
        "track_window": None,
        "log_level": None,
        "log_file": None,
        "verbose": False,
    }


def test_representative_override_contract() -> None:
    """Capture variadic, tri-state, optional-value, and logging options."""
    args = get_parser().parse_args(
        [
            "1234",
            "--env",
            "production",
            "--config",
            "config.json",
            "--inference",
            "model-a.pt",
            "model-b.pt",
            "--gallery",
            "15",
            "--batch",
            "8",
            "--segmentation",
            "--cp4",
            "--model",
            "cp4:cpsam",
            "--benchmark",
            "--stitch",
            "--no-stream-stitch",
            "--track",
            "custom.ckpt",
            "--track-mode",
            "ilp",
            "--track-batch-size",
            "2",
            "--track-device",
            "cpu",
            "--track-window",
            "6",
            "--log-level",
            "DEBUG",
            "--log-file",
            "none",
            "--verbose",
        ]
    )

    assert args.ID == [1234]
    assert args.inference == ["model-a.pt", "model-b.pt"]
    assert args.segmentation is True
    assert args.stream_stitch is False
    assert args.track == "custom.ckpt"
    assert args.track_mode == "ilp"
    assert args.track_device == "cpu"
    assert args.verbose is True


def test_bare_track_uses_general_2d() -> None:
    """The optional-valued tracking flag is a load-bearing compatibility case."""
    args = get_parser().parse_args(["1234", "--track"])
    assert args.track == "general_2d"


@pytest.mark.parametrize(
    ("flag", "attribute"),
    [
        ("--no-segmentation", "segmentation"),
        ("--no-benchmark", "benchmark"),
        ("--no-stitch", "stitch"),
    ],
)
def test_boolean_optional_negative_forms(flag: str, attribute: str) -> None:
    """Preserve argparse's generated negative boolean option names."""
    args = get_parser().parse_args(["1234", flag])
    assert getattr(args, attribute) is False


def test_invalid_tracking_mode_exits_two() -> None:
    """Invalid choices remain command-line usage errors."""
    with pytest.raises(SystemExit, match="2"):
        get_parser().parse_args(["1234", "--track-mode", "invalid"])
