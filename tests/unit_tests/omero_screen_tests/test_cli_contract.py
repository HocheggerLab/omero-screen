"""Behaviour contract for the ``omero-screen`` Click CLI.

The pipeline itself is never run here. ``_apply_environment`` and ``_run`` are
patched out, so what is under test is the command surface: names, types,
defaults, the two awkward argparse spellings, and the environment variables
each option sets.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

import bin.run_omero_screen as runner_module
from bin.run_omero_screen import TRACK_DEFAULT_MODEL, cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def parsed(runner):
    """Invoke the CLI and return the kwargs each collaborator received."""

    def invoke(argv, expect_exit=0):
        with (
            patch.object(runner_module, "_apply_environment") as env_mock,
            patch.object(runner_module, "_run") as run_mock,
        ):
            result = runner.invoke(cli, argv)
        assert result.exit_code == expect_exit, result.output
        return {
            "env": env_mock.call_args.kwargs if env_mock.call_args else None,
            "run": run_mock.call_args.kwargs if run_mock.call_args else None,
            "output": result.output,
        }

    return invoke


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_default_argument_contract(parsed) -> None:
    """Freeze the names, types and defaults the runner depends on."""
    got = parsed(["1234", "5678"])
    assert got["run"] == {
        "ids": [1234, 5678],
        "segmentation": False,
        "stitch": False,
        "delete": False,
        "benchmark": False,
    }
    assert got["env"] == {
        "env": None,
        "config": None,
        "stitch_config": None,
        "inference": (),
        "gallery": 10,
        "batch": 16,
        "track": None,
        "track_mode": "greedy",
        "track_batch_size": 4,
        "track_device": None,
        "track_window": None,
        "stream_stitch": None,
        "verbose": False,
        "log_level": None,
        "log_file": None,
    }


def test_at_least_one_plate_id_is_required(runner) -> None:
    assert runner.invoke(cli, []).exit_code == 2


def test_plate_ids_must_be_integers(runner) -> None:
    assert runner.invoke(cli, ["not-a-plate"]).exit_code == 2


# --------------------------------------------------------------------------
# --track: argparse nargs="?" with a const
# --------------------------------------------------------------------------


class TestTrackOptionalValue:
    """`--track [MODEL]` must keep working in all four spellings."""

    def test_absent_is_none(self, parsed) -> None:
        assert parsed(["1234"])["env"]["track"] is None

    def test_bare_flag_uses_the_default_model(self, parsed) -> None:
        got = parsed(["1234", "--track"])
        assert got["env"]["track"] == TRACK_DEFAULT_MODEL

    def test_space_separated_value(self, parsed) -> None:
        assert parsed(["1234", "--track", "ilp2d"])["env"]["track"] == "ilp2d"

    def test_equals_separated_value(self, parsed) -> None:
        assert parsed(["1234", "--track=ilp2d"])["env"]["track"] == "ilp2d"

    def test_bare_flag_before_another_option(self, parsed) -> None:
        """A bare --track must not swallow the following flag."""
        got = parsed(["1234", "--track", "--stitch"])
        assert got["env"]["track"] == TRACK_DEFAULT_MODEL
        assert got["run"]["stitch"] is True

    def test_greedy_value_matches_argparse(self, parsed) -> None:
        """argparse's nargs="?" also consumed a following plate ID.

        Verified against the pre-migration parser: `--track 1234 5678`
        gave track='1234' and ID=[5678], and `--track 1234` alone was a
        usage error. Click reproduces both.
        """
        got = parsed(["--track", "1234", "5678"])
        assert got["env"]["track"] == "1234"
        assert got["run"]["ids"] == [5678]

    def test_bare_track_with_no_plate_id_is_a_usage_error(
        self, parsed
    ) -> None:
        parsed(["--track", "1234"], expect_exit=2)


# --------------------------------------------------------------------------
# --stream-stitch: tri-state
# --------------------------------------------------------------------------


class TestStreamStitchTriState:
    """Omission means 'decide automatically' and must stay distinct."""

    def test_omitted_is_none(self, parsed) -> None:
        assert parsed(["1234"])["env"]["stream_stitch"] is None

    def test_enabled_is_true(self, parsed) -> None:
        assert parsed(["1234", "--stream-stitch"])["env"]["stream_stitch"] is (
            True
        )

    def test_disabled_is_false(self, parsed) -> None:
        got = parsed(["1234", "--no-stream-stitch"])
        assert got["env"]["stream_stitch"] is False


# --------------------------------------------------------------------------
# Variadic and boolean pairs
# --------------------------------------------------------------------------


class TestVariadicInference:
    """argparse nargs='+' and Click's repeated form must both work."""

    def test_argparse_spelling(self, parsed) -> None:
        got = parsed(["1234", "--inference", "a.pth", "b.pth"])
        assert got["env"]["inference"] == ("a.pth", "b.pth")

    def test_click_spelling(self, parsed) -> None:
        got = parsed(
            ["1234", "--inference", "a.pth", "--inference", "b.pth"]
        )
        assert got["env"]["inference"] == ("a.pth", "b.pth")

    def test_stops_at_the_next_option(self, parsed) -> None:
        got = parsed(["1234", "--inference", "a.pth", "--stitch"])
        assert got["env"]["inference"] == ("a.pth",)
        assert got["run"]["stitch"] is True


@pytest.mark.parametrize(
    ("flag", "key", "where"),
    [
        ("--segmentation", "segmentation", "run"),
        ("--stitch", "stitch", "run"),
        ("--benchmark", "benchmark", "run"),
    ],
)
def test_boolean_optional_action_pairs(parsed, flag, key, where) -> None:
    """BooleanOptionalAction became --flag/--no-flag; both sides work."""
    assert parsed(["1234", flag])[where][key] is True
    assert parsed(["1234", f"--no-{flag[2:]}"])[where][key] is False


def test_delete_is_a_plain_flag(parsed) -> None:
    assert parsed(["1234", "--delete"])["run"]["delete"] is True


# --------------------------------------------------------------------------
# Validation and overrides
# --------------------------------------------------------------------------


def test_representative_override_contract(parsed) -> None:
    """Capture variadic, tri-state, optional-value and logging together."""
    got = parsed(
        [
            "1234",
            "--inference",
            "model_a.pth",
            "model_b.pth",
            "--gallery",
            "4",
            "--batch",
            "8",
            "--stitch",
            "--no-stream-stitch",
            "--track",
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
            "-v",
        ]
    )
    assert got["env"] == {
        "env": None,
        "config": None,
        "stitch_config": None,
        "inference": ("model_a.pth", "model_b.pth"),
        "gallery": 4,
        "batch": 8,
        "track": TRACK_DEFAULT_MODEL,
        "track_mode": "ilp",
        "track_batch_size": 2,
        "track_device": "cpu",
        "track_window": 6,
        "stream_stitch": False,
        "verbose": True,
        "log_level": "DEBUG",
        "log_file": "none",
    }
    assert got["run"]["stitch"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["1234", "--track-mode", "bogus"],
        ["1234", "--track-device", "tpu"],
        ["1234", "--gallery", "not-a-number"],
    ],
)
def test_invalid_choices_and_types_exit_two(runner, argv) -> None:
    assert runner.invoke(cli, argv).exit_code == 2


@pytest.mark.parametrize("flag", ["--config", "--stitch-config"])
def test_missing_config_file_fails_before_import(runner, flag) -> None:
    """A bad path must fail loudly rather than fall back to defaults."""
    result = runner.invoke(cli, ["1234", flag, "/definitely/missing.json"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_existing_config_file_is_accepted(parsed, tmp_path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    got = parsed(["1234", "--config", str(cfg)])
    assert got["env"]["config"] == str(cfg)


# --------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------


def test_help_exits_zero_and_lists_the_hard_options(runner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for opt in ("--track", "--stream-stitch", "--inference", "--stitch"):
        assert opt in result.output


def test_help_does_not_import_the_pipeline() -> None:
    """Great Docs discovery and --help must stay cheap."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "import bin.run_omero_screen;"
        "heavy=[m for m in ('torch','cellpose','omero','napari')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip() == "", (
        f"heavy imports at module scope: {out.stdout}"
    )
