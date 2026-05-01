# ------------------------------------------------------------------------------
# Tests for cellclass.bin.sbatch_training.create_job_script().
# ------------------------------------------------------------------------------

"""Tests for sbatch_training.create_job_script()."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from cellclass.bin.sbatch_training import create_job_script

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAINING_PROG = "run_training.py"


def _make_args(
    tmp_path: Path,
    gpu: bool = True,
    exec_flag: bool = False,
    submit: bool = False,
    extra_args: list[str] | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace for create_job_script.

    Args:
        tmp_path: Directory in which the stub run_training.py is created.
        gpu: Whether to include a GPU SBATCH directive.
        exec_flag: Whether to mark commands as executable (True) or commented (#).
        submit: Whether sbatch submission is requested.
        extra_args: Additional training arguments.

    Returns:
        Configured Namespace instance.

    """
    return argparse.Namespace(
        job_class="gpu",
        username="testuser",
        hours=12,
        memory=32,
        gpu=gpu,
        exec=exec_flag,
        submit=submit,
        args=extra_args or ["rois.npz", "--model", "efficientnetb3s"],
    )


@pytest.fixture(autouse=True)
def stub_training_prog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a stub run_training.py and chdir to tmp_path so create_job_script finds it.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    """
    prog = tmp_path / _TRAINING_PROG
    prog.touch()
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateJobScript:
    """Tests for sbatch_training.create_job_script()."""

    def test_script_file_is_created(self, tmp_path: Path) -> None:
        """create_job_script() must produce a .sh file on disk."""
        script = create_job_script(_make_args(tmp_path))
        assert Path(script).exists()

    def test_script_contains_sbatch_directives(self, tmp_path: Path) -> None:
        """Generated script must contain at least one #SBATCH line."""
        script = create_job_script(_make_args(tmp_path))
        content = Path(script).read_text()
        assert "#SBATCH" in content

    def test_script_contains_training_command(self, tmp_path: Path) -> None:
        """Generated script must invoke run_training.py via uv run."""
        script = create_job_script(_make_args(tmp_path))
        content = Path(script).read_text()
        assert _TRAINING_PROG in content
        assert "uv run" in content

    def test_gpu_flag_adds_gres_directive(self, tmp_path: Path) -> None:
        """--gpu=True must add a #SBATCH --gres=gpu line."""
        script = create_job_script(_make_args(tmp_path, gpu=True))
        content = Path(script).read_text()
        assert "--gres=gpu" in content

    def test_no_gpu_flag_omits_gres_directive(self, tmp_path: Path) -> None:
        """--gpu=False must not include an active #SBATCH --gres=gpu line."""
        script = create_job_script(_make_args(tmp_path, gpu=False))
        content = Path(script).read_text()
        # Line must not be present as an active SBATCH directive
        active_gpu = any(
            line.startswith("#SBATCH") and "--gres=gpu" in line
            for line in content.splitlines()
        )
        assert not active_gpu

    def test_script_contains_memory_directive(self, tmp_path: Path) -> None:
        """Generated script must declare memory via #SBATCH --mem."""
        script = create_job_script(_make_args(tmp_path))
        content = Path(script).read_text()
        assert "--mem=" in content

    def test_script_contains_time_directive(self, tmp_path: Path) -> None:
        """Generated script must declare wall-time via #SBATCH --time."""
        script = create_job_script(_make_args(tmp_path))
        content = Path(script).read_text()
        assert "--time=" in content

    def test_script_contains_training_arguments(self, tmp_path: Path) -> None:
        """Training arguments passed via args.args must appear in the script."""
        script = create_job_script(
            _make_args(tmp_path, extra_args=["rois.npz", "--model", "densenet121"])
        )
        content = Path(script).read_text()
        assert "densenet121" in content
