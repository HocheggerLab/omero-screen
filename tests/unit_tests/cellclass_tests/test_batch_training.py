# ------------------------------------------------------------------------------
# Tests for cellclass.bin.batch_training.run() script-generation path.
# ------------------------------------------------------------------------------

"""Tests for batch_training.run() in script-generation mode (no background)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cellclass.bin.batch_training import run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(batch: Path, script: Path, dry_run: bool = False) -> argparse.Namespace:
    """Build an argparse.Namespace for batch_training.run() with --background=False.

    Args:
        batch: Path to the batch arguments file.
        script: Path where the generated shell script should be written.
        dry_run: If True, perform a dry run (no script written).

    Returns:
        Configured Namespace instance.

    """
    return argparse.Namespace(
        batch=str(batch),
        script=str(script),
        cmd="run_training.py",
        background=False,
        dry_run=dry_run,
    )


def _write_batch(path: Path, content: str) -> None:
    """Write content to a batch text file.

    Args:
        path: Destination path.
        content: Text content for the file.

    """
    path.write_text(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchTrainingRun:
    """Tests for the script-generation code path of batch_training.run()."""

    def test_script_has_one_line_per_uncommented_input(
        self, tmp_path: Path, npz_file: Path
    ) -> None:
        """Two uncommented lines produce exactly two command lines in the script."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        _write_batch(
            batch,
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n"
            f"{npz_file} --model densenet121 --lr 1e-4\n",
        )
        run(_make_args(batch, script))
        lines = [l for l in script.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_comment_lines_are_skipped(self, tmp_path: Path, npz_file: Path) -> None:
        """Lines starting with '#' must be excluded from the generated script."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        _write_batch(
            batch,
            f"# comment line\n"
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n",
        )
        run(_make_args(batch, script))
        lines = [l for l in script.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        # The single command line must start with the training program, not a comment marker
        assert lines[0].startswith("run_training.py")

    def test_blank_lines_are_skipped(self, tmp_path: Path, npz_file: Path) -> None:
        """Blank lines must not produce empty commands in the generated script."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        _write_batch(
            batch,
            f"\n{npz_file} --model efficientnetb3s --lr 1e-3\n\n",
        )
        run(_make_args(batch, script))
        lines = [l for l in script.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_wandb_flag_in_every_command(
        self, tmp_path: Path, npz_file: Path
    ) -> None:
        """Every generated command must include the --wandb flag."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        _write_batch(
            batch,
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n"
            f"{npz_file} --model densenet121 --lr 1e-4\n",
        )
        run(_make_args(batch, script))
        for line in script.read_text().splitlines():
            if line.strip():
                assert "--wandb" in line

    def test_checkpoint_and_state_flags_in_every_command(
        self, tmp_path: Path, npz_file: Path
    ) -> None:
        """Every generated command must include -n (checkpoint) and -s (state) flags."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        _write_batch(
            batch,
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n"
            f"{npz_file} --model densenet121 --lr 1e-4\n",
        )
        run(_make_args(batch, script))
        for line in script.read_text().splitlines():
            if line.strip():
                assert " -n " in line
                assert " -s " in line

    def test_duplicate_dataset_produces_unique_checkpoint_names(
        self, tmp_path: Path, npz_file: Path
    ) -> None:
        """The same dataset path on two lines must generate unique checkpoint names."""
        batch = tmp_path / "train.txt"
        script = tmp_path / "batch.sh"
        # Same npz path twice → should disambiguate with .1 / .2
        _write_batch(
            batch,
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n"
            f"{npz_file} --model efficientnetb3s --lr 1e-3\n",
        )
        run(_make_args(batch, script))
        lines = [l for l in script.read_text().splitlines() if l.strip()]
        # Extract -n arguments
        checkpoints: list[str] = []
        for line in lines:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "-n" and i + 1 < len(parts):
                    checkpoints.append(parts[i + 1])
        assert len(checkpoints) == 2
        assert checkpoints[0] != checkpoints[1]
