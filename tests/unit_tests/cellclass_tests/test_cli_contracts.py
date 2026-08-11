"""Compatibility contracts for the unified CellClass Click command suite."""

from __future__ import annotations

import getpass
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import click
from cellclass.bin.run_training import Existing, LossFunction, LrScheduler
from cellclass.cli import (
    batch,
    dataset,
    extract,
    sbatch,
    train,
)
from cellclass.cli import (
    test_command as model_test_command,
)
from cellclass.models import Model
from click.testing import CliRunner


def _file(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.touch()
    return str(path)


def _run_args(
    command: click.Command, argv: list[str], target: str
) -> Namespace:
    with patch(target) as run:
        result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    return run.call_args.args[0]  # type: ignore[no-any-return]


def _sbatch_args(argv: list[str]) -> Namespace:
    completed = Mock(stdout="")
    with (
        patch(
            "cellclass.bin.sbatch_training.create_job_script",
            return_value="training.sh",
        ) as create_job_script,
        patch("subprocess.run", return_value=completed),
    ):
        result = CliRunner().invoke(sbatch, argv)
    assert result.exit_code == 0, result.output
    return create_job_script.call_args.args[0]  # type: ignore[no-any-return]


def test_train_default_contract(tmp_path: Path) -> None:
    """Freeze all training defaults and destination names passed to the app."""
    input_file = _file(tmp_path, "rois.npz")
    args = _run_args(train, [input_file], "cellclass.bin.run_training.run")

    assert vars(args) == {
        "input": input_file,
        "size": 0,
        "data_seed": 42,
        "device": "cuda",
        "state": "training.json",
        "model": Model.densenet121,
        "name": "model.pt",
        "existing": Existing.error,
        "weights": "DEFAULT",
        "freeze_weights": False,
        "epochs": 2000,
        "batch_size": 32,
        "num_workers": 0,
        "seed": 0xDEADBEEF,
        "validation_size": 0.2,
        "learning_rate": 1e-4,
        "lr_scheduler": LrScheduler.step,
        "lr_gamma": 0.1,
        "lr_step_size": 5,
        "lr_plat_factor": 0.1,
        "lr_plat_patience": 2,
        "patience": 10,
        "delta": 0,
        "rel_delta": 1e-4,
        "weight_decay": 1e-4,
        "flip": 1,
        "rotate": 180,
        "translate": 0.1,
        "loss_function": LossFunction.focal_loss,
        "focal_gamma": 2,
        "loss_weights": False,
        "dropout": 0.4,
        "testing_size": 0.2,
        "testing_interval": 5,
        "log_level": 20,
        "cudnn_benchmark": False,
        "pin_memory": False,
        "wandb": False,
        "entity": None,
        "project": "cellclass",
        "run_name": None,
        "tags": [],
    }


def test_train_representative_override_contract(tmp_path: Path) -> None:
    """Capture enum, boolean-pair, and variadic training overrides."""
    input_file = _file(tmp_path, "rois.npz")
    args = _run_args(
        train,
        [
            input_file,
            "--model",
            "efficientnetb3s",
            "--existing",
            "overwrite",
            "--no-freeze-weights",
            "--lr-scheduler",
            "plateau",
            "--loss-function",
            "cross_entropy",
            "--loss-weights",
            "--cudnn-benchmark",
            "--pin-memory",
            "--wandb",
            "--tags",
            "screen",
            "mitosis",
        ],
        "cellclass.bin.run_training.run",
    )

    assert args.model is Model.efficientnetb3s
    assert args.existing is Existing.overwrite
    assert args.lr_scheduler is LrScheduler.plateau
    assert args.loss_function is LossFunction.cross_entropy
    assert args.tags == ["screen", "mitosis"]
    assert args.wandb is True


def test_test_model_default_contract(tmp_path: Path) -> None:
    """Freeze evaluation defaults and destination names passed to the app."""
    input_file = _file(tmp_path, "rois.npz")
    args = _run_args(
        model_test_command, [input_file], "cellclass.bin.test_model.run"
    )

    assert vars(args) == {
        "input": input_file,
        "size": 0,
        "data_seed": 42,
        "device": "cuda",
        "model": Model.densenet121,
        "name": None,
        "weights": None,
        "script": None,
        "testing_size": 0.2,
        "batch_size": 128,
        "num_workers": 0,
        "log_level": 20,
        "pin_memory": False,
        "reorder": None,
        "plot_matrix": False,
        "matrix_file": None,
        "matrix_csv": None,
    }


def test_dataset_contract(tmp_path: Path) -> None:
    """Capture dataset path, variadic, and boolean-pair parsing."""
    args = _run_args(
        dataset,
        [
            str(tmp_path),
            "--name",
            "screen",
            "--ignore",
            "unassigned",
            "blurred",
            "--channels",
            "DAPI",
            "Tub",
            "--duplicates",
            "--no-single-label",
        ],
        "cellclass.bin.create_dataset.run",
    )

    assert vars(args) == {
        "dir": str(tmp_path),
        "name": "screen",
        "out": None,
        "ignore": ["unassigned", "blurred"],
        "duplicates": True,
        "channels": ["DAPI", "Tub"],
        "single_label": False,
    }


def test_extract_contract(tmp_path: Path) -> None:
    """Capture model-extraction paths, enums, and variadic values."""
    settings = _file(tmp_path, "training.json")
    args = _run_args(
        extract,
        [
            settings,
            "--name",
            "mitosis",
            "--save",
            "--overwrite",
            "--model",
            "densenet121",
            "--channels",
            "DAPI",
            "Tub",
            "--labels",
            "interphase",
            "mitosis",
        ],
        "cellclass.bin.extract_model.run",
    )

    assert args.file == settings
    assert args.save is True
    assert args.overwrite is True
    assert args.model is Model.densenet121
    assert args.channels == ["DAPI", "Tub"]
    assert args.labels == ["interphase", "mitosis"]


def test_batch_contract(tmp_path: Path) -> None:
    """Capture batch-runner defaults and boolean options."""
    batch_file = _file(tmp_path, "train.txt")
    args = _run_args(
        batch,
        [batch_file, "--dry-run", "--background", "--script", "jobs.sh"],
        "cellclass.bin.batch_training.run",
    )

    assert args.batch == batch_file
    assert args.dry_run is True
    assert args.background is True
    assert args.script == "jobs.sh"
    assert Path(args.cmd).name == "cellclass-train"


def test_sbatch_default_and_remainder_contract() -> None:
    """Freeze SLURM defaults and the training-argument remainder boundary."""
    defaults = _sbatch_args([])
    assert vars(defaults) == {
        "args": None,
        "job_class": "gpu",
        "results_dir": "/mnt/lustre/users/gdsc/",
        "username": getpass.getuser(),
        "hours": 12,
        "memory": 32,
        "gpu": True,
        "exec": True,
        "submit": True,
    }

    forwarded = _sbatch_args(
        [
            "--hours",
            "24",
            "--no-gpu",
            "--no-exec",
            "--no-submit",
            "--args",
            "rois.npz",
            "--model",
            "efficientnetb3s",
        ]
    )
    assert forwarded.hours == 24
    assert forwarded.gpu is False
    assert forwarded.exec is False
    assert forwarded.submit is False
    assert forwarded.args == ["rois.npz", "--model", "efficientnetb3s"]
