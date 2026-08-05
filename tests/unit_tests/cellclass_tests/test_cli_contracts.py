"""Compatibility contracts for the pre-Click CellClass command suite."""

from __future__ import annotations

import getpass
from pathlib import Path

from cellclass.bin.batch_training import get_parser as batch_parser
from cellclass.bin.create_dataset import get_parser as dataset_parser
from cellclass.bin.extract_model import get_parser as extract_parser
from cellclass.bin.run_training import (
    Existing,
    LossFunction,
    LrScheduler,
)
from cellclass.bin.run_training import (
    get_parser as train_parser,
)
from cellclass.bin.sample_images import get_parser as sample_parser
from cellclass.bin.sbatch_training import get_parser as sbatch_parser
from cellclass.bin.test_model import get_parser as model_test_parser
from cellclass.models import Model


def _file(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.touch()
    return str(path)


def test_train_default_contract(tmp_path: Path) -> None:
    """Freeze all training defaults and argparse destination names."""
    dataset = _file(tmp_path, "rois.npz")
    args = train_parser().parse_args([dataset])

    assert vars(args) == {
        "input": dataset,
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
        "wandb_id": None,
    }


def test_train_representative_override_contract(tmp_path: Path) -> None:
    """Capture enum, boolean-pair, and variadic training overrides."""
    dataset = _file(tmp_path, "rois.npz")
    args = train_parser().parse_args(
        [
            dataset,
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
        ]
    )

    assert args.model is Model.efficientnetb3s
    assert args.existing is Existing.overwrite
    assert args.lr_scheduler is LrScheduler.plateau
    assert args.loss_function is LossFunction.cross_entropy
    assert args.tags == ["screen", "mitosis"]
    assert args.wandb is True


def test_test_model_default_contract(tmp_path: Path) -> None:
    """Freeze evaluation defaults and argparse destination names."""
    dataset = _file(tmp_path, "rois.npz")
    args = model_test_parser().parse_args([dataset])

    assert vars(args) == {
        "input": dataset,
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
    args = dataset_parser().parse_args(
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
        ]
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
    args = extract_parser().parse_args(
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
        ]
    )

    assert args.file == settings
    assert args.save is True
    assert args.overwrite is True
    assert args.model is Model.densenet121
    assert args.channels == ["DAPI", "Tub"]
    assert args.labels == ["interphase", "mitosis"]


def test_batch_contract(tmp_path: Path) -> None:
    """Capture batch-runner defaults and boolean options."""
    batch = _file(tmp_path, "train.txt")
    args = batch_parser().parse_args(
        [batch, "--dry-run", "--background", "--script", "jobs.sh"]
    )

    assert args.batch == batch
    assert args.dry_run is True
    assert args.background is True
    assert args.script == "jobs.sh"
    assert Path(args.cmd).name == "cellclass-train"


def test_sample_contract(tmp_path: Path) -> None:
    """Capture sample-export path and numeric options."""
    dataset = _file(tmp_path, "rois.npz")
    args = sample_parser().parse_args(
        [dataset, "--output", str(tmp_path), "--samples", "25", "--crop", "64"]
    )

    assert vars(args) == {
        "dataset": dataset,
        "output": str(tmp_path),
        "samples": 25,
        "crop": 64,
    }


def test_sbatch_default_and_remainder_contract() -> None:
    """Freeze SLURM defaults and the training-argument remainder boundary."""
    defaults = sbatch_parser().parse_args([])
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

    forwarded = sbatch_parser().parse_args(
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
