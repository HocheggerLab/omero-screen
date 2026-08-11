"""Behaviour tests for the unified CellClass Click interface."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from cellclass.cli import (
    batch,
    cli,
    dataset,
    extract,
    sample,
    sbatch,
    train,
)
from cellclass.cli import (
    test_command as model_test_command,
)
from click.testing import CliRunner
from pytest import MonkeyPatch


def test_unified_sample_command_calls_application(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The new grouped command passes typed values to the application."""
    dataset = tmp_path / "rois.npz"
    dataset.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.sample_images.run", run)

    result = CliRunner().invoke(
        cli,
        [
            "sample",
            str(dataset),
            "--output",
            str(tmp_path),
            "--samples",
            "25",
            "--crop",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        dataset=str(dataset), output=str(tmp_path), samples=25, crop=64
    )


def test_legacy_sample_command_uses_same_click_command(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The legacy console script can invoke the shared command object directly."""
    dataset = tmp_path / "rois.npz"
    dataset.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.sample_images.run", run)

    result = CliRunner().invoke(sample, [str(dataset)])

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        dataset=str(dataset), output=None, samples=10, crop=0
    )


def test_sample_command_writes_images(tmp_path: Path) -> None:
    """The Click boundary executes the sampling workflow end to end."""
    dataset = tmp_path / "rois.npz"
    np.savez(
        dataset,
        X=np.ones((2, 1, 4, 4), dtype=np.uint8),
        y_names=np.array(["interphase", "mitosis"]),
    )

    result = CliRunner().invoke(cli, ["sample", str(dataset)])

    assert result.exit_code == 0, result.output
    assert "interphase = 1" in result.output
    assert "mitosis = 1" in result.output
    assert (tmp_path / "interphase.tif").is_file()
    assert (tmp_path / "mitosis.tif").is_file()


def test_sample_rejects_invalid_values(tmp_path: Path) -> None:
    """Path and integer failures remain usage errors."""
    runner = CliRunner()

    missing = runner.invoke(cli, ["sample", str(tmp_path / "missing.npz")])
    invalid_integer = runner.invoke(
        cli, ["sample", __file__, "--samples", "not-an-integer"]
    )

    assert missing.exit_code == 2
    assert "does not exist" in missing.output
    assert invalid_integer.exit_code == 2
    assert "not a valid integer" in invalid_integer.output


def test_dataset_preserves_variadic_and_boolean_contract(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Dataset keeps argparse-style multi-value options and boolean pairs."""
    run = Mock()
    monkeypatch.setattr("cellclass.bin.create_dataset.run", run)

    result = CliRunner().invoke(
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
    )

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert vars(args) == {
        "dir": str(tmp_path),
        "name": "screen",
        "out": None,
        "ignore": ["unassigned", "blurred"],
        "duplicates": True,
        "channels": ["DAPI", "Tub"],
        "single_label": False,
    }


def test_extract_preserves_enum_and_variadic_contract(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Extract converts model choices and preserves multi-value overrides."""
    settings = tmp_path / "training.json"
    settings.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.extract_model.run", run)

    result = CliRunner().invoke(
        extract,
        [
            str(settings),
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
    )

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.model.value == "densenet121"
    assert args.channels == ["DAPI", "Tub"]
    assert args.labels == ["interphase", "mitosis"]
    assert args.save is True
    assert args.overwrite is True


def test_batch_preserves_defaults(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Batch passes legacy defaults through the Click boundary."""
    batch_file = tmp_path / "train.txt"
    batch_file.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.batch_training.run", run)

    result = CliRunner().invoke(batch, [str(batch_file), "--dry-run"])

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.batch == str(batch_file)
    assert args.dry_run is True
    assert args.background is False
    assert args.script == "batch.sh"
    assert Path(args.cmd).name == "cellclass-train"


def test_model_test_preserves_options(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Model evaluation keeps aliases, defaults, and variadic label ordering."""
    input_file = tmp_path / "rois.npz"
    input_file.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.test_model.run", run)

    result = CliRunner().invoke(
        model_test_command,
        [
            str(input_file),
            "-d",
            "cpu",
            "--model",
            "efficientnetb3s",
            "--reorder",
            "1",
            "2",
            "0",
            "--pin-memory",
            "--plot-matrix",
        ],
    )

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.device == "cpu"
    assert args.model.value == "efficientnetb3s"
    assert args.reorder == [1, 2, 0]
    assert args.pin_memory is True
    assert args.plot_matrix is True


def test_sbatch_preserves_remainder_boundary(monkeypatch: MonkeyPatch) -> None:
    """Everything after --args remains ordered training input."""
    create_job_script = Mock(return_value="training.sh")
    monkeypatch.setattr(
        "cellclass.bin.sbatch_training.create_job_script", create_job_script
    )

    result = CliRunner().invoke(
        sbatch,
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
        ],
    )

    assert result.exit_code == 0, result.output
    args = create_job_script.call_args.args[0]
    assert args.hours == 24
    assert args.gpu is False
    assert args.exec is False
    assert args.submit is False
    assert args.args == ["rois.npz", "--model", "efficientnetb3s"]


def test_train_preserves_defaults_and_variadic_tags(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Training converts enums and retains all representative options."""
    input_file = tmp_path / "rois.npz"
    input_file.touch()
    run = Mock()
    monkeypatch.setattr("cellclass.bin.run_training.run", run)

    result = CliRunner().invoke(
        train,
        [
            str(input_file),
            "--model",
            "efficientnetb3s",
            "--existing",
            "overwrite",
            "--lr-scheduler",
            "plateau",
            "--loss-function",
            "cross_entropy",
            "--loss-weights",
            "--tags",
            "screen",
            "mitosis",
        ],
    )

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.model.value == "efficientnetb3s"
    assert args.existing.value == "overwrite"
    assert args.lr_scheduler.value == "plateau"
    assert args.loss_function.value == "cross_entropy"
    assert args.tags == ["screen", "mitosis"]
    assert args.epochs == 2000
    assert not hasattr(args, "wandb_id")


def test_train_resume_uses_click_parameter_sources(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Only explicitly supplied continuation options override saved state."""
    state = tmp_path / "training.json"
    state.write_text(
        '{"model": "densenet121", "epochs": 7, "device": "cpu", '
        '"wandb": false, "wandb_id": "saved-id"}'
    )
    run = Mock()
    monkeypatch.setattr("cellclass.bin.run_training.run", run)

    result = CliRunner().invoke(train, [str(state), "--epochs", "9"])

    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.epochs == 9
    assert args.device == "cpu"
    assert args.wandb_id == "saved-id"
    assert args.restart is True
    assert args.existing.value == "load"


def test_all_legacy_aliases_are_shared_click_commands() -> None:
    """Every legacy entry point targets the canonical command object."""
    pyproject = (
        Path(__file__).parents[3] / "packages" / "cellclass" / "pyproject.toml"
    )
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]

    assert scripts == {
        "cellclass": "cellclass.cli:cli",
        "cellclass-train": "cellclass.cli:train",
        "cellclass-test": "cellclass.cli:test_command",
        "cellclass-dataset": "cellclass.cli:dataset",
        "cellclass-extract": "cellclass.cli:extract",
        "cellclass-batch": "cellclass.cli:batch",
        "cellclass-sbatch": "cellclass.cli:sbatch",
        "cellclass-sample": "cellclass.cli:sample",
    }


def test_lazy_package_exports_remain_available() -> None:
    """The lightweight package root preserves the existing public imports."""
    from cellclass import Model

    assert Model.densenet121.value == "densenet121"


def test_help_import_does_not_load_torch() -> None:
    """CLI discovery stays independent of the Torch runtime stack."""
    env = os.environ.copy()
    src = str(Path(__file__).parents[3] / "packages" / "cellclass" / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [src, env.get("PYTHONPATH", "")] if part
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from click.testing import CliRunner; "
                "from cellclass.cli import cli; "
                "assert 'torch' not in sys.modules; "
                "commands = [[], ['dataset'], ['extract'], ['batch'], "
                "['test'], ['sbatch'], ['train'], ['sample']]; "
                "results = [CliRunner().invoke(cli, [*args, '--help']) "
                "for args in commands]; "
                "assert all(result.exit_code == 0 for result in results); "
                "assert 'torch' not in sys.modules; "
                "print(results[0].output)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "sample" in result.stdout
