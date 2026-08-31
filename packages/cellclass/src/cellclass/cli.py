"""Click command-line interface for CellClass."""

from __future__ import annotations

import getpass
import pathlib
import sys
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import click
from click.core import ParameterSource

from cellclass.options import Existing, LossFunction, LrScheduler, Model

_MODEL_CHOICES = tuple(model.value for model in Model)

_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str)
_DIRECTORY = click.Path(
    exists=True, file_okay=False, dir_okay=True, path_type=str
)


class VariadicOptionCommand(click.Command):
    """Preserve argparse-style ``--option VALUE [VALUE ...]`` options."""

    def __init__(
        self,
        *args: Any,
        variadic_options: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialise a command with options that consume values until the next flag."""
        self._variadic_options = frozenset(variadic_options)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Expand variadic values into Click's repeatable-option representation."""
        expanded: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token not in self._variadic_options:
                expanded.append(token)
                index += 1
                continue

            index += 1
            if index == len(args) or args[index].startswith("-"):
                expanded.append(token)
                continue
            while index < len(args) and not args[index].startswith("-"):
                expanded.extend((token, args[index]))
                index += 1
        return super().parse_args(ctx, expanded)


class RemainderCommand(click.Command):
    """Preserve ``--args`` as the boundary for forwarded training arguments."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Translate the legacy remainder marker to Click's standard separator."""
        if "--args" in args:
            args = list(args)
            args[args.index("--args")] = "--"
        return super().parse_args(ctx, args)


def _none_or_str(value: str | None) -> str | None:
    """Convert the literal string ``None`` to None."""
    return None if value == "None" else value


def _explicit(ctx: click.Context, name: str) -> bool:
    """Return whether an option was supplied on the command line."""
    return ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE


@click.group(name="cellclass")
def cli() -> None:
    """Train and evaluate cell-image classifiers."""


@cli.command(name="sample")
@click.argument("dataset", type=_FILE)
@click.option(
    "--output",
    type=_DIRECTORY,
    help="Output path (default: same directory as dataset).",
)
@click.option("--samples", type=int, default=10, show_default=True)
@click.option(
    "--crop",
    type=int,
    default=0,
    help="Crop NxN dimension (default: full image).",
)
def sample(dataset: str, output: str | None, samples: int, crop: int) -> None:
    """Sample example images from DATASET."""
    from cellclass.bin.sample_images import run

    run(dataset=dataset, output=output, samples=samples, crop=crop)


@cli.command(
    name="dataset",
    cls=VariadicOptionCommand,
    variadic_options=("--ignore", "--channels"),
)
@click.argument("directory", metavar="DIR", type=_DIRECTORY)
@click.option(
    "--name", default="rois", show_default=True, help="Output dataset name."
)
@click.option("--out", type=_DIRECTORY, help="Output dataset directory.")
@click.option(
    "--ignore",
    multiple=True,
    default=("unassigned",),
    help="Labels to ignore. Accepts one or more values.",
)
@click.option(
    "--duplicates/--no-duplicates", default=False, help="Log duplicates."
)
@click.option(
    "--channels",
    multiple=True,
    help="Channel names. Accepts one or more values.",
)
@click.option(
    "--single-label/--no-single-label",
    default=True,
    help="Only use single-label masks.",
)
def dataset(
    directory: str,
    name: str,
    out: str | None,
    ignore: tuple[str, ...],
    duplicates: bool,
    channels: tuple[str, ...],
    single_label: bool,
) -> None:
    """Convert cell images in DIR to a training dataset."""
    from cellclass.bin.create_dataset import run

    run(
        Namespace(
            dir=directory,
            name=name,
            out=out,
            ignore=list(ignore),
            duplicates=duplicates,
            channels=list(channels) or None,
            single_label=single_label,
        )
    )


@cli.command(
    name="extract",
    cls=VariadicOptionCommand,
    variadic_options=("--channels", "--labels"),
)
@click.argument("file", type=_FILE)
@click.option(
    "--name", help="Model file prefix (default uses model metadata)."
)
@click.option("--save/--no-save", default=False, help="Save model files.")
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing model files.",
)
@click.option(
    "--model", type=click.Choice(_MODEL_CHOICES), help="Model override."
)
@click.option("--channels", multiple=True, help="Input channel overrides.")
@click.option("--labels", multiple=True, help="Class label overrides.")
def extract(
    file: str,
    name: str | None,
    save: bool,
    overwrite: bool,
    model: str | None,
    channels: tuple[str, ...],
    labels: tuple[str, ...],
) -> None:
    """Extract a TorchScript model from training settings FILE."""
    from cellclass.bin.extract_model import run

    run(
        Namespace(
            file=file,
            name=name,
            save=save,
            overwrite=overwrite,
            model=Model(model) if model else None,
            channels=list(channels) or None,
            labels=list(labels) or None,
        )
    )


@cli.command(name="batch")
@click.argument("batch_file", metavar="BATCH", type=_FILE)
@click.option(
    "--dry-run/--no-dry-run", default=False, help="Perform a dry run."
)
@click.option(
    "--background/--no-background",
    default=False,
    help="Run each training process in the background.",
)
@click.option("--script", default="batch.sh", show_default=True)
@click.option(
    "--cmd",
    default=str(pathlib.Path(sys.executable).parent / "cellclass-train"),
    help="Training program.",
)
def batch(
    batch_file: str,
    dry_run: bool,
    background: bool,
    script: str,
    cmd: str,
) -> None:
    """Generate training runs from BATCH."""
    from cellclass.bin.batch_training import run

    run(
        Namespace(
            batch=batch_file,
            dry_run=dry_run,
            background=background,
            script=script,
            cmd=cmd,
        )
    )


@cli.command(
    name="test",
    cls=VariadicOptionCommand,
    variadic_options=("--reorder",),
)
@click.argument("input_file", metavar="dataset.npz", type=_FILE)
@click.option("--size", type=int, default=0, help="Number of images to use.")
@click.option("--data-seed", type=int, default=42, show_default=True)
@click.option("-d", "--device", default="cuda", show_default=True)
@click.option(
    "--model", type=click.Choice(_MODEL_CHOICES), default="densenet121"
)
@click.option("-n", "--name", type=_FILE, help="Training checkpoint name.")
@click.option("-w", "--weights", type=_FILE, help="Model weights.")
@click.option("-s", "--script", type=_FILE, help="Model script.")
@click.option("--testing-size", type=float, default=0.2, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--log-level", type=int, default=20, show_default=True)
@click.option("--pin-memory/--no-pin-memory", default=False)
@click.option(
    "--reorder", type=int, multiple=True, help="Reorder class labels."
)
@click.option("--plot-matrix/--no-plot-matrix", default=False)
@click.option("--matrix-file", help="Save the confusion matrix image.")
@click.option("--matrix-csv", help="Save the confusion matrix CSV.")
def test_command(
    input_file: str,
    size: int,
    data_seed: int,
    device: str,
    model: str,
    name: str | None,
    weights: str | None,
    script: str | None,
    testing_size: float,
    batch_size: int,
    num_workers: int,
    log_level: int,
    pin_memory: bool,
    reorder: tuple[int, ...],
    plot_matrix: bool,
    matrix_file: str | None,
    matrix_csv: str | None,
) -> None:
    """Evaluate a trained model against a dataset."""
    from cellclass.bin.test_model import run

    run(
        Namespace(
            input=input_file,
            size=size,
            data_seed=data_seed,
            device=device,
            model=Model(model),
            name=name,
            weights=weights,
            script=script,
            testing_size=testing_size,
            batch_size=batch_size,
            num_workers=num_workers,
            log_level=log_level,
            pin_memory=pin_memory,
            reorder=list(reorder) or None,
            plot_matrix=plot_matrix,
            matrix_file=matrix_file,
            matrix_csv=matrix_csv,
        )
    )


@cli.command(
    name="sbatch",
    cls=RemainderCommand,
    context_settings={"ignore_unknown_options": True},
)
@click.option("--class", "job_class", default="gpu", show_default=True)
@click.option(
    "--results-dir", default="/mnt/lustre/users/gdsc/", show_default=True
)
@click.option("-u", "--username", default=getpass.getuser(), show_default=True)
@click.option("--hours", type=int, default=12, show_default=True)
@click.option("-m", "--memory", type=int, default=32, show_default=True)
@click.option("--no-gpu", is_flag=True, default=False)
@click.option("--no-exec", is_flag=True, default=False)
@click.option("--no-submit", is_flag=True, default=False)
@click.argument("training_args", nargs=-1, type=click.UNPROCESSED)
def sbatch(
    job_class: str,
    results_dir: str,
    username: str,
    hours: int,
    memory: int,
    no_gpu: bool,
    no_exec: bool,
    no_submit: bool,
    training_args: tuple[str, ...],
) -> None:
    """Submit a CellClass training job to SLURM.

    Place arguments forwarded to training after ``--args``.
    """
    import subprocess

    from cellclass.bin.sbatch_training import create_job_script

    args = Namespace(
        args=list(training_args) if training_args else None,
        job_class=job_class,
        results_dir=results_dir,
        username=username,
        hours=hours,
        memory=memory,
        gpu=not no_gpu,
        exec=not no_exec,
        submit=not no_submit,
    )
    script = create_job_script(args)
    if args.submit:
        click.echo(
            subprocess.run(
                ["sbatch", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout
        )


@cli.command(
    name="train",
    cls=VariadicOptionCommand,
    variadic_options=("--tags",),
)
@click.argument("input_file", metavar="dataset.npz | state.json", type=_FILE)
@click.option("--size", type=int, default=0, help="Number of images to use.")
@click.option("--data-seed", type=int, default=42, show_default=True)
@click.option("-d", "--device", default="cuda", show_default=True)
@click.option("-s", "--state", default="training.json", show_default=True)
@click.option(
    "--model", type=click.Choice(_MODEL_CHOICES), default="densenet121"
)
@click.option("-n", "--name", default="model.pt", show_default=True)
@click.option(
    "--existing",
    type=click.Choice(("overwrite", "load", "error")),
    default="error",
)
@click.option("--weights", default="DEFAULT", show_default=True)
@click.option("--freeze-weights/--no-freeze-weights", default=False)
@click.option("--epochs", type=int, default=2000, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=0xDEADBEEF, show_default=True)
@click.option("--validation-size", type=float, default=0.2, show_default=True)
@click.option(
    "--lr", "learning_rate", type=float, default=1e-4, show_default=True
)
@click.option(
    "--lr-scheduler", type=click.Choice(("step", "plateau")), default="step"
)
@click.option("--lr-gamma", type=float, default=0.1, show_default=True)
@click.option(
    "--lr-step", "lr_step_size", type=int, default=5, show_default=True
)
@click.option(
    "--lr-factor", "lr_plat_factor", type=float, default=0.1, show_default=True
)
@click.option(
    "--lr-patience", "lr_plat_patience", type=int, default=2, show_default=True
)
@click.option("--patience", type=int, default=10, show_default=True)
@click.option("--delta", type=float, default=0.0, show_default=True)
@click.option("--rel-delta", type=float, default=1e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--flip", type=int, default=1, show_default=True)
@click.option("--rotate", type=int, default=180, show_default=True)
@click.option("--translate", type=float, default=0.1, show_default=True)
@click.option(
    "--loss-function",
    type=click.Choice(("focal_loss", "cross_entropy")),
    default="focal_loss",
)
@click.option("--focal-gamma", type=float, default=2.0, show_default=True)
@click.option("--loss-weights/--no-loss-weights", default=False)
@click.option("--dropout", type=float, default=0.4, show_default=True)
@click.option("--testing-size", type=float, default=0.2, show_default=True)
@click.option("--testing-interval", type=int, default=5, show_default=True)
@click.option("--log-level", type=int, default=20, show_default=True)
@click.option("--cudnn-benchmark/--no-cudnn-benchmark", default=False)
@click.option("--pin-memory/--no-pin-memory", default=False)
@click.option("--wandb/--no-wandb", default=False)
@click.option("--entity", help="Weights and Biases team/entity.")
@click.option("--project", default="cellclass", show_default=True)
@click.option("--run-name", help="Weights and Biases run display name.")
@click.option("--tags", multiple=True, help="Weights and Biases tags.")
@click.option("--wandb-id", help="Weights and Biases run ID.")
@click.pass_context
def train(
    ctx: click.Context,
    input_file: str,
    size: int,
    data_seed: int,
    device: str,
    state: str,
    model: str,
    name: str,
    existing: str,
    weights: str,
    freeze_weights: bool,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    validation_size: float,
    learning_rate: float,
    lr_scheduler: str,
    lr_gamma: float,
    lr_step_size: int,
    lr_plat_factor: float,
    lr_plat_patience: int,
    patience: int,
    delta: float,
    rel_delta: float,
    weight_decay: float,
    flip: int,
    rotate: int,
    translate: float,
    loss_function: str,
    focal_gamma: float,
    loss_weights: bool,
    dropout: float,
    testing_size: float,
    testing_interval: int,
    log_level: int,
    cudnn_benchmark: bool,
    pin_memory: bool,
    wandb: bool,
    entity: str | None,
    project: str,
    run_name: str | None,
    tags: tuple[str, ...],
    wandb_id: str | None,
) -> None:
    """Train a model using an ROI dataset or resume from saved state."""
    import json

    from cellclass.bin.run_training import run

    args = Namespace(
        input=input_file,
        size=size,
        data_seed=data_seed,
        device=device,
        state=state,
        model=Model(model),
        name=name,
        existing=Existing(existing),
        weights=_none_or_str(weights),
        freeze_weights=freeze_weights,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        validation_size=validation_size,
        learning_rate=learning_rate,
        lr_scheduler=LrScheduler(lr_scheduler),
        lr_gamma=lr_gamma,
        lr_step_size=lr_step_size,
        lr_plat_factor=lr_plat_factor,
        lr_plat_patience=lr_plat_patience,
        patience=patience,
        delta=delta,
        rel_delta=rel_delta,
        weight_decay=weight_decay,
        flip=flip,
        rotate=rotate,
        translate=translate,
        loss_function=LossFunction(loss_function),
        focal_gamma=focal_gamma,
        loss_weights=loss_weights,
        dropout=dropout,
        testing_size=testing_size,
        testing_interval=testing_interval,
        log_level=log_level,
        cudnn_benchmark=cudnn_benchmark,
        pin_memory=pin_memory,
        wandb=wandb,
        entity=_none_or_str(entity),
        project=project,
        run_name=run_name,
        tags=list(tags),
        wandb_id=wandb_id,
    )

    if input_file.endswith(".json"):
        with open(input_file) as stream:
            saved_state = json.load(stream)
        saved_state["existing"] = Existing.load
        saved_state["model"] = Model(saved_state["model"])
        explicit_names = (
            "log_level",
            "epochs",
            "wandb",
            "device",
            "patience",
            "delta",
            "rel_delta",
            "testing_size",
            "testing_interval",
        )
        overrides = {
            option: getattr(args, option)
            for option in explicit_names
            if _explicit(ctx, option)
        }
        vars(args).update(saved_state)
        vars(args).update(overrides)
        args.restart = True

    if args.wandb_id is None:
        del args.wandb_id
    run(args)
