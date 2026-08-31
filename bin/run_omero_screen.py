#!/usr/bin/env python3
"""Command-line entry point for the OMERO-Screen analysis pipeline.

Running a plate is the whole point of the package: this module configures the
environment from the command line, then hands the plate IDs to
``omero_screen.loops.plate_loop``.

The exported :data:`cli` command is what Great Docs renders as CLI reference
and what ``CliRunner`` drives in tests. Every heavy import — OMERO, Torch,
Cellpose, the pipeline itself — happens inside the callback, after parsing and
after the environment variables have been set, because the package ``__init__``
reads several of them at import time.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

import click

_CP4_MODELS: dict[str, str] = {
    "nuclei": "cp4:cpsam",
    "RPE": "cp4:cpsam",
    "HELA": "cp4:cpsam",
    "U2OS": "cp4:cpsam",
    "HCC1143": "cp4:cpsam",
    "MM231": "cp4:cpsam",
    "PALB": "cp4:cpsam",
}

TRACK_DEFAULT_MODEL = "general_2d"


class ArgparseCompatCommand(click.Command):
    """Preserve two argparse spellings Click does not accept natively.

    ``--inference A B C``
        argparse's ``nargs="+"`` takes several values after one flag; Click's
        ``multiple=True`` wants the flag repeated. Values are expanded into
        the repeated form before parsing, so both spellings work.

    ``--track [MODEL]``
        argparse's ``nargs="?"`` with a ``const`` makes the value optional:
        a bare ``--track`` means ``general_2d``. Click 8.3 rejects a bare
        flag even when ``flag_value`` is set, so a bare occurrence is
        rewritten to its default value here.

    A cousin of this class lives in ``cellclass.cli`` and
    ``omero_screen_napari.trainingdata_db.cli``. The duplication is
    deliberate: ``omero_utils`` — the obvious shared home — calls
    ``set_env_vars()`` at import, which would drag OMERO configuration into
    every ``--help``.
    """

    def __init__(
        self,
        *args: Any,
        variadic_options: Sequence[str] = (),
        optional_value_options: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record which options need argparse-compatible rewriting."""
        self._variadic_options = frozenset(variadic_options)
        self._optional_value_options = dict(optional_value_options or {})
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Rewrite argparse-style tokens into Click's representation."""
        expanded: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]

            if token in self._variadic_options:
                index += 1
                if index == len(args) or args[index].startswith("-"):
                    expanded.append(token)
                    continue
                while index < len(args) and not args[index].startswith("-"):
                    expanded.extend((token, args[index]))
                    index += 1
                continue

            if token in self._optional_value_options:
                index += 1
                if index == len(args) or args[index].startswith("-"):
                    # Bare flag: supply the argparse `const`.
                    expanded.append(
                        f"{token}={self._optional_value_options[token]}"
                    )
                    continue
                expanded.append(f"{token}={args[index]}")
                index += 1
                continue

            expanded.append(token)
            index += 1
        return super().parse_args(ctx, expanded)


@click.command(
    cls=ArgparseCompatCommand,
    variadic_options=("--inference",),
    optional_value_options={"--track": TRACK_DEFAULT_MODEL},
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("ids", metavar="ID...", nargs=-1, required=True, type=int)
@click.option(
    "--env",
    default=None,
    help="Environment name (requires configuration file .env.{name}).",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    metavar="PATH",
    help=(
        "Path to an OMERO_SCREEN_CONFIG JSON (MODEL_DICT / FEATURELIST / "
        "CHANNEL_SEG_PROFILES). Overrides the OMERO_SCREEN_CONFIG env var. "
        "Errors if the path does not exist (no silent fallback to defaults)."
    ),
)
@click.option(
    "--inference",
    multiple=True,
    metavar="MODEL",
    help="Inference model filename(s). Accepts one or more values.",
)
@click.option(
    "--gallery",
    type=int,
    default=10,
    show_default=True,
    help="Width N for the inference NxN example gallery.",
)
@click.option(
    "--batch",
    type=int,
    default=16,
    show_default=True,
    help="Classification batch size.",
)
@click.option(
    "--segmentation/--no-segmentation",
    default=False,
    show_default=True,
    help="Only perform image segmentation.",
)
@click.option(
    "--delete",
    is_flag=True,
    default=False,
    help=(
        "Delete the plate's existing segmentation masks and segment from "
        "scratch. Without this, a re-run reuses the stored masks (both "
        "stitched and per-field) and only recomputes the measurements. "
        "Use it after changing segmentation settings, or to repair a "
        "plate whose stored masks are wrong or empty."
    ),
)
@click.option(
    "--cp4",
    is_flag=True,
    default=False,
    help=(
        "Use Cellpose 4 (cpsam) for segmentation instead of the default "
        "Cellpose 3 models."
    ),
)
@click.option(
    "--model",
    default=None,
    metavar="MODEL",
    help=(
        "Override all segmentation models with a single model name "
        "(e.g. 'cp4:cpsam', 'cp3:cyto3'). Overrides --cp4."
    ),
)
@click.option(
    "--benchmark/--no-benchmark",
    default=False,
    show_default=True,
    help="Record per-image timing data and write a JSON benchmark report.",
)
@click.option(
    "--stitch/--no-stitch",
    default=False,
    show_default=True,
    help=(
        "Run stitched-well segmentation: assemble all fields per well into "
        "one canvas, segment that canvas, and exclude border objects only at "
        "the outer edge."
    ),
)
@click.option(
    "--stream-stitch/--no-stream-stitch",
    default=None,
    help=(
        "Stitch one timepoint at a time to bound host RAM on long "
        "multi-channel timelapses (peak ~= canvas + one frame, not all "
        "fields + canvas; costs n_fields x T OMERO reads). Default: "
        "auto-enable when the estimated peak exceeds the host-RAM budget. "
        "Use --stream-stitch / --no-stream-stitch to force. Requires --stitch."
    ),
)
@click.option(
    "--stitch-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    metavar="PATH",
    help=(
        "Path to an OMERO_SCREEN_STITCH_CONFIG JSON configuration. "
        "Overrides the OMERO_SCREEN_STITCH_CONFIG env var. "
        "Errors if the path does not exist (no silent fallback to defaults)."
    ),
)
@click.option(
    "--track",
    is_flag=False,
    flag_value=TRACK_DEFAULT_MODEL,
    default=None,
    metavar="MODEL",
    help=(
        "Track nuclei across time with Trackastra. Optional MODEL is a "
        f"pretrained name or checkpoint path (default when flag given: "
        f"{TRACK_DEFAULT_MODEL}). Requires --stitch and a timelapse (T>1); "
        "a no-op on single-timepoint plates."
    ),
)
@click.option(
    "--track-mode",
    type=click.Choice(["greedy", "greedy_nodiv", "ilp"]),
    default="greedy",
    show_default=True,
    help="Trackastra linking mode.",
)
@click.option(
    "--track-batch-size",
    type=int,
    default=4,
    show_default=True,
    help=(
        "Attention windows Trackastra scores per forward pass. Caps GPU "
        "memory during tracking; lower this if tracking hits CUDA OOM, raise "
        "it for faster scoring when VRAM allows (Trackastra's own GPU "
        "default is 16)."
    ),
)
@click.option(
    "--track-device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help=(
        "Force the tracking device (default: auto-detect). Use 'cpu' when a "
        "dense well exceeds GPU VRAM — runs the identical computation in host "
        "RAM (slower, but no 44 GiB ceiling and no loss of accuracy)."
    ),
)
@click.option(
    "--track-window",
    type=int,
    default=None,
    help=(
        "Override Trackastra's temporal window (frames per attention "
        "window). Smaller cuts GPU memory ~quadratically at the cost of "
        "temporal context; default keeps the model's trained window."
    ),
)
@click.option(
    "--log-level",
    default=None,
    help=(
        "Log level (DEBUG/INFO/WARNING/ERROR). Overrides "
        "$OMERO_SCREEN_LOG_LEVEL; default INFO."
    ),
)
@click.option(
    "--log-file",
    default=None,
    help=(
        "Log file path, or 'none' to disable file logging. Overrides "
        "$OMERO_SCREEN_LOG_FILE; default logs/app.log."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help=(
        "Verbose debugging: DEBUG level, console logging on, and rich "
        "tracebacks. Console is off by default so logs don't clutter the "
        "progress display."
    ),
)
def cli(
    ids: tuple[int, ...],
    env: str | None,
    config: str | None,
    inference: tuple[str, ...],
    gallery: int,
    batch: int,
    segmentation: bool,
    delete: bool,
    cp4: bool,
    model: str | None,
    benchmark: bool,
    stitch: bool,
    stream_stitch: bool | None,
    stitch_config: str | None,
    track: str | None,
    track_mode: str,
    track_batch_size: int,
    track_device: str | None,
    track_window: int | None,
    log_level: str | None,
    log_file: str | None,
    verbose: bool,
) -> None:
    """Run the OMERO-Screen analysis pipeline on one or more plates.

    ID... are OMERO plate IDs. Each plate is segmented, measured and, where
    an EdU channel is present, assigned cell-cycle phases; results are
    attached back to the plate in OMERO.
    """
    _apply_environment(
        env=env,
        config=config,
        stitch_config=stitch_config,
        inference=inference,
        gallery=gallery,
        batch=batch,
        track=track,
        track_mode=track_mode,
        track_batch_size=track_batch_size,
        track_device=track_device,
        track_window=track_window,
        stream_stitch=stream_stitch,
        verbose=verbose,
        log_level=log_level,
        log_file=log_file,
    )

    if model or cp4:
        from omero_screen import default_config

        model_name = model if model else "cp4:cpsam"
        default_config.MODEL_DICT = {
            k: model_name for k in default_config.MODEL_DICT
        }

    _run(
        ids=list(ids),
        segmentation=segmentation,
        stitch=stitch,
        delete=delete,
        benchmark=benchmark,
    )


def _apply_environment(
    *,
    env: str | None,
    config: str | None,
    stitch_config: str | None,
    inference: tuple[str, ...],
    gallery: int,
    batch: int,
    track: str | None,
    track_mode: str,
    track_batch_size: int,
    track_device: str | None,
    track_window: int | None,
    stream_stitch: bool | None,
    verbose: bool,
    log_level: str | None,
    log_file: str | None,
) -> None:
    """Translate the parsed options into environment variables.

    Order matters. The config paths must be set before anything imports
    ``omero_screen``, whose package ``__init__`` reads OMERO_SCREEN_CONFIG at
    import time, and logging must be configured before the pipeline runs.
    """
    if env:
        os.environ["ENV"] = env

    # An explicit flag fails loudly on a bad path rather than silently
    # falling back to the built-in defaults; Click's exists=True does the
    # check during parsing, before any import.
    if config:
        os.environ["OMERO_SCREEN_CONFIG"] = config
    if stitch_config:
        os.environ["OMERO_SCREEN_STITCH_CONFIG"] = stitch_config

    # Importing config triggers set_env_vars() (loads .env.{ENV}) so an
    # OMERO_SCREEN_LOG_* override placed there is honoured; an explicit flag
    # still wins over the env var.
    from omero_screen.config import configure_logging

    configure_logging(
        level="DEBUG" if verbose else log_level,
        console=verbose,
        diagnose=verbose,
        log_file=log_file,
    )

    if inference:
        os.environ["OMERO_SCREEN_INFERENCE_MODEL"] = ":".join(inference)
    if gallery:
        os.environ["OMERO_SCREEN_INFERENCE_GALLERY_WIDTH"] = str(gallery)
    if batch:
        os.environ["OMERO_SCREEN_INFERENCE_BATCH_SIZE"] = str(batch)
    if track:
        os.environ["OMERO_SCREEN_TRACKING_MODEL"] = track
        os.environ["OMERO_SCREEN_TRACKING_MODE"] = track_mode
        os.environ["OMERO_SCREEN_TRACKING_BATCH_SIZE"] = str(track_batch_size)
        if track_device:
            os.environ["OMERO_SCREEN_TRACKING_DEVICE"] = track_device
        if track_window:
            os.environ["OMERO_SCREEN_TRACKING_WINDOW"] = str(track_window)
    if stream_stitch is not None:
        os.environ["OMERO_SCREEN_STITCH_STREAMING"] = (
            "1" if stream_stitch else "0"
        )


def _run(
    *,
    ids: list[int],
    segmentation: bool,
    stitch: bool,
    delete: bool,
    benchmark: bool,
) -> None:
    """Connect to OMERO and run the pipeline over every requested plate."""
    from omero.gateway import BlitzGateway
    from omero_utils.omero_connect import omero_connect

    from omero_screen.benchmarking import get_benchmark, init_benchmark
    from omero_screen.loops import plate_loop

    @omero_connect
    def run_plate_loop(
        plate_ids: list[int], conn: BlitzGateway | None = None
    ) -> None:
        assert conn is not None
        for plate_id in plate_ids:
            init_benchmark(enabled=benchmark, plate_id=plate_id)
            timer = get_benchmark()
            plate_loop(
                conn,
                plate_id,
                segmentation_mode=segmentation,
                stitch_mode=stitch,
                delete_existing=delete,
            )
            report_path = timer.save_report()
            if benchmark:
                click.echo(f"Benchmark report saved to {report_path}")

    run_plate_loop(ids)


def main() -> None:
    """Entry point for the ``omero-screen`` console script."""
    try:
        cli.main(standalone_mode=False)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()
