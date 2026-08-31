#!/usr/bin/env python3
"""Script to submit Omero Screen jobs to a SLURM cluster."""

import argparse
import getpass
import inspect
import os
import subprocess


def _create_job_script(args: argparse.Namespace, plate_ids: list[int]) -> str:
    """Create the SLURM job script.

    Args:
        args: Program arguments
        plate_ids: Plates IDs to process

    Returns:
        The name of the script file
    """
    # Validate installation
    omero_screen = "omero-screen"
    omero_screen_prog = "run_omero_screen.py"
    send_mail = "send_mail.py"
    torch_test = "torch_test.py"

    if not os.path.isfile(omero_screen_prog):
        raise Exception(f"Missing program: {omero_screen_prog}")
    if not os.path.isfile(send_mail):
        raise Exception(f"Missing program: {send_mail}")
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if omero_screen != os.path.basename(parent_dir):
        raise Exception(f"Not within an '{omero_screen}' installation")

    # Check for an environment file
    env_file = f"../.env.{args.env}" if args.env else "../.env"
    if not os.path.exists(env_file):
        raise Exception(f"Missing env file: {env_file}")

    # Job name uses first plate ID and PID to avoid script name clashes
    pid = os.getpid()
    name = f"os{str(plate_ids[0])}.{str(pid)}"

    # Options
    prog_options = (
        f"--inference {' '.join(args.inference)}" if args.inference else ""
    )
    if args.env:
        prog_options += f" --env {args.env}"
    if args.segmentation:
        prog_options += " --segmentation"
    if args.delete:
        prog_options += " --delete"
    if args.model:
        prog_options += f" --model {args.model}"
    elif args.cp4:
        prog_options += " --cp4"
    if args.stitch:
        prog_options += " --stitch"
    if args.stream_stitch is not None:
        prog_options += (
            " --stream-stitch" if args.stream_stitch else " --no-stream-stitch"
        )
    if args.track:
        prog_options += f" --track {args.track}"
        prog_options += f" --track-mode {args.track_mode}"
        prog_options += f" --track-batch-size {args.track_batch_size}"
        if args.track_device:
            prog_options += f" --track-device {args.track_device}"
        if args.track_window:
            prog_options += f" --track-window {args.track_window}"
    if args.config:
        # Resolve to an absolute path now: the job runs from this directory
        # (bin/), so a relative --config would otherwise resolve against bin/.
        # Validate at submission time to fail fast rather than after queueing.
        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            raise Exception(f"Missing config file: {config_path}")
        prog_options += f" --config {config_path}"
    if args.stitch_config:
        # Resolve to an absolute path now.
        stitch_config_path = os.path.abspath(args.stitch_config)
        if not os.path.exists(stitch_config_path):
            raise Exception(
                f"Missing stitch config file: {stitch_config_path}"
            )
        prog_options += f" --stitch-config {stitch_config_path}"

    # Create the job file
    script = f"{name}.sh"
    with open(script, "w") as f:
        # job options
        # The -l option to bash is to make bash act as if a login shell (enables conda init)
        print(
            inspect.cleandoc(f"""\
      #!/bin/bash -l
      #SBATCH -J {name}
      #SBATCH -o {name}."%j".out
      #SBATCH -p {args.job_class}
      #SBATCH --mail-user {args.username}@sussex.ac.uk
      #SBATCH --mail-type=END,FAIL
      #SBATCH --mem={args.memory}G
      #SBATCH --time={args.hours}:00:00
      """),
            file=f,
        )
        if args.threads > 1:
            print(
                inspect.cleandoc(f"""\
        #SBATCH -n {args.threads}
        """),
                file=f,
            )
        if args.gpu:
            print(
                inspect.cleandoc("""\
        #SBATCH --gres=gpu
        """),
                file=f,
            )
        # job script
        run = "exec" if args.exec else "cmd"
        comment = "" if args.exec else "#"
        print(
            inspect.cleandoc(
                f"""
      function msg {{
        echo $(date "+[%F %T]") $@
      }}
      function runcmd {{
        msg {run}: $@
        {comment}$@
      }}
      set -e
      export PYTHONPATH=$(cd ../ && pwd)
      msg PYTHONPATH=$PYTHONPATH
      """
            ),
            file=f,
        )
        # Test for gpu
        if args.gpu:
            print(
                inspect.cleandoc(
                    """
        set +e
        runcmd uv run python {torch_test}
        code=$?
        if [ $code -ne 0 ]; then
          msg Torch test exit code: $code
          uv run python {send_mail} -m "{msg}" -s "{subject}" {username}@sussex.ac.uk
          exit $code
        fi
        set -e
        """.format(
                        torch_test=torch_test,
                        send_mail=send_mail,
                        msg=f"Torch GPU unavailable for {script}",
                        subject=f"{script} failed",
                        username=args.username,
                    )
                ),
                file=f,
            )
        for plate_id in set(plate_ids):
            print(
                f"runcmd uv run python {omero_screen_prog} {plate_id} {prog_options}",
                file=f,
            )
        # E-mail the user when the job has finished.
        # Here we use a custom python script which sends immediately.
        subject = f"Job results: {name}"
        msg = f"""
          Job results: {name}
          Plate: {", ".join([str(x) for x in plate_ids])}
          """
        print(f"msg Sending result e-mail using {send_mail}", file=f)
        print(
            f"python {send_mail} -m '{msg}' -s '{subject}' "
            f"{args.username}@sussex.ac.uk",
            file=f,
        )
        print("msg Done", file=f)
        print(f"rm {script}", file=f)

        return script


def _parse_args() -> argparse.Namespace:
    """Parse the script arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Program to run Omero Screen on a SLURM cluster.",
        epilog=inspect.cleandoc("""Note:

      This program makes assumptions on the installation of Omero Screen and
      the run environment."""),
    )
    parser.add_argument("ID", type=int, nargs="+", help="Screen ID")
    group = parser.add_argument_group("Job submission")
    group.add_argument(
        "--class",
        dest="job_class",
        default="gpu",
        help="Job class (default: %(default)s)",
    )
    group.add_argument(
        "-u",
        "--username",
        dest="username",
        default=getpass.getuser(),
        help="Username (default: %(default)s)",
    )
    group.add_argument(
        "-t",
        "--threads",
        type=int,
        dest="threads",
        default=1,
        help="Threads (default: %(default)s). Use when not executing on the GPU",
    )
    group.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Expected maximum job hours (default: %(default)s)",
    )
    group.add_argument(
        "-m",
        "--memory",
        type=int,
        dest="memory",
        default=32,
        help="Memory in Gb (default: %(default)s)",
    )
    group.add_argument(
        "--gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use a GPU node (default: %(default)s)",
    )
    group.add_argument(
        "--exec",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Execute script statements. "
        "Disable this to submit a job without running Omero Screen (default: %(default)s)",
    )
    group.add_argument(
        "--submit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Disable this to create the script but not submit using sbatch (default: %(default)s)",
    )
    group.add_argument(
        "--multi-submit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Submit a single job for each Screen ID (default: %(default)s)",
    )
    group = parser.add_argument_group("Omero Screen overrides")
    group.add_argument(
        "--inference",
        type=str,
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Inference model(s).",
    )
    group.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (requires configuration file .env.{name}).",
    )
    group.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an OMERO_SCREEN_CONFIG JSON (MODEL_DICT / FEATURELIST / "
        "CHANNEL_SEG_PROFILES). Resolved to an absolute path and validated at "
        "submission time, then forwarded to omero-screen --config.",
    )
    group.add_argument(
        "--segmentation",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only perform image segmentation (default: %(default)s)",
    )
    group.add_argument(
        "--delete",
        default=False,
        action="store_true",
        help=(
            "Delete the plate's existing segmentation masks and segment from "
            "scratch. Without this, a re-run reuses the stored masks (both "
            "stitched and per-field) and only recomputes the measurements. "
            "Use it after changing segmentation settings, or to repair a "
            "plate whose stored masks are wrong or empty."
        ),
    )
    group.add_argument(
        "--cp4",
        default=False,
        action="store_true",
        help="Use Cellpose 4 (cpsam) for segmentation instead of the default Cellpose 3 models.",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Override all segmentation models with a single model name (e.g. 'cp4:cpsam'). Overrides --cp4.",
    )
    group.add_argument(
        "--stitch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run stitched-well segmentation: assemble all fields per well into one canvas, segment that canvas, and exclude border objects only at the outer edge (default: %(default)s)",
    )
    group.add_argument(
        "--stream-stitch",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Stitch one timepoint at a time to bound host RAM on long "
        "multi-channel timelapses (costs n_fields x T OMERO reads). Default: "
        "auto-enable when the estimated peak exceeds the RAM budget; use "
        "--stream-stitch / --no-stream-stitch to force. Requires --stitch.",
    )
    group.add_argument(
        "--stitch-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an OMERO_SCREEN_STITCH_CONFIG JSON configuration. "
        "Overrides the OMERO_SCREEN_STITCH_CONFIG env var. "
        "Errors if the path does not exist (no silent fallback to defaults).",
    )
    group.add_argument(
        "--track",
        type=str,
        nargs="?",
        const="general_2d",
        default=None,
        metavar="MODEL",
        help="Track nuclei across time with Trackastra. Optional MODEL is a pretrained name or checkpoint path (default when flag given: %(const)s). Requires --stitch and a timelapse (T>1); a no-op on single-timepoint plates.",
    )
    group.add_argument(
        "--track-mode",
        type=str,
        default="greedy",
        choices=["greedy", "greedy_nodiv", "ilp"],
        help="Trackastra linking mode (default: %(default)s).",
    )
    group.add_argument(
        "--track-batch-size",
        type=int,
        default=4,
        help="Attention windows Trackastra scores per forward pass "
        "(default: %(default)s). Lower if tracking hits CUDA OOM on dense "
        "wells; raise for faster scoring when GPU VRAM allows.",
    )
    group.add_argument(
        "--track-device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force the tracking device (default: auto-detect). Use 'cpu' when "
        "a dense well exceeds GPU VRAM — same computation in host RAM, slower "
        "but no VRAM ceiling and no accuracy loss.",
    )
    group.add_argument(
        "--track-window",
        type=int,
        default=None,
        help="Override Trackastra's temporal window (frames per attention "
        "window). Smaller cuts GPU memory ~quadratically at the cost of "
        "temporal context; default keeps the model's trained window.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    plate_batches = [[x] for x in args.ID] if args.multi_submit else [args.ID]
    del args.ID

    for plate_ids in plate_batches:
        script = _create_job_script(args, plate_ids)

        # job submission
        if args.submit:
            print(
                subprocess.run(
                    ["sbatch", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
            )
