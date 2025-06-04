#!/usr/bin/env python3
"""Script to submit Omero Screen jobs to a SLURM cluster."""

import argparse
import getpass
import inspect
import os
import subprocess


def _create_job_script(args: argparse.Namespace) -> str:
    """Create the SLURM job script.

    Args:
        args: Program arguments

    Returns:
        The name of the script file
    """
    # Validate installation
    omero_screen = "omero-screen"
    omero_screen_prog = "run_omero_screen.py"
    send_mail = "send-mail.py"
    torch_test = "torch-test.py"

    if not os.path.isfile(omero_screen_prog):
        raise Exception(f"Missing program: {omero_screen_prog}")
    if not os.path.isfile(send_mail):
        raise Exception(f"Missing program: {send_mail}")
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if omero_screen != os.path.basename(parent_dir):
        raise Exception(f"Not within an '{omero_screen}' installation")

    # Check for an environment file
    env_file = "../.env." + (args.env if args.env else "development")
    if not os.path.exists(env_file):
        raise Exception(f"Missing env file: {env_file}")

    # Job name uses first plate ID and PID to avoid script name clashes
    pid = os.getpid()
    name = f"os{str(args.ID[0])}.{str(pid)}"

    # Options
    prog_options = (
        f"--inference {' '.join(args.inference)}" if args.inference else ""
    )
    if args.env:
        prog_options += f"--env {args.env}"

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
      runcmd module add proxy
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
        for plate_id in set(args.ID):
            print(
                f"runcmd uv run python {omero_screen_prog} {prog_options} {plate_id}",
                file=f,
            )
        # E-mail the user when the job has finished.
        # Here we use a custom python script which sends immediately.
        subject = f"Job results: {name}"
        msg = f"""
          Job results: {name}
          Plate: {", ".join([str(x) for x in args.ID])}
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
        help="Use a GPU node",
    )
    group.add_argument(
        "--exec",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Execute script statements. "
        "Disable this to submit a job without running Omero Screen",
    )
    group.add_argument(
        "--submit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Disable this to create the script but not submit using sbatch",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    script = _create_job_script(args)

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
