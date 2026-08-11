#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""Submit cellclass training jobs to a SLURM HPC cluster via sbatch."""

import argparse
import inspect
import os


def create_job_script(args: argparse.Namespace) -> str:
    """Write a SLURM job script and return its filename."""
    # Validate installation
    training_prog = "run_training.py"

    if not os.path.isfile(training_prog):
        raise Exception(f"Missing program: {training_prog}")

    # Job name
    pid = os.getpid()
    name_width = 14 - len(str(pid))
    name = "training"[0:name_width] + "." + str(pid)

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
        if args.gpu:
            # Note: constraint option is not valid although it is specified in the artemis docs
            print(
                inspect.cleandoc("""\
        ##SBATCH --constraint="gpu"
        #SBATCH --gres=gpu
        """),
                file=f,
            )
        # job script
        run = "exec" if args.exec else "cmd"
        comment = "" if args.exec else "#"
        print(
            inspect.cleandoc(f"""
      function msg {{
        echo $(date "+[%F %T]") $@
      }}
      function runcmd {{
        msg {run}: $@
        {comment}$@
      }}
      set -e
      runcmd module add proxy
      """),
            file=f,
        )

        arguments = " ".join(args.args)
        print(f"runcmd uv run {training_prog} {arguments}", file=f)
        print(f"rm {script}", file=f)

        return script


def main() -> None:
    """Entry point for direct execution of the sbatch command."""
    from cellclass.cli import sbatch

    sbatch.main(prog_name="cellclass-sbatch")


if __name__ == "__main__":
    main()
