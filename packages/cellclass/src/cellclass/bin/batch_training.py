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

"""Generate and run batch training shell scripts from a training plan file."""

import argparse

from cellclass.bin_utils import file_path


def run(args: argparse.Namespace) -> None:
    """Execute training runs defined in a batch file."""
    import logging
    import os
    import re
    import subprocess

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s", level=logging.INFO
    )

    # Load the training settings
    logging.info(f"Loading training settings: {args.batch}")
    with open(args.batch) as f:
        training = f.readlines()
    used = set()
    script = []
    for line in training:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        arguments = re.split(r"\s+", line)
        cmd = [args.cmd, "--wandb"]
        cmd.extend(arguments)
        # Identify the output file
        prefix = os.path.basename(arguments[0])
        n = 0
        while True:
            n += 1
            base = f"{prefix}.{n}"
            if base in used:
                continue
            if not (
                os.path.isfile(base + ".pt") or os.path.isfile(base + ".json")
            ):
                # This output file is OK
                break
        used.add(base)
        cmd.extend(["-n", base + ".pt", "-s", base + ".json"])
        out = base + ".out"

        # Create a name
        if "--run-name" not in arguments:
            # Start with the dataset name
            name = [os.path.basename(os.path.splitext(arguments[0])[0])]
            for a in arguments[1:]:
                # Detect new argument
                if a[0] == "-":
                    size = 0
                    # Ignore some arguments
                    if a in ["-d", "--device", "--project"]:
                        size -= 1
                    else:
                        name.append(a.lstrip("-"))
                    continue
                if size < 0:
                    continue
                if size > 0:
                    name[-1] = name[-1] + ","
                name[-1] = name[-1] + a
                size += 1

            cmd.extend(["--run-name", "_".join(name)])

        logging.info(f"Run {cmd} > {out}")
        if args.dry_run:
            continue
        if args.background:
            # Runs in a new background process
            with open(out, "w") as f:
                subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=f)
        else:
            # Print to a script
            script.append(" ".join(cmd) + " > " + out)

    if args.dry_run:
        # https://patorjk.com/software/taag/#p=display&f=Alligator2&t=Dry%20run
        font = """
    :::::::::  :::::::::  :::   :::      :::::::::  :::    ::: ::::    :::
    :+:    :+: :+:    :+: :+:   :+:      :+:    :+: :+:    :+: :+:+:   :+:
    +:+    +:+ +:+    +:+  +:+ +:+       +:+    +:+ +:+    +:+ :+:+:+  +:+
    +#+    +:+ +#++:++#:    +#++:        +#++:++#:  +#+    +:+ +#+ +:+ +#+
    +#+    +#+ +#+    +#+    +#+         +#+    +#+ +#+    +#+ +#+  +#+#+#
    #+#    #+# #+#    #+#    #+#         #+#    #+# #+#    #+# #+#   #+#+#
    #########  ###    ###    ###         ###    ###  ########  ###    ####
"""
        print(font)

    if not args.background:
        logging.info("Creating batch script: %s", args.script)
        with open(args.script, "w") as f:
            for line in script:
                f.write(f"{line}\n")

    logging.info("Done")


def main() -> None:
    """Entry point for cellclass-batch CLI."""
    parser = argparse.ArgumentParser(
        description="Program to run the training script for a batch file of arguments."
    )

    parser.add_argument(
        "batch", metavar="BATCH", type=file_path, help="Batch arguments file"
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Perform a dry run (default: %(default)s)",
    )
    parser.add_argument(
        "--background",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Perform each run as a new background process (default: %(default)s)",
    )
    parser.add_argument(
        "--script",
        default="batch.sh",
        help="Batch script (default: %(default)s)",
    )
    parser.add_argument(
        "--cmd",
        default="run_training.py",
        help="Program (default: %(default)s)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
