#!/usr/bin/env python3

"""Tests performing a computation on the GPU using pytorch."""

import argparse

CODE_TORCH_GPU = 0
CODE_TORCH_CPU = 1
CODE_NO_TORCH = -1


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Program to perform a computation on the GPU using pytorch.",
        epilog="Note: The exit status is non-zero if GPU computation failed.",
    )

    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Print torch results",
    )
    parser.add_argument(
        "-m",
        "--memory",
        dest="memory",
        action="store_true",
        help="Print memory usage at each step",
    )
    parser.add_argument(
        "-c",
        "--convolution",
        dest="convolution",
        action="store_true",
        help="Perform a convolution",
    )

    args = parser.parse_args()

    # Set-up functions for output
    if args.debug:
        # Print out information on torch usage
        def info(msg: str) -> None:
            print(f"INFO   - {msg}")
    else:

        def info(msg: str) -> None:
            pass

    if args.memory:
        # Print out current resident and virtual memory
        import psutil

        def mem(msg: str) -> None:
            process = psutil.Process()
            i = process.memory_info()
            print(
                f"MEMORY - {msg}: rss={i.rss / 1024**3:.3}G, vms={i.vms / 1024**3:.3}G"
            )
    else:

        def mem(msg: str) -> None:
            pass

    mem("start-up")

    code = CODE_NO_TORCH
    try:
        import torch

        mem("import torch")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

        mem("torch.device")

        info(f"torch.device = {str(device)}")

        t = torch.rand(5, 3)
        mem("torch.rand(5, 3)")

        t = t.to(device)
        mem("t.to(device)")

        info(str(t))

        if args.convolution:
            import torch.nn as nn

            mem("import torch.nn")

            m = nn.Conv2d(16, 33, 3, stride=2)
            m = m.to(device)
            mem("nn.Conv2d")

            t = torch.randn(20, 16, 50, 100)
            t = t.to(device)
            mem("torch.randn(20, 16, 50, 100)")

            output = m(t)
            mem("convolution done")

            info(output)

        # Reach here if all torch computations worked.
        # Set the exit code to distinguish GPU or CPU
        code = CODE_TORCH_GPU if str(device) != "cpu" else CODE_TORCH_CPU

    except Exception as e:  # noqa: BLE001
        info(str(e))

    mem(f"exit({code})")

    # Report if pytorch computation succeeded
    if code == CODE_TORCH_GPU:
        info("torch GPU computation available")
    elif code == CODE_TORCH_CPU:
        print("WARN   - torch CPU computation available")
    else:
        print("ERROR  - torch GPU not available")

    exit(code)


# Standard boilerplate to call the function to begin the program.
if __name__ == "__main__":
    _main()
