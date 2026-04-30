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

"""PyTorch utility helpers: optimizer device transfer and CUDA memory reporting."""

from __future__ import annotations

import torch


def optimizer_to(optim: torch.optim.Optimizer, device: torch.device) -> None:
    """Move the optimizer state to a device.

    Args:
        optim: Optimizer whose state to move.
        device: Target device.

    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def memory(
    device: torch.device | None = None,
) -> tuple[int, int, int, int]:
    """Return CUDA memory statistics for a device.

    Args:
        device: CUDA device. Defaults to the current device.

    Returns:
        Tuple of (allocated, max_allocated, reserved, max_reserved) bytes.

    """
    return (
        torch.cuda.memory_allocated(device=device),
        torch.cuda.max_memory_allocated(device=device),
        torch.cuda.memory_reserved(device=device),
        torch.cuda.max_memory_reserved(device=device),
    )
