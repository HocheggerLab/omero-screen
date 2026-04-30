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

"""Evaluation utilities: single-epoch test pass returning true and predicted labels."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test_epoch(
    model: nn.Module,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one evaluation pass and return true and predicted labels.

    Args:
        model: Network model (must already be on ``device``).
        validation_loader: DataLoader yielding (images, labels) batches.
        device: Device to run inference on.

    Returns:
        Tuple of (y_true, y_pred) as 1-D CPU tensors.

    """
    model.eval()

    yy: list[torch.Tensor] = []
    yy_pred: list[torch.Tensor] = []

    for x, y in validation_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
            _, y_pred = torch.max(y_pred, 1)
        del x
        yy.append(y.detach().cpu())
        yy_pred.append(y_pred.detach().cpu())

    return torch.cat(yy), torch.cat(yy_pred)
