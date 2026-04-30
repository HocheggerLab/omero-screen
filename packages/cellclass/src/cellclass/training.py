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

"""Training loop utilities: focal loss, train/validation epoch helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# FocalLoss with optional alpha weights tensor for class imbalance.
# Adapted from: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):  # type: ignore[misc]
    """Focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialise FocalLoss.

        Args:
            alpha: Per-class weight tensor, e.g. inverse class frequencies.
            gamma: Focusing parameter. Higher values down-weight easy examples.
            reduction: ``'mean'``, ``'sum'``, or ``'none'``.

        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            x: Logits tensor of shape (N, C).
            target: Class indices tensor of shape (N,).

        Returns:
            Scalar loss (or per-sample if reduction is ``'none'``).

        """
        logpt = F.log_softmax(x, dim=-1)
        pt = torch.exp(logpt)
        return F.nll_loss(
            ((1 - pt) ** self.gamma) * logpt,
            target,
            weight=self.alpha,
            reduction=self.reduction,
        )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    classes: bool = False,
) -> tuple[
    float, float, float, float, torch.Tensor | None, torch.Tensor | None
]:
    """Perform one training epoch.

    Args:
        model: Network model (must be on ``device``).
        train_loader: DataLoader for training data.
        validation_loader: DataLoader for validation data.
        loss_fn: Loss function with signature ``(logits, labels) -> loss``.
        optimiser: Optimiser used to update model parameters.
        device: Compute device.
        classes: If True, return true and predicted class tensors.

    Returns:
        Tuple of (train_loss, val_loss, train_acc, val_acc, y_true, y_pred).
        y_true and y_pred are None when ``classes=False``.

    """
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    n = 0

    for x, y in train_loader:
        m = len(x)
        n += m
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        del x
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() * m
        _, pred = torch.max(y_pred, 1)
        train_acc += int((y == pred).sum().item())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        del y, y_pred

    train_loss /= n
    train_acc /= n

    val_loss, val_acc, y, y_pred = val_epoch(
        model, validation_loader, loss_fn, device, classes=classes
    )

    return train_loss, val_loss, train_acc, val_acc, y, y_pred


def val_epoch(
    model: nn.Module,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    classes: bool = False,
) -> tuple[float, float, torch.Tensor | None, torch.Tensor | None]:
    """Perform one validation epoch.

    Args:
        model: Network model (must be on ``device``).
        validation_loader: DataLoader for validation data.
        loss_fn: Loss function with signature ``(logits, labels) -> loss``.
        device: Compute device.
        classes: If True, return true and predicted class tensors.

    Returns:
        Tuple of (val_loss, val_acc, y_true, y_pred).
        y_true and y_pred are None when ``classes=False``.

    """
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    n = 0
    yy: list[torch.Tensor] = []
    yy_pred: list[torch.Tensor] = []

    for x, y in validation_loader:
        m = len(x)
        n += m
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        del x
        loss = loss_fn(y_pred, y)
        val_loss += loss.item() * m
        _, pred = torch.max(y_pred, 1)
        val_acc += int((y == pred).sum().item())
        if classes:
            yy.append(y.detach().cpu())
            yy_pred.append(pred.detach().cpu())
        else:
            del y
        del y_pred

    val_loss /= n
    val_acc /= n

    if classes:
        return val_loss, val_acc, torch.cat(yy), torch.cat(yy_pred)
    return val_loss, val_acc, None, None
