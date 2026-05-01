# ------------------------------------------------------------------------------
# Tests for FocalLoss from cellclass.training.
# ------------------------------------------------------------------------------

"""Tests for training.FocalLoss using CPU tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cellclass.training import FocalLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SAMPLES: int = 8
NUM_CLASSES: int = 3
LARGE_LOGIT: float = 100.0  # simulates very confident, correct prediction


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_output_is_scalar_tensor(self) -> None:
        """Default reduction='mean' must return a zero-dimensional scalar tensor."""
        loss_fn = FocalLoss()
        logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
        result = loss_fn(logits, targets)
        assert result.shape == torch.Size([])

    def test_perfect_predictions_yield_near_zero_loss(self) -> None:
        """Very high logit for the correct class should push loss close to zero."""
        loss_fn = FocalLoss(gamma=2.0)
        targets = torch.zeros(NUM_SAMPLES, dtype=torch.long)
        logits = torch.full((NUM_SAMPLES, NUM_CLASSES), -LARGE_LOGIT)
        logits[:, 0] = LARGE_LOGIT
        result = loss_fn(logits, targets)
        assert float(result) < 1e-3

    def test_uniform_logits_yield_positive_loss(self) -> None:
        """Uniform logits correspond to maximum uncertainty — loss must be positive."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.zeros(NUM_SAMPLES, NUM_CLASSES)
        targets = torch.zeros(NUM_SAMPLES, dtype=torch.long)
        result = loss_fn(logits, targets)
        assert float(result) > 0.0

    def test_gamma_zero_approximates_cross_entropy(self) -> None:
        """With gamma=0 FocalLoss degenerates to cross-entropy loss."""
        loss_fn_focal = FocalLoss(gamma=0.0, reduction="mean")
        logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
        focal_val = float(loss_fn_focal(logits, targets))
        ce_val = float(F.cross_entropy(logits, targets))
        assert abs(focal_val - ce_val) < 1e-4

    def test_alpha_weighting_changes_loss_value(self) -> None:
        """Providing an alpha tensor must produce a different result than no alpha."""
        logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
        loss_no_alpha = float(FocalLoss(gamma=2.0)(logits, targets))
        alpha = torch.tensor([2.0, 1.0, 0.5])
        loss_with_alpha = float(FocalLoss(alpha=alpha, gamma=2.0)(logits, targets))
        assert loss_no_alpha != loss_with_alpha

    def test_reduction_none_returns_per_sample_tensor(self) -> None:
        """reduction='none' must return a 1-D tensor of length N."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
        result = loss_fn(logits, targets)
        assert result.shape == torch.Size([NUM_SAMPLES])
