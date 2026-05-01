# ------------------------------------------------------------------------------
# Tests for StoppingCriteria from cellclass.bin.run_training.
# ------------------------------------------------------------------------------

"""Tests for run_training.StoppingCriteria early-stopping logic."""

from __future__ import annotations

import pytest

from cellclass.bin.run_training import StoppingCriteria

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PATIENCE: int = 3
INITIAL_LOSS: float = 1.0


class TestStoppingCriteria:
    """Tests for the StoppingCriteria class."""

    def test_does_not_stop_before_patience_exhausted(self) -> None:
        """Fewer than patience non-improving calls must not trigger a stop."""
        sc = StoppingCriteria(patience=DEFAULT_PATIENCE)
        sc.check(INITIAL_LOSS)  # first call counts as improvement
        for _ in range(DEFAULT_PATIENCE - 1):
            result = sc.check(INITIAL_LOSS + 1.0)
        assert result is False

    def test_stops_exactly_when_patience_is_reached(self) -> None:
        """Exactly patience non-improving calls after the first must trigger stop."""
        sc = StoppingCriteria(patience=DEFAULT_PATIENCE)
        sc.check(INITIAL_LOSS)  # initialise min_value
        for i in range(DEFAULT_PATIENCE - 1):
            sc.check(INITIAL_LOSS + 1.0)
        result = sc.check(INITIAL_LOSS + 1.0)
        assert result is True

    def test_improvement_resets_counter_and_does_not_stop(self) -> None:
        """An improvement after near-patience state must reset the counter."""
        sc = StoppingCriteria(patience=DEFAULT_PATIENCE)
        sc.check(INITIAL_LOSS)
        # Build up counter to patience-1
        for _ in range(DEFAULT_PATIENCE - 1):
            sc.check(INITIAL_LOSS + 1.0)
        # Now improve — counter resets
        result = sc.check(INITIAL_LOSS * 0.5)
        assert result is False
        assert sc.counter == 0

    def test_first_call_always_counts_as_improvement(self) -> None:
        """The very first check call must not increment the counter."""
        sc = StoppingCriteria(patience=1)
        result = sc.check(INITIAL_LOSS)
        assert result is False
        assert sc.counter == 0

    def test_min_delta_prevents_marginal_improvement_counting(self) -> None:
        """Value must improve by more than min_delta absolute to reset counter."""
        min_delta = 0.1
        sc = StoppingCriteria(patience=DEFAULT_PATIENCE, min_delta=min_delta)
        sc.check(INITIAL_LOSS)
        # Improve by less than min_delta — should not count
        result = sc.check(INITIAL_LOSS - min_delta * 0.5)
        assert sc.counter == 1  # treated as no improvement

    def test_rel_delta_prevents_marginal_relative_improvement(self) -> None:
        """Value must improve by more than min_rel_delta relative to count."""
        rel_delta = 0.5  # require 50% relative improvement
        sc = StoppingCriteria(patience=DEFAULT_PATIENCE, min_rel_delta=rel_delta)
        sc.check(INITIAL_LOSS)
        # Improve by only 10% — below rel_delta threshold
        result = sc.check(INITIAL_LOSS * 0.9)
        assert sc.counter == 1  # not counted as improvement

    def test_patience_of_one_stops_on_first_non_improvement(self) -> None:
        """With patience=1, a single non-improving value triggers stop."""
        sc = StoppingCriteria(patience=1)
        sc.check(INITIAL_LOSS)
        result = sc.check(INITIAL_LOSS + 1.0)
        assert result is True
