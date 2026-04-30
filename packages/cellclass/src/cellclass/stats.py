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

"""Incremental statistics (mean, variance, std, min, max) for streaming data."""

# Incrementing statistics (mean, variance).
# Ref: Chan, Golub and Levesque (1983)
# Algorithms for Computing the Sample Variance: Analysis and Recommendations.
# American Statistician, 37, 242-247.
# https://doi.org/10.2307/2683386

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Union

import numpy as np


class Statistics:
    """Incremental mean, variance, std, min and max statistics."""

    def __init__(
        self,
        x: Union[float, Iterable[float]] | None = None,
        ddof: int = 1,
    ) -> None:
        """Create an instance, optionally seeded with initial data.

        Args:
            x: Initial value or iterable of values.
            ddof: Delta degrees of freedom. Divisor is (N - ddof).

        """
        self._n: int = 0
        self._m1: float = 0.0
        self._ssd: float = 0.0
        self._ddof: int = ddof
        self._min: float = float("inf")
        self._max: float = float("-inf")
        if x is not None:
            if isinstance(x, Iterable):
                arr = np.array(list(x), dtype=float).flatten()
                self._n = len(arr)
                if self._n > 0:
                    self._m1 = float(np.mean(arr))
                    self._ssd = float(np.sum((arr - self._m1) ** 2))
                    self._min = float(np.min(arr))
                    self._max = float(np.max(arr))
            else:
                self.add(float(x))

    def set_ddof(self, ddof: int) -> None:
        """Set the Delta Degrees of Freedom.

        Args:
            ddof: Divisor used in the calculation is (N - ddof).

        """
        self._ddof = ddof

    def add(self, x: float) -> None:
        """Update statistics with a single value.

        Args:
            x: New value.

        """
        n1 = self._n + 1
        dev = x - self._m1
        ndev = dev / n1
        self._m1 += ndev
        self._ssd += self._n * dev * ndev
        self._n = n1
        self._min = min(self._min, x)
        self._max = max(self._max, x)

    def add_all(self, x: Iterable[float]) -> None:
        """Update statistics with all values in an iterable.

        Args:
            x: Iterable of values.

        """
        for i in x:
            self.add(i)

    def add_stats(self, x: Statistics) -> None:
        """Merge another Statistics instance into this one.

        Args:
            x: Statistics to merge.

        """
        n1 = self._n
        n2 = x._n
        if n1 == 0:
            self._n = n2
            self._m1 = x._m1
            self._ssd = x._ssd
            self._min = x._min
            self._max = x._max
        elif n2 != 0:
            dm = self._m1 - x._m1
            self._ssd = (self._ssd + x._ssd) + dm**2 * ((n1 * n2) / (n1 + n2))
            mu1 = self._m1
            mu2 = x._m1
            self._n = n1 + n2
            self._min = min(self._min, x._min)
            self._max = max(self._max, x._max)
            if n1 == n2:
                self._m1 = (mu1 + mu2) * 0.5
            elif n2 < n1:
                self._m1 = mu1 + (mu2 - mu1) * (n2 / (n1 + n2))
            else:
                self._m1 = mu2 + (mu1 - mu2) * (n1 / (n1 + n2))

    def count(self) -> int:
        """Return the count."""
        return self._n

    def min(self) -> float:
        """Return the minimum."""
        return self._min

    def max(self) -> float:
        """Return the maximum."""
        return self._max

    def mean(self) -> float:
        """Return the mean."""
        return self._m1 if self._n != 0 else float("nan")

    def var(self) -> float:
        """Return the variance."""
        if self._n == 0:
            return float("nan")
        if self._n == 1:
            return 0.0
        return self._ssd / (self._n - self._ddof)

    def std(self) -> float:
        """Return the standard deviation."""
        if self._n == 0:
            return float("nan")
        if self._n == 1:
            return 0.0
        return math.sqrt(self._ssd / (self._n - self._ddof))
