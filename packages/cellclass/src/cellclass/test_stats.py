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

"""Tests for the Statistics incremental statistics class."""

import math

import numpy as np
from stats import Statistics


class TestStatistics:
    """Unit tests for the Statistics class."""

    def test_empty(self) -> None:
        """Test Statistics initialised with no data."""
        v = Statistics()
        assert v.count() == 0
        assert v.min() == float("inf")
        assert v.max() == float("-inf")
        assert math.isnan(v.mean())
        assert math.isnan(v.var())
        assert math.isnan(v.std())

    def test_one(self) -> None:
        """Test Statistics with a single value."""
        v = Statistics()
        x = 77
        v.add(x)
        assert v.count() == 1
        assert v.min() == x
        assert v.max() == x
        assert v.mean() == x
        assert v.var() == 0
        assert v.std() == 0
        # Add an empty stats
        v.add_stats(Statistics())
        assert v.count() == 1
        assert v.min() == x
        assert v.max() == x
        assert v.mean() == x
        assert v.var() == 0
        assert v.std() == 0

    def test_two(self) -> None:
        """Test Statistics with two values."""
        x = [77, 123]
        v = Statistics(x)
        assert v.count() == 2
        assert v.min() == np.min(x)
        assert v.max() == np.max(x)
        assert v.mean() == np.mean(x)
        assert v.var() == np.var(x, correction=1)
        assert v.std() == np.std(x, correction=1)
        v.set_ddof(0)
        assert v.var() == np.var(x, correction=0)
        assert v.std() == np.std(x, correction=0)
        # Add to an empty stats
        v1 = v
        v = Statistics()
        v.add_stats(v1)
        assert v.count() == 2
        assert v.min() == np.min(x)
        assert v.max() == np.max(x)
        assert v.mean() == np.mean(x)
        assert v.var() == np.var(x, correction=1)
        assert v.std() == np.std(x, correction=1)

    def test_random(self) -> None:
        """Test Statistics against numpy for random data arrays."""
        rng = np.random.default_rng()
        for i in range(5):
            x = rng.standard_normal(10 + i)
            v = Statistics(x)
            n = len(x)
            min_val = np.min(x)
            max_val = np.max(x)
            mean = np.mean(x)
            var = np.var(x, correction=1)
            std = math.sqrt(var)
            assert v.count() == n
            assert v.min() == min_val
            assert v.max() == max_val
            assert math.isclose(v.mean(), mean, rel_tol=1e-10)
            assert math.isclose(v.var(), var, rel_tol=1e-10)
            assert math.isclose(v.std(), std, rel_tol=1e-10)
            # Combining
            i = len(x) // 2
            v = Statistics(x[:i])
            v.add_stats(Statistics(x[i:]))
            assert v.count() == n
            assert v.min() == min_val
            assert v.max() == max_val
            assert math.isclose(v.mean(), mean, rel_tol=1e-10)
            assert math.isclose(v.var(), var, rel_tol=1e-10)
            assert math.isclose(v.std(), std, rel_tol=1e-10)
            if len(x) % 2 == 1:
                # Symmetric combine
                v1 = Statistics(x[i:])
                v1.add_stats(Statistics(x[:i]))
                assert v.count() == n
                assert v.min() == min_val
                assert v.max() == max_val
                assert v.mean() == v1.mean()
                assert v.var() == v1.var()
                assert v.std() == v1.std()
