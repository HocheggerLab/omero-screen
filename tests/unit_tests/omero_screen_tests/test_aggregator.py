import numpy as np
import pytest

from omero_screen.aggregator import ImageAggregator

@pytest.mark.parametrize(
    "size, frames, block_size, sigma",
    [
        (1080, 5, 60, 30), # Standard 1080x1080 plate
        (987, 7, 31, 17),  # Non-integral block size
    ]
)
def test_aggregator(size: int, frames: int, block_size: int, sigma: int):
    """Test the image aggregator using the Gaussian filter."""
    rng = np.random.default_rng(seed=12367841628)
    min = 100
    max = 200
    agg = ImageAggregator(block_size)
    assert agg.get_image() is None
    assert agg.get_gaussian_image(sigma) is None
    for _ in range(frames):
        agg.add_image(rng.uniform(min, max, (size, size)))
    # The aggregator collates the minimum of the image within blocks.
    # Test the values are close to the minimum.
    assert agg.get_image() is not None
    a = agg.get_gaussian_image(sigma)
    assert a is not None
    upper = min + (max - min) * 0.05
    assert np.any((min >= a) | (a < upper))
