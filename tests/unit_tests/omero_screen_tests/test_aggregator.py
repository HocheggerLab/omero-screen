"""Unit tests for aggregator module using synthetic data."""

import numpy as np
import pytest
import scipy.ndimage

from omero_screen.aggregator import (
    ImageAggregator,
    block,
    fixup_scipy_ndimage_result,
    gaussian_filter,
    median_filter,
    strel_disk,
)


@pytest.mark.parametrize(
    "size, frames, block_size, sigma",
    [
        (1080, 5, 60, 30),  # Standard 1080x1080 plate
        (987, 7, 31, 17),  # Non-integral block size
    ],
)
def test_aggregator(size: int, frames: int, block_size: int, sigma: int):
    """Test the image aggregator using the Gaussian filter."""
    rng = np.random.default_rng(seed=12367841628)
    min_val = 100
    max_val = 200
    agg = ImageAggregator(block_size)
    assert agg.get_image() is None
    assert agg.get_gaussian_image(sigma) is None
    for _ in range(frames):
        agg.add_image(rng.uniform(min_val, max_val, (size, size)))
    # The aggregator collates the minimum of the image within blocks.
    # Test the values are close to the minimum.
    assert agg.get_image() is not None
    a = agg.get_gaussian_image(sigma)
    assert a is not None
    upper = min_val + (max_val - min_val) * 0.05
    assert np.any((min_val >= a) | (a < upper))


class TestStrelDisk:
    """Test the strel_disk structuring element function."""

    def test_strel_disk_radius_1(self):
        """Test disk with radius 1."""
        disk = strel_disk(1.0)
        expected = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64
        )
        np.testing.assert_array_equal(disk, expected)

    def test_strel_disk_radius_2(self):
        """Test disk with radius 2."""
        disk = strel_disk(2.0)
        # Should be 5x5 array
        assert disk.shape == (5, 5)
        # Center should be 1
        assert disk[2, 2] == 1
        # Corners should be 0
        assert disk[0, 0] == 0
        assert disk[0, 4] == 0
        assert disk[4, 0] == 0
        assert disk[4, 4] == 0

    def test_strel_disk_radius_5(self):
        """Test disk with radius 5."""
        disk = strel_disk(5.0)
        # Should be 11x11 array
        assert disk.shape == (11, 11)
        # Center should be 1
        assert disk[5, 5] == 1
        # All values should be 0 or 1
        assert np.all((disk == 0) | (disk == 1))

    def test_strel_disk_symmetry(self):
        """Test that disk is symmetric."""
        disk = strel_disk(3.0)
        # Should be symmetric horizontally
        np.testing.assert_array_equal(disk, np.fliplr(disk))
        # Should be symmetric vertically
        np.testing.assert_array_equal(disk, np.flipud(disk))

    def test_strel_disk_dtype(self):
        """Test that disk has correct dtype."""
        disk = strel_disk(2.5)
        assert disk.dtype == np.float64


class TestMedianFilter:
    """Test the median filter function."""

    def test_median_filter_basic(self):
        """Test basic median filtering."""
        rng = np.random.default_rng(seed=42)
        # Create image with some noise
        image = rng.uniform(100, 200, (100, 100))

        filtered = median_filter(image, radius=2.0)

        # Output shape should match input
        assert filtered.shape == image.shape
        # Output should be smoothed (less extreme values)
        assert np.std(filtered) <= np.std(image)

    def test_median_filter_preserves_dtype(self):
        """Test that median filter preserves the input dtype."""
        rng = np.random.default_rng(seed=42)
        image = rng.uniform(100, 200, (50, 50)).astype(np.float32)

        filtered = median_filter(image, radius=1.5)

        assert filtered.dtype == image.dtype

    def test_median_filter_removes_outliers(self):
        """Test that median filter removes isolated outliers."""
        # Create image with uniform background
        image = np.ones((50, 50), dtype=np.float64) * 100
        # Add a single hot pixel
        image[25, 25] = 10000

        filtered = median_filter(image, radius=2.0)

        # Hot pixel should be reduced
        assert filtered[25, 25] < image[25, 25]
        # Hot pixel should be closer to background
        assert abs(filtered[25, 25] - 100) < abs(image[25, 25] - 100)


class TestGaussianFilter:
    """Test the gaussian filter function."""

    def test_gaussian_filter_basic(self):
        """Test basic Gaussian filtering."""
        rng = np.random.default_rng(seed=42)
        image = rng.uniform(100, 200, (100, 100))

        filtered = gaussian_filter(image, sigma=2.0)

        # Output shape should match input
        assert filtered.shape == image.shape
        # Output should be smoothed
        assert np.std(filtered) < np.std(image)

    def test_gaussian_filter_edge_handling(self):
        """Test that Gaussian filter handles edges correctly."""
        # Create image with bright spot in corner
        image = np.zeros((50, 50), dtype=np.float64)
        image[0, 0] = 1000

        filtered = gaussian_filter(image, sigma=3.0)

        # Edge artifacts should be corrected
        # The corner pixel should still be brightest but not infinite
        assert np.isfinite(filtered).all()
        assert filtered[0, 0] > filtered[25, 25]

    def test_gaussian_filter_sigma_effect(self):
        """Test that larger sigma produces more smoothing."""
        rng = np.random.default_rng(seed=42)
        image = rng.uniform(100, 200, (100, 100))

        filtered_small = gaussian_filter(image, sigma=1.0)
        filtered_large = gaussian_filter(image, sigma=5.0)

        # Larger sigma should produce smoother result (lower std)
        assert np.std(filtered_large) < np.std(filtered_small)

    def test_gaussian_filter_constant_image(self):
        """Test Gaussian filter on constant image."""
        image = np.ones((50, 50), dtype=np.float64) * 150

        filtered = gaussian_filter(image, sigma=2.0)

        # Constant image should remain constant
        np.testing.assert_allclose(filtered, image, rtol=1e-10)


class TestBlock:
    """Test the block function for image subdivision."""

    def test_block_single_block(self):
        """Test block division with single block."""
        labels, indexes = block((100, 100), (100, 100))

        # Should create labels array same size as image
        assert labels.shape == (100, 100)
        # All labels should be 0 (single block)
        assert np.all(labels == 0)
        # Should have one index
        assert len(indexes) == 1
        assert indexes[0] == 0

    def test_block_four_blocks(self):
        """Test block division into 2x2 grid."""
        labels, indexes = block((100, 100), (50, 50))

        # Should create labels array same size as image
        assert labels.shape == (100, 100)
        # Should have 4 blocks (0, 1, 2, 3)
        assert len(indexes) == 4
        # Labels should range from 0 to 3
        unique_labels = np.unique(labels)
        np.testing.assert_array_equal(unique_labels, [0, 1, 2, 3])

    def test_block_rectangular(self):
        """Test block division with rectangular blocks."""
        labels, indexes = block((120, 80), (40, 40))

        # Should create labels array same size as image
        assert labels.shape == (120, 80)
        # Should have 3x2 = 6 blocks
        assert len(indexes) == 6

    def test_block_non_divisible(self):
        """Test block division when image size is not divisible by block size."""
        labels, indexes = block((100, 100), (30, 30))

        # Should still create proper labels
        assert labels.shape == (100, 100)
        # Should have at least 9 blocks (3x3)
        assert len(indexes) >= 9


class TestFixupScipyNdimageResult:
    """Test the fixup_scipy_ndimage_result function."""

    def test_fixup_single_value(self):
        """Test conversion of single scalar value."""
        result = fixup_scipy_ndimage_result(42.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 42.0

    def test_fixup_list(self):
        """Test conversion of list."""
        result = fixup_scipy_ndimage_result([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_fixup_array(self):
        """Test that array is preserved."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fixup_scipy_ndimage_result(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_fixup_tuple(self):
        """Test conversion of tuple."""
        result = fixup_scipy_ndimage_result((10, 20, 30))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestImageAggregator:
    """Test the ImageAggregator class."""

    def test_aggregator_initialization(self):
        """Test that aggregator initializes correctly."""
        agg = ImageAggregator(block_size=60)

        # Should return None before any images added
        assert agg.get_image() is None
        assert agg.get_median_image(5.0) is None
        assert agg.get_gaussian_image(10.0) is None

    def test_aggregator_single_image(self):
        """Test aggregator with single image."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)  # No block processing

        image = rng.uniform(100, 200, (100, 100))
        agg.add_image(image)

        result = agg.get_image()
        assert result is not None
        # With one image, result should equal input
        np.testing.assert_array_equal(result, image)

    def test_aggregator_multiple_images_no_blocks(self):
        """Test aggregator averaging multiple images without block processing."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        images = [rng.uniform(100, 200, (50, 50)) for _ in range(5)]
        for img in images:
            agg.add_image(img)

        result = agg.get_image()
        assert result is not None

        # Result should be the mean of all images
        expected = np.mean(images, axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_aggregator_with_blocks(self):
        """Test aggregator with block-wise minimum aggregation."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=25)

        # Add several images
        for _ in range(3):
            agg.add_image(rng.uniform(100, 200, (100, 100)))

        result = agg.get_image()
        assert result is not None
        assert result.shape == (100, 100)

        # With block minimum, values should be lower than simple mean
        # All values should be positive
        assert np.all(result > 0)

    def test_aggregator_median_filter(self):
        """Test aggregator with median filter output."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        for _ in range(3):
            agg.add_image(rng.uniform(100, 200, (100, 100)))

        result = agg.get_median_image(radius=2.0)
        assert result is not None
        assert result.shape == (100, 100)

    def test_aggregator_gaussian_filter(self):
        """Test aggregator with Gaussian filter output."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        for _ in range(3):
            agg.add_image(rng.uniform(100, 200, (100, 100)))

        result = agg.get_gaussian_image(sigma=3.0)
        assert result is not None
        assert result.shape == (100, 100)

    def test_aggregator_reset(self):
        """Test aggregator reset functionality."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=30)

        # Add images
        for _ in range(3):
            agg.add_image(rng.uniform(100, 200, (100, 100)))

        # Should have result
        assert agg.get_image() is not None

        # Reset
        agg.reset()

        # Should return None after reset
        assert agg.get_image() is None

    def test_aggregator_caching(self):
        """Test that aggregator caches results correctly."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        agg.add_image(rng.uniform(100, 200, (100, 100)))

        # Get image twice
        result1 = agg.get_image()
        result2 = agg.get_image()

        # Should return same cached result
        assert result1 is result2

    def test_aggregator_cache_invalidation(self):
        """Test that cache is invalidated when new image is added."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        agg.add_image(rng.uniform(100, 150, (50, 50)))
        result1 = agg.get_image()
        assert result1 is not None

        # Add another image
        agg.add_image(rng.uniform(150, 200, (50, 50)))
        result2 = agg.get_image()
        assert result2 is not None

        # Results should be different
        assert not np.array_equal(result1, result2)

    def test_aggregator_different_sizes_error(self):
        """Test that adding images of different sizes works with proper initialization."""
        rng = np.random.default_rng(seed=42)
        agg = ImageAggregator(block_size=0)

        # First image sets the size
        agg.add_image(rng.uniform(100, 200, (50, 50)))

        # Second image of different size should cause an error during addition
        with pytest.raises(ValueError):
            agg.add_image(rng.uniform(100, 200, (60, 60)))
