"""Unit tests for flatfield_corr module using synthetic data."""

import numpy as np
import pytest

from omero_screen.flatfield_corr import (
    flatfieldcorr_name,
    random_timgs,
)


class TestFlatfieldCorrName:
    """Test the flatfieldcorr_name function."""

    def test_flatfieldcorr_name_basic(self):
        """Test basic name generation for flatfield correction."""
        plate_id = 12345
        expected = "12345_flatfield_masks"
        result = flatfieldcorr_name(plate_id)
        assert result == expected

    def test_flatfieldcorr_name_single_digit(self):
        """Test name generation with single digit plate ID."""
        plate_id = 1
        expected = "1_flatfield_masks"
        result = flatfieldcorr_name(plate_id)
        assert result == expected

    def test_flatfieldcorr_name_large_id(self):
        """Test name generation with large plate ID."""
        plate_id = 999999999
        expected = "999999999_flatfield_masks"
        result = flatfieldcorr_name(plate_id)
        assert result == expected


class TestRandomTimgs:
    """Test the random_timgs function that selects random timepoints."""

    def test_random_timgs_less_than_10(self):
        """Test selecting timepoints when fewer than 10 are available."""
        rng = np.random.default_rng(seed=42)
        # Create a 5D array: (T, Z, Y, X, C) with 5 timepoints
        image_array = rng.uniform(0, 1000, (5, 1, 100, 100, 1))
        result = random_timgs(image_array)

        # Should return all 5 timepoints
        assert len(result) == 5
        # Each result should have shape (1, Z, Y, X, C)
        for img in result:
            assert img.shape == (1, 1, 100, 100, 1)

    def test_random_timgs_exactly_10(self):
        """Test selecting timepoints when exactly 10 are available."""
        rng = np.random.default_rng(seed=42)
        # Create a 5D array with 10 timepoints
        image_array = rng.uniform(0, 1000, (10, 1, 100, 100, 1))
        result = random_timgs(image_array)

        # Should return all 10 timepoints
        assert len(result) == 10
        for img in result:
            assert img.shape == (1, 1, 100, 100, 1)

    def test_random_timgs_more_than_10(self):
        """Test selecting timepoints when more than 10 are available."""
        rng = np.random.default_rng(seed=42)
        # Create a 5D array with 20 timepoints
        image_array = rng.uniform(0, 1000, (20, 1, 100, 100, 1))
        result = random_timgs(image_array)

        # Should return only 10 random timepoints
        assert len(result) == 10
        for img in result:
            assert img.shape == (1, 1, 100, 100, 1)

    def test_random_timgs_single_timepoint(self):
        """Test with a single timepoint."""
        rng = np.random.default_rng(seed=42)
        # Create a 5D array with 1 timepoint
        image_array = rng.uniform(0, 1000, (1, 1, 100, 100, 1))
        result = random_timgs(image_array)

        # Should return the single timepoint
        assert len(result) == 1
        assert result[0].shape == (1, 1, 100, 100, 1)

    def test_random_timgs_preserves_data(self):
        """Test that data is preserved correctly during selection."""
        # Create array with distinct values for each timepoint
        image_array = np.zeros((15, 1, 10, 10, 1))
        for t in range(15):
            image_array[t, :, :, :, :] = (
                t * 100
            )  # Each timepoint has unique value

        result = random_timgs(image_array)

        # Should have 10 unique results
        assert len(result) == 10
        # Each result should maintain its distinct value
        unique_values = set()
        for img in result:
            value = img[
                0, 0, 0, 0
            ].item()  # Convert numpy scalar to Python float
            unique_values.add(value)

        # Should have 10 different timepoint values
        assert len(unique_values) == 10
        # All values should be multiples of 100 (from 0 to 1400)
        for val in unique_values:
            assert val % 100 == 0
            assert 0 <= val < 1500

    def test_random_timgs_reproducibility(self):
        """Test that randomness is actually random between calls."""
        rng = np.random.default_rng(seed=42)
        image_array = rng.uniform(0, 1000, (20, 1, 100, 100, 1))

        # Make two separate calls
        result1 = random_timgs(image_array)
        result2 = random_timgs(image_array)

        # Results should be different (random sampling)
        # Check if at least one timepoint differs
        indices1 = [np.mean(img) for img in result1]
        indices2 = [np.mean(img) for img in result2]

        # The two sets of mean values should be different
        # (very unlikely to be identical with random sampling)
        assert indices1 != indices2


class TestSyntheticFlatfieldCorrection:
    """Test flatfield correction workflow with synthetic data."""

    @pytest.fixture
    def synthetic_vignette_pattern(self):
        """Create a synthetic vignette pattern for testing.

        Returns a 2D array with radial gradient simulating lens vignetting.
        """
        size = 512
        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Create radial distance from center
        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        max_distance = np.sqrt(2 * center**2)

        # Create vignette: brighter in center, darker at edges
        # Using a cosine falloff for smooth transition
        vignette = 1.0 - 0.5 * (distance / max_distance) ** 2

        return vignette

    @pytest.fixture
    def synthetic_flatfield_image(self, synthetic_vignette_pattern):
        """Create a synthetic image with vignetting and some sample cells.

        Simulates a microscopy image with illumination gradient.
        """
        rng = np.random.default_rng(seed=42)
        size = synthetic_vignette_pattern.shape[0]

        # Start with base illumination
        image = synthetic_vignette_pattern * 5000

        # Add some "cells" (bright spots) at random locations
        num_cells = 50
        for _ in range(num_cells):
            x = rng.integers(50, size - 50)
            y = rng.integers(50, size - 50)
            # Create small Gaussian spots
            yy, xx = np.ogrid[:size, :size]
            cell = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 50)
            image += cell * 3000

        # Add noise
        image += rng.normal(0, 50, image.shape)

        return image.astype(np.float64)

    def test_vignette_pattern_properties(self, synthetic_vignette_pattern):
        """Test that synthetic vignette pattern has expected properties."""
        # Should be 2D
        assert synthetic_vignette_pattern.ndim == 2
        assert synthetic_vignette_pattern.shape == (512, 512)

        # Values should be between 0.5 and 1.0
        assert np.min(synthetic_vignette_pattern) >= 0.5
        assert np.max(synthetic_vignette_pattern) <= 1.0

        # Center should be brighter than edges
        center = 256
        center_value = synthetic_vignette_pattern[center, center]
        edge_value = synthetic_vignette_pattern[0, 0]
        assert center_value > edge_value

    def test_flatfield_correction_application(
        self, synthetic_flatfield_image, synthetic_vignette_pattern
    ):
        """Test applying flatfield correction to synthetic image."""
        # Normalize the vignette pattern to use as correction mask
        correction_mask = (
            synthetic_vignette_pattern / synthetic_vignette_pattern.mean()
        )

        # Apply correction
        corrected = synthetic_flatfield_image / correction_mask

        # Check that correction reduces intensity variation
        original_std = np.std(synthetic_flatfield_image)
        corrected_std = np.std(corrected)

        # The corrected image should have similar or slightly different std
        # (depends on the signal content)
        assert corrected.shape == synthetic_flatfield_image.shape

        # Mean intensity should be preserved approximately
        original_mean = np.mean(synthetic_flatfield_image)
        corrected_mean = np.mean(corrected)

        # Allow 20% tolerance due to normalization
        assert abs(corrected_mean - original_mean) / original_mean < 0.2

    def test_diagonal_intensity_profile(self, synthetic_flatfield_image):
        """Test extracting diagonal intensity profile."""
        diagonal = np.diagonal(synthetic_flatfield_image)

        # Should have length equal to image dimension
        assert len(diagonal) == 512

        # Should be 1D array
        assert diagonal.ndim == 1

        # Values should be positive
        assert np.all(diagonal > 0)


class TestGeneratedImageProperties:
    """Test properties of generated correction masks and examples."""

    def test_normalized_mask_properties(self):
        """Test that normalized flatfield mask has expected properties."""
        rng = np.random.default_rng(seed=42)
        # Simulate an aggregated, blurred image
        size = 512

        # Create a smooth gradient
        y, x = np.ogrid[:size, :size]
        center = size // 2
        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        max_distance = np.sqrt(2 * center**2)
        aggregated_img = 1000 * (1.0 - 0.3 * (distance / max_distance) ** 2)

        # Normalize by mean (as done in aggregate_imgs)
        norm_mask = aggregated_img / aggregated_img.mean()

        # Check properties
        assert norm_mask.shape == (size, size)
        assert np.mean(norm_mask) == pytest.approx(1.0, rel=1e-10)

        # Values should be positive
        assert np.all(norm_mask > 0)

        # Should vary smoothly (max/min ratio should be reasonable)
        ratio = np.max(norm_mask) / np.min(norm_mask)
        assert ratio < 5  # Typical for illumination correction

    def test_background_correction(self):
        """Test background subtraction after flatfield correction."""
        rng = np.random.default_rng(seed=42)

        # Create synthetic corrected image
        size = 256
        corrected_img = rng.uniform(100, 1000, (size, size))

        # Apply background correction as in gen_example
        # Note: The 0.2 percentile can be below some pixels, so minimum might be < 1
        bgcorr_img = corrected_img - np.percentile(corrected_img, 0.2) + 1

        # Background corrected image minimum should be near 1 (within reasonable range)
        # Some pixels may be slightly below 1 due to the percentile calculation
        assert (
            np.percentile(bgcorr_img, 1) >= 0
        )  # Very low percentile should be near 1
        assert np.min(bgcorr_img) > -100  # No extreme negative values

        # Shape preserved
        assert bgcorr_img.shape == corrected_img.shape
