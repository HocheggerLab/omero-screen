"""Unit tests for direct OMERO loader functionality."""

import numpy as np
import pytest

from omero_screen_napari.direct_omero_loader import (
    _generate_crops,
    _parse_image_input,
)


class TestParseImageInput:
    """Test image input parsing function."""

    def test_parse_all_uppercase(self):
        """Test parsing 'All' returns all indices."""
        result = _parse_image_input("All", num_images=5)

        assert result == [0, 1, 2, 3, 4]

    def test_parse_all_lowercase(self):
        """Test parsing 'all' returns all indices."""
        result = _parse_image_input("all", num_images=5)

        assert result == [0, 1, 2, 3, 4]

    def test_parse_all_mixed_case(self):
        """Test parsing 'AlL' returns all indices."""
        result = _parse_image_input("AlL", num_images=3)

        assert result == [0, 1, 2]

    def test_parse_all_with_whitespace(self):
        """Test parsing '  All  ' with surrounding whitespace."""
        result = _parse_image_input("  All  ", num_images=5)

        assert result == [0, 1, 2, 3, 4]

    def test_parse_single_index(self):
        """Test parsing single index '0'."""
        result = _parse_image_input("0", num_images=3)

        assert result == [0]

    def test_parse_single_middle_index(self):
        """Test parsing single middle index '2'."""
        result = _parse_image_input("2", num_images=5)

        assert result == [2]

    def test_parse_single_last_index(self):
        """Test parsing single last valid index."""
        result = _parse_image_input("4", num_images=5)

        assert result == [4]

    def test_parse_comma_separated_with_spaces(self):
        """Test parsing '0, 1, 2' with spaces."""
        result = _parse_image_input("0, 1, 2", num_images=5)

        assert result == [0, 1, 2]

    def test_parse_comma_separated_without_spaces(self):
        """Test parsing '0,1,2' without spaces."""
        result = _parse_image_input("0,1,2", num_images=5)

        assert result == [0, 1, 2]

    def test_parse_comma_separated_mixed_spacing(self):
        """Test parsing '0,1, 2,  3' with mixed spacing."""
        result = _parse_image_input("0,1, 2,  3", num_images=10)

        assert result == [0, 1, 2, 3]

    def test_parse_range(self):
        """Test parsing range '3-5'."""
        result = _parse_image_input("3-5", num_images=10)

        assert result == [3, 4, 5]

    def test_parse_range_start_to_start(self):
        """Test parsing range '0-0'."""
        result = _parse_image_input("0-0", num_images=5)

        assert result == [0]

    def test_parse_range_full(self):
        """Test parsing range covering all images."""
        result = _parse_image_input("0-4", num_images=5)

        assert result == [0, 1, 2, 3, 4]

    def test_parse_duplicates_are_deduped_and_sorted(self):
        """Test parsing '1, 1, 2, 0' removes duplicates and sorts."""
        result = _parse_image_input("1, 1, 2, 0", num_images=5)

        assert result == [0, 1, 2]

    def test_parse_unordered_list_is_sorted(self):
        """Test parsing '3, 1, 2' returns sorted list."""
        result = _parse_image_input("3, 1, 2", num_images=5)

        assert result == [1, 2, 3]

    def test_parse_invalid_text_raises_value_error(self):
        """Test parsing 'abc' raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("abc", num_images=5)

    def test_parse_empty_string_raises_value_error(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("", num_images=5)

    def test_parse_whitespace_only_raises_value_error(self):
        """Test parsing whitespace-only string raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("   ", num_images=5)

    def test_parse_out_of_range_index_raises_value_error(self):
        """Test parsing index beyond num_images raises ValueError."""
        with pytest.raises(ValueError, match="Image index 5 out of range"):
            _parse_image_input("5", num_images=3)

    def test_parse_out_of_range_in_list_raises_value_error(self):
        """Test parsing list with out-of-range index raises ValueError."""
        with pytest.raises(ValueError, match="Image index 10 out of range"):
            _parse_image_input("0, 1, 10", num_images=5)

    def test_parse_range_end_out_of_range_raises_value_error(self):
        """Test parsing range with end beyond num_images raises ValueError."""
        with pytest.raises(ValueError, match="Image index .* out of range"):
            _parse_image_input("0-10", num_images=5)

    def test_parse_negative_index_raises_value_error(self):
        """Test parsing negative index raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("-1", num_images=5)

    def test_parse_range_with_negative_raises_value_error(self):
        """Test parsing range with negative start raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("-1-5", num_images=10)

    def test_parse_mixed_range_and_comma_raises_value_error(self):
        """Test parsing mixed format '0-2, 4-5' raises ValueError."""
        # The regex actually DOES match this pattern, but the parsing logic
        # doesn't handle it (tries to parse "0-2" as an int)
        with pytest.raises(ValueError, match="invalid literal for int"):
            _parse_image_input("0-2, 4-5", num_images=10)

    def test_parse_special_characters_raises_value_error(self):
        """Test parsing input with special characters raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("0; 1; 2", num_images=5)

    def test_parse_range_with_spaces_raises_value_error(self):
        """Test parsing range with spaces '0 - 5' raises ValueError."""
        with pytest.raises(
            ValueError, match="doesn't match any of the expected patterns"
        ):
            _parse_image_input("0 - 5", num_images=10)


class TestGenerateCrops:
    """Test crop generation function."""

    def test_generate_crops_nucleus_segmentation(self):
        """Test generating crops with nucleus segmentation."""
        # Create 50x50x3 test image
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        # Create 50x50x2 label array (nucleus=channel 0, cell=channel 1)
        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        # Add a nucleus at center
        labels[20:30, 20:30, 0] = 1  # Nucleus channel

        # Centroid at (25, 25) - center of the label
        centroids = np.array([[25, 25]])

        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)
        assert crop_labels[0].shape == (10, 10)
        assert np.all(crops[0] == 100)
        # Check that label was extracted from nucleus channel
        assert np.any(crop_labels[0] == 1)

    def test_generate_crops_cell_segmentation(self):
        """Test generating crops with cell segmentation."""
        # Create 50x50x3 test image
        image = np.ones((50, 50, 3), dtype=np.uint8) * 200

        # Create 50x50x2 label array
        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        # Add a cell at center in channel 1
        labels[20:30, 20:30, 1] = 1  # Cell channel

        centroids = np.array([[25, 25]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="cell"
        )

        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)
        assert crop_labels[0].shape == (10, 10)
        # Check that label was extracted from cell channel
        assert np.any(crop_labels[0] == 1)

    def test_generate_crops_multiple_centroids(self):
        """Test generating crops from multiple centroids."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 150

        # Create labels with 3 separate regions
        labels = np.zeros((100, 100, 2), dtype=np.uint8)
        labels[10:20, 10:20, 0] = 1
        labels[40:50, 40:50, 0] = 2
        labels[70:80, 70:80, 0] = 3

        # Centroids at center of each region
        centroids = np.array([[15, 15], [45, 45], [75, 75]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        assert len(crops) == 3
        for crop in crops:
            assert crop.shape == (10, 10, 3)
        for label in crop_labels:
            assert label.shape == (10, 10)
            assert np.any(label > 0)

    def test_generate_crops_edge_centroid_needs_padding(self):
        """Test generating crop at edge requires padding."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        # Place label near edge
        labels[0:5, 0:5, 0] = 1

        # Centroid at (2, 2) - near corner
        centroids = np.array([[2, 2]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Should still generate a crop (padded to 10x10)
        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)
        assert crop_labels[0].shape == (10, 10)

    def test_generate_crops_empty_label_at_centroid_excluded(self):
        """Test that centroids with empty labels are excluded."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        # Add label at one location
        labels[10:20, 10:20, 0] = 1

        # But centroid points to empty region
        centroids = np.array([[35, 35]])  # No label here
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Empty label crop should be excluded
        assert len(crops) == 0
        assert len(crop_labels) == 0

    def test_generate_crops_mixed_empty_and_valid(self):
        """Test that only valid (non-empty) crops are returned."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100

        labels = np.zeros((100, 100, 2), dtype=np.uint8)
        # Only one region has label
        labels[40:50, 40:50, 0] = 1

        # Three centroids, only middle one has label
        centroids = np.array([[10, 10], [45, 45], [80, 80]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Only the middle centroid should generate a crop
        assert len(crops) == 1
        assert len(crop_labels) == 1

    def test_generate_crops_2d_labels_squeezed(self):
        """Test that 2D labels (no channel dim) are handled correctly."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        # 2D label array (will be squeezed)
        labels = np.zeros((50, 50), dtype=np.uint8)
        labels[20:30, 20:30] = 1

        centroids = np.array([[25, 25]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)
        assert crop_labels[0].shape == (10, 10)
        assert np.any(crop_labels[0] == 1)

    def test_generate_crops_3d_labels_single_channel_squeezed(self):
        """Test that 3D labels with single channel are squeezed."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        # 3D with single channel (shape: 50, 50, 1)
        labels = np.zeros((50, 50, 1), dtype=np.uint8)
        labels[20:30, 20:30, 0] = 1

        centroids = np.array([[25, 25]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)
        assert crop_labels[0].shape == (10, 10)

    def test_generate_crops_no_centroids_returns_empty(self):
        """Test that no centroids returns empty lists."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        labels[20:30, 20:30, 0] = 1

        centroids = np.array([]).reshape(0, 2)  # Empty array with correct shape
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        assert len(crops) == 0
        assert len(crop_labels) == 0

    def test_generate_crops_large_crop_size(self):
        """Test generating crops with large crop size."""
        # Small image
        image = np.ones((20, 20, 3), dtype=np.uint8) * 100

        labels = np.zeros((20, 20, 2), dtype=np.uint8)
        labels[5:15, 5:15, 0] = 1

        centroids = np.array([[10, 10]])
        crop_size = 30  # Larger than image

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Should pad to requested size
        assert len(crops) == 1
        assert crops[0].shape == (30, 30, 3)
        assert crop_labels[0].shape == (30, 30)

    def test_generate_crops_centroid_coordinates_rounded(self):
        """Test that float centroids are converted to int coordinates."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        labels[20:30, 20:30, 0] = 1

        # Float centroids
        centroids = np.array([[25.7, 25.3]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Should handle float coordinates by converting to int
        assert len(crops) == 1
        assert crops[0].shape == (10, 10, 3)

    def test_generate_crops_preserves_image_values(self):
        """Test that crop values match original image values."""
        # Create image with specific values
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[20:30, 20:30, 0] = 255  # Red channel
        image[20:30, 20:30, 1] = 128  # Green channel
        image[20:30, 20:30, 2] = 64  # Blue channel

        labels = np.zeros((50, 50, 2), dtype=np.uint8)
        labels[20:30, 20:30, 0] = 1

        centroids = np.array([[25, 25]])
        crop_size = 10

        crops, crop_labels = _generate_crops(
            image, labels, centroids, crop_size, segmentation="nucleus"
        )

        # Check that RGB values are preserved
        assert len(crops) == 1
        assert np.any(crops[0][:, :, 0] == 255)  # Red
        assert np.any(crops[0][:, :, 1] == 128)  # Green
        assert np.any(crops[0][:, :, 2] == 64)  # Blue
