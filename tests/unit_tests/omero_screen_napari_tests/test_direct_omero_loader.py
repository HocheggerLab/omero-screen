"""Unit tests for direct OMERO loader functionality."""

import numpy as np
import pytest

from omero_screen_napari.direct_omero_loader import (
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
