"""Unit tests for direct OMERO loader functionality."""

import pytest
from omero_screen_napari.direct_omero_loader import (
    _parse_image_input,
    _resolve_n_crops,
    metadata_default_n_crops,
)


class TestResolveNCrops:
    """Test how the crop count is resolved from request + metadata."""

    def test_explicit_n_crops_overrides_metadata(self):
        """A user-supplied count wins over the metadata default."""
        metadata = {"n_crops": 25, "user_data": {"rows": 5, "columns": 5}}

        assert _resolve_n_crops(60, metadata) == 60

    def test_none_falls_back_to_metadata_n_crops(self):
        """An empty field uses the metadata's n_crops."""
        metadata = {"n_crops": 25, "user_data": {"rows": 3, "columns": 3}}

        assert _resolve_n_crops(None, metadata) == 25

    def test_none_falls_back_to_gallery_grid(self):
        """Without n_crops, the default is the gallery rows * columns."""
        metadata = {"user_data": {"rows": 4, "columns": 5}}

        assert _resolve_n_crops(None, metadata) == 20

    def test_none_without_metadata_means_no_cap(self):
        """Metadata that constrains nothing loads every crop."""
        assert _resolve_n_crops(None, {"user_data": {}}) == 0

    def test_zero_request_means_no_cap(self):
        """An explicit 0 loads every crop, ignoring the metadata."""
        metadata = {"n_crops": 25, "user_data": {}}

        assert _resolve_n_crops(0, metadata) == 0

    def test_negative_request_is_clamped_to_no_cap(self):
        """A negative count is clamped rather than sampled against."""
        assert _resolve_n_crops(-5, {"n_crops": 25, "user_data": {}}) == 0

    def test_metadata_default_prefers_n_crops_over_grid(self):
        """n_crops takes precedence over the gallery grid."""
        metadata = {"n_crops": 12, "user_data": {"rows": 5, "columns": 5}}

        assert metadata_default_n_crops(metadata) == 12

    def test_metadata_default_ignores_partial_grid(self):
        """A grid with a zero dimension yields no default."""
        metadata = {"user_data": {"rows": 4, "columns": 0}}

        assert metadata_default_n_crops(metadata) == 0


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
