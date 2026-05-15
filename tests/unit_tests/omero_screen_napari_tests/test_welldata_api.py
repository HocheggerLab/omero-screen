
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.welldata_api import (
    ImageParser,
    PixelSizeParser,
    ScaleIntensityParser,
    UserInput,
)


@pytest.fixture
def mock_omero_data():
    mock = MagicMock(spec=OmeroData)
    mock.plate_data = None
    mock.plate = MagicMock()
    return mock

@pytest.fixture
def mock_conn():
    return MagicMock()

class TestUserInput:
    def test_well_data_parser_valid(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1, B2", "All", mock_conn)
        user_input._well_data_parser()
        assert user_input._well_pos_list == ["A1", "B2"]

    def test_well_data_parser_invalid(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "Invalid", "All", mock_conn)
        with pytest.raises(ValueError, match="Invalid well input format"):
            user_input._well_data_parser()

    def test_image_index_parser_all(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "All", mock_conn)
        user_input._image_number = 5
        user_input._image_index_parser()
        assert user_input._image_index == [0, 1, 2, 3, 4]

    def test_image_index_parser_range(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "1-3", mock_conn)
        user_input._image_index_parser()
        assert user_input._image_index == [1, 2, 3]

    def test_image_index_parser_list(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "1, 3, 5", mock_conn)
        user_input._image_index_parser()
        assert user_input._image_index == [1, 3, 5]

    def test_image_index_parser_single(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "2", mock_conn)
        user_input._image_index_parser()
        assert user_input._image_index == [2]

    def test_image_index_parser_invalid(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "invalid", mock_conn)
        with pytest.raises(ValueError, match="Image input 'invalid' doesn't match"):
            user_input._image_index_parser()

    def test_image_time_parser_all(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "All", mock_conn)
        user_input._image_time_parser()
        assert mock_omero_data.crop_start == ()
        assert mock_omero_data.crop_length == ()

    def test_image_time_parser_range(self, mock_omero_data, mock_conn):
        user_input = UserInput(mock_omero_data, 123, "A1", "All", mock_conn, time="1-3")

        # Mock plate and image to return dimensions
        mock_plate = MagicMock()
        mock_well = MagicMock()
        mock_image = MagicMock()
        mock_image.getSizeX.return_value = 100
        mock_image.getSizeY.return_value = 100
        mock_image.getSizeZ.return_value = 1
        mock_image.getSizeC.return_value = 3
        mock_image.getSizeT.return_value = 10

        mock_well.getImage.return_value = mock_image
        mock_plate.listChildren.return_value = [mock_well]
        user_input._plate = mock_plate

        user_input._image_time_parser()

        # Start is 0-indexed, so 1 becomes 0. Length is 3-1 = 2? No, range is inclusive in parser logic?
        # Looking at code: start, end = map(int, time.split("-"))
        # start -= 1
        # length = end - start
        # if time is "1-3", start=1, end=3. start becomes 0. length = 3 - 0 = 3.
        # crop_start = (0, 0, 0, 0, 0)
        # crop_length = (100, 100, 1, 3, 3)

        assert mock_omero_data.crop_start == (0, 0, 0, 0, 0)
        assert mock_omero_data.crop_length == (100, 100, 1, 3, 3)

class TestScaleIntensityParser:
    def test_get_values_with_cell_columns(self, mock_omero_data):
        """Test that parser prefers cell columns when available and non-null."""
        mock_omero_data.channel_data = {"DAPI": 0}
        parser = ScaleIntensityParser(mock_omero_data)

        # Create a DataFrame with cell columns
        df = pl.DataFrame({
            "intensity_max_DAPI_cell": [100, 200],
            "intensity_min_DAPI_cell": [10, 20]
        })
        parser._plate_data = df.lazy()

        parser._get_values()

        # Mean of max: (100+200)/2 = 150
        # Min of min: 10
        assert parser._intensities == {0: (10, 150)}

    def test_get_values_with_nucleus_columns(self, mock_omero_data):
        """Test that parser uses nucleus columns when cell columns not available."""
        mock_omero_data.channel_data = {"DAPI": 0}
        parser = ScaleIntensityParser(mock_omero_data)

        # Create a DataFrame with only nucleus columns
        df = pl.DataFrame({
            "intensity_max_DAPI_nucleus": [100, 200],
            "intensity_min_DAPI_nucleus": [10, 20]
        })
        parser._plate_data = df.lazy()

        parser._get_values()

        # Mean of max: (100+200)/2 = 150
        # Min of min: 10
        assert parser._intensities == {0: (10, 150)}

    def test_get_values_fallback_to_nucleus(self, mock_omero_data):
        """Test that parser falls back to nucleus when cell columns contain only nulls."""
        mock_omero_data.channel_data = {"DAPI": 0}
        parser = ScaleIntensityParser(mock_omero_data)

        # Create a DataFrame with null cell columns and valid nucleus columns
        df = pl.DataFrame({
            "intensity_max_DAPI_cell": [None, None],
            "intensity_min_DAPI_cell": [None, None],
            "intensity_max_DAPI_nucleus": [100, 200],
            "intensity_min_DAPI_nucleus": [10, 20]
        })
        parser._plate_data = df.lazy()

        parser._get_values()

        # Should use nucleus columns
        assert parser._intensities == {0: (10, 150)}

    def test_get_values_missing_columns_raises_error(self, mock_omero_data):
        """Test that parser raises ValueError when neither cell nor nucleus columns exist."""
        mock_omero_data.channel_data = {"DAPI": 0}
        parser = ScaleIntensityParser(mock_omero_data)

        # Create a DataFrame without the required columns
        df = pl.DataFrame({
            "intensity_max_DAPI": [100, 200],
            "some_other_column": [1, 2]
        })
        parser._plate_data = df.lazy()

        with pytest.raises(ValueError, match="Neither cell nor nucleus intensity columns found"):
            parser._get_values()

class TestPixelSizeParser:
    def test_check_pixel_values_matching(self, mock_omero_data):
        parser = PixelSizeParser(mock_omero_data)

        mock_image1 = MagicMock()
        mock_image1.getPixelSizeX.return_value = 0.5
        mock_image1.getPixelSizeY.return_value = 0.5

        mock_image2 = MagicMock()
        mock_image2.getPixelSizeX.return_value = 0.5
        mock_image2.getPixelSizeY.return_value = 0.5

        parser._random_images = [mock_image1, mock_image2]

        parser._check_pixel_values()
        assert parser._pixel_size == (0.5, 0.5)

    def test_check_pixel_values_mismatch(self, mock_omero_data):
        parser = PixelSizeParser(mock_omero_data)

        mock_image1 = MagicMock()
        mock_image1.getPixelSizeX.return_value = 0.5
        mock_image1.getPixelSizeY.return_value = 0.5

        mock_image2 = MagicMock()
        mock_image2.getPixelSizeX.return_value = 0.6
        mock_image2.getPixelSizeY.return_value = 0.6

        parser._random_images = [mock_image1, mock_image2]

        with pytest.raises(ValueError, match="Pixel sizes are not identical between wells"):
            parser._check_pixel_values()

    def test_check_pixel_values_zero(self, mock_omero_data):
        parser = PixelSizeParser(mock_omero_data)

        mock_image1 = MagicMock()
        mock_image1.getPixelSizeX.return_value = 0.0
        mock_image1.getPixelSizeY.return_value = 0.5

        mock_image2 = MagicMock()
        mock_image2.getPixelSizeX.return_value = 0.5
        mock_image2.getPixelSizeY.return_value = 0.5

        parser._random_images = [mock_image1, mock_image2]

        with pytest.raises(ValueError, match="One of the pixel sizes is 0"):
            parser._check_pixel_values()


def _make_mask_child(name: str, image_id: int = 999):
    """Build a mock OMERO mask image with a given name and ID."""
    m = MagicMock()
    m.getName.return_value = name
    m.getId.return_value = image_id
    return m


def _make_image_parser_with_dataset(
    omero_data: OmeroData,
    dataset_children: list,
    image_ids: list[int],
) -> ImageParser:
    """Construct an ImageParser wired to a mocked screen_dataset."""
    omero_data.screen_dataset = MagicMock()
    omero_data.screen_dataset.listChildren.return_value = iter(dataset_children)
    omero_data.crop_start = ()
    omero_data.crop_length = ()
    omero_data.plate_id = 0
    parser = ImageParser.__new__(ImageParser)
    parser._omero_data = omero_data
    parser._image_ids = image_ids
    parser._label_arrays = []
    return parser


class TestCollectLabelsStitchedDetection:
    """``_collect_labels`` sets ``omero_data.label_stitched_mode`` from mask names."""

    def test_sets_flag_true_for_stitched_masks(self):
        omero = OmeroData()
        children = [_make_mask_child(f"{img_id}_stitched_segmentation")
                    for img_id in (10, 11)]
        parser = _make_image_parser_with_dataset(
            omero, dataset_children=children, image_ids=[10, 11]
        )

        with patch(
            "omero_screen_napari.welldata_api.get_image",
            return_value=np.zeros((1, 1, 16, 16, 1), dtype=np.uint16),
        ):
            parser._collect_labels()

        assert omero.label_stitched_mode is True

    def test_sets_flag_false_for_legacy_masks(self):
        omero = OmeroData()
        # Pre-set to True to make sure _collect_labels resets it.
        omero.label_stitched_mode = True
        children = [_make_mask_child(f"{img_id}_segmentation")
                    for img_id in (10, 11)]
        parser = _make_image_parser_with_dataset(
            omero, dataset_children=children, image_ids=[10, 11]
        )

        with patch(
            "omero_screen_napari.welldata_api.get_image",
            return_value=np.zeros((1, 1, 16, 16, 1), dtype=np.uint16),
        ):
            parser._collect_labels()

        assert omero.label_stitched_mode is False

    def test_mixed_masks_prefer_stitched(self):
        """If a plate somehow has both, stitched wins and the flag is True."""
        omero = OmeroData()
        children = [
            _make_mask_child("10_segmentation"),
            _make_mask_child("10_stitched_segmentation"),
            _make_mask_child("11_stitched_segmentation"),
        ]
        parser = _make_image_parser_with_dataset(
            omero, dataset_children=children, image_ids=[10, 11]
        )

        with patch(
            "omero_screen_napari.welldata_api.get_image",
            return_value=np.zeros((1, 1, 16, 16, 1), dtype=np.uint16),
        ):
            parser._collect_labels()

        assert omero.label_stitched_mode is True
