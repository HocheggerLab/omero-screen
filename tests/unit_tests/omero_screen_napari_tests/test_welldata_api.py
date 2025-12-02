
import pytest
from unittest.mock import MagicMock, patch
import polars as pl
from omero_screen_napari.welldata_api import UserInput, ScaleIntensityParser, PixelSizeParser
from omero_screen_napari.omero_data import OmeroData

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
    def test_set_keyword_cell(self, mock_omero_data):
        parser = ScaleIntensityParser(mock_omero_data)
        parser._plate_data = pl.LazyFrame({"intensity_max_DAPI_cell": [1]})
        parser._set_keyword()
        assert parser._keyword == "_cell"

    def test_set_keyword_nucleus(self, mock_omero_data):
        parser = ScaleIntensityParser(mock_omero_data)
        parser._plate_data = pl.LazyFrame({"intensity_max_DAPI_nucleus": [1]})
        parser._set_keyword()
        assert parser._keyword == "_nucleus"

    def test_set_keyword_invalid(self, mock_omero_data):
        parser = ScaleIntensityParser(mock_omero_data)
        parser._plate_data = pl.LazyFrame({"intensity_max_DAPI": [1]})
        with pytest.raises(ValueError, match="Neither '_cell' nor '_nucleus' is present"):
            parser._set_keyword()

    def test_get_values(self, mock_omero_data):
        mock_omero_data.channel_data = {"DAPI": 0}
        parser = ScaleIntensityParser(mock_omero_data)
        parser._keyword = "_cell"

        # Create a DataFrame with necessary columns
        df = pl.DataFrame({
            "intensity_max_DAPI_cell": [100, 200],
            "intensity_min_DAPI_cell": [10, 20]
        })
        parser._plate_data = df.lazy()

        parser._get_values()

        # Mean of max: (100+200)/2 = 150
        # Min of min: 10
        assert parser._intensities == {0: (10, 150)}

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
