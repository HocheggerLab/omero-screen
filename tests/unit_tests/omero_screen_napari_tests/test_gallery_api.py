
import numpy as np
import pytest
from omero_screen_napari.gallery_api import (
    calculate_crop_coordinates,
    erase_masks,
    fill_missing_channels,
    pad_region,
)


class TestGalleryHelpers:
    def test_calculate_crop_coordinates(self):
        # crop_size=10, centroid=50 -> 45 to 55
        start, end = calculate_crop_coordinates(50, 100, 10)
        assert start == 45
        assert end == 55

    def test_calculate_crop_coordinates_min_boundary(self):
        # crop_size=10, centroid=2 -> 0 to 7
        start, end = calculate_crop_coordinates(2, 100, 10)
        assert start == 0
        assert end == 7

    def test_calculate_crop_coordinates_max_boundary(self):
        # crop_size=10, centroid=98 -> 93 to 100
        start, end = calculate_crop_coordinates(98, 100, 10)
        assert start == 93
        assert end == 100

    def test_pad_region(self):
        # 8x8 image, crop_size=10 -> pad to 10x10
        img = np.zeros((8, 8, 3))
        lbl = np.zeros((8, 8))
        padded_img, padded_lbl = pad_region(img, lbl, 10)
        assert padded_img.shape == (10, 10, 3)
        assert padded_lbl.shape == (10, 10)

    def test_erase_masks_single_id(self):
        # Label with 1 and 2. Keep 1.
        label = np.array([[1, 1], [2, 2]])
        results = erase_masks(label, 1)
        assert len(results) == 1
        assert np.array_equal(results[0], np.array([[1, 1], [0, 0]]))

    def test_erase_masks_list_ids(self):
        # Label with 1 and 2. Keep both.
        label = np.array([[1, 1], [2, 2]])
        results = erase_masks(label, [1, 2])
        assert len(results) == 2
        # Order depends on iteration, but should find both masks
        found_1 = any(np.array_equal(r, np.array([[1, 1], [0, 0]])) for r in results)
        found_2 = any(np.array_equal(r, np.array([[0, 0], [2, 2]])) for r in results)
        assert found_1 and found_2

    def test_erase_masks_float_tolerance(self):
        # Simulate legacy float handling with centroid distance
        # 10x10 array, center is (5,5)
        label = np.zeros((10, 10), dtype=int)
        # Add a blob near center
        label[4:6, 4:6] = 1 # Centroid ~ (4.5, 4.5)
        # Add a blob far away
        label[0:2, 0:2] = 2 # Centroid ~ (0.5, 0.5)

        # Passing a float ID triggers the centroid distance logic
        results = erase_masks(label, 1.0)
        # Should only keep the central one
        assert len(results) == 1
        assert np.sum(results[0] == 1) > 0
        assert np.sum(results[0] == 2) == 0

    def test_fill_missing_channels_3_channels(self):
        # 3 channels: [ch0, ch1, ch2] -> RGB direct mapping
        img = np.zeros((10, 10, 5))
        img[..., 0] = 1
        img[..., 1] = 2
        img[..., 2] = 3

        indices = [0, 1, 2]
        res = fill_missing_channels(img, indices)

        assert res.shape == (10, 10, 3)
        assert res[0, 0, 0] == 1  # Red = ch0
        assert res[0, 0, 1] == 2  # Green = ch1
        assert res[0, 0, 2] == 3  # Blue = ch2

    def test_fill_missing_channels_missing(self):
        # 2 channels: [ch0, ch1] -> [Red=ch0, Green=ch1, Blue=0]
        img = np.zeros((10, 10, 5))
        img[..., 0] = 1
        img[..., 1] = 2

        indices = [0, 1]
        res = fill_missing_channels(img, indices)

        assert res.shape == (10, 10, 3)
        assert res[0, 0, 0] == 1  # Red = ch0
        assert res[0, 0, 1] == 2  # Green = ch1
        assert res[0, 0, 2] == 0  # Blue = empty

from unittest.mock import MagicMock

import polars as pl
from omero_screen_napari.gallery_api import (
    OmeroData,
    RandomImageParser,
    UserData,
    _filter_well_centroids,
)


@pytest.fixture
def mock_omero_data():
    mock = MagicMock(spec=OmeroData)
    mock.plate_data = pl.DataFrame()
    mock.image_ids = []
    mock.images = []
    mock.labels = []
    mock.cropped_images = []
    mock.cropped_labels = []
    return mock

@pytest.fixture
def mock_user_data():
    return UserData()

class TestFilterWellCentroids:
    """_filter_well_centroids ports the old CroppedImageParser polars
    filtering: well → cellcycle → classifier → loaded-image intersection →
    timepoint, returning a pandas DataFrame for CropPipeline.
    """

    def test_returns_loaded_well_rows(self, mock_omero_data, mock_user_data):
        mock_user_data.well = "A1"
        mock_user_data.segmentation = "nucleus"
        mock_user_data.cellcycle = "All"

        df = pl.DataFrame({
            "well": ["A1", "A1"],
            "image_id": [101, 102],
            "centroid-0-nuc": [10, 20],
            "centroid-1-nuc": [10, 20],
            "label": [1, 2],
        })
        mock_omero_data.plate_data = df.lazy()
        mock_omero_data.image_ids = [101, 102]

        result = _filter_well_centroids(mock_omero_data, mock_user_data)

        assert set(result["image_id"].tolist()) == {101, 102}
        assert result["label"].tolist() == [1, 2]

    def test_filters_to_loaded_images_only(self, mock_omero_data, mock_user_data):
        mock_user_data.well = "A1"
        mock_user_data.segmentation = "nucleus"
        mock_user_data.cellcycle = "All"

        df = pl.DataFrame({
            "well": ["A1", "A1", "A1"],
            "image_id": [101, 102, 103],
            "centroid-0-nuc": [10, 20, 30],
            "centroid-1-nuc": [10, 20, 30],
            "label": [1, 2, 3],
            "cell_cycle": ["G1", "G1", "G1"],
        })
        mock_omero_data.plate_data = df.lazy()
        # Only 101 and 103 are loaded
        mock_omero_data.image_ids = [101, 103]

        result = _filter_well_centroids(mock_omero_data, mock_user_data)

        # 102 filtered out (not loaded)
        assert set(result["image_id"].tolist()) == {101, 103}
        assert 102 not in result["image_id"].tolist()

class TestRandomImageParser:
    def test_parse_random_index_all(self, mock_omero_data, mock_user_data):
        parser = RandomImageParser(mock_omero_data, mock_user_data, False)

        # No rows/cols specified -> select all
        mock_user_data.rows = 0
        mock_user_data.columns = 0

        mock_omero_data.cropped_images = [1, 2, 3, 4, 5] # Dummy list

        parser._parse_random_index()
        assert len(parser._chosen_indices) == 5
        assert sorted(parser._chosen_indices) == [0, 1, 2, 3, 4]

    def test_parse_random_index_subset(self, mock_omero_data, mock_user_data):
        parser = RandomImageParser(mock_omero_data, mock_user_data, False)

        # 2x2 grid -> 4 images
        mock_user_data.rows = 2
        mock_user_data.columns = 2

        # 10 available images
        mock_omero_data.cropped_images = list(range(10))

        parser._parse_random_index()
        assert len(parser._chosen_indices) == 4
        # Check uniqueness
        assert len(set(parser._chosen_indices)) == 4

    def test_remove_chosen_crops(self, mock_omero_data, mock_user_data):
        parser = RandomImageParser(mock_omero_data, mock_user_data, False)

        images = ["a", "b", "c", "d", "e"]
        parser._chosen_indices = [1, 3] # remove b and d

        remaining = parser._remove_chosen_crops(images)
        assert remaining == ["a", "c", "e"]


class TestChannelResolution:
    """Regression tests for gallery RGB-slot resolution.

    Guards the bug where a single-channel selection (which must render in
    grayscale) was silently expanded to a 3-channel RGB image by auto-filling
    the blank slots — e.g. DAPI -> [DAPI, EdU, DAPI] (magenta nucleus + green).
    """

    available = ["DAPI", "Tub", "H2AX", "EdU"]

    def test_single_channel_stays_single(self):
        # The regression: picking only DAPI must NOT pull in other channels.
        from omero_screen_napari._gallery_widget import _resolve_channels

        assert _resolve_channels("DAPI", "", "", self.available) == ["DAPI"]

    def test_single_channel_renders_grayscale(self):
        # A single resolved channel must produce a (H, W, 1) grayscale image.
        from omero_screen_napari._gallery_widget import _resolve_channels

        channels = _resolve_channels("DAPI", "", "", self.available)
        idx = [self.available.index(c) for c in channels]
        img = np.zeros((8, 8, len(self.available)), dtype=np.uint8)
        out = fill_missing_channels(img, idx)
        assert out.shape == (8, 8, 1)

    def test_two_channels_preserved_in_order(self):
        from omero_screen_napari._gallery_widget import _resolve_channels

        assert _resolve_channels("DAPI", "EdU", "", self.available) == [
            "DAPI",
            "EdU",
        ]

    def test_stale_value_is_dropped(self):
        # A value not on the plate is ignored, not forced into the list.
        from omero_screen_napari._gallery_widget import _resolve_channels

        assert _resolve_channels("RFP", "", "", self.available) == [
            "EdU",
            "DAPI",
        ]  # falls back to auto-pick (no red hint -> EdU green, DAPI blue)

    def test_all_blank_falls_back_to_auto_defaults(self):
        from omero_screen_napari._gallery_widget import _resolve_channels

        channels = _resolve_channels("", "", "", self.available)
        assert channels  # non-empty
        assert all(c in self.available for c in channels)
