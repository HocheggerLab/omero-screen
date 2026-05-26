
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
    CroppedImageParser,
    OmeroData,
    RandomImageParser,
    UserData,
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

class TestCroppedImageParser:
    def test_prepare_data_for_cropping_valid(self, mock_omero_data, mock_user_data):
        # Setup mocks
        mock_user_data.well = "A1"
        mock_user_data.segmentation = "nucleus"
        mock_user_data.cellcycle = "All"

        # Mock well data
        df = pl.DataFrame({
            "well": ["A1", "A1"],
            "image_id": [101, 102],
            "centroid-0-nuc": [10, 20],
            "centroid-1-nuc": [10, 20],
            "label": [1, 2]
        })
        # Important: The code calls _get_well_data which filters plate_data.
        # We can bypass that by setting _well_data directly or mocking filter chain.
        # Let's mock the internal state after _get_well_data would run,
        # OR mock the plate_data so the filter works.
        # Mocking plate_data is cleaner if we can make the LazyFrame logic work.
        # But _get_well_data calls .collect().
        # Let's just set _well_data directly and test _prepare_data_for_cropping directly.

        parser = CroppedImageParser(mock_omero_data, mock_user_data)
        parser._well_data = df

        # Set loaded image IDs in OmeroData
        mock_omero_data.image_ids = [101, 102]

        parser._prepare_data_for_cropping()

        assert parser._image_ids == [101, 102]
        # Centroids are now resolved per-image in _crop_data; verify the
        # stored dataframe carries the right rows for downstream lookup.
        assert set(parser._df["image_id"].to_list()) == {101, 102}
        assert parser._df["label"].to_list() == [1, 2]

    def test_prepare_data_for_cropping_filter_loaded_only(self, mock_omero_data, mock_user_data):
        mock_user_data.well = "A1"
        mock_user_data.segmentation = "nucleus"
        mock_user_data.cellcycle = "All"

        # 3 images in metadata
        df = pl.DataFrame({
            "image_id": [101, 102, 103],
            "centroid-0-nuc": [10, 20, 30],
            "centroid-1-nuc": [10, 20, 30],
            "label": [1, 2, 3],
            "cell_cycle": ["G1", "G1", "G1"]
        })
        parser = CroppedImageParser(mock_omero_data, mock_user_data)
        parser._well_data = df

        # Only 101 and 103 are loaded
        mock_omero_data.image_ids = [101, 103]

        parser._prepare_data_for_cropping()

        # Should filter out 102; _df retains only the loaded images
        assert parser._image_ids == [101, 103]
        assert set(parser._df["image_id"].to_list()) == {101, 103}
        assert 102 not in parser._df["image_id"].to_list()

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

class TestZarrCropDispatch:
    """The _try_zarr_crop_path short-circuit: falls back cleanly when zarr
    is unavailable, and consumes fetch_crop / fetch_label_crop when it is.
    """

    def _make_parser(self, plate_id=1, well="A1", n_channels=2):
        omero_data = MagicMock(spec=OmeroData)
        omero_data.plate_id = plate_id
        omero_data.intensities = {i: (0, 1000) for i in range(n_channels)}
        omero_data.image_ids = [555]
        omero_data.images = []
        omero_data.labels = []
        omero_data.cropped_images = []
        omero_data.cropped_labels = []
        ud = UserData()
        ud.well = well
        ud.crop_size = 32
        ud.timepoint = 0
        ud.segmentation = "nucleus"
        parser = CroppedImageParser(omero_data, ud)
        parser._centroids_row = [100]
        parser._centroids_col = [100]
        parser._ids = [7]
        return parser

    def test_falls_back_when_no_zarr(self, monkeypatch):
        parser = self._make_parser()
        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.resolve_to_zarr", lambda _p: None
        )
        assert parser._try_zarr_crop_path(555) is False

    def test_falls_back_when_intensities_empty(self, monkeypatch):
        parser = self._make_parser()
        parser._omero_data.intensities = {}
        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.resolve_to_zarr",
            lambda _p: MagicMock(),
        )
        assert parser._try_zarr_crop_path(555) is False

    def test_zarr_path_produces_crops(self, monkeypatch):
        parser = self._make_parser(n_channels=2)
        # Fake crops: (C, Y, X) image, (Y, X) labels containing target id 7.
        fake_image = np.full((2, 32, 32), 500, dtype=np.uint16)
        fake_label = np.zeros((32, 32), dtype=np.uint32)
        fake_label[10:20, 10:20] = 7

        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.resolve_to_zarr",
            lambda _p: MagicMock(),
        )
        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.fetch_crop",
            lambda *a, **k: fake_image,
        )
        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.fetch_label_crop",
            lambda *a, **k: fake_label,
        )

        assert parser._try_zarr_crop_path(555) is True
        assert len(parser._cropped_images) == 1
        img = parser._cropped_images[0]
        # Y, X, C output, scaled to float32 [0, 1].
        assert img.shape == (32, 32, 2)
        assert img.dtype == np.float32
        # 500 / max(1000-0, 1) = 0.5
        assert abs(float(img.mean()) - 0.5) < 1e-3
        # Label crop isolated to id 7.
        lbl = parser._cropped_labels[0]
        assert int(lbl[15, 15]) == 7
        assert int(lbl[0, 0]) == 0

    def test_zarr_path_falls_back_when_fetch_raises(self, monkeypatch):
        parser = self._make_parser()
        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.resolve_to_zarr",
            lambda _p: MagicMock(),
        )

        def boom(*a, **k):
            raise FileNotFoundError("store evicted")

        monkeypatch.setattr(
            "omero_screen_napari.gallery_api.fetch_crop", boom
        )
        assert parser._try_zarr_crop_path(555) is False
        assert parser._cropped_images == []
