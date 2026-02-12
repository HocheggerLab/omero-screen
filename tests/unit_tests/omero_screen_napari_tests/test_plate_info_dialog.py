"""Tests for the plate info dialog data helpers."""

from unittest.mock import patch

import pytest


# --------------- Fixtures ---------------


@pytest.fixture
def sample_meta() -> dict:
    return {
        "channel_data": {"DAPI": "0", "Tub": "1", "EdU": "2"},
        "pixel_size": (0.3, 0.3),
        "intensities": {0: (100, 5000), 1: (50, 3000), 2: (0, 65535)},
        "plate_name": "TestPlate",
    }


@pytest.fixture
def sample_wells() -> dict:
    return {
        "A1": {
            "well_id": 10,
            "metadata": {"cell_line": "RPE", "condition": "ctrl"},
            "images": [
                {"image_id": 100, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                {"image_id": 101, "size_t": 1, "index": 1, "pos_x": 1.0, "pos_y": 0.0},
                {"image_id": 102, "size_t": 3, "index": 2, "pos_x": 2.0, "pos_y": 0.0},
                {"image_id": 103, "size_t": 1, "index": 3, "pos_x": 3.0, "pos_y": 0.0},
            ],
        },
        "A2": {
            "well_id": 11,
            "metadata": {"cell_line": "RPE", "condition": "drug"},
            "images": [
                {"image_id": 200, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
            ],
        },
        "B1": {
            "well_id": 12,
            "metadata": {"cell_line": "HeLa", "condition": "ctrl"},
            "images": [
                {"image_id": 300, "size_t": 2, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                {"image_id": 301, "size_t": 2, "index": 1, "pos_x": 1.0, "pos_y": 0.0},
            ],
        },
    }


@pytest.fixture
def sample_label_map() -> dict:
    return {
        "A1": [500, 501, 502, 503],
        "A2": [],
        "B1": [600, 601],
    }


def _patch_cache_fns(meta, wells, label_map=None):
    """Return a context manager that patches the cache accessor functions."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("omero_screen_napari._plate_info_dialog.is_plate_cached", return_value=True), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_plate_metadata", return_value=meta), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_well_data", return_value=wells), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_label_map", return_value=label_map):
            yield

    return _ctx()


# --------------- build_table_data from cache ---------------


class TestBuildTableDataFromCache:
    def test_returns_correct_header(self, sample_meta, sample_wells, sample_label_map):
        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import _build_from_cache

            header, meta_keys, rows, is_cached = _build_from_cache(42)

            assert is_cached is True
            assert header["plate_name"] == "TestPlate"
            assert "DAPI" in header["channels"]
            assert "Tub" in header["channels"]
            assert "EdU" in header["channels"]
            assert header["total_wells"] == 3
            assert header["total_images"] == 7  # 4 + 1 + 2

    def test_returns_correct_rows(self, sample_meta, sample_wells, sample_label_map):
        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import _build_from_cache

            _header, _keys, rows, _is_cached = _build_from_cache(42)

            assert len(rows) == 3

            # Rows sorted: A1, A2, B1
            assert rows[0]["well"] == "A1"
            assert rows[1]["well"] == "A2"
            assert rows[2]["well"] == "B1"

            # A1: 4 images, max timepoint 3, has labels
            assert rows[0]["images"] == 4
            assert rows[0]["timepoints"] == 3
            assert rows[0]["metadata"]["cell_line"] == "RPE"
            assert rows[0]["metadata"]["condition"] == "ctrl"
            assert rows[0]["labels"] == "Yes"

            # A2: 1 image, timepoint 1, no labels (empty list)
            assert rows[1]["images"] == 1
            assert rows[1]["timepoints"] == 1
            assert rows[1]["labels"] == "No"

            # B1: 2 images, max timepoint 2, has labels
            assert rows[2]["images"] == 2
            assert rows[2]["timepoints"] == 2
            assert rows[2]["metadata"]["cell_line"] == "HeLa"
            assert rows[2]["labels"] == "Yes"

    def test_no_label_map_shows_no(self, sample_meta, sample_wells):
        with _patch_cache_fns(sample_meta, sample_wells, label_map=None):
            from omero_screen_napari._plate_info_dialog import _build_from_cache

            _header, _keys, rows, _is_cached = _build_from_cache(42)

            for row in rows:
                assert row["labels"] == "No"


# --------------- Dynamic metadata keys ---------------


class TestCollectMetadataKeys:
    def test_cell_line_always_first(self):
        from omero_screen_napari._plate_info_dialog import _collect_metadata_keys

        wells = {
            "A1": {"metadata": {"treatment": "DMSO", "cell_line": "RPE", "timepoint": "24h"}},
            "A2": {"metadata": {"cell_line": "HeLa", "treatment": "Drug"}},
        }
        keys = _collect_metadata_keys(wells)
        assert keys[0] == "cell_line"
        assert "treatment" in keys
        assert "timepoint" in keys

    def test_remaining_keys_sorted(self):
        from omero_screen_napari._plate_info_dialog import _collect_metadata_keys

        wells = {
            "A1": {"metadata": {"cell_line": "RPE", "z_extra": "1", "a_first": "2"}},
        }
        keys = _collect_metadata_keys(wells)
        assert keys == ["cell_line", "a_first", "z_extra"]

    def test_empty_metadata(self):
        from omero_screen_napari._plate_info_dialog import _collect_metadata_keys

        wells = {
            "A1": {"metadata": {}},
            "A2": {"metadata": {}},
        }
        keys = _collect_metadata_keys(wells)
        assert keys == []

    def test_keys_union_across_wells(self):
        from omero_screen_napari._plate_info_dialog import _collect_metadata_keys

        wells = {
            "A1": {"metadata": {"cell_line": "RPE", "siRNA": "siCtrl"}},
            "A2": {"metadata": {"cell_line": "RPE", "drug": "Noc"}},
        }
        keys = _collect_metadata_keys(wells)
        assert keys == ["cell_line", "drug", "siRNA"]


class TestKeyToHeader:
    def test_underscore_key(self):
        from omero_screen_napari._plate_info_dialog import _key_to_header

        assert _key_to_header("cell_line") == "Cell Line"

    def test_no_underscore_key(self):
        from omero_screen_napari._plate_info_dialog import _key_to_header

        assert _key_to_header("siRNA") == "siRNA"

    def test_multi_underscore(self):
        from omero_screen_napari._plate_info_dialog import _key_to_header

        assert _key_to_header("time_in_hours") == "Time In Hours"


# --------------- _well_sort_key ---------------


class TestWellSortKey:
    def test_sorts_correctly(self):
        from omero_screen_napari._plate_info_dialog import _well_sort_key

        wells = ["B2", "A1", "A10", "A2", "B1"]
        sorted_wells = sorted(wells, key=_well_sort_key)
        assert sorted_wells == ["A1", "A2", "A10", "B1", "B2"]


# --------------- build_table_data routing ---------------


class TestBuildTableData:
    def test_routes_to_cache(self, sample_meta, sample_wells):
        with _patch_cache_fns(sample_meta, sample_wells):
            from omero_screen_napari._plate_info_dialog import build_table_data

            _header, _keys, _rows, is_cached = build_table_data(42)
            assert is_cached is True

    def test_routes_to_omero_when_not_cached(self):
        with patch("omero_screen_napari._plate_info_dialog.is_plate_cached", return_value=False), \
             patch("omero_screen_napari._plate_info_dialog._build_from_omero") as mock_omero:
            mock_omero.return_value = ({"plate_name": "X"}, [], [], False)

            from omero_screen_napari._plate_info_dialog import build_table_data

            _header, _keys, _rows, is_cached = build_table_data(999)
            assert is_cached is False
            mock_omero.assert_called_once_with(999)

    def test_metadata_keys_returned(self, sample_meta, sample_wells):
        with _patch_cache_fns(sample_meta, sample_wells):
            from omero_screen_napari._plate_info_dialog import build_table_data

            _header, keys, _rows, _cached = build_table_data(42)
            assert keys == ["cell_line", "condition"]


# --------------- Missing metadata graceful handling ---------------


class TestMissingMetadata:
    def test_empty_metadata_dict(self, sample_meta):
        """Wells with empty metadata dicts produce rows with empty metadata."""
        wells = {
            "A1": {
                "well_id": 10,
                "metadata": {},
                "images": [
                    {"image_id": 100, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                ],
            },
        }

        with _patch_cache_fns(sample_meta, wells):
            from omero_screen_napari._plate_info_dialog import _build_from_cache

            _header, keys, rows, _ = _build_from_cache(42)

            assert keys == []
            assert rows[0]["metadata"] == {}

    def test_mixed_metadata_keys(self, sample_meta):
        """Wells with different metadata keys still display all columns."""
        wells = {
            "A1": {
                "well_id": 10,
                "metadata": {"cell_line": "RPE", "siRNA": "siCtrl"},
                "images": [{"image_id": 100, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0}],
            },
            "A2": {
                "well_id": 11,
                "metadata": {"cell_line": "HeLa"},
                "images": [{"image_id": 200, "size_t": 1, "index": 0, "pos_x": 0.0, "pos_y": 0.0}],
            },
        }

        with _patch_cache_fns(sample_meta, wells):
            from omero_screen_napari._plate_info_dialog import _build_from_cache

            _header, keys, rows, _ = _build_from_cache(42)

            assert keys == ["cell_line", "siRNA"]
            # A1 has both keys
            assert rows[0]["metadata"].get("siRNA") == "siCtrl"
            # A2 is missing siRNA — table will show ""
            assert rows[1]["metadata"].get("siRNA") is None
