"""Tests for the plate info dialog data helpers."""

import os
from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance exists for Qt widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# --------------- Fixtures ---------------


@pytest.fixture
def sample_meta() -> dict:
    return {
        "channel_data": {"DAPI": "0", "Tub": "1", "EdU": "2"},
        "pixel_size": (0.3, 0.3),
        "intensities": {0: (100, 5000), 1: (50, 3000), 2: (0, 65535)},
        "plate_name": "TestPlate",
        "ff_mask_id": 999,
    }


@pytest.fixture
def sample_wells() -> dict:
    return {
        "A1": {
            "well_id": 10,
            "metadata": {"cell_line": "RPE", "condition": "ctrl"},
            "images": [
                {"image_id": 100, "dims": (1,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                {"image_id": 101, "dims": (1,2,3,4,5), "index": 1, "pos_x": 1.0, "pos_y": 0.0},
                {"image_id": 102, "dims": (3,2,3,4,5), "index": 2, "pos_x": 2.0, "pos_y": 0.0},
                {"image_id": 103, "dims": (1,2,3,4,5), "index": 3, "pos_x": 3.0, "pos_y": 0.0},
            ],
        },
        "A2": {
            "well_id": 11,
            "metadata": {"cell_line": "RPE", "condition": "drug"},
            "images": [
                {"image_id": 200, "dims": (1,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0},
            ],
        },
        "B1": {
            "well_id": 12,
            "metadata": {"cell_line": "HeLa", "condition": "ctrl"},
            "images": [
                {"image_id": 300, "dims": (2,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0},
                {"image_id": 301, "dims": (2,2,3,4,5), "index": 1, "pos_x": 1.0, "pos_y": 0.0},
            ],
        },
    }


@pytest.fixture
def sample_label_map() -> dict:
    return {
        "A1": [
            {"image_id": 500, "dims": (1,2,3,4,5)},
            {"image_id": 501, "dims": (1,2,3,4,5)},
            {"image_id": 502, "dims": (3,2,3,4,5)},
            {"image_id": 503, "dims": (1,2,3,4,5)}
        ],
        "A2": [],
        "B1": [
            {"image_id": 600, "dims": (2,2,3,4,5)},
            {"image_id": 601, "dims": (2,2,3,4,5)},
        ],
    }


def _patch_cache_fns(meta, wells, label_map=None, well_cache_status=None):
    """Return a context manager that patches the cache accessor functions."""
    import contextlib

    if label_map is None:
        label_map = {}
    if well_cache_status is None:
        well_cache_status = {}

    @contextlib.contextmanager
    def _ctx():
        with patch("omero_screen_napari._plate_info_dialog.is_plate_cached", return_value=True), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_plate_metadata", return_value=meta), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_well_data", return_value=wells), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_label_map", return_value=label_map), \
             patch("omero_screen_napari._plate_info_dialog.get_well_cache_status", return_value=well_cache_status), \
             patch("omero_screen_napari._plate_info_dialog.PlateInfoDialog._start_cache_monitoring"):
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

    def test_missing_label_map_shows_no(self, sample_meta, sample_wells):
        label_map = {
            "A1": [None, None, None, None],
            "A2": [None],
            "B1": [None, None],
        }
        with _patch_cache_fns(sample_meta, sample_wells, label_map=label_map):
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
                    {"image_id": 100, "dims": (1,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0},
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
                "images": [{"image_id": 100, "dims": (1,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0}],
            },
            "A2": {
                "well_id": 11,
                "metadata": {"cell_line": "HeLa"},
                "images": [{"image_id": 200, "dims": (1,2,3,4,5), "index": 0, "pos_x": 0.0, "pos_y": 0.0}],
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


# --------------- Checkbox selection (Qt widget tests) ---------------


class TestPlateInfoDialogCheckboxes:
    """Test the checkbox column in PlateInfoDialog."""

    def test_table_has_checkbox_column(self, qapp, sample_meta, sample_wells, sample_label_map):
        """Table should have a Select checkbox column at index 0."""
        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=lambda _: None)
            table = dialog.table

            # First column header should be empty (checkbox column)
            header_item = table.horizontalHeaderItem(0)
            assert header_item is not None
            assert header_item.text() == ""

            # Well column is now at index 1
            header_item_well = table.horizontalHeaderItem(1)
            assert header_item_well is not None
            assert header_item_well.text() == "Well"

            # Each row should have a checkbox widget in column 0
            assert len(dialog._row_checkboxes) == 3
            for row_idx in range(table.rowCount()):
                cb_widget = table.cellWidget(row_idx, 0)
                assert cb_widget is not None

    def test_load_selected_collects_checked_wells(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """_on_load_selected should collect well positions from checked rows."""
        loaded: list[str] = []

        def callback(wells: str) -> None:
            loaded.append(wells)

        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=callback)

            # Check rows 0 and 2 (A1 and B1 after sorting)
            dialog._row_checkboxes[0].setChecked(True)
            dialog._row_checkboxes[2].setChecked(True)

            dialog._on_load_selected()

            assert len(loaded) == 1
            wells_str = loaded[0]
            # The checked wells should appear in the callback
            assert "A1" in wells_str
            assert "B1" in wells_str

    def test_load_selected_falls_back_to_row_selection(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """When no checkboxes are checked, fall back to Qt row selection."""
        loaded: list[str] = []

        def callback(wells: str) -> None:
            loaded.append(wells)

        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=callback)

            # No checkboxes checked — select row 1 via Qt selection
            dialog.table.selectRow(1)
            dialog._on_load_selected()

            assert len(loaded) == 1
            assert "A2" in loaded[0]

    def test_select_all_toggles_all_checkboxes(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """The Select All checkbox should toggle all row checkboxes."""
        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=lambda _: None)

            # Initially none checked
            assert not any(cb.isChecked() for cb in dialog._row_checkboxes)

            # Check Select All
            dialog._select_all_cb.setChecked(True)
            assert all(cb.isChecked() for cb in dialog._row_checkboxes)

            # Uncheck Select All
            dialog._select_all_cb.setChecked(False)
            assert not any(cb.isChecked() for cb in dialog._row_checkboxes)

    def test_double_click_uses_well_column_index_1(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """Double-click should read well from column 1 (shifted by checkbox)."""
        loaded: list[str] = []

        def callback(wells: str) -> None:
            loaded.append(wells)

        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=callback)

            # Double-click row 0 — should load A1 from column 1
            dialog._on_double_click(0, 1)

            assert len(loaded) == 1
            assert loaded[0] == "A1"


# --------------- Cached column tests ---------------


class TestPlateInfoDialogCacheStatus:
    """Test the Cached column in PlateInfoDialog."""

    def test_cached_column_exists(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """Table should have a 'Cached' column as the last column."""
        with _patch_cache_fns(sample_meta, sample_wells, sample_label_map):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=lambda _: None)
            table = dialog.table

            last_col = table.columnCount() - 1
            header_item = table.horizontalHeaderItem(last_col)
            assert header_item is not None
            assert header_item.text() == "Cached"

    def test_cached_shows_yes_when_fully_cached(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """All wells show 'Yes' when all images are in cache."""
        all_cached = {"A1": True, "A2": True, "B1": True}
        with _patch_cache_fns(
            sample_meta, sample_wells, sample_label_map,
            well_cache_status=all_cached,
        ):
            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=lambda _: None)
            table = dialog.table
            cached_col = table.columnCount() - 1

            for row_idx in range(table.rowCount()):
                item = table.item(row_idx, cached_col)
                assert item is not None
                assert item.text() == "Yes"

    def test_cached_shows_no_when_not_cached(
        self, qapp, sample_meta, sample_wells, sample_label_map
    ):
        """Wells show 'No' when not cached per get_well_cache_status."""
        # Patch is_plate_cached to False and provide partial well status
        well_status = {"A1": True, "A2": False, "B1": False}
        with patch("omero_screen_napari._plate_info_dialog.is_plate_cached", return_value=False), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_plate_metadata", return_value=sample_meta), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_well_data", return_value=sample_wells), \
             patch("omero_screen_napari._plate_info_dialog.get_cached_label_map", return_value=sample_label_map), \
             patch("omero_screen_napari._plate_info_dialog.get_well_cache_status", return_value=well_status), \
             patch("omero_screen_napari._plate_info_dialog._build_from_omero") as mock_omero, \
             patch("omero_screen_napari._plate_info_dialog.PlateInfoDialog._start_cache_monitoring"):
            # Make _build_from_omero return proper data
            from omero_screen_napari._plate_info_dialog import (
                _build_rows,
                _collect_metadata_keys,
                _build_header_info,
            )
            header_info = _build_header_info(sample_meta, sample_wells)
            metadata_keys = _collect_metadata_keys(sample_wells)
            rows = _build_rows(sample_wells, sample_label_map, label_unknown=True)
            mock_omero.return_value = (header_info, metadata_keys, rows, False)

            from omero_screen_napari._plate_info_dialog import PlateInfoDialog

            dialog = PlateInfoDialog(42, on_load_callback=lambda _: None)
            table = dialog.table
            cached_col = table.columnCount() - 1

            # Build a lookup: well_pos -> cached text
            well_cached: dict[str, str] = {}
            for row_idx in range(table.rowCount()):
                well_item = table.item(row_idx, 1)
                cached_item = table.item(row_idx, cached_col)
                if well_item and cached_item:
                    well_cached[well_item.text()] = cached_item.text()

            assert well_cached["A1"] == "Yes"
            assert well_cached["A2"] == "No"
            assert well_cached["B1"] == "No"
