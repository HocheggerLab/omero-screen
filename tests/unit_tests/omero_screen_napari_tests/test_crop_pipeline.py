"""Tests for the unified crop-generation core (crop_pipeline)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from omero_screen_napari.crop_pipeline import (
    CropPipeline,
    CropSourceError,
    WelldataSource,
    ZarrSource,
    _resolve_centroids,
)
from omero_screen_napari.omero_data import OmeroData


# ---------------------------------------------------------------------------
# Fake source for exercising the pipeline core without zarr/OMERO/in-memory.
# ---------------------------------------------------------------------------


class _FakeSource:
    """Returns a fixed crop + a label crop containing the given labels."""

    def __init__(self, label_field: np.ndarray) -> None:
        self._label_field = label_field
        self.calls: list[tuple] = []

    def fetch(self, image_id, centroid, size, t, mask_name):
        self.calls.append((image_id, centroid, size, t, mask_name))
        crop = np.full((size, size, 2), 0.5, dtype=np.float32)
        return crop, self._label_field.copy()


def _nucleus_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CropPipeline core loop
# ---------------------------------------------------------------------------


class TestCropPipeline:
    def test_isolates_label_and_collects_cell_meta(self):
        # Label field has two cells (7 and 8); each centroid row isolates one.
        label_field = np.zeros((32, 32), dtype=np.uint32)
        label_field[5:10, 5:10] = 7
        label_field[20:25, 20:25] = 8
        source = _FakeSource(label_field)
        df = _nucleus_df(
            {
                "image_id": [1, 1],
                "centroid-0-nuc": [7, 22],
                "centroid-1-nuc": [7, 22],
                "label": [7, 8],
            }
        )
        result = CropPipeline(
            source=source,
            centroids_df=df,
            segmentation="nucleus",
            crop_size=32,
            timepoint=0,
        ).run()

        assert len(result.crops) == 2
        assert len(result.labels) == 2
        assert result.image_ids == [1]
        # Each label crop isolated to its own id (no cross-contamination).
        assert set(np.unique(result.labels[0])) == {0, 7}
        assert set(np.unique(result.labels[1])) == {0, 8}
        assert result.cell_meta[0] == {
            "centroid_row": 7,
            "centroid_col": 7,
            "image_id": 1,
        }

    def test_excluded_centroids_are_skipped(self):
        label_field = np.zeros((32, 32), dtype=np.uint32)
        label_field[5:10, 5:10] = 7
        source = _FakeSource(label_field)
        df = _nucleus_df(
            {
                "image_id": [1],
                "centroid-0-nuc": [7],
                "centroid-1-nuc": [7],
                "label": [7],
            }
        )
        result = CropPipeline(
            source=source,
            centroids_df=df,
            segmentation="nucleus",
            crop_size=32,
            timepoint=0,
            excluded_centroids={(1, 7, 7)},
        ).run()
        assert result.crops == []
        assert source.calls == []  # never even fetched

    def test_empty_label_isolation_drops_crop(self):
        # Centroid id 99 isn't present in the label field → isolated mask is
        # all-zero → crop dropped.
        label_field = np.zeros((32, 32), dtype=np.uint32)
        label_field[5:10, 5:10] = 7
        source = _FakeSource(label_field)
        df = _nucleus_df(
            {
                "image_id": [1],
                "centroid-0-nuc": [7],
                "centroid-1-nuc": [7],
                "label": [99],
            }
        )
        result = CropPipeline(
            source=source,
            centroids_df=df,
            segmentation="nucleus",
            crop_size=32,
            timepoint=0,
        ).run()
        assert result.crops == []
        assert result.image_ids == []

    def test_empty_dataframe_returns_empty_result(self):
        source = _FakeSource(np.zeros((32, 32), dtype=np.uint32))
        result = CropPipeline(
            source=source,
            centroids_df=pd.DataFrame(),
            segmentation="nucleus",
            crop_size=32,
            timepoint=0,
        ).run()
        assert result.crops == []


# ---------------------------------------------------------------------------
# Centroid resolution
# ---------------------------------------------------------------------------


class TestResolveCentroids:
    def test_nucleus_keeps_every_row(self):
        df = pd.DataFrame(
            {
                "centroid-0-nuc": [1, 2, 2],
                "centroid-1-nuc": [1, 2, 2],
                "label": [10, 20, 20],
            }
        )
        rows, cols, ids = _resolve_centroids(df, "nucleus")
        assert ids == [10, 20, 20]

    def test_nucleus_literal_eval_of_stringified_lists(self):
        df = pd.DataFrame(
            {
                "centroid-0-nuc": [1],
                "centroid-1-nuc": [1],
                "label": ["[10, 11]"],  # multi-nucleate id list as a string
            }
        )
        _, _, ids = _resolve_centroids(df, "nucleus")
        assert ids == [[10, 11]]

    def test_cell_dedups_on_centroid_pair(self):
        df = pd.DataFrame(
            {
                "centroid-0-cell": [5, 5, 9],
                "centroid-1-cell": [5, 5, 9],
                "Cyto_ID": [3, 3, 4],
            }
        )
        rows, cols, ids = _resolve_centroids(df, "cell")
        assert len(rows) == 2
        assert ids == [3, 4]


# ---------------------------------------------------------------------------
# ZarrSource
# ---------------------------------------------------------------------------


class TestZarrSource:
    def _patch_zarr(self, monkeypatch, *, resolve, wells, crop=None, label=None):
        import omero_screen_napari.zarr_cache as zc

        monkeypatch.setattr(zc, "resolve_to_zarr", lambda _p: resolve)
        monkeypatch.setattr(zc, "cached_wells", lambda _p: wells)
        if crop is not None:
            monkeypatch.setattr(zc, "fetch_crop", lambda *a, **k: crop)
        if label is not None:
            monkeypatch.setattr(zc, "fetch_label_crop", lambda *a, **k: label)

    def test_raises_when_no_zarr(self, monkeypatch):
        self._patch_zarr(monkeypatch, resolve=None, wells=[])
        with pytest.raises(CropSourceError, match="No zarr cache"):
            ZarrSource(1, "A1")

    def test_raises_when_well_not_built(self, monkeypatch):
        self._patch_zarr(monkeypatch, resolve=object(), wells=["B2"])
        with pytest.raises(CropSourceError, match="not built"):
            ZarrSource(1, "A1")

    def test_fetch_normalises_with_intensity_window(self, monkeypatch):
        crop = np.full((2, 32, 32), 500, dtype=np.uint16)  # (C, Y, X)
        label = np.zeros((32, 32), dtype=np.uint32)
        label[10:20, 10:20] = 7
        self._patch_zarr(
            monkeypatch, resolve=object(), wells=["A1"], crop=crop, label=label
        )
        src = ZarrSource(1, "A1", intensities={0: (0, 1000), 1: (0, 1000)})
        out_crop, out_label = src.fetch(555, (15.0, 15.0), 32, 0, "nuclei")

        assert out_crop.shape == (32, 32, 2)
        assert out_crop.dtype == np.float32
        # 500 / (1000 - 0) = 0.5
        assert abs(float(out_crop.mean()) - 0.5) < 1e-3
        # Label crop is returned raw (isolation happens in the pipeline).
        assert int(out_label[15, 15]) == 7

    def test_fetch_raises_cropsourceerror_on_fetch_failure(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("store evicted")

        import omero_screen_napari.zarr_cache as zc

        monkeypatch.setattr(zc, "resolve_to_zarr", lambda _p: object())
        monkeypatch.setattr(zc, "cached_wells", lambda _p: ["A1"])
        monkeypatch.setattr(zc, "fetch_crop", boom)
        monkeypatch.setattr(zc, "fetch_label_crop", boom)

        src = ZarrSource(1, "A1", intensities={0: (0, 1000)})
        with pytest.raises(CropSourceError, match="Zarr fetch failed"):
            src.fetch(555, (15.0, 15.0), 32, 0, "nuclei")


# ---------------------------------------------------------------------------
# WelldataSource (in-memory fields)
# ---------------------------------------------------------------------------


class TestWelldataSource:
    def _omero_data(self, image, labels, image_id=555):
        od = MagicMock(spec=OmeroData)
        od.image_ids = [image_id]
        od.images = [image]
        od.labels = [labels]
        od.intensities = {0: (0, 1000), 1: (0, 1000)}
        return od

    def test_fetch_scales_crops_and_selects_label_channel(self):
        # (Y, X, 2) field of constant 500; (Y, X, 2) labels [nucleus, cell].
        image = np.full((64, 64, 2), 500, dtype=np.uint16)
        labels = np.zeros((64, 64, 2), dtype=np.uint32)
        labels[28:36, 28:36, 0] = 7  # nucleus label
        labels[20:44, 20:44, 1] = 9  # cell label
        src = WelldataSource(self._omero_data(image, labels))

        crop, label_crop = src.fetch(555, (32.0, 32.0), 32, 0, "nuclei")
        assert crop.shape == (32, 32, 2)
        assert crop.dtype == np.float32
        # 500 / (1000 - 0) = 0.5
        assert abs(float(crop.mean()) - 0.5) < 1e-3
        # nucleus channel selected
        assert 7 in np.unique(label_crop)
        assert 9 not in np.unique(label_crop)

        # cell mask_name selects the cell channel
        _, cell_label = src.fetch(555, (32.0, 32.0), 32, 0, "cells")
        assert 9 in np.unique(cell_label)

    def test_fetch_pads_edge_crops(self):
        image = np.full((64, 64, 2), 500, dtype=np.uint16)
        labels = np.zeros((64, 64, 2), dtype=np.uint32)
        labels[0:4, 0:4, 0] = 3
        src = WelldataSource(self._omero_data(image, labels))
        crop, label_crop = src.fetch(555, (1.0, 1.0), 32, 0, "nuclei")
        # Centre-padded back up to the full crop size.
        assert crop.shape == (32, 32, 2)
        assert label_crop.shape == (32, 32)

    def test_fetch_unknown_image_raises(self):
        image = np.zeros((64, 64, 2), dtype=np.uint16)
        labels = np.zeros((64, 64, 2), dtype=np.uint32)
        src = WelldataSource(self._omero_data(image, labels))
        with pytest.raises(CropSourceError, match="not loaded"):
            src.fetch(999, (32.0, 32.0), 32, 0, "nuclei")
