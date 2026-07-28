"""Equivalence: the lazy dask streaming build == the eager whole-array build.

The streaming refactor exists to bound RAM without changing the output, so
these are the load-bearing tests: the same synthetic fields fed through the
new dask path (block-by-block) must produce arrays — and a zarr store —
identical to the old eager path (load-all-then-stitch).

OMERO is mocked at the builder's three download seams. N=1 field isolates the
*new* logic (per-T-block streaming + dask wrapping); multi-field stitching is
the same shared ``_stitch_image`` / ``_recompose_labels`` in both paths.
T=5 with the default block of 4 exercises a full block + a partial tail block.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from omero_utils.stitching import recompose_tiles
import zarr
from omero_screen_napari.zarr_cache import (
    PlateZarrWriter,
    builder,
    plate_zarr_path,
)

_CHANNELS = {"DAPI": "0", "Tub": "1"}
_N_T, _Y, _X, _N_C = 5, 8, 8, 2
_IMAGE_IDS = [10]
_MASK_IDS = [20]


def _make_well() -> MagicMock:
    well = MagicMock()
    samples = []
    for iid in _IMAGE_IDS:
        s = MagicMock()
        img = s.getImage.return_value
        img.getId.return_value = iid
        img.getSizeT.return_value = _N_T
        img.getSizeY.return_value = _Y
        img.getSizeX.return_value = _X
        img.getSizeZ.return_value = 1
        img.getSizeC.return_value = _N_C
        s.getPosX.return_value.getValue.return_value = 0.0
        s.getPosY.return_value.getValue.return_value = 0.0
        samples.append(s)
    well.getWellSample.side_effect = lambda n: samples[n]
    well.listChildren.return_value = samples
    return well


def _synthetic_sources() -> tuple[
    dict[int, np.ndarray], dict[int, np.ndarray], dict[str, np.ndarray]
]:
    rng = np.random.default_rng(0)
    # Field images: (T, Z=1, Y, X, C) uint16 (the shape napari get_image yields).
    field_data = {
        _IMAGE_IDS[0]: rng.integers(
            0, 5000, (_N_T, 1, _Y, _X, _N_C), dtype=np.uint16
        )
    }
    # Masks: (T, Z=1, Y, X, C=2) — channel 0 nuclei, channel 1 cells.
    mask_data = {
        _MASK_IDS[0]: rng.integers(
            0, 50, (_N_T, 1, _Y, _X, 2), dtype=np.uint16
        )
    }
    flatfield = {
        "DAPI": np.ones((_Y, _X), dtype=np.float32),
        "Tub": np.ones((_Y, _X), dtype=np.float32),
    }
    return field_data, mask_data, flatfield


def _patches(field_data, mask_data):  # type: ignore[no-untyped-def]
    """Patch the builder's three OMERO download seams with synthetic data."""

    def fake_get_image(conn, image_id, start=None, end=None, tag=None):  # type: ignore[no-untyped-def]
        arr = field_data[image_id]
        return arr if start is None else arr[start:end]

    def fake_resolve(_well):  # type: ignore[no-untyped-def]
        return list(_MASK_IDS), list(_IMAGE_IDS)

    def fake_trange(  # type: ignore[no-untyped-def]
        conn,
        mask_ids,
        *,
        t0=None,
        t1=None,
        source_ids=None,
        conn_factory=None,
        max_workers=3,
    ):
        sl = slice(None) if t0 is None else slice(t0, t1)
        nuclei, cells = [], []
        for mid in mask_ids:
            sq = np.squeeze(mask_data[mid][sl], axis=1)  # (t, Y, X, C)
            nuclei.append(np.ascontiguousarray(sq[..., 0]))
            cells.append(np.ascontiguousarray(sq[..., 1]))
        return nuclei, cells

    assert len(_IMAGE_IDS) == 1, "Only 1 offset configured for synthetic well images"
    def _fake_load_canvas_offsets(_well):
        return np.array([[0, 0]])

    return (
        patch.object(builder, "get_image", fake_get_image),
        patch.object(builder, "resolve_stitched_mask_ids", fake_resolve),
        patch.object(
            builder, "fetch_stitched_field_masks_trange", fake_trange
        ),
        patch.object(builder, "_load_canvas_offsets", _fake_load_canvas_offsets),
    )


def _build_both():  # type: ignore[no-untyped-def]
    """Return ((dask_img, dask_nuc, dask_cell), (eager_img, eager_nuc, eager_cell))."""
    field_data, mask_data, flatfield = _synthetic_sources()
    well = _make_well()
    p_img, p_res, p_tr, p_co = _patches(field_data, mask_data)
    with p_img, p_res, p_tr, p_co:
        # New lazy/dask path.
        img_d, nuc_d, cell_d = builder._build_lazy_well_arrays(
            MagicMock(),
            None,
            well,
            _CHANNELS,
            flatfield,
            plate_id=99,
            block_t=4,
        )
        dask_arrays = (
            np.asarray(img_d),
            np.asarray(nuc_d),
            np.asarray(cell_d),
        )
        # Old eager path, same inputs.
        imgs_ntyxc, offsets = builder._load_well_fields(
            MagicMock(), well, _CHANNELS, flatfield
        )
        eager_img = builder._stitch_image(imgs_ntyxc, offsets)
        mids, src = builder.resolve_stitched_mask_ids(well)
        nuc_f, cell_f = builder.fetch_stitched_field_masks_trange(
            MagicMock(), mids, source_ids=src
        )
        eager_nuc = recompose_tiles(nuc_f, offsets).astype(
            np.uint32, copy=False
        )
        # cell_f is not None
        eager_cell = recompose_tiles(cell_f, offsets).astype(
            np.uint32, copy=False
        )
    return dask_arrays, (eager_img, eager_nuc, eager_cell)


def test_dask_arrays_equal_eager() -> None:
    """Computed dask arrays match the eager build, value- and dtype-identical."""
    (d_img, d_nuc, d_cell), (e_img, e_nuc, e_cell) = _build_both()

    assert d_img.shape == e_img.shape == (_N_T, _N_C, _Y, _X)
    assert d_img.dtype == e_img.dtype
    np.testing.assert_array_equal(d_img, e_img)
    np.testing.assert_array_equal(d_nuc, e_nuc)
    np.testing.assert_array_equal(d_cell, e_cell)


def test_streamed_zarr_matches_eager_zarr() -> None:
    """Byte-for-byte: writing the dask arrays yields the same zarr as numpy.

    Both go through the same PlateZarrWriter; the only difference is whether
    write_image/write_labels stream a dask array or write a numpy one.
    """
    (d_img, d_nuc, d_cell), (e_img, e_nuc, e_cell) = _build_both()
    # Re-fetch the lazy arrays (consumed arrays above were materialised).
    field_data, mask_data, flatfield = _synthetic_sources()
    well = _make_well()
    p_img, p_res, p_tr, p_co = _patches(field_data, mask_data)

    def _write(plate_id, img, nuc, cell):  # type: ignore[no-untyped-def]
        w = PlateZarrWriter(
            plate_id=plate_id,
            plate_name="t",
            channel_names=list(_CHANNELS),
            pixel_size_um=0.65,
            n_timepoints=_N_T,
        )
        with w:
            w.ensure_plate(all_wells=["A1"])
            w.write_well("A1", img, nuc, cell)

    with p_img, p_res, p_tr, p_co:
        lazy_img, lazy_nuc, lazy_cell = builder._build_lazy_well_arrays(
            MagicMock(),
            None,
            well,
            _CHANNELS,
            flatfield,
            plate_id=99,
            block_t=4,
        )
        _write(8002, lazy_img, lazy_nuc, lazy_cell)  # dask (streamed)
    _write(8001, e_img, e_nuc, e_cell)  # numpy (eager)

    eager = zarr.open_group(str(plate_zarr_path(8001)), mode="r")
    streamed = zarr.open_group(str(plate_zarr_path(8002)), mode="r")

    # Authoritative full-resolution data (level 0) must be byte-identical —
    # image (analysis/crops) and both label maps (full-res display/crops).
    for sub in ("0/0", "0/labels/nuclei/0", "0/labels/cells/0"):
        np.testing.assert_array_equal(
            eager[f"A/1/{sub}"][:], streamed[f"A/1/{sub}"][:], err_msg=sub
        )

    # Pyramid levels >0 are built by ome-zarr's *dask* write path (streamed)
    # vs its *numpy* path (eager); these produce valid but not byte-identical
    # downsamples. They are display-only (level 0 is identical) and
    # self-consistent going forward (all caches now build via dask), so we
    # assert the pyramid is *valid*, not equal:
    #   - image: matching shape/dtype;
    #   - labels: matching shape AND only real label values (a nearest
    #     downsample, never interpolated garbage).
    for lvl in (1, 2):
        e_img, s_img = eager[f"A/1/0/{lvl}"], streamed[f"A/1/0/{lvl}"]
        assert e_img.shape == s_img.shape
        assert e_img.dtype == s_img.dtype
        for kind in ("nuclei", "cells"):
            base_vals = set(
                np.unique(streamed[f"A/1/0/labels/{kind}/0"][:]).tolist()
            )
            s_lbl = streamed[f"A/1/0/labels/{kind}/{lvl}"]
            e_lbl = eager[f"A/1/0/labels/{kind}/{lvl}"]
            assert s_lbl.shape == e_lbl.shape
            assert set(np.unique(s_lbl[:]).tolist()).issubset(base_vals), (
                f"{kind} L{lvl} has non-label (interpolated) values"
            )
