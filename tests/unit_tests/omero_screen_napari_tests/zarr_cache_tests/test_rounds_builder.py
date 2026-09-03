"""The 4i build: restain rounds land on the master's canvas, correctly shifted.

Mocks the builder's OMERO download seams (the same three
``test_builder_streaming`` uses) and feeds a restain round whose pixels are a
known shift of the master's. Two properties matter:

* the restain channels appear on the *master's* canvas, offset by exactly the
  alignment shift -- a transposed or negated shift is the failure mode that
  survives visual inspection, since real shifts are only a few pixels;
* the declared dask block shape equals what is actually written. ``write_image``
  streams from the dask array, so a mismatch corrupts the store rather than
  raising.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from omero_screen_napari.zarr_cache import builder

_MASTER_CH = {"DAPI": "0", "Tub": "1"}
_ROUND_CH = {"DAPI": "0", "EdU": "1"}
_N_T, _Y, _X = 2, 32, 32
_MASTER_IDS = [10]
_ROUND_IDS = [110]
_MASK_IDS = [20]


def _make_well(image_ids: list[int]) -> MagicMock:
    well = MagicMock()
    samples = []
    for iid in image_ids:
        s = MagicMock()
        img = s.getImage.return_value
        img.getId.return_value = iid
        img.getSizeT.return_value = _N_T
        img.getSizeY.return_value = _Y
        img.getSizeX.return_value = _X
        img.getSizeZ.return_value = 1
        img.getSizeC.return_value = 2
        samples.append(s)
    well.getWellSample.side_effect = lambda n: samples[n]
    well.listChildren.return_value = samples
    well.countWellSample.return_value = len(samples)
    return well


def _sources(shift_x: int, shift_y: int):  # type: ignore[no-untyped-def]
    """Master field, plus a restain field that is the master rolled by the shift.

    A restain coordinate maps into master frame by subtracting (x, y), so the
    restain image is the master content displaced by +(x, y).
    """
    rng = np.random.default_rng(3)
    master = rng.integers(1, 5000, (_N_T, 1, _Y, _X, 2), dtype=np.uint16)
    restain = np.roll(
        np.roll(master, shift_y, axis=2), shift_x, axis=3
    ).astype(np.uint16)
    field_data = {_MASTER_IDS[0]: master, _ROUND_IDS[0]: restain}
    mask_data = {
        _MASK_IDS[0]: rng.integers(0, 20, (_N_T, 1, _Y, _X, 2), dtype=np.uint16)
    }
    ff = {
        name: np.ones((_Y, _X), dtype=np.float32)
        for name in {*_MASTER_CH, *_ROUND_CH}
    }
    return field_data, mask_data, ff


def _patches(field_data, mask_data):  # type: ignore[no-untyped-def]
    def fake_get_image(conn, image_id, start=None, end=None, tag=None):  # type: ignore[no-untyped-def]
        arr = field_data[image_id]
        return arr if start is None else arr[start:end]

    def fake_resolve(_well, _fields):  # type: ignore[no-untyped-def]
        return list(_MASK_IDS), list(_MASTER_IDS)

    def fake_trange(  # type: ignore[no-untyped-def]
        conn, mask_ids, *, t0=None, t1=None, source_ids=None,
        conn_factory=None, max_workers=3,
    ):
        sl = slice(None) if t0 is None else slice(t0, t1)
        nuclei, cells = [], []
        for mid in mask_ids:
            sq = np.squeeze(mask_data[mid][sl], axis=1)
            nuclei.append(np.ascontiguousarray(sq[..., 0]))
            cells.append(np.ascontiguousarray(sq[..., 1]))
        return nuclei, cells

    return (
        patch.object(builder, "get_image", fake_get_image),
        patch.object(builder, "resolve_stitched_mask_ids", fake_resolve),
        patch.object(builder, "fetch_stitched_field_masks_trange", fake_trange),
        patch.object(
            builder, "_load_canvas_offsets", lambda _well: np.array([[0, 0]])
        ),
    )


def _build(shift_x: int, shift_y: int):  # type: ignore[no-untyped-def]
    field_data, mask_data, ff = _sources(shift_x, shift_y)
    master_well = _make_well(_MASTER_IDS)
    round_well = _make_well(_ROUND_IDS)
    spec = builder.RoundSpec(
        plate_id=4130,
        well=round_well,
        shifts=np.array([[shift_x, shift_y]]),
        image_ids=list(_ROUND_IDS),
        channel_data=_ROUND_CH,
        flatfield_dict=ff,
    )
    p_img, p_res, p_tr, p_co = _patches(field_data, mask_data)
    with p_img, p_res, p_tr, p_co:
        img, nuc, cell = builder._build_lazy_well_arrays(
            MagicMock(),
            None,
            master_well,
            _MASTER_CH,
            ff,
            plate_id=4127,
            block_t=1,
            round_specs=[spec],
        )
        return img, np.asarray(img.compute()), field_data


class TestChannelStacking:
    def test_channel_count_is_the_sum_of_rounds(self) -> None:
        lazy, arr, _ = _build(0, 0)
        assert arr.shape[1] == len(_MASTER_CH) + len(_ROUND_CH)

    def test_declared_dask_shape_matches_what_is_computed(self) -> None:
        """A mismatch here corrupts the written store rather than raising."""
        lazy, arr, _ = _build(3, -2)
        assert lazy.shape == arr.shape

    def test_canvas_is_the_masters(self) -> None:
        _, arr, _ = _build(5, 5)
        assert arr.shape[-2:] == (_Y, _X)

    def test_master_channels_come_first_and_are_unshifted(self) -> None:
        _, arr, field_data = _build(4, 3)
        expected = field_data[_MASTER_IDS[0]][:, 0, :, :, 0]
        np.testing.assert_array_equal(arr[:, 0], expected)


class TestShiftApplication:
    @pytest.mark.parametrize(
        "shift_x, shift_y", [(0, 0), (3, 0), (0, 3), (4, -3), (-4, 3)]
    )
    def test_restain_lands_registered_with_the_master(
        self, shift_x: int, shift_y: int
    ) -> None:
        """After the shift, restain DAPI must match master DAPI pixel-for-pixel.

        The restain source is the master rolled by +(x, y); placing it at
        ``master_offset - (x, y)`` must undo exactly that.
        """
        _, arr, field_data = _build(shift_x, shift_y)
        master_dapi = arr[:, 0]
        restain_dapi = arr[:, len(_MASTER_CH)]
        # Compare only the interior, away from the strip the shift leaves
        # uncovered (and from np.roll's wraparound in the synthetic source).
        pad = max(abs(shift_x), abs(shift_y))
        if pad:
            master_dapi = master_dapi[:, pad:-pad, pad:-pad]
            restain_dapi = restain_dapi[:, pad:-pad, pad:-pad]
        np.testing.assert_array_equal(restain_dapi, master_dapi)

    def test_a_transposed_shift_would_not_register(self) -> None:
        """Guard the axis order: swapping x and y must break registration.

        Without this, a transposed shift passes every other test whenever the
        two components happen to be similar -- exactly how the alignment.csv
        transposition survived for so long.
        """
        _, arr, _ = _build(6, -2)
        _, transposed, _ = _build_transposed(6, -2)
        interior = (slice(None), slice(6, -6), slice(6, -6))
        correct = np.array_equal(
            arr[:, len(_MASTER_CH)][interior], arr[:, 0][interior]
        )
        wrong = np.array_equal(
            transposed[:, len(_MASTER_CH)][interior],
            transposed[:, 0][interior],
        )
        assert correct and not wrong


def _build_transposed(shift_x: int, shift_y: int):  # type: ignore[no-untyped-def]
    """Build with the shift deliberately transposed, for the guard test."""
    field_data, mask_data, ff = _sources(shift_x, shift_y)
    spec = builder.RoundSpec(
        plate_id=4130,
        well=_make_well(_ROUND_IDS),
        shifts=np.array([[shift_y, shift_x]]),  # deliberately wrong
        image_ids=list(_ROUND_IDS),
        channel_data=_ROUND_CH,
        flatfield_dict=ff,
    )
    p_img, p_res, p_tr, p_co = _patches(field_data, mask_data)
    with p_img, p_res, p_tr, p_co:
        img, _, _ = builder._build_lazy_well_arrays(
            MagicMock(), None, _make_well(_MASTER_IDS), _MASTER_CH, ff,
            plate_id=4127, block_t=1, round_specs=[spec],
        )
        return img, np.asarray(img.compute()), field_data


class TestFieldResolution:
    def test_image_ids_preferred_over_listchildren_order(self) -> None:
        """Resolving by ID is what makes correspondence order-independent."""
        spec = builder.RoundSpec(
            plate_id=4130,
            well=_make_well([999]),
            shifts=np.array([[0, 0]]),
            image_ids=[110],
            channel_data=_ROUND_CH,
            flatfield_dict={},
        )
        assert spec.field_ids([0]) == [110]

    def test_falls_back_to_position_without_ids(self) -> None:
        spec = builder.RoundSpec(
            plate_id=4130,
            well=_make_well([777]),
            shifts=np.array([[0, 0]]),
            image_ids=None,
            channel_data=_ROUND_CH,
            flatfield_dict={},
        )
        assert spec.field_ids([0]) == [777]
