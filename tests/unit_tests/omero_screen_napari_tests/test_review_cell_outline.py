"""Per-frame crops of the reviewed cell and lazy Follow panning (#76)."""

import numpy as np
import pytest
from omero_screen_napari.review.cell_outline import (
    cell_crop,
    needs_recentre,
    patch_crop,
)
from omero_screen_napari.review.masks import load_patch, save_patch


def stack() -> np.ndarray:
    """Two frames, 1000²: a small nucleus 5 and a large nucleus 7."""
    nuclei = np.zeros((2, 1000, 1000), dtype=np.uint32)
    nuclei[0, 495:505, 495:505] = 5
    nuclei[1, 300:600, 300:600] = 7
    nuclei[1, 100:110, 100:110] = 5
    return nuclei


def test_crop_holds_only_the_cell_at_its_canvas_position() -> None:
    """The crop is 1 on the cell, and (y0, x0) maps it back to the canvas."""
    nuclei = stack()
    crop = cell_crop(nuclei, 0, 5, 500, 500)
    assert crop is not None
    mask, y0, x0 = crop
    assert mask.dtype == np.uint8
    assert mask.shape == (128, 128)
    canvas = np.zeros((1000, 1000), dtype=np.uint8)
    canvas[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
    assert np.array_equal(canvas, (nuclei[0] == 5).astype(np.uint8))


def test_crop_grows_until_a_large_nucleus_fits() -> None:
    """A nucleus wider than the first window is not clipped."""
    nuclei = stack()
    crop = cell_crop(nuclei, 1, 7, 450, 450)
    assert crop is not None
    mask, _, _ = crop
    assert mask.sum() == 300 * 300


def test_crop_stops_at_the_largest_window() -> None:
    """The window never grows past max_half."""
    nuclei = stack()
    crop = cell_crop(nuclei, 1, 7, 450, 450, half=16, max_half=64)
    assert crop is not None
    assert crop[0].shape == (128, 128)


def test_crop_at_the_canvas_edge_is_not_grown() -> None:
    """A nucleus touching the canvas edge counts as whole."""
    nuclei = np.zeros((1, 200, 200), dtype=np.uint32)
    nuclei[0, 0:10, 0:10] = 3
    crop = cell_crop(nuclei, 0, 3, 5, 5)
    assert crop is not None
    mask, y0, x0 = crop
    assert (y0, x0) == (0, 0)
    assert mask.shape == (69, 69)
    assert mask.sum() == 100


def test_no_crop_when_the_label_is_absent() -> None:
    """A label not near the centroid gives no outline."""
    assert cell_crop(stack(), 0, 9, 500, 500) is None


def test_crop_reads_a_lazy_stack() -> None:
    """Dask stacks (corrected nuclei) give the same crop as numpy."""
    da = pytest.importorskip("dask.array")
    nuclei = stack()
    lazy = da.from_array(nuclei, chunks=(1, 256, 256))
    a, b = cell_crop(nuclei, 0, 5, 500, 500), cell_crop(lazy, 0, 5, 500, 500)
    assert a is not None and b is not None
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_patch_crop_is_uint8_at_the_patch_origin() -> None:
    """A reviewer mask is shown as stored."""
    mask = np.array([[True, False], [True, True]])
    out, y0, x0 = patch_crop((10, 20, mask))
    assert (y0, x0) == (10, 20)
    assert out.dtype == np.uint8 and out.sum() == 3


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ((0.0, 0.0), False),
        ((25.0, 0.0), False),
        ((35.0, 0.0), True),
        ((0.0, -61.0), True),
    ],
)
def test_recentre_only_outside_the_central_region(
    cell: tuple[float, float], expected: bool
) -> None:
    """View 100 × 200 centred at 0: the cell may move 30 / 60 before a pan."""
    assert needs_recentre(cell, (0.0, 0.0), (100.0, 200.0)) is expected


def test_recentre_without_a_view_size() -> None:
    """An unknown view size always recentres, as before."""
    assert needs_recentre((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def test_patch_reads_are_cached_until_the_file_changes(tmp_path) -> None:
    """A patch is read once, and read again when it is saved again."""
    patch_dir = tmp_path / "edits"
    rel = save_patch(patch_dir, "C2", 3, 7, 1, 2, np.ones((2, 2), bool))
    first = load_patch(tmp_path, rel)
    assert load_patch(tmp_path, rel)[2] is first[2]
    assert not first[2].flags.writeable
    save_patch(patch_dir, "C2", 3, 7, 4, 5, np.ones((3, 3), bool))
    again = load_patch(tmp_path, rel)
    assert again[:2] == (4, 5) and again[2].shape == (3, 3)


def test_outline_traces_the_nucleus_on_the_canvas() -> None:
    """The polygon sits on the nucleus' border, in canvas coordinates, at its frame."""
    from omero_screen_napari.review.cell_outline import crop_outline

    nuclei = stack()
    crop = cell_crop(nuclei, 0, 5, 500, 500)
    assert crop is not None
    polygons = crop_outline(crop, 0)
    assert len(polygons) == 1
    poly = polygons[0]
    assert poly.shape[1] == 3
    assert np.all(poly[:, 0] == 0)
    # The nucleus covers rows/cols 495..504; the 0.5 iso-line runs half a pixel outside.
    assert poly[:, 1].min() == pytest.approx(494.5)
    assert poly[:, 1].max() == pytest.approx(504.5)
    assert poly[:, 2].min() == pytest.approx(494.5)
    assert poly[:, 2].max() == pytest.approx(504.5)


def test_outline_of_a_nucleus_at_the_crop_border_is_closed() -> None:
    """A mask touching the crop edge still gives one closed polygon."""
    from omero_screen_napari.review.cell_outline import crop_outline

    mask = np.ones((4, 4), dtype=np.uint8)
    (poly,) = crop_outline((mask, 10, 20), 3)
    assert np.allclose(poly[0], poly[-1])
    assert poly[:, 1].min() == pytest.approx(9.5)
    assert poly[:, 2].max() == pytest.approx(23.5)


def test_outline_of_an_empty_mask_is_empty() -> None:
    """No nucleus, no polygon."""
    from omero_screen_napari.review.cell_outline import crop_outline

    assert crop_outline((np.zeros((5, 5), dtype=np.uint8), 0, 0), 0) == []


def test_frames_are_ordered_outwards_from_the_centre() -> None:
    """The frame on screen first, then its neighbours, earlier first on ties."""
    from omero_screen_napari.review.cell_outline import frames_from

    assert frames_from([1, 2, 3, 4, 5, 9], 3) == [3, 2, 4, 1, 5, 9]
