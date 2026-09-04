"""Tests for the per-well cell-cycle montage.

Selection and scaling are what decide what the figure *claims*, so they carry
the weight here:

* the draw must be reproducible from the seed, or a figure cannot be regenerated
  and the panels are effectively hand-picked;
* display limits must be shared across phases, since per-crop scaling would
  normalise away the size and intensity differences the montage exists to show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from omero_screen_napari.phase_montage import (
    DEFAULT_PHASES,
    _close,
    MontageConfig,
    MontageError,
    _crop_pixels,
    _outline,
    _resolve_overlay,
    build_montage,
    channel_limits,
    axis_column,
    build_pages,
    condition_column,
    export_plate_pdfs,
    export_well_pdf,
    plate_wells,
    render_montage,
    resolve_mask_label,
    select_cells,
    select_grid,
)
from omero_screen_napari.zarr_cache import PlateZarrWriter
from omero_screen_napari.zarr_cache.registry import ZarrPlateEntry, upsert


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OMERO_SCREEN_CACHE_PATH", str(tmp_path))
    return tmp_path


DEFAULT_PER_PHASE = {"G1": 10, "S": 8, "G2/M": 6, "Polyploid": 5}


def _cell_positions(
    per_phase: dict[str, int] | None = None,
) -> dict[int, tuple[int, int]]:
    """``{label: (cy, cx)}`` — the single source of truth for cell placement."""
    per_phase = per_phase or DEFAULT_PER_PHASE
    rng = np.random.default_rng(0)
    positions: dict[int, tuple[int, int]] = {}
    label = 0
    for count in per_phase.values():
        for _ in range(count):
            label += 1
            positions[label] = (
                int(rng.integers(80, 430)),
                int(rng.integers(80, 430)),
            )
    return positions


def _measurements(
    well: str = "C3", per_phase: dict[str, int] | None = None
) -> pl.DataFrame:
    per_phase = per_phase or DEFAULT_PER_PHASE
    positions = _cell_positions(per_phase)
    rows = []
    label = 0
    for phase, count in per_phase.items():
        for _ in range(count):
            label += 1
            cy, cx = positions[label]
            rows.append(
                {
                    "plate_id": 99,
                    "well": well,
                    "label": label,
                    "cell_cycle": phase,
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "equivalent_diameter_area_cell": (
                        34.0 if phase != "Polyploid" else 56.0
                    ),
                }
            )
    return pl.DataFrame(rows)


CHANNELS = ["DAPI_R1", "Tub_R1", "Y15_R1", "p21_R2", "TP53_R3"]


#: Realistic for these plates: plate 4127 is 1.187 um/px. The value matters —
#: at 0.3 um/px a 20 um scale bar is wider than half the crop and is correctly
#: suppressed, which silently disables the scale-bar assertions.
PIXEL_SIZE_UM = 1.2


def _build_plate(plate_id: int = 99, well: str = "C3", size: int = 512):
    """Write a synthetic plate whose masks match ``_measurements()``.

    The measurement centroids and the drawn masks are generated from the same
    RNG stream, so a crop centred on a measurement actually contains that cell.
    An earlier version drew masks at unrelated positions; every render test then
    passed against panels with no cell in them, which is precisely the class of
    fixture unrealism that hid the real-data bugs.

    The cell mask is deliberately labelled **differently** from the nuclei mask
    (offset by 1000), mirroring plate 4127 where nucleus 2184 sits in cell 2152.
    """
    rng = np.random.default_rng(1)
    image = rng.integers(
        200, 3000, (1, len(CHANNELS), size, size), dtype=np.uint16
    )
    nuc = np.zeros((1, size, size), dtype=np.uint32)
    cells = np.zeros((1, size, size), dtype=np.uint32)
    yy, xx = np.ogrid[:size, :size]
    for label, (cy, cx) in _cell_positions().items():
        nuc[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= 10**2] = label
        cells[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= 18**2] = label + 1000
    writer = PlateZarrWriter(plate_id, "demo", CHANNELS, PIXEL_SIZE_UM, 1)
    with writer:
        writer.ensure_plate(all_wells=[well])
        writer.write_well(well, image, nuc, cells)
    upsert(ZarrPlateEntry(plate_id=plate_id))


class TestSelection:
    def test_takes_the_requested_number_per_phase(self) -> None:
        cells, warnings = select_cells(
            _measurements(), "C3", MontageConfig(cells_per_phase=3)
        )
        assert {p: len(c) for p, c in cells.items()} == {
            "G1": 3,
            "S": 3,
            "G2/M": 3,
            "Polyploid": 3,
        }
        assert warnings == []

    def test_is_reproducible_from_the_seed(self) -> None:
        """Without this a published figure cannot be regenerated."""
        config = MontageConfig(seed=7)
        first, _ = select_cells(_measurements(), "C3", config)
        second, _ = select_cells(_measurements(), "C3", config)
        assert {p: [c.label for c in v] for p, v in first.items()} == {
            p: [c.label for c in v] for p, v in second.items()
        }

    def test_a_different_seed_draws_different_cells(self) -> None:
        a, _ = select_cells(_measurements(), "C3", MontageConfig(seed=1))
        b, _ = select_cells(_measurements(), "C3", MontageConfig(seed=2))
        assert [c.label for c in a["G1"]] != [c.label for c in b["G1"]]

    def test_draw_does_not_depend_on_row_order(self) -> None:
        """The DB may return rows in any order; the draw must not follow it."""
        df = _measurements()
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        config = MontageConfig(seed=3)
        a, _ = select_cells(df, "C3", config)
        b, _ = select_cells(shuffled, "C3", config)
        assert [c.label for c in a["G1"]] == [c.label for c in b["G1"]]

    def test_no_duplicate_cells_within_a_phase(self) -> None:
        cells, _ = select_cells(
            _measurements(), "C3", MontageConfig(cells_per_phase=5)
        )
        for refs in cells.values():
            labels = [c.label for c in refs]
            assert len(labels) == len(set(labels))

    def test_cells_come_from_the_phase_they_are_filed_under(self) -> None:
        df = _measurements()
        cells, _ = select_cells(df, "C3", MontageConfig())
        by_label = dict(
            zip(df["label"].to_list(), df["cell_cycle"].to_list(), strict=True)
        )
        for phase, refs in cells.items():
            for ref in refs:
                assert by_label[ref.label] == phase

    def test_sparse_phase_reports_rather_than_fails(self) -> None:
        cells, warnings = select_cells(
            _measurements(per_phase={"G1": 10, "S": 1, "G2/M": 0}),
            "C3",
            MontageConfig(cells_per_phase=4, phases=("G1", "S", "G2/M")),
        )
        assert len(cells["S"]) == 1
        assert cells["G2/M"] == []
        assert any("only 1 S" in w for w in warnings)
        assert any("no G2/M" in w for w in warnings)

    def test_edge_filter_drops_cells_whose_crop_overruns(self) -> None:
        df = pl.DataFrame(
            [
                {
                    "plate_id": 99,
                    "well": "C3",
                    "label": 1,
                    "cell_cycle": "G1",
                    "centroid_y": 5.0,
                    "centroid_x": 5.0,
                    "equivalent_diameter_area_cell": 34.0,
                },
                {
                    "plate_id": 99,
                    "well": "C3",
                    "label": 2,
                    "cell_cycle": "G1",
                    "centroid_y": 250.0,
                    "centroid_x": 250.0,
                    "equivalent_diameter_area_cell": 34.0,
                },
            ]
        )
        cells, _ = select_cells(
            df,
            "C3",
            MontageConfig(cells_per_phase=2, phases=("G1",)),
            canvas_hw=(512, 512),
            crop_px=128,
        )
        assert [c.label for c in cells["G1"]] == [2]

    def test_unknown_well_raises(self) -> None:
        with pytest.raises(MontageError, match="no measurements"):
            select_cells(_measurements(), "Z9", MontageConfig())

    def test_missing_phase_column_raises(self) -> None:
        df = _measurements().drop("cell_cycle")
        with pytest.raises(MontageError, match="cell-cycle column"):
            select_cells(df, "C3", MontageConfig())


class TestCropSizing:
    def test_large_cells_give_a_larger_crop(self) -> None:
        """A size tuned for G1 would clip exactly the polyploid phenotype."""
        small = _crop_pixels(
            MontageConfig(phases=("G1",)),
            _measurements(per_phase={"G1": 20}),
            "C3",
            0.3,
        )
        big = _crop_pixels(
            MontageConfig(phases=("Polyploid",)),
            _measurements(per_phase={"Polyploid": 20}),
            "C3",
            0.3,
        )
        assert big > small

    def test_does_not_depend_on_the_draw(self) -> None:
        """Sized from the plate, so the crop is stable across seeds and wells."""
        df = _measurements()
        a = _crop_pixels(MontageConfig(seed=1), df, "C3", 0.3)
        b = _crop_pixels(MontageConfig(seed=99), df, "C3", 0.3)
        assert a == b

    def test_only_shown_phases_count(self) -> None:
        """Excluding Polyploid must not leave the crop sized for it."""
        df = _measurements()
        with_poly = _crop_pixels(
            MontageConfig(phases=("G1", "Polyploid")), df, "C3", 0.3
        )
        without = _crop_pixels(MontageConfig(phases=("G1",)), df, "C3", 0.3)
        assert with_poly > without

    def test_explicit_microns_win(self) -> None:
        assert (
            _crop_pixels(
                MontageConfig(crop_um=30.0), _measurements(), "C3", 0.5
            )
            == 60
        )

    def test_always_even_and_bounded(self) -> None:
        px = _crop_pixels(MontageConfig(), _measurements(), "C3", 0.3)
        assert px % 2 == 0
        assert 32 <= px <= 1024


class TestOverlayResolution:
    def test_splits_composite_from_greyscale(self) -> None:
        overlay, grey = _resolve_overlay(CHANNELS, ("dapi", "tub"))
        assert [CHANNELS[i] for i in overlay] == ["DAPI_R1", "Tub_R1"]
        assert [CHANNELS[i] for i in grey] == ["Y15_R1", "p21_R2", "TP53_R3"]

    def test_matches_through_the_round_suffix(self) -> None:
        overlay, _ = _resolve_overlay(["DAPI_R1", "Tub_R1"], ("dapi",))
        assert overlay == [0]

    def test_repeated_stain_stays_greyscale(self) -> None:
        """Only the first match per role joins the composite."""
        names = ["DAPI_R1", "Tub_R1", "DAPI_R2"]
        overlay, grey = _resolve_overlay(names, ("dapi", "tub"))
        assert overlay == [0, 1]
        assert grey == [2]

    def test_absent_overlay_channel_is_skipped(self) -> None:
        overlay, grey = _resolve_overlay(["p21_R1", "TP53_R1"], ("dapi",))
        assert overlay == []
        assert grey == [0, 1]


class TestOutline:
    def test_outlines_only_the_target_cell(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[5:15, 5:15] = 1
        mask[25:35, 25:35] = 2
        outline = _outline(mask, 1)
        assert outline[:20, :20].any()
        assert not outline[20:, 20:].any()

    def test_absent_label_outlines_nothing(self) -> None:
        """The centroid can land on a neighbour; outline nothing, not the wrong cell."""
        mask = np.zeros((20, 20), dtype=np.uint32)
        mask[5:10, 5:10] = 3
        assert not _outline(mask, 99).any()


class TestLimitsAndRender:
    def test_limits_are_per_channel_and_shared(self) -> None:
        _build_plate()
        limits = channel_limits(99, ["C3"])
        assert set(limits) == set(range(len(CHANNELS)))
        for lo, hi in limits.values():
            assert lo < hi

    def test_build_montage_resolves_everything(self) -> None:
        _build_plate()
        montage = build_montage(99, "C3", _measurements(), MontageConfig())
        assert montage.channel_names == CHANNELS
        assert [montage.channel_names[i] for i in montage.overlay_indices] == [
            "DAPI_R1",
            "Tub_R1",
        ]
        assert len(montage.grey_indices) == 3
        assert set(montage.cells) == set(DEFAULT_PHASES)
        assert montage.pixel_size_um == pytest.approx(PIXEL_SIZE_UM)

    def test_missing_zarr_store_raises(self) -> None:
        with pytest.raises(MontageError, match="no zarr cache"):
            build_montage(12345, "C3", _measurements(), MontageConfig())

    def test_exports_a_pdf(self, tmp_path: Path) -> None:
        _build_plate()
        out = tmp_path / "figs"
        path = export_well_pdf(
            99, "C3", _measurements(), out, MontageConfig(cells_per_phase=2)
        )
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.read_bytes()[:4] == b"%PDF"

    def test_export_is_reproducible(self, tmp_path: Path) -> None:
        """Same seed, same plate: the same cells are drawn."""
        _build_plate()
        config = MontageConfig(seed=5, cells_per_phase=2)
        a = build_montage(99, "C3", _measurements(), config)
        b = build_montage(99, "C3", _measurements(), config)
        assert {p: [c.label for c in v] for p, v in a.cells.items()} == {
            p: [c.label for c in v] for p, v in b.cells.items()
        }


class TestRealSchemaColumns:
    """Column names as omero-screen actually writes them.

    Plate 4127 exports `centroid-0-nuc` / `centroid-1-nuc`, not `centroid_y` /
    `centroid_x`, so the first run against real data failed with "No centroid
    columns". regionprops names are per compartment and the montage has to
    accept all of them.
    """

    def _regionprops_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "plate_id": 4127,
                    "well": "C3",
                    "label": 2184,
                    "cell_cycle": "G1",
                    "centroid-0-nuc": 900.0,
                    "centroid-1-nuc": 800.0,
                    "centroid-0-cell": 905.0,
                    "centroid-1-cell": 812.0,
                    "equivalent_diameter_area_cell": 41.0,
                    "equivalent_diameter_area_nucleus": 14.0,
                }
            ]
        )

    def test_regionprops_centroid_columns_resolve(self) -> None:
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert len(cells["G1"]) == 1

    def test_nucleus_centroid_is_preferred_over_cell(self) -> None:
        """The nucleus is the segmentation anchor the phase call is made on."""
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert cells["G1"][0].centroid == (900.0, 800.0)

    def test_cell_extent_preferred_over_nuclear(self) -> None:
        """The crop must hold the cell; in polyploid cells the two diverge."""
        cells, _ = select_cells(
            self._regionprops_df(), "C3", MontageConfig(phases=("G1",))
        )
        assert cells["G1"][0].diameter == 41.0

    def test_error_names_the_columns_it_found(self) -> None:
        df = (
            self._regionprops_df()
            .rename(
                {
                    "centroid-0-nuc": "y_position",
                    "centroid-1-nuc": "x_position",
                }
            )
            .drop(["centroid-0-cell", "centroid-1-cell"])
        )
        with pytest.raises(MontageError, match="no centroid columns at all"):
            select_cells(df, "C3", MontageConfig())

    def test_area_columns_still_work(self) -> None:
        """Plates without an equivalent-diameter column fall back to area."""
        df = (
            self._regionprops_df()
            .drop(
                [
                    "equivalent_diameter_area_cell",
                    "equivalent_diameter_area_nucleus",
                ]
            )
            .with_columns(pl.lit(1256.0).alias("area_cell"))
        )
        cells, _ = select_cells(df, "C3", MontageConfig(phases=("G1",)))
        # area 1256 -> equivalent diameter 2*sqrt(1256/pi) ~= 40
        assert cells["G1"][0].diameter == pytest.approx(40.0, abs=0.5)


class TestMaskLabelResolution:
    """CellView's `label` is the nucleus label; the cell mask is labelled independently.

    On plate 4127, nucleus 2184 sits inside cell 2152. Matching on the ID alone
    found nothing and every panel was drawn with no outline at all — silently,
    because an empty boundary is a valid array.
    """

    def _mask(self, label: int) -> np.ndarray:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[10:30, 10:30] = label
        return mask

    def test_matching_label_is_used_directly(self) -> None:
        assert resolve_mask_label(self._mask(2184), 2184) == 2184

    def test_falls_back_to_the_label_under_the_centre(self) -> None:
        """The nucleus centroid is the crop centre by construction."""
        assert resolve_mask_label(self._mask(2152), 2184) == 2152

    def test_background_centre_resolves_to_nothing(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint32)
        mask[0:5, 0:5] = 7
        assert resolve_mask_label(mask, 2184) is None

    def test_outline_is_drawn_for_the_resolved_label(self) -> None:
        mask = self._mask(2152)
        outline = _outline(mask, resolve_mask_label(mask, 2184))
        assert outline.any(), "the cell must be outlined via the centre lookup"

    def test_outline_of_none_is_empty_not_an_error(self) -> None:
        assert not _outline(self._mask(5), None).any()

    def test_outline_does_not_cover_the_cell_interior(self) -> None:
        """It is an outline, not a filled overlay — the pixels must stay visible."""
        mask = self._mask(3)
        outline = _outline(mask, 3, width=3)
        interior = np.zeros_like(mask, dtype=bool)
        interior[15:25, 15:25] = True
        assert not (outline & interior).any()

    def test_outline_width_is_thicker_than_one_pixel(self) -> None:
        """A 1px boundary is invisible once scaled to a montage panel."""
        mask = self._mask(3)
        assert (
            _outline(mask, 3, width=3).sum() > _outline(mask, 3, width=1).sum()
        )


class TestPanelAnnotation:
    """The contour and scale bar go on *every* panel, not just the composite.

    A montage is read panel by panel — a reader looking at one marker in
    isolation has no other cue to which cell is the subject, or to scale.
    """

    def _figure(self):  # type: ignore[no-untyped-def]
        _build_plate()
        montage = build_montage(
            99, "C3", _measurements(), MontageConfig(cells_per_phase=1)
        )
        return render_montage(montage), montage

    def test_every_panel_has_a_scale_bar(self) -> None:
        fig, montage = self._figure()
        try:
            for ax in fig.axes:
                assert ax.patches, "panel has no scale bar"
        finally:
            _close(fig)

    def test_every_panel_carries_an_image(self) -> None:
        fig, montage = self._figure()
        try:
            n_cols = 1 + len(montage.grey_indices)
            assert len(fig.axes) == n_cols * sum(
                len(v) for v in montage.cells.values()
            )
            for ax in fig.axes:
                assert ax.images
        finally:
            _close(fig)

    def test_greyscale_panels_are_rgb_so_the_contour_can_be_coloured(
        self,
    ) -> None:
        fig, montage = self._figure()
        try:
            # Column 0 is the composite; the rest are the greyscale markers.
            for ax in fig.axes:
                data = ax.images[0].get_array()
                assert data.ndim == 3 and data.shape[2] == 3
        finally:
            _close(fig)

    def test_contour_colour_present_on_a_grey_panel(self) -> None:
        """The grey panel must actually carry the outline pixels."""
        _build_plate()
        montage = build_montage(
            99, "C3", _measurements(), MontageConfig(cells_per_phase=1)
        )
        fig = render_montage(montage)
        try:
            n_cols = 1 + len(montage.grey_indices)
            grey_ax = fig.axes[1]  # first row, first greyscale marker
            data = np.asarray(grey_ax.images[0].get_array())
            # Outline pixels are the only ones where the three bands differ.
            assert (data[..., 0] != data[..., 2]).any(), (
                "greyscale panel carries no coloured contour"
            )
            assert n_cols > 1
        finally:
            _close(fig)

    def test_no_scale_bar_without_a_pixel_size(self) -> None:
        """A bar with no known scale would be a lie."""
        from omero_screen_napari.phase_montage import _add_scale_bar

        fig = None
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            _add_scale_bar(ax, 128, None)
            assert not ax.patches
        finally:
            plt.close(fig)

    def test_no_scale_bar_when_it_would_span_the_panel(self) -> None:
        from omero_screen_napari.phase_montage import _add_scale_bar

        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            # 20 um at 1 um/px is 20px, against a 24px crop.
            _add_scale_bar(ax, 24, 1.0)
            assert not ax.patches
        finally:
            plt.close(fig)


class TestBatchExport:
    """Whole-plate export. One bad well must not abandon the other twenty."""

    def test_all_wells_by_default(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = _measurements(well="C3")
        paths, failures = export_plate_pdfs(
            99, df, tmp_path, MontageConfig(cells_per_phase=1)
        )
        # The filename carries the axis: plate 4127 has both a well G1 and a
        # phase G1, so the value alone would be ambiguous.
        assert [p.name for p in paths] == ["plate99_well-C3_montage.pdf"]
        assert failures == []

    def test_plate_wells_lists_every_well(self) -> None:
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        assert plate_wells(df) == ["A1", "C3"]

    def test_explicit_subset_is_honoured(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        paths, _ = export_plate_pdfs(
            99,
            df,
            tmp_path,
            MontageConfig(cells_per_phase=1),
            wells=["C3"],
        )
        assert len(paths) == 1

    def test_an_uncached_well_is_excluded_and_reported(
        self, tmp_path: Path, caplog
    ) -> None:
        """The whole point of the batch: good pages survive one bad well.

        A well vanishing because nobody cached it looks identical to a well with
        no cells in that phase, so it has to be said out loud.
        """
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="Z9")])
        pages = build_pages(99, df, MontageConfig(cells_per_phase=1))
        assert [p.page_label for p in pages] == ["C3"]
        assert any("not in the zarr cache" in w for w in pages[0].missing)
        paths, _ = export_plate_pdfs(
            99, df, tmp_path, MontageConfig(cells_per_phase=1)
        )
        assert len(paths) == 1

    def test_progress_callback_reports_each_page(self, tmp_path: Path) -> None:
        _build_plate(well="C3")
        df = pl.concat([_measurements(well="C3"), _measurements(well="A1")])
        seen: list[tuple[str, int, int]] = []
        export_plate_pdfs(
            99,
            df,
            tmp_path,
            MontageConfig(cells_per_phase=1),
            on_progress=lambda w, i, n: seen.append((w, i, n)),
        )
        # Only C3 is cached, so only C3 becomes a page.
        assert seen == [("C3", 0, 1)]


class TestTypography:
    """Arial 7pt where available, embedded editable, never DejaVu by accident.

    The style context has to wrap *saving*, not just drawing: matplotlib bakes a
    font size into a Text artist at creation but resolves the family at draw
    time, so a context around figure construction alone still wrote DejaVu Sans
    into the PDF.

    Arial is a system font on macOS but is **not** installed on the Linux CI
    runner, so asserting "Arial is embedded" outright is an assertion about the
    machine, not about this code. The portable check is that the embedded font
    is whatever matplotlib resolves for our configured family — which is Arial
    on a machine that has it, and the documented fallback on one that does not.
    """

    def _embedded_fonts(self, tmp_path: Path) -> set[bytes]:
        import re

        _build_plate()
        path = export_well_pdf(
            99,
            "C3",
            _measurements(),
            tmp_path,
            MontageConfig(cells_per_phase=1),
        )
        return set(
            re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", path.read_bytes())
        )

    def _configured_font_name(self) -> str:
        """The family matplotlib actually resolves for our rc settings."""
        from matplotlib import font_manager

        from omero_screen_napari.phase_montage import FONT_FAMILY

        path = font_manager.findfont(
            font_manager.FontProperties(
                family=[FONT_FAMILY, "Helvetica", "DejaVu Sans"]
            )
        )
        return str(font_manager.get_font(path).family_name)

    def test_pdf_embeds_the_configured_font(self, tmp_path: Path) -> None:
        """Catches the real bug: DejaVu embedded while Arial was available."""
        fonts = self._embedded_fonts(tmp_path)
        expected = self._configured_font_name().replace(" ", "")
        assert any(expected.encode() in f for f in fonts), (
            f"expected {expected}, got {fonts}"
        )

    @pytest.mark.skipif(
        "Arial"
        not in {
            f.name
            for f in __import__(
                "matplotlib.font_manager", fromlist=["fontManager"]
            ).fontManager.ttflist
        },
        reason="Arial is not installed on this machine (e.g. the Linux CI runner)",
    )
    def test_pdf_embeds_arial_where_available(self, tmp_path: Path) -> None:
        fonts = self._embedded_fonts(tmp_path)
        assert any(b"Arial" in f for f in fonts), f"got {fonts}"
        assert not any(b"DejaVu" in f for f in fonts), f"got {fonts}"

    def test_text_is_truetype_not_outlines(self, tmp_path: Path) -> None:
        """Type 3 would arrive in Illustrator as uneditable outlines."""
        _build_plate()
        path = export_well_pdf(
            99,
            "C3",
            _measurements(),
            tmp_path,
            MontageConfig(cells_per_phase=1),
        )
        assert b"/Type3" not in path.read_bytes()

    def test_arial_is_requested_first(self) -> None:
        """Environment-independent: what we ask for, whatever the machine has."""
        import matplotlib.pyplot as plt

        from omero_screen_napari.phase_montage import (
            FONT_FAMILY,
            FONT_SIZE,
            montage_style,
        )

        with montage_style():
            assert plt.rcParams["font.sans-serif"][0] == FONT_FAMILY
            assert plt.rcParams["font.size"] == FONT_SIZE
            assert plt.rcParams["pdf.fonttype"] == 42

    def test_style_does_not_leak(self) -> None:
        """A montage export must not restyle every other plot in the session."""
        import matplotlib.pyplot as plt

        from omero_screen_napari.phase_montage import montage_style

        before = plt.rcParams["font.size"]
        with montage_style():
            assert plt.rcParams["font.size"] == 7
        assert plt.rcParams["font.size"] == before


def _two_condition_df() -> pl.DataFrame:
    """Two siRNAs x two replicate wells each, so replicate spread is testable."""
    frames = []
    for sirna, wells in (("Scrm", ["A1", "A2"]), ("PPP4C", ["B1", "B2"])):
        for well in wells:
            frames.append(
                _measurements(well=well).with_columns(
                    pl.lit(sirna).alias("sirna"),
                    pl.lit("RPE-1").alias("cell_line"),
                )
            )
    # Labels must be unique per well for the spread assertions to mean anything.
    out = []
    for offset, frame in enumerate(frames):
        out.append(frame.with_columns(pl.col("label") + offset * 1000))
    return pl.concat(out)


class TestConditionAxis:
    def test_detects_the_column_that_varies(self) -> None:
        """cell_line is constant on 4127; sirna is the real phenotype axis."""
        assert condition_column(_two_condition_df()) == "sirna"

    def test_constant_column_is_not_a_condition(self) -> None:
        df = _measurements().with_columns(pl.lit("RPE-1").alias("cell_line"))
        assert condition_column(df) is None

    def test_explicit_column_wins(self) -> None:
        df = _two_condition_df()
        cfg = MontageConfig(condition_col="cell_line")
        assert condition_column(df, cfg) == "cell_line"

    def test_unknown_explicit_column_raises(self) -> None:
        with pytest.raises(MontageError, match="not in the CellView data"):
            condition_column(
                _two_condition_df(), MontageConfig(condition_col="nope")
            )

    def test_condition_axis_without_a_variable_explains_itself(self) -> None:
        cfg = MontageConfig(pages="phase", rows="condition")
        with pytest.raises(MontageError, match="nothing to compare"):
            axis_column(_measurements(), "condition", cfg)


class TestGridSelection:
    def test_pages_and_rows_must_differ(self) -> None:
        cfg = MontageConfig(pages="phase", rows="phase")
        with pytest.raises(MontageError, match="must differ"):
            select_grid(_two_condition_df(), cfg)

    def test_phase_pages_condition_rows(self) -> None:
        """The requested layout: a page per phase, a row per phenotype."""
        cfg = MontageConfig(
            pages="phase", rows="condition", cells_per_phase=1
        )
        grid, _ = select_grid(_two_condition_df(), cfg)
        assert set(grid) == set(DEFAULT_PHASES)
        assert set(grid["G1"]) == {"Scrm", "PPP4C"}
        assert len(grid["G1"]["Scrm"]) == 1

    def test_rows_carry_their_condition(self) -> None:
        cfg = MontageConfig(
            pages="phase", rows="condition", cells_per_phase=1
        )
        grid, _ = select_grid(_two_condition_df(), cfg)
        assert grid["G1"]["PPP4C"][0].condition == "PPP4C"

    def test_cells_come_from_the_phase_of_their_page(self) -> None:
        cfg = MontageConfig(
            pages="phase", rows="condition", cells_per_phase=2
        )
        grid, _ = select_grid(_two_condition_df(), cfg)
        for phase, rows in grid.items():
            for cells in rows.values():
                assert all(c.phase == phase for c in cells)

    def test_replicates_are_spread_across_wells(self) -> None:
        """Two cells for a condition must not both come from one well.

        Otherwise the row is a portrait of that well, and a well-specific
        artefact reads as a phenotype.
        """
        cfg = MontageConfig(
            pages="phase", rows="condition", cells_per_phase=2
        )
        grid, _ = select_grid(_two_condition_df(), cfg)
        cells = grid["G1"]["Scrm"]
        assert len(cells) == 2
        assert {c.well for c in cells} == {"A1", "A2"}

    def test_single_well_group_still_works(self) -> None:
        """pages=well confines a group to one well; the rotation must no-op."""
        cfg = MontageConfig(pages="well", rows="phase", cells_per_phase=2)
        grid, _ = select_grid(_two_condition_df(), cfg)
        assert {c.well for c in grid["A1"]["G1"]} == {"A1"}

    def test_condition_pages_phase_rows(self) -> None:
        cfg = MontageConfig(
            pages="condition", rows="phase", cells_per_phase=1
        )
        grid, _ = select_grid(_two_condition_df(), cfg)
        assert set(grid) == {"Scrm", "PPP4C"}
        assert set(grid["Scrm"]) == set(DEFAULT_PHASES)

    def test_grid_is_reproducible_from_the_seed(self) -> None:
        cfg = MontageConfig(
            pages="phase", rows="condition", cells_per_phase=2, seed=5
        )
        df = _two_condition_df()
        first, _ = select_grid(df, cfg)
        second, _ = select_grid(df, cfg)
        assert [c.label for c in first["S"]["Scrm"]] == [
            c.label for c in second["S"]["Scrm"]
        ]


class TestPageFilenames:
    def test_filename_carries_the_axis(self, tmp_path: Path) -> None:
        """Plate 4127 has both a well G1 and a phase G1."""
        from omero_screen_napari.phase_montage import _safe

        assert _safe("G2/M") == "G2-M"
        _build_plate(well="C3")
        paths, _ = export_plate_pdfs(
            99, _measurements(well="C3"), tmp_path,
            MontageConfig(cells_per_phase=1),
        )
        assert paths[0].name == "plate99_well-C3_montage.pdf"


class TestLabelCollision:
    """A neighbour's cell label can equal the target's nucleus label.

    Labels are dense small integers and a crop holds ten to twenty cells, so on
    plate 4127 this happened for 3.9% of cells — and matching on the ID first
    outlined an unrelated neighbour in 12 of those 13 cases. The centre pixel is
    authoritative because the nucleus centroid is the crop centre by
    construction.
    """

    def _colliding_mask(self) -> np.ndarray:
        """Target cell 2152 at the centre; an unrelated cell 2184 at the edge."""
        mask = np.zeros((60, 60), dtype=np.uint32)
        mask[20:40, 20:40] = 2152  # contains the centre (30, 30)
        mask[0:8, 0:8] = 2184  # coincidentally equals the nucleus label
        return mask

    def test_centre_wins_over_a_coincidental_id_match(self) -> None:
        assert resolve_mask_label(self._colliding_mask(), 2184) == 2152

    def test_the_outline_encloses_the_centre(self) -> None:
        """The failure was visible as an outline nowhere near the cell."""
        mask = self._colliding_mask()
        label = resolve_mask_label(mask, 2184)
        outline = _outline(mask, label)
        ys, xs = np.nonzero(outline)
        assert ys.min() >= 15 and ys.max() <= 45, "outline is at the panel edge"

    def test_nuclei_mask_still_resolves_to_itself(self) -> None:
        """On the nuclei mask the ID does match, and the centre agrees."""
        mask = np.zeros((60, 60), dtype=np.uint32)
        mask[25:35, 25:35] = 2184
        assert resolve_mask_label(mask, 2184) == 2184

    def test_background_centre_falls_back_only_when_near(self) -> None:
        """A far-away ID match is the same coincidence without a contradiction."""
        far = np.zeros((60, 60), dtype=np.uint32)
        far[0:5, 0:5] = 2184
        assert resolve_mask_label(far, 2184) is None

        near = np.zeros((60, 60), dtype=np.uint32)
        near[28:33, 28:33] = 0
        near[26:29, 26:29] = 2184  # within the centre radius
        assert resolve_mask_label(near, 2184) == 2184
