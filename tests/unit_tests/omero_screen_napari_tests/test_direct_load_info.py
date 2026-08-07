"""Unit tests for the plate/well/timepoint coverage summary."""

from omero_screen_napari.direct_load_info import (
    format_info_html,
    summarise_selection,
)


def _cells(*specs):
    """Build a cell population from (image_id, row, col, predicted) tuples."""
    return list(specs)


class TestFlatSummary:
    """Tests for the no-classifier reading."""

    def test_counts_labelled_and_available(self):
        """Cells split into labelled and still-available."""
        cells = _cells(
            (1, 10, 10, None),
            (1, 20, 20, None),
            (1, 30, 30, None),
        )
        annotations = {(1, 10, 10): "metaphase"}

        info = summarise_selection(cells, annotations, split_by_class=False)

        assert info.total_cells == 3
        assert info.labelled == 1
        assert info.available == 2

    def test_label_breakdown_is_counted(self):
        """Each saved label is tallied."""
        cells = _cells(
            (1, 1, 1, None),
            (1, 2, 2, None),
            (1, 3, 3, None),
            (1, 4, 4, None),
        )
        annotations = {
            (1, 1, 1): "metaphase",
            (1, 2, 2): "metaphase",
            (1, 3, 3): "prophase",
        }

        info = summarise_selection(cells, annotations, split_by_class=False)

        assert info.labels == {"metaphase": 2, "prophase": 1}
        assert info.available == 1

    def test_saved_unassigned_is_a_label_not_availability(self):
        """A saved 'unassigned' counts as labelled, not as still-available.

        The word is overloaded: 'unassigned' is a real stored label AND
        the informal name for untouched cells. They must not be conflated
        — a saved 'unassigned' is excluded from future loads.
        """
        cells = _cells((1, 1, 1, None), (1, 2, 2, None))
        annotations = {(1, 1, 1): "unassigned"}

        info = summarise_selection(cells, annotations, split_by_class=False)

        assert info.labelled == 1
        assert info.available == 1
        assert info.labels == {"unassigned": 1}

    def test_no_breakdown_without_classifier(self):
        """The per-class table is absent when no classifier is selected."""
        info = summarise_selection(
            _cells((1, 1, 1, None)), {}, split_by_class=False
        )

        assert info.by_predicted_class is None

    def test_annotations_outside_the_selection_are_ignored(self):
        """Labels for cells in another field don't inflate the count."""
        cells = _cells((1, 1, 1, None))
        annotations = {(1, 1, 1): "metaphase", (99, 5, 5): "metaphase"}

        info = summarise_selection(cells, annotations, split_by_class=False)

        assert info.total_cells == 1
        assert info.labelled == 1

    def test_empty_population(self):
        """An empty selection reports zeros rather than failing."""
        info = summarise_selection([], {}, split_by_class=False)

        assert info.total_cells == 0
        assert info.available == 0
        assert info.labels == {}


class TestClassSplitSummary:
    """Tests for the relabelling reading."""

    def test_splits_availability_by_predicted_class(self):
        """Each predicted class reports its own remaining pool."""
        cells = _cells(
            (1, 1, 1, "metaphase"),
            (1, 2, 2, "metaphase"),
            (1, 3, 3, "metaphase"),
            (1, 4, 4, "interphase"),
        )
        annotations = {(1, 1, 1): "metaphase"}

        info = summarise_selection(cells, annotations, split_by_class=True)
        by_class = {b.predicted_class: b for b in info.by_predicted_class}

        assert by_class["metaphase"].total == 3
        assert by_class["metaphase"].labelled == 1
        assert by_class["metaphase"].available == 2
        assert by_class["interphase"].available == 1

    def test_shows_what_predictions_were_relabelled_as(self):
        """Disagreement with the model is visible per class."""
        cells = _cells(
            (1, 1, 1, "metaphase"),
            (1, 2, 2, "metaphase"),
            (1, 3, 3, "metaphase"),
        )
        annotations = {
            (1, 1, 1): "metaphase",
            (1, 2, 2): "prometaphase",
        }

        info = summarise_selection(cells, annotations, split_by_class=True)
        metaphase = info.by_predicted_class[0]

        assert metaphase.labels == {"metaphase": 1, "prometaphase": 1}

    def test_cells_without_a_prediction_are_grouped(self):
        """Unpredicted cells get their own row rather than vanishing."""
        cells = _cells((1, 1, 1, None), (1, 2, 2, "metaphase"))

        info = summarise_selection(cells, {}, split_by_class=True)
        names = {b.predicted_class for b in info.by_predicted_class}

        assert "(no prediction)" in names

    def test_classes_are_ordered_by_size(self):
        """The biggest pool is listed first."""
        cells = _cells(
            (1, 1, 1, "small"),
            (1, 2, 2, "big"),
            (1, 3, 3, "big"),
            (1, 4, 4, "big"),
        )

        info = summarise_selection(cells, {}, split_by_class=True)

        assert [b.predicted_class for b in info.by_predicted_class] == [
            "big",
            "small",
        ]

    def test_totals_match_the_flat_summary(self):
        """The split never disagrees with the headline numbers."""
        cells = _cells(
            (1, 1, 1, "a"),
            (1, 2, 2, "b"),
            (1, 3, 3, "b"),
        )
        annotations = {(1, 2, 2): "x"}

        info = summarise_selection(cells, annotations, split_by_class=True)

        assert sum(b.total for b in info.by_predicted_class) == info.total_cells
        assert (
            sum(b.available for b in info.by_predicted_class) == info.available
        )
        assert (
            sum(b.labelled for b in info.by_predicted_class) == info.labelled
        )


class TestFormatting:
    """Tests for the rendered output."""

    def test_reports_headline_numbers(self):
        """Totals appear in the rendered text."""
        info = summarise_selection(
            _cells((1, 1, 1, None), (1, 2, 2, None)),
            {(1, 1, 1): "metaphase"},
            split_by_class=False,
        )

        html = format_info_html(info, "Plate 1 · Well A1 · t=0")

        assert "Plate 1 · Well A1 · t=0" in html
        assert "metaphase" in html

    def test_empty_selection_says_so(self):
        """An empty result explains itself instead of showing a blank table."""
        info = summarise_selection([], {}, split_by_class=False)

        assert "No cells found" in format_info_html(info, "header")

    def test_class_table_rendered_when_split(self):
        """The per-class table appears only in the split reading."""
        info = summarise_selection(
            _cells((1, 1, 1, "metaphase")), {}, split_by_class=True
        )

        assert "By predicted class" in format_info_html(info, "header")
