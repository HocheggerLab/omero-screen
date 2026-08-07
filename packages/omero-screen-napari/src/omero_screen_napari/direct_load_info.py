"""Coverage summary for a plate/well/timepoint selection.

Answers "how much of this population have I already labelled, and what is
left?" before a load, by crossing the cell population in CellView with the
annotations in the training database.

Two readings, matching the two ways the widget is used:

* **No classifier column selected** — a flat count: how many cells exist,
  how many are still untouched, and how the labelled ones break down.
* **A classifier column selected** — the same, split by the class that
  classifier predicted, which is what you want when relabelling the output
  of an earlier model (e.g. "of the 510 cells it called metaphase, which
  have I confirmed, and what did I call them?").

Note the word *unassigned* is overloaded: it is both a real saved label
and the informal name for "not yet looked at". Here the two are strictly
separate — ``available`` counts cells with no annotation at all, while a
saved ``unassigned`` appears in ``labels`` like any other class.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# (image_id, centroid_row, centroid_col) — the key CropPipeline matches on.
CellKey = tuple[int, int, int]


@dataclass(frozen=True)
class ClassBreakdown:
    """Coverage of one predicted class.

    Attributes:
        predicted_class: Class predicted by the selected classifier.
        total: Cells the classifier assigned to this class.
        labelled: How many of those carry an annotation.
        available: How many are still unannotated (and so loadable).
        labels: Annotation labels applied within this class, by count.
    """

    predicted_class: str
    total: int
    labelled: int
    available: int
    labels: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionInfo:
    """Coverage of a whole plate/well/timepoint selection.

    Attributes:
        total_cells: Cells present in CellView for the selection.
        labelled: Cells carrying an annotation for this classifier.
        available: Cells with no annotation yet.
        labels: Annotation labels applied, by count.
        by_predicted_class: Per-class breakdown, or ``None`` when no
            classifier column was selected. Ordered by descending total.
    """

    total_cells: int
    labelled: int
    available: int
    labels: dict[str, int] = field(default_factory=dict)
    by_predicted_class: list[ClassBreakdown] | None = None


def summarise_selection(
    cells: Iterable[tuple[int, int, int, str | None]],
    annotations: Mapping[CellKey, str],
    *,
    split_by_class: bool,
) -> SelectionInfo:
    """Cross a cell population with existing annotations.

    Args:
        cells: ``(image_id, centroid_row, centroid_col, predicted_class)``
            per cell. ``predicted_class`` may be ``None`` when the cell has
            no prediction, or always ``None`` when no classifier is in play.
        annotations: Saved label per cell key, as returned by
            :meth:`TrainingDB.get_annotated_labels`.
        split_by_class: Whether to produce the per-predicted-class
            breakdown.

    Returns:
        The coverage summary. Annotations whose key is absent from ``cells``
        are ignored — they describe cells outside this selection.
    """
    total = 0
    labelled = 0
    labels: dict[str, int] = {}
    per_class: dict[str, dict[str, int]] = {}
    per_class_total: dict[str, int] = {}
    per_class_labelled: dict[str, int] = {}

    for image_id, row, col, predicted in cells:
        total += 1
        key = (image_id, row, col)
        label = annotations.get(key)

        if split_by_class:
            name = predicted if predicted is not None else "(no prediction)"
            per_class_total[name] = per_class_total.get(name, 0) + 1
            per_class.setdefault(name, {})

        if label is None:
            continue

        labelled += 1
        labels[label] = labels.get(label, 0) + 1
        if split_by_class:
            per_class_labelled[name] = per_class_labelled.get(name, 0) + 1
            per_class[name][label] = per_class[name].get(label, 0) + 1

    breakdown: list[ClassBreakdown] | None = None
    if split_by_class:
        breakdown = [
            ClassBreakdown(
                predicted_class=name,
                total=per_class_total[name],
                labelled=per_class_labelled.get(name, 0),
                available=per_class_total[name]
                - per_class_labelled.get(name, 0),
                labels=dict(
                    sorted(
                        per_class[name].items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                ),
            )
            for name in per_class_total
        ]
        breakdown.sort(key=lambda b: (-b.total, b.predicted_class))

    return SelectionInfo(
        total_cells=total,
        labelled=labelled,
        available=total - labelled,
        labels=dict(sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))),
        by_predicted_class=breakdown,
    )


def format_info_html(info: SelectionInfo, header: str) -> str:
    """Render a summary as HTML for display in a message box.

    Args:
        info: Summary to render.
        header: Context line naming the selection.

    Returns:
        An HTML fragment.
    """
    parts = [f"<b>{header}</b><br><br>"]

    if info.total_cells == 0:
        parts.append("No cells found in CellView for this selection.")
        return "".join(parts)

    parts.append(
        f"<b>{info.total_cells}</b> cells · "
        f"<b>{info.available}</b> not yet labelled · "
        f"<b>{info.labelled}</b> labelled<br>"
    )

    if info.labels:
        parts.append("<br><u>Labels applied</u><br>")
        parts.append("<table cellpadding='2'>")
        for label, count in info.labels.items():
            parts.append(
                f"<tr><td>{label}</td><td align='right'>{count}</td></tr>"
            )
        parts.append("</table>")

    if info.by_predicted_class is not None:
        parts.append("<br><u>By predicted class</u><br>")
        parts.append(
            "<table cellpadding='3'><tr>"
            "<th align='left'>predicted</th>"
            "<th align='right'>total</th>"
            "<th align='right'>left</th>"
            "<th align='right'>labelled</th>"
            "<th align='left'>as</th></tr>"
        )
        for row in info.by_predicted_class:
            as_text = (
                ", ".join(f"{k} {v}" for k, v in row.labels.items()) or "—"
            )
            parts.append(
                f"<tr><td>{row.predicted_class}</td>"
                f"<td align='right'>{row.total}</td>"
                f"<td align='right'>{row.available}</td>"
                f"<td align='right'>{row.labelled}</td>"
                f"<td>{as_text}</td></tr>"
            )
        parts.append("</table>")

    return "".join(parts)
