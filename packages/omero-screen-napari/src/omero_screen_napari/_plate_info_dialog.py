"""Plate Info dialog showing a summary table of all wells in a plate.

Reads from diskcache (fast path) or fetches from OMERO (slow path) to
display plate metadata, per-well cell line and dynamic annotation keys,
image counts, timepoints, and label availability.
"""

import os
from collections.abc import Callable
from typing import Any

from omero_screen.config import get_logger
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omero_screen_napari.plate_cache import (
    get_cached_label_map,
    get_cached_plate_metadata,
    get_cached_well_data,
    is_plate_cached,
)

logger = get_logger(__name__)


# --------------- Data helpers (testable without Qt) ---------------


def _collect_metadata_keys(wells: dict[str, dict[str, Any]]) -> list[str]:
    """Discover all unique metadata keys across wells, ordered consistently.

    ``cell_line`` always comes first; remaining keys are sorted alphabetically.

    Args:
        wells: Well map from cache or OMERO.

    Returns:
        Ordered list of metadata key names.
    """
    all_keys: set[str] = set()
    for well_info in wells.values():
        all_keys.update(well_info.get("metadata", {}).keys())

    # cell_line first, then the rest alphabetically
    ordered: list[str] = []
    if "cell_line" in all_keys:
        ordered.append("cell_line")
        all_keys.discard("cell_line")
    ordered.extend(sorted(all_keys))
    return ordered


def _key_to_header(key: str) -> str:
    """Convert a metadata key to a human-readable column header.

    Examples:
        ``cell_line`` -> ``Cell Line``
        ``siRNA``     -> ``siRNA``  (no underscores, kept as-is)
    """
    if "_" in key:
        return key.replace("_", " ").title()
    return key


def build_table_data(
    plate_id: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], bool]:
    """Build the header info and row data for the plate info table.

    Returns:
        Tuple of (header_info, metadata_keys, rows, is_cached).
        header_info: dict with plate_name, channels, pixel_size, total_wells,
                     total_images.
        metadata_keys: ordered list of metadata keys found across wells.
        rows: list of dicts, one per well, with keys: well, metadata (dict),
              images, timepoints, labels.
        is_cached: whether data came from cache.
    """
    if is_plate_cached(plate_id):
        return _build_from_cache(plate_id)
    return _build_from_omero(plate_id)


def _build_rows(
    wells: dict[str, dict[str, Any]],
    label_map: dict[str, list[int]] | None,
    label_unknown: bool = False,
) -> list[dict[str, Any]]:
    """Build row dicts from well data.

    Args:
        wells: Well map keyed by well position.
        label_map: Label availability map, or None.
        label_unknown: If True, use "?" for labels (OMERO slow path).
    """
    rows: list[dict[str, Any]] = []
    for well_pos in sorted(wells.keys(), key=_well_sort_key):
        well_info = wells[well_pos]
        metadata = well_info.get("metadata", {})
        images = well_info.get("images", [])
        max_t = max((img.get("size_t", 1) for img in images), default=1)

        if label_unknown:
            labels = "?"
        else:
            has_labels = bool(label_map and label_map.get(well_pos))
            labels = "Yes" if has_labels else "No"

        rows.append(
            {
                "well": well_pos,
                "metadata": metadata,
                "images": len(images),
                "timepoints": max_t,
                "labels": labels,
            }
        )
    return rows


def _build_header_info(
    meta: dict[str, Any],
    wells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the header summary dict."""
    channel_data: dict[str, str] = meta["channel_data"]
    pixel_size: tuple[float, float] = meta["pixel_size"]
    plate_name: str = meta.get("plate_name", str(id))
    total_images = sum(len(w["images"]) for w in wells.values())
    return {
        "plate_name": plate_name,
        "channels": ", ".join(channel_data.keys()),
        "pixel_size": f"{pixel_size[0]} x {pixel_size[1]} \u00b5m",
        "total_wells": len(wells),
        "total_images": total_images,
    }


def _build_from_cache(
    plate_id: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], bool]:
    """Build table data from diskcache."""
    meta = get_cached_plate_metadata(plate_id)
    wells = get_cached_well_data(plate_id)
    label_map = get_cached_label_map(plate_id)

    if meta is None or wells is None:
        raise ValueError(f"Plate {plate_id} cache incomplete")

    header_info = _build_header_info(meta, wells)
    metadata_keys = _collect_metadata_keys(wells)
    rows = _build_rows(wells, label_map)

    return header_info, metadata_keys, rows, True


def _build_from_omero(
    plate_id: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], bool]:
    """Build table data by fetching from OMERO (slow path)."""
    from omero.gateway import BlitzGateway

    from omero_screen_napari.plate_cache import (
        _fetch_plate_metadata,
        _fetch_well_map,
    )

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")

    conn = BlitzGateway(username, password, host=host)
    conn.connect()
    if not conn.isConnected():
        raise RuntimeError(f"Failed to connect to OMERO server at {host}")

    try:
        meta = _fetch_plate_metadata(conn, plate_id)
        wells = _fetch_well_map(conn, plate_id)
    finally:
        conn.close(hard=True)

    header_info = _build_header_info(meta, wells)
    metadata_keys = _collect_metadata_keys(wells)
    rows = _build_rows(wells, label_map=None, label_unknown=True)

    return header_info, metadata_keys, rows, False


def _well_sort_key(well_pos: str) -> tuple[str, int]:
    """Sort wells by letter then number (A1, A2, ..., B1, ...)."""
    letter = well_pos[0]
    try:
        number = int(well_pos[1:])
    except ValueError:
        number = 0
    return (letter, number)


# --------------- Qt Dialog ---------------


class PlateInfoDialog(QDialog):  # type: ignore[misc]
    """Dialog showing a summary table of all wells in a plate.

    Args:
        plate_id: OMERO plate ID.
        on_load_callback: Callback receiving a well position string (e.g. "A1").
        parent: Parent widget.
    """

    def __init__(
        self,
        plate_id: int,
        on_load_callback: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plate_id = plate_id
        self.on_load_callback = on_load_callback

        try:
            header_info, metadata_keys, rows, is_cached = build_table_data(
                plate_id
            )
        except Exception as e:
            logger.exception(
                "Failed to load plate info for %d: %s", plate_id, e
            )
            self._show_error(str(e))
            return

        self._rows = rows

        self.setWindowTitle(
            f"Plate Info - {plate_id} ({header_info['plate_name']})"
        )
        self.setMinimumSize(700, 450)

        main_layout = QVBoxLayout()

        # Header
        main_layout.addWidget(self._build_header(header_info, is_cached))

        # Table
        self.table = self._build_table(rows, metadata_keys)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        main_layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(self._on_load_selected)
        button_layout.addWidget(load_btn)
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _build_header(
        self, header_info: dict[str, Any], is_cached: bool
    ) -> QLabel:
        """Build the header label with plate summary."""
        if is_cached:
            status = "<span style='color:green'>Cached</span>"
        else:
            status = "<span style='color:orange'>Not cached</span>"

        text = (
            f"<b>Channels:</b> {header_info['channels']}  |  "
            f"<b>Status:</b> {status}<br>"
            f"<b>Wells:</b> {header_info['total_wells']}  |  "
            f"<b>Images:</b> {header_info['total_images']}  |  "
            f"<b>Pixel size:</b> {header_info['pixel_size']}"
        )
        label = QLabel(text)
        label.setTextFormat(Qt.RichText)  # type: ignore[arg-type]
        return label

    def _build_table(
        self,
        rows: list[dict[str, Any]],
        metadata_keys: list[str],
    ) -> QTableWidget:
        """Build the QTableWidget from row data with dynamic metadata columns.

        Column layout: Well | <metadata keys...> | Images | Timepoints | Labels
        """
        # Build column headers: Well + dynamic metadata + fixed tail columns
        meta_headers = [_key_to_header(k) for k in metadata_keys]
        columns = ["Well"] + meta_headers + ["Images", "Timepoints", "Labels"]
        n_meta = len(metadata_keys)

        table = QTableWidget(len(rows), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Sorting must be disabled during population — Qt re-sorts after
        # each setItem(), scrambling row indices for subsequent columns.
        table.setSortingEnabled(False)

        for row_idx, row_data in enumerate(rows):
            col = 0
            # Well
            table.setItem(row_idx, col, QTableWidgetItem(row_data["well"]))
            col += 1

            # Dynamic metadata columns
            meta = row_data.get("metadata", {})
            for key in metadata_keys:
                table.setItem(
                    row_idx, col, QTableWidgetItem(str(meta.get(key, "")))
                )
                col += 1

            # Images (numeric for sorting)
            img_item = QTableWidgetItem()
            img_item.setData(Qt.ItemDataRole.DisplayRole, row_data["images"])  # type: ignore[arg-type]
            table.setItem(row_idx, col, img_item)
            col += 1

            # Timepoints (numeric for sorting)
            tp_item = QTableWidgetItem()
            tp_item.setData(
                Qt.ItemDataRole.DisplayRole, row_data["timepoints"]
            )  # type: ignore[arg-type]
            table.setItem(row_idx, col, tp_item)
            col += 1

            # Labels
            table.setItem(row_idx, col, QTableWidgetItem(row_data["labels"]))

        table.setSortingEnabled(True)
        table.sortItems(0, Qt.AscendingOrder)  # type: ignore[arg-type]

        # Column resize modes
        header = table.horizontalHeader()
        if header:
            # Well column
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            # Metadata columns — last one stretches, rest resize to contents
            for i in range(1, 1 + n_meta):
                if i == n_meta:  # last metadata column
                    header.setSectionResizeMode(i, QHeaderView.Stretch)
                else:
                    header.setSectionResizeMode(
                        i, QHeaderView.ResizeToContents
                    )
            # If no metadata columns, stretch the Well column instead
            if n_meta == 0:
                header.setSectionResizeMode(0, QHeaderView.Stretch)
            # Tail columns: Images, Timepoints, Labels
            for i in range(1 + n_meta, len(columns)):
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        return table

    def _on_load_selected(self) -> None:
        """Load wells for the selected table rows."""
        if self.on_load_callback is None:
            return

        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        well_positions = []
        for index in selected_rows:
            item = self.table.item(index.row(), 0)
            if item:
                well_positions.append(item.text())

        if well_positions:
            self.on_load_callback(", ".join(well_positions))
            self.accept()

    def _on_double_click(self, row: int, _column: int) -> None:
        """Load the well from the double-clicked row."""
        if self.on_load_callback is None:
            return

        item = self.table.item(row, 0)
        if item:
            self.on_load_callback(item.text())
            self.accept()

    def _show_error(self, message: str) -> None:
        """Set up a minimal error-state layout."""
        self.setWindowTitle("Plate Info - Error")
        self.setMinimumSize(400, 150)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Failed to load plate info:\n{message}"))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)
        self.setLayout(layout)
