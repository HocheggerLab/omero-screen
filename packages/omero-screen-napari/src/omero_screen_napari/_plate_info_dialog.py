"""Plate Info dialog showing a summary table of all wells in a plate.

Reads from diskcache (fast path) or fetches from OMERO (slow path) to
display plate metadata, per-well cell line and dynamic annotation keys,
image counts, timepoints, and label availability.
"""

from collections.abc import Callable
from typing import Any

from omero_screen.config import get_logger
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
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
    filter_empty_wells,
    get_cached_label_map,
    get_cached_plate_metadata,
    get_cached_well_data,
    get_well_cache_status,
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
    label_map: dict[str, list[dict[str, int | tuple[int, ...]] | None]] | None,
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
        max_t = max((img.get("dims", (1,))[0] for img in images), default=1)

        if label_unknown:
            labels = "?"
        elif label_map is not None:
            well_label_entries = label_map.get(well_pos)
            if well_label_entries:
                # Label entries are None if there is no label for corresponding image
                count = sum(x is not None for x in well_label_entries)
                if count == len(well_label_entries):
                    labels = "Yes"
                elif count == 0:
                    labels = "No"
                else:
                    labels = "Partial"
            else:
                labels = "No"

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
        "pixel_size": f"{pixel_size[0]:.3f} x {pixel_size[1]:.3f} \u00b5m",
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

    if meta is None or wells is None or label_map is None:
        raise ValueError(f"Plate {plate_id} cache incomplete")

    wells = filter_empty_wells(wells)
    header_info = _build_header_info(meta, wells)
    metadata_keys = _collect_metadata_keys(wells)
    rows = _build_rows(wells, label_map)

    return header_info, metadata_keys, rows, True


def _build_from_omero(
    plate_id: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], bool]:
    """Build table data by fetching from OMERO (slow path)."""
    from omero_screen_napari.omero_data import OmeroConnection
    from omero_screen_napari.plate_cache import (
        get_label_map,
        get_plate_metadata,
        get_well_data,
    )

    connection = OmeroConnection()
    try:
        meta = get_plate_metadata(connection, plate_id)
        wells = get_well_data(connection, plate_id)
        labels = get_label_map(connection, plate_id)
    finally:
        connection.close(hard=True)

    wells = filter_empty_wells(wells)
    header_info = _build_header_info(meta, wells)
    metadata_keys = _collect_metadata_keys(wells)
    rows = _build_rows(wells, label_map=labels)

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
        on_build_callback: Callback used when the plate info has been built (receives plate ID).
        on_load_callback: Callback receiving a well position string (e.g. "A1").
        on_cache_callback: Callback receiving plate ID to cache.
        parent: Parent widget.
    """

    def __init__(
        self,
        plate_id: int,
        on_build_callback: Callable[[int], None] | None = None,
        on_load_callback: Callable[[str], None] | None = None,
        on_cache_callback: Callable[[int], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plate_id = plate_id
        self.on_load_callback = on_load_callback
        self.on_cache_callback = on_cache_callback

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

        if on_build_callback is not None:
            on_build_callback(plate_id)

        self._rows = rows
        self._is_cached = is_cached
        self._cached_wells: set[str] = set()

        self.setWindowTitle(
            f"Plate Info - {plate_id} ({header_info['plate_name']})"
        )
        self.setMinimumSize(700, 450)

        main_layout = QVBoxLayout()

        # Header
        main_layout.addWidget(self._build_header(header_info, is_cached))

        # Table
        self.table = self._build_table(rows, metadata_keys, is_cached)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        main_layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self._select_all_cb)
        cache_btn = QPushButton("Cache Plate")
        cache_btn.clicked.connect(self._on_cache_selected)
        button_layout.addWidget(cache_btn)
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(self._on_load_selected)
        button_layout.addWidget(load_btn)
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Live cache monitoring
        self._cache_timer: QTimer | None = None
        self._start_cache_monitoring()

    def _build_header(
        self, header_info: dict[str, Any], is_cached: bool
    ) -> QLabel:
        """Build the header label with plate summary."""
        if is_cached:
            status = "<span style='color:green'>Cached</span>"
        else:
            status = "<span style='color:orange'>OMERO</span>"

        text = (
            f"<b>Channels:</b> {header_info['channels']}  |  "
            f"<b>Metadata Status:</b> {status}<br>"
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
        is_cached: bool = False,
    ) -> QTableWidget:
        """Build the QTableWidget from row data with dynamic metadata columns.

        Column layout: Select | Well | <metadata keys...> | Images | Timepoints | Labels | Cached
        """
        # Build column headers: Select + Well + dynamic metadata + fixed tail
        meta_headers = [_key_to_header(k) for k in metadata_keys]
        columns = (
            ["Select", "Well"]
            + meta_headers
            + ["Images", "Timepoints", "Labels", "Cached"]
        )
        n_meta = len(metadata_keys)
        self._cached_col_idx = len(columns) - 1

        # Always check actual per-image cache status — is_plate_cached()
        # returns True as soon as metadata is cached, before images download.
        well_cache_status = get_well_cache_status(self.plate_id)

        table = QTableWidget(len(rows), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Sorting must be disabled during population — Qt re-sorts after
        # each setItem(), scrambling row indices for subsequent columns.
        table.setSortingEnabled(False)

        self._row_checkboxes: list[QCheckBox] = []

        for row_idx, row_data in enumerate(rows):
            col = 0

            # Checkbox
            cb = QCheckBox()
            table.setCellWidget(row_idx, col, cb)
            self._row_checkboxes.append(cb)
            col += 1

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
            col += 1

            # Cached status
            well_pos = row_data["well"]
            cached_text = "Yes" if well_cache_status.get(well_pos) else "No"
            table.setItem(row_idx, col, QTableWidgetItem(cached_text))
            if well_cache_status.get(well_pos):
                self._cached_wells.add(well_pos)

        table.setSortingEnabled(True)
        table.sortItems(1, Qt.AscendingOrder)  # type: ignore[arg-type]

        # Place "Select All" checkbox in the header (column 0)
        header = table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.Fixed)
            table.setColumnWidth(0, 40)
            # Well column
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            # Metadata columns — last one stretches, rest resize to contents
            for i in range(2, 2 + n_meta):
                if i == 1 + n_meta:  # last metadata column
                    header.setSectionResizeMode(i, QHeaderView.Stretch)
                else:
                    header.setSectionResizeMode(
                        i, QHeaderView.ResizeToContents
                    )
            # If no metadata columns, stretch the Well column instead
            if n_meta == 0:
                header.setSectionResizeMode(1, QHeaderView.Stretch)
            # Tail columns: Images, Timepoints, Labels, Cached
            for i in range(2 + n_meta, len(columns)):
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # Replace "Select" header text with a checkbox widget
        self._select_all_cb = QCheckBox()
        self._select_all_cb.setToolTip("Select / Deselect All")
        self._select_all_cb.stateChanged.connect(self._on_select_all_toggled)
        table.setHorizontalHeaderItem(0, QTableWidgetItem(""))
        table.horizontalHeader().setMinimumSectionSize(40)

        return table

    def _on_select_all_toggled(self, state: int) -> None:
        """Toggle all row checkboxes when the Select All checkbox changes."""
        checked = bool(state)
        for cb in self._row_checkboxes:
            cb.setChecked(checked)

    def _on_load_selected(self) -> None:
        """Load wells for the checked rows (falls back to row selection)."""
        if self.on_load_callback is None:
            return

        # Primary: collect wells from checked checkboxes
        well_positions: list[str] = []
        for row_idx, cb in enumerate(self._row_checkboxes):
            if cb.isChecked():
                item = self.table.item(row_idx, 1)  # Well column at index 1
                if item:
                    well_positions.append(item.text())

        # Fallback: use Qt row selection if no checkboxes are checked
        if not well_positions:
            selected_rows = self.table.selectionModel().selectedRows()
            for index in selected_rows:
                item = self.table.item(index.row(), 1)
                if item:
                    well_positions.append(item.text())

        if well_positions:
            self.on_load_callback(", ".join(well_positions))
            self.accept()

    def _on_double_click(self, row: int, _column: int) -> None:
        """Load the well from the double-clicked row."""
        if self.on_load_callback is None:
            return

        item = self.table.item(row, 1)  # Well column shifted to index 1
        if item:
            self.on_load_callback(item.text())
            self.accept()

    def _on_cache_selected(self) -> None:
        """Cache the plate."""
        if self.on_cache_callback is None:
            return
        self.on_cache_callback(self.plate_id)
        self._start_cache_monitoring()

    def _start_cache_monitoring(self) -> None:
        """Start a QTimer to poll cache status if a worker is active."""
        from omero_screen_napari._welldata_widget import get_active_download

        if get_active_download() == self.plate_id:
            self._cache_timer = QTimer(self)
            self._cache_timer.timeout.connect(self._poll_cache_status)
            self._cache_timer.start(1000)

    def _poll_cache_status(self) -> None:
        """Check for newly cached wells and update the table."""
        from omero_screen_napari._welldata_widget import get_active_download

        status = get_well_cache_status(self.plate_id)
        cached_col = self._cached_col_idx

        for row_idx in range(self.table.rowCount()):
            well_item = self.table.item(row_idx, 1)
            if well_item is None:
                continue
            well_pos = well_item.text()
            if well_pos in self._cached_wells:
                continue  # already marked
            if status.get(well_pos):
                cached_item = self.table.item(row_idx, cached_col)
                if cached_item is not None:
                    cached_item.setText("Yes")
                self._cached_wells.add(well_pos)

        # Stop timer when all wells cached or worker stopped
        all_done = len(self._cached_wells) == self.table.rowCount()
        if (
            all_done or get_active_download() != self.plate_id
        ) and self._cache_timer is not None:
            self._cache_timer.stop()
            self._cache_timer = None

    def closeEvent(self, a0: Any) -> None:
        """Stop the cache timer when the dialog closes."""
        if self._cache_timer is not None:
            self._cache_timer.stop()
            self._cache_timer = None
        super().closeEvent(a0)

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
