Context

 After Phase 1 cached all plate data (metadata, images, labels) for offline well navigation,
 the user now wants a browsable summary of all wells in a plate. Currently, users must
 know well positions by heart or guess. A popup dialog showing all wells with their metadata,
 image counts, and cache status makes exploring 96-well plates much more practical.

 Goal: Add a "Plate Info" button to the welldata widget that opens a dialog showing a
 table of all wells, with the ability to load selected wells into the viewer.

 ---
 Design

 UI Layout

 +------------------------------------------------------------+
 | Plate Info - 12345 (TestPlate)                         [X] |
 +------------------------------------------------------------+
 | Channels: DAPI, Tub, EdU  |  Status: Cached (green)       |
 | Wells: 96  |  Images: 384  |  Pixel size: 0.3 x 0.3 um   |
 +------------------------------------------------------------+
 | Well | Cell Line | Condition | Images | Timepoints | Labels|
 |------|-----------|-----------|--------|------------|-------|
 | A1   | RPE       | ctrl      | 4      | 1          | Yes   |
 | A2   | RPE       | drug      | 4      | 1          | No    |
 | ...  |           |           |        |            |       |
 +------------------------------------------------------------+
 | [Load Selected]                                   [Close]  |
 +------------------------------------------------------------+

 Data Sources

 Cached plate (fast path — no OMERO connection):
 - plate:{plate_id}:meta → plate_name, channel_data, pixel_size
 - plate:{plate_id}:wells → well_pos, metadata (cell_line, condition), images list
 - plate:{plate_id}:labels → label_ids per well (presence = "Yes")

 Uncached plate (slow path — requires OMERO):
 - Reuse _fetch_plate_metadata() and _fetch_well_map() from plate_cache.py
 - Show a brief "Loading..." message while fetching
 - Label map not available without caching (show "?" in Labels column)

 ---
 Step 1: Create _plate_info_dialog.py

 File: packages/omero-screen-napari/src/omero_screen_napari/_plate_info_dialog.py

 Class: PlateInfoDialog(QDialog)

 Follows the same pattern as _session_manager_widget.py:AnnotationSessionManager:
 - QVBoxLayout with header labels, QTableWidget, and action buttons
 - QTableWidget.SelectRows selection behavior, NoEditTriggers, sorting enabled

 class PlateInfoDialog(QDialog):
     def __init__(
         self,
         plate_id: int,
         on_load_callback: Callable[[str], None] | None = None,
         parent: QWidget | None = None,
     ) -> None:
         ...

 Constructor logic:
 1. Check is_plate_cached(plate_id) to determine data source
 2. If cached: load from cache (instant)
 3. If not cached: open OMERO connection, fetch metadata + well map, close connection
 4. Build header labels and table

 Table columns (7 columns):
 ┌─────┬────────────┬───────────────────────────┬──────────────────┐
 │  #  │   Column   │          Source           │      Width       │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 0   │ Well       │ well_pos key              │ ResizeToContents │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 1   │ Cell Line  │ metadata["cell_line"]     │ ResizeToContents │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 2   │ Condition  │ metadata["condition"]     │ Stretch          │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 3   │ Images     │ len(images)               │ ResizeToContents │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 4   │ Timepoints │ max(size_t) from images   │ ResizeToContents │
 ├─────┼────────────┼───────────────────────────┼──────────────────┤
 │ 5   │ Labels     │ "Yes"/"No" from label_map │ ResizeToContents │
 └─────┴────────────┴───────────────────────────┴──────────────────┘
 Methods:
 - _build_header() → summary labels (plate name, channels, cache status, counts)
 - _populate_table() → fill QTableWidget rows from well data
 - _on_load_selected() → get selected well(s), call on_load_callback with comma-joined positions
 - _fetch_data_from_omero() → slow path for uncached plates

 Cache status indicator: Use colored text — <span style='color:green'>Cached</span>
 or <span style='color:orange'>Not cached</span> in the header QLabel (HTML supported).

 Loading wells from the dialog

 The on_load_callback receives a well position string (e.g. "A1" or "A1, B2").
 The parent (_welldata_widget.py) sets the welldata widget's well_pos_list field
 and programmatically triggers the load:

 def _make_load_callback(welldata_instance):
     def _load_well(well_pos: str):
         welldata_instance.well_pos_list.value = well_pos
         welldata_instance()  # triggers the call_button action
     return _load_well

 Double-click to load

 Connect self.table.cellDoubleClicked to load the well from that row directly,
 as a convenience shortcut.

 ---
 Step 2: Modify _welldata_widget.py

 File: packages/omero-screen-napari/src/omero_screen_napari/_welldata_widget.py

 Change well_widget_combined() return type

 Currently returns Container (magicgui). Change to return a native QWidget
 that embeds the magic_factory widgets plus a "Plate Info" QPushButton.
 Napari accepts both Container and QWidget as dock widgets.

 def well_widget_combined() -> QWidget:
     welldata_instance = welldata_widget()
     stitched_instance = stitched_data_widget()

     widget = QWidget()
     layout = QVBoxLayout(widget)
     layout.addWidget(welldata_instance.native)

     plate_info_btn = QPushButton("Plate Info")
     plate_info_btn.clicked.connect(
         lambda: _open_plate_info(welldata_instance, widget)
     )
     layout.addWidget(plate_info_btn)

     layout.addWidget(stitched_instance.native)
     return widget

 Helper function _open_plate_info()

 def _open_plate_info(welldata_instance, parent):
     plate_id_str = welldata_instance.plate_id.value
     try:
         plate_id = int(plate_id_str)
     except (ValueError, TypeError):
         QMessageBox.warning(parent, "Invalid Plate ID", "Enter a valid plate ID first")
         return

     def load_callback(well_pos: str):
         welldata_instance.well_pos_list.value = well_pos
         welldata_instance()

     dialog = PlateInfoDialog(plate_id, on_load_callback=load_callback, parent=parent)
     dialog.exec_()

 New imports

 from omero_screen_napari._plate_info_dialog import PlateInfoDialog
 from qtpy.QtWidgets import QPushButton, QMessageBox, QVBoxLayout  # add to existing imports

 ---
 Step 3: Tests

 File: tests/unit_tests/omero_screen_napari_tests/test_plate_info_dialog.py

 Test the data-fetching logic (not the Qt UI itself):

 1. test_build_table_data_from_cache — mock _cache, verify table rows are correct
 2. test_build_table_data_uncached — mock OMERO connection, verify metadata fetch
 3. test_cache_status_indicator — verify cached vs uncached status string

 Since the dialog is UI-heavy, focus tests on the data extraction helper functions.
 Extract a _build_table_data() function that returns a list of row dicts — this
 is testable independently of Qt.

 ---
 Files to Create/Modify
 ┌───────────────────────────┬────────┬─────────────────────────────────────────────────────────────────┐
 │           File            │ Action │                           Description                           │
 ├───────────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
 │ _plate_info_dialog.py     │ CREATE │ PlateInfoDialog QDialog with well summary table                 │
 ├───────────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
 │ _welldata_widget.py       │ MODIFY │ Change well_widget_combined() to QWidget, add Plate Info button │
 ├───────────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
 │ test_plate_info_dialog.py │ CREATE │ Tests for data extraction logic                                 │
 └───────────────────────────┴────────┴─────────────────────────────────────────────────────────────────┘
 ---
 What's NOT in Scope

 - CellView enrichment (cell counts per well) — can be added later
 - Thumbnail previews per well — future enhancement
 - Plate map visual layout (8x12 grid) — future enhancement (table is sufficient for now)
 - Editing metadata from the dialog — read-only view

 ---
 Verification

 1. Unit tests: pytest tests/unit_tests/omero_screen_napari_tests/test_plate_info_dialog.py
 2. Manual test (cached plate): Open widget, enter cached plate ID, click "Plate Info" — dialog shows instantly
 3. Manual test (uncached plate): Enter uncached plate ID, click "Plate Info" — dialog fetches from OMERO and shows data
 4. Manual test (load well): Double-click a well row in the dialog — well loads in viewer
 5. Existing tests: pytest tests/unit_tests/ — all pass (no regressions)
