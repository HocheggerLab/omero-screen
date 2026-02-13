Phase 5: Cached Plates Table & Per-Well Cache Status

 Context

 After Phase 4 (multi-well stitching with checkboxes), users can select and load
 multiple wells. But they have no visibility into:
 - Which plates are already cached — they must remember plate IDs manually
 - Which wells are ready during background caching — they wait for the entire
 plate to finish before they can start working

 This phase adds:
 1. A compact "Cached Plates" table in the widget showing what's available
 2. A "Cached" column in PlateInfoDialog that updates live during background
 caching, so users can load cached wells immediately
 3. Well-ordered downloads so wells complete sequentially (A1 first, then A2, ...)

 ---
 Step 1: Cache helper functions

 File: packages/omero-screen-napari/src/omero_screen_napari/plate_cache.py

 1a. get_all_cached_plates() -> list[tuple[int, str]]

 Scan _cache keys for plate:*:meta entries, fetch each meta dict for
 plate_name, return (plate_id, plate_name) pairs sorted by plate_id desc.

 Use _cache.iterkeys() to enumerate. Skip any corrupt entries gracefully.

 1b. get_well_cache_status(plate_id) -> dict[str, bool]

 For each well in cached wells data, check whether all its images × timepoints
 exist in cache using f"{image_id}:{t}" in _cache (fast SQLite index lookup,
 no data deserialization).

 Returns {well_pos: True/False}. Empty dict if plate not in cache.

 ---
 Step 2: Well-ordered downloads in cache_plate()

 File: packages/omero-screen-napari/src/omero_screen_napari/plate_cache.py

 Problem

 Currently _partition_round_robin() scatters images across workers:
 - Worker 0 gets image 0 (well A1), image 3 (well A2), image 6 (well B1), ...
 - Worker 1 gets image 1 (well A1), image 4 (well A2), ...

 This means well A1 isn't fully cached until ALL workers finish their first images.

 Solution

 Replace round-robin with well-grouped partitioning. Build the download list
 organized by well (sorted A1, A2, ...), including each well's labels right after
 its images. Distribute whole wells to workers round-robin.

 Change in cache_plate() (around line 109-154):

 # Build downloads grouped by well, sorted by well position
 sorted_well_keys = sorted(wells.keys(), key=_well_sort_key)
 well_groups: list[list[dict[str, Any]]] = []

 for well_pos in sorted_well_keys:
     group: list[dict[str, Any]] = []
     # Well images (need flatfield)
     for img_info in wells[well_pos]["images"]:
         for t in range(img_info["size_t"]):
             group.append({"image_id": img_info["image_id"], "timepoint": t,
                           "apply_flatfield": True})
     # Well labels (no flatfield, skip if cached)
     if well_pos in label_map:
         for label_id in label_map[well_pos]:
             if f"{label_id}:0" not in _cache:
                 group.append({"image_id": label_id, "timepoint": 0,
                               "apply_flatfield": False})
     if group:
         well_groups.append(group)

 # Distribute whole well groups to workers
 batches: list[list[dict[str, Any]]] = [[] for _ in range(max_workers)]
 for i, group in enumerate(well_groups):
     batches[i % max_workers].extend(group)
 batches = [b for b in batches if b]

 total = sum(len(b) for b in batches)

 Import _well_sort_key from _plate_info_dialog or duplicate locally (it's 5
 lines — better to extract to a shared utility or duplicate to avoid circular
 import).

 Keep _partition_round_robin as-is (don't remove — it's not used elsewhere
 but removing it is unnecessary risk).

 ---
 Step 3: Cached Plates table widget

 File: packages/omero-screen-napari/src/omero_screen_napari/_welldata_widget.py

 Add a CachedPlatesTable(QWidget) class with:
 - A compact QTableWidget (2 columns: Plate ID, Name; max height ~120px)
 - A "Refresh" button
 - On double-click: populate the plate_id field in welldata_widget

 Integration in well_widget_combined()

 QWidget → QVBoxLayout
   ├── CachedPlatesTable          ← NEW: compact table at top
   ├── welldata_widget.native     (plate_id, [Plate Info btn], well_pos, ...)
   └── stitched_data_widget.native

 Double-click on a cached plate → sets welldata_instance.plate_id.value.

 Refresh after caching completes

 Update start_cache_worker's on_finished callback to also refresh the table:
 store a reference to the table widget and call table.refresh() when caching
 finishes.

 ---
 Step 4: Per-well "Cached" column in PlateInfoDialog

 File: packages/omero-screen-napari/src/omero_screen_napari/_plate_info_dialog.py

 4a. Add "Cached" column

 Append a "Cached" column as the last column in _build_table().

 New column layout:
 Select | Well | <metadata...> | Images | Timepoints | Labels | Cached

 Initial values:
 - If plate fully cached (is_cached=True from build_table_data): all "Yes"
 - Otherwise: compute from get_well_cache_status() → "Yes" / "No"

 4b. QTimer for live updates during caching

 Add to __init__:
 self._cached_wells: set[str] = set()
 self._cache_timer: QTimer | None = None
 self._start_cache_monitoring()

 _start_cache_monitoring():
 - Import _active_cache_worker and _active_cache_plate_id from _welldata_widget
 - If worker is running for this plate → start QTimer(interval=1000ms)
 - Timer calls _poll_cache_status() which:
   a. Calls get_well_cache_status(self.plate_id)
   b. For any newly-cached wells, update the "Cached" cell to "Yes"
   c. If all wells cached or worker stopped → stop timer

 closeEvent() override: stop timer on dialog close.

 QTimer works in modal dialogs because exec_() runs a local event loop
 that still processes timer events.

 4c. Import QTimer

 from qtpy.QtCore import Qt, QTimer

 ---
 Step 5: Tests

 test_plate_cache.py — new helper functions

 class TestGetAllCachedPlates:
     - test_empty_cache_returns_empty
     - test_finds_cached_plates
     - test_sorted_by_plate_id_descending
     - test_skips_corrupt_metadata

 class TestGetWellCacheStatus:
     - test_uncached_plate_returns_empty
     - test_all_images_cached_returns_true
     - test_missing_images_returns_false
     - test_multi_timepoint_check

 test_plate_cache.py — well-grouped partitioning

 class TestWellGroupedPartitioning:
     - test_wells_complete_sequentially_within_worker
     - test_workers_get_sorted_wells

 test_plate_info_dialog.py — cached column

 class TestPlateInfoDialogCacheStatus:
     - test_cached_column_exists (qapp)
     - test_cached_shows_yes_when_fully_cached (qapp)
     - test_cached_shows_no_when_not_cached (qapp)

 ---
 Files Summary
 File: plate_cache.py
 Action: MODIFY
 Description: Add get_all_cached_plates(), get_well_cache_status(), well-grouped downloads
 ────────────────────────────────────────
 File: _welldata_widget.py
 Action: MODIFY
 Description: Add CachedPlatesTable widget, integrate in well_widget_combined(), refresh on cache complete
 ────────────────────────────────────────
 File: _plate_info_dialog.py
 Action: MODIFY
 Description: Add "Cached" column, QTimer polling, live status updates
 ────────────────────────────────────────
 File: test_plate_cache.py
 Action: MODIFY
 Description: Tests for new helpers and partitioning
 ────────────────────────────────────────
 File: test_plate_info_dialog.py
 Action: MODIFY
 Description: Tests for cached column
 ---
 Verification

 1. pytest tests/unit_tests/ — all pass
 2. Manual: Open napari → see cached plates table (empty or populated)
 3. Manual: Enter plate ID → tick "cache" → press Enter → open Plate Info
 → watch "Cached" column update from "No" to "Yes" well by well
 4. Manual: Double-click a cached plate row → plate_id field populated
 5. Manual: Load a partially-cached well → verify it works (fast path for cached images)
