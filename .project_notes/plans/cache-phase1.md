Phase 1: Self-Contained Plate Cache with Concurrent Downloads

 Context

 When navigating between wells in the napari welldata widget, every interaction triggers
 parse_omero_data() which opens a fresh OMERO connection and runs 6+ parser classes with
 network calls — even when all image data is already in diskcache. The result is noticeable
 lag on every well switch.

 Goal: Cache ALL data needed for display (metadata + flatfield-corrected images + labels +
 stage positions) so that well navigation is fully offline once a plate is cached. Also add
 concurrent downloads (3 workers) and progress reporting.

 ---
 Cache Key Structure

 All stored in the existing diskcache.Cache instance from omero_image.py:

 plate:{plate_id}:meta     → dict with channel_data, pixel_size, intensities, plate_name
 plate:{plate_id}:wells    → dict mapping well_pos → {well_id, metadata, images: [{image_id, size_t, pos_x, pos_y}]}
 plate:{plate_id}:labels   → dict mapping well_pos → [label_image_id, ...] (or empty if no masks)
 {image_id}:{timepoint}    → flatfield-corrected float32 ZYXC numpy array

 - Images cached as float32 (not float64) — sufficient for display, halves storage vs float64
 - Labels stay as integer arrays with same key pattern (their image_ids won't collide with plate image_ids)
 - Flatfield mask is NOT cached — held in memory during the download pass only

 ---
 Step 1: Create plate_cache.py — Cache Orchestration Module

 File: packages/omero-screen-napari/src/omero_screen_napari/plate_cache.py

 New module that orchestrates the complete plate download. Uses the shared _cache from omero_image.py.

 Functions

 def is_plate_cached(plate_id: int) -> bool:
     """Check if plate metadata exists in cache."""
     return _cache.get(f"plate:{plate_id}:meta") is not None

 def get_cached_plate_metadata(plate_id: int) -> dict | None:
     """Return cached plate metadata or None."""

 def get_cached_well_data(plate_id: int) -> dict | None:
     """Return cached wells dict or None."""

 def cache_plate(plate_id: int, max_workers: int = 3) -> Generator[tuple[int, int], None, None]:
     """Cache entire plate: metadata + flatfield-corrected images + labels.

     Yields (images_done, images_total) for progress reporting.
     Opens one OMERO connection for metadata, then spawns workers for images.
     """

 cache_plate() flow:

 1. Open one OMERO connection for metadata
 2. Fetch and cache plate metadata (channel_data, pixel_size, intensities, plate_name)
 3. Fetch well data for ALL wells (well_pos, well_id, metadata, image list with stage positions)
 4. Download flatfield mask into memory (NOT cached)
 5. Cache plate metadata and well data dicts
 6. Close metadata connection
 7. Partition images across workers, launch concurrent downloads
 8. Each worker: open connection, download batch, apply flatfield, store float32, close connection
 9. Yield progress after each image

 HQL query for images + stage positions (extend existing _get_image_ids):

 select w.row, w.column, ws.posX, ws.posY, i.id, pi.sizeT
 from Plate as p
   left join p.wells as w
   left join w.wellSamples as ws
   left join ws.image as i
   left join i.pixels as pi
 where p.id = :plate_id
 This returns well row/column (to compute well_pos), stage positions, and image info in one query.

 Concurrent download structure:

 def _download_batch(batch, flatfield_masks, channel_data, timepoint=0, conn=None):
     """Download and flatfield-correct a batch of images. One conn per worker."""
     for item in batch:
         arr = _download_and_correct(conn, item.image_id, timepoint, flatfield_masks, channel_data)
         _cache[f"{item.image_id}:{timepoint}"] = arr
 - 3 workers, round-robin partitioning
 - Each worker gets its own @omero_connect connection
 - Flatfield mask passed by reference (shared, read-only)

 ---
 Step 2: Metadata Fetching Functions

 File: packages/omero-screen-napari/src/omero_screen_napari/plate_cache.py

 Extract the metadata fetching logic from welldata_api.py parser classes into standalone functions
 that can be called independently:

 def _fetch_plate_metadata(conn, plate_id) -> dict:
     """Fetch channel_data, pixel_size, intensities, plate_name from OMERO.
     Reuses logic from ChannelDataParser, PixelSizeParser, ScaleIntensityParser."""

 def _fetch_well_map(conn, plate_id) -> dict:
     """Fetch all wells with metadata, image lists, and stage positions.
     Single HQL query for images + positions, plus well annotations."""

 def _fetch_flatfield_mask(conn, plate_id, screen_dataset) -> NDArray:
     """Download flatfield mask. Reuses FlatfieldMaskParser logic."""

 def _fetch_label_map(conn, plate_id, screen_dataset) -> dict:
     """Map well image_ids to their segmentation label image_ids in the dataset."""

 Key design: These functions call OMERO directly but DO NOT write to omero_data. They return
 plain dicts/arrays that cache_plate() stores in diskcache.

 For channel_data and well metadata, we reuse the existing parsing logic from welldata_api.py
 where possible (call into the existing parser classes or extract their core logic).

 ---
 Step 3: Flatfield Correction During Download

 In the download worker, apply correction before caching:

 def _download_and_correct(conn, image_id, timepoint, flatfield_masks, channel_data) -> NDArray:
     """Download raw image, apply flatfield correction, return float32 ZYXC."""
     raw = _get_omero_image_timepoint(image, timepoint)  # uint16 ZYXC
     corrected = raw.astype(np.float32) / flatfield_masks.astype(np.float32)
     return corrected

 - Flatfield mask shape: (1, T, Y, X, C) or (Y, X, C) — need to verify and broadcast correctly
 - Result dtype: float32 (4 bytes/pixel vs 2 for uint16 — acceptable tradeoff)
 - Labels are NOT corrected — stored as-is (integer masks)

 ---
 Step 4: Fast-Path in Widget

 File: packages/omero-screen-napari/src/omero_screen_napari/_welldata_widget.py

 Modify welldata_widget() to check cache before calling parse_omero_data():

 def welldata_widget(viewer, plate_id, well_pos_list, images, time, cache):
     plate_num = int(plate_id)

     if cache:
         # Start background caching if not already running
         start_cache_worker(plate_num)

     # Try fast path: load from cache without OMERO
     if is_plate_cached(plate_num):
         load_from_cache(omero_data, plate_num, well_pos_list, images, time)
     else:
         # Fall back to current OMERO path
         parse_omero_data(omero_data, plate_id, well_pos_list, images, time=time)

     clear_viewer_layers(viewer)
     add_image_to_viewer(viewer)
     ...

 load_from_cache() function:

 def load_from_cache(omero_data, plate_id, well_pos_list, images, time):
     """Populate OmeroData entirely from diskcache. No OMERO connection needed."""
     meta = get_cached_plate_metadata(plate_id)
     wells = get_cached_well_data(plate_id)

     omero_data.plate_id = plate_id
     omero_data.channel_data = meta["channel_data"]
     omero_data.pixel_size = meta["pixel_size"]
     omero_data.intensities = meta["intensities"]
     omero_data.plate_name = meta["plate_name"]

     # Parse user well/image selection (no OMERO needed)
     selected_wells = parse_well_positions(well_pos_list)
     selected_images = parse_image_input(images)

     # Load per-well data from cache
     for well_pos in selected_wells:
         well_info = wells[well_pos]
         omero_data.well_metadata_list.append(well_info["metadata"])
         omero_data.well_id_list.append(well_info["well_id"])

         for img_info in well_info["images"]:
             if img_info["index"] in selected_images:
                 arr = _cache.get(f"{img_info['image_id']}:{timepoint}")
                 image_arrays.append(arr)

     omero_data.images = np.stack(image_arrays)
     # Similar for labels

 ---
 Step 5: Progress Reporting

 File: packages/omero-screen-napari/src/omero_screen_napari/_welldata_widget.py

 Modify the background cache worker to report progress:

 def start_cache_worker(plate_id: int) -> None:
     """Start background plate caching with progress reporting."""
     # ... deduplication logic (existing) ...

     worker = create_worker(cache_plate, plate_id, max_workers=3)

     # Progress callback
     def on_progress(progress: tuple[int, int]):
         done, total = progress
         # Update napari status bar or activity widget

     worker.yielded.connect(on_progress)
     worker.start()

 The cache_plate() generator yields (images_done, images_total) tuples.

 ---
 Step 6: Update omero_image.py

 File: packages/omero-screen-napari/src/omero_screen_napari/omero_image.py

 Minimal changes:
 - Export _cache so plate_cache.py can use the same Cache instance
 - Export _get_omero_image_timepoint() for use by download workers
 - Keep existing get_image() / get_image_timepoint() for backward compatibility
 - The cache_plate_images() generator becomes DEPRECATED (replaced by plate_cache.cache_plate())

 ---
 Files to Create/Modify
 ┌─────────────────────┬───────────┬────────────────────────────────────────────────────────────────────────────────┐
 │        File         │  Action   │                                  Description                                   │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ plate_cache.py      │ CREATE    │ Cache orchestration: metadata fetch, concurrent download, flatfield correction │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ omero_image.py      │ MODIFY    │ Export _cache and _get_omero_image_timepoint; deprecate cache_plate_images     │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ _welldata_widget.py │ MODIFY    │ Add cache-first fast path before parse_omero_data()                            │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ omero_data.py       │ MINOR     │ No changes needed — existing fields sufficient                                 │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ welldata_api.py     │ NO CHANGE │ Keep existing OMERO path as fallback (no refactoring needed)                   │
 ├─────────────────────┼───────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ test_plate_cache.py │ CREATE    │ Tests for metadata caching, concurrent downloads, cache-first path             │
 └─────────────────────┴───────────┴────────────────────────────────────────────────────────────────────────────────┘
 ---
 What's NOT in Scope

 - Stitching UI (future work — stage positions are cached for later use)
 - Compression option for diskcache (can be added later)
 - CellView/polars data caching (stays from DB — it's already fast)
 - Refactoring welldata_api.py parser classes (keep as fallback, don't touch)

 ---
 Verification

 1. Unit tests: Mock OMERO, verify metadata stored/retrieved from cache correctly
 2. Unit tests: Verify concurrent download with flatfield correction produces float32 arrays
 3. Unit tests: Verify is_plate_cached() returns True after caching, False before
 4. Integration test: Cache a real plate with cache=True, then switch wells — no OMERO calls
 5. Benchmark: Compare well-switch latency with and without cache (target: <100ms from cache)
 6. Run existing tests: pytest tests/unit_tests/ — all must pass (no regressions)
