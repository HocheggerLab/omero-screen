Context

 When loading a well from cache, images display as a scrollable stack. The user
 must then manually click the "Stitch" widget, which uses hardcoded Operetta
 patterns supporting only 2/4/9/21 images. The cache already stores stage
 positions (pos_x, pos_y in µm) per image but they're unused.

 Goal: Auto-stitch images using their absolute stage positions when loading
 from cache. This works for any number of images per well and removes the need
 for the separate stitch step.

 ---
 Design

 Position-to-grid conversion

 Stage positions define a regular microscope grid. The algorithm:
 1. Cluster unique X positions (within 1µm tolerance) → column indices
 2. Cluster unique Y positions → row indices
 3. Map each image to (col, row)
 4. Compute overlap: overlap_px = tile_size_px - spacing_px where
 spacing_px = position_spacing_µm / pixel_size_µm
 5. Return grid_map[col][row] = image_index + overlap values

 Then delegate to existing compose_tiles() / compose_labels() for blending.

 Display path (cache + single well)

 welldata_widget (cached plate, one well)
   → load_from_cache (populates omero_data.images + .image_positions)
   → stitch_from_positions (positions → grid → compose_tiles)
   → _display_stitched (same pattern as stitched_data_widget)

 Falls back to stack view when: not cached, multi-well, or no valid positions.

 ---
 Step 1: Add image_positions to OmeroData

 File: omero_data.py

 Add after image_ids (line 76):
 image_positions: list[tuple[float, float] | None] = field(default_factory=list)

 Reset in reset() (line ~115) and reset_well_and_image_data() (line ~136):
 self.image_positions = []

 ---
 Step 2: Populate positions in load_from_cache()

 File: plate_cache.py

 In the image collection loop (around line 688), alongside image_ids.append():
 px, py = img_info.get("pos_x"), img_info.get("pos_y")
 positions.append((float(px), float(py)) if px is not None and py is not None else None)

 After building all arrays: omero_data.image_positions = positions

 ---
 Step 3: Create position_stitching.py (NEW)

 File: packages/omero-screen-napari/src/omero_screen_napari/position_stitching.py

 Functions

 has_valid_positions(positions) -> bool
 - True if len ≥ 2 and all non-None.

 _cluster_values(values, tolerance) -> list[float]
 - Sort values, walk through grouping consecutive values within tolerance.
 - Return list of cluster centroids (representative values).

 positions_to_grid(positions, tile_shape_yx, pixel_size, tolerance_um=1.0)
 - Cluster X and Y positions separately.
 - Map each image to (col, row) based on closest cluster.
 - Calculate spacing between adjacent columns/rows.
 - Compute overlap = tile_size - spacing (in pixels).
 - Return (grid_map: dict[int, dict[int, int]], overlap_x: int, overlap_y: int).

 stitch_from_positions(images, positions, pixel_size, rotation=0.0, edge=0, mode="reflect")
 - Calls positions_to_grid().
 - Builds tiles[col][row] = images[idx] from grid_map.
 - Handles 4D (NYXC): calls compose_tiles(tiles, ...) directly.
 - Handles 5D (NTYXC): loops over T, composes per-timepoint, stacks.
 - Returns stitched (YXC) or (TYXC).
 - Imports: compose_tiles from welldata_api.

 stitch_labels_from_positions(labels, positions, pixel_size, rotation=0.0)
 - Same grid logic, delegates to compose_labels() from welldata_api.

 ---
 Step 4: Add _display_stitched() helper in _welldata_widget.py

 File: _welldata_widget.py

 Extract from the existing stitched_data_widget pattern:

 def _display_stitched(viewer, stitched_images, stitched_labels=None):
     names = [k for k, v in sorted(omero_data.channel_data.items(), key=lambda x: int(x[1]))]
     viewer.add_image(
         stitched_images, contrast_limits=list(omero_data.intensities[0]),
         gamma=1, channel_axis=-1, scale=omero_data.pixel_size, name=names,
     )
     set_color_maps(viewer)
     if stitched_labels is not None:
         add_label_layers(viewer, labels=stitched_labels[np.newaxis, ...])
     viewer.scale_bar.visible = True
     viewer.scale_bar.unit = "µm"

 ---
 Step 5: Modify welldata_widget() cache path

 File: _welldata_widget.py

 Branch the cache path to auto-stitch for single-well + valid positions:

 if is_plate_cached(plate_num):
     load_from_cache(omero_data, plate_num, well_pos_list, images, time=time)
     if (len(omero_data.well_id_list) == 1
             and has_valid_positions(omero_data.image_positions)):
         stitched = stitch_from_positions(
             omero_data.images, omero_data.image_positions, omero_data.pixel_size)
         stitched_lbl = (stitch_labels_from_positions(
             omero_data.labels, omero_data.image_positions, omero_data.pixel_size)
             if omero_data.labels.size > 0 else None)
         clear_viewer_layers(viewer)
         _display_stitched(viewer, stitched, stitched_lbl)
     else:
         clear_viewer_layers(viewer)
         add_image_to_viewer(viewer)
         set_color_maps(viewer)
         add_label_layers(viewer)
 else:
     parse_omero_data(...)
     clear_viewer_layers(viewer)
     add_image_to_viewer(viewer)
     set_color_maps(viewer)
     add_label_layers(viewer)

 Skip the metadata slider for stitched view — show first well's metadata once.

 ---
 Step 6: Tests

 File: tests/unit_tests/omero_screen_napari_tests/test_position_stitching.py (NEW)
 ┌───────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
 │          Test class           │                                 Cases                                 │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
 │ TestHasValidPositions         │ all valid → True; single → False; None in list → False; empty → False │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
 │ TestClusterValues             │ regular spacing; tolerance grouping; single value                     │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
 │ TestPositionsToGrid           │ 2×2 grid; 3×3 grid; single row; overlap calculation                   │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
 │ TestStitchFromPositions       │ 2×2 synthetic → correct output shape; 5D time series                  │
 ├───────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
 │ TestStitchLabelsFromPositions │ 2×2 labels → correct shape                                            │
 └───────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
 Update test_plate_cache.py:
 - test_load_from_cache_stores_positions — verify positions populated
 - test_load_from_cache_null_positions — verify None handling

 ---
 Files Summary
 ┌────────────────────────────┬────────┬─────────────────────────────────────────────────┐
 │            File            │ Action │                   Description                   │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ omero_data.py              │ MODIFY │ Add image_positions field + reset               │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ plate_cache.py             │ MODIFY │ Collect positions in load_from_cache()          │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ position_stitching.py      │ CREATE │ Position-to-grid + stitch orchestration         │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ _welldata_widget.py        │ MODIFY │ Auto-stitch in cache path + _display_stitched() │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ test_position_stitching.py │ CREATE │ Unit tests for position stitching               │
 ├────────────────────────────┼────────┼─────────────────────────────────────────────────┤
 │ test_plate_cache.py        │ MODIFY │ Test position collection                        │
 └────────────────────────────┴────────┴─────────────────────────────────────────────────┘
 ---
 Verification

 1. pytest tests/unit_tests/omero_screen_napari_tests/test_position_stitching.py
 2. pytest tests/unit_tests/ — all pass (no regressions)
 3. Manual (cached, single well): Load from Plate Info → stitched view appears
 4. Manual (cached, multi-well): Select 2+ wells → stack view (fallback)
5. Manual (not cached): Unchanged — stack + manual stitch widget
