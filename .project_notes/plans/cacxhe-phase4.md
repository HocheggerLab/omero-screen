Context

 Phase 3 implemented auto-stitching for single wells loaded from cache. The user
 now wants to compare conditions across multiple wells: select several wells in
 the PlateInfo dialog, stitch each individually, and slide between them using
 napari's dimension slider. This requires:

 1. A more intuitive well selection UI (checkboxes instead of Ctrl/Shift-click)
 2. Per-well stitching when multiple wells are selected
 3. Stacking stitched results so the napari slider navigates between wells
 4. Metadata display that tracks which well is currently shown

 ---
 Step 1: Add checkbox column to PlateInfoDialog

 File: packages/omero-screen-napari/src/omero_screen_napari/_plate_info_dialog.py

 Changes

 - Import QCheckBox from qtpy.QtWidgets
 - In _build_table(): add a "Select" column at index 0 with QCheckBox widgets
 via table.setCellWidget(row, 0, checkbox). Shift all data columns right by 1.
 - Add a header checkbox ("Select All") that toggles all row checkboxes.
 - Update _on_load_selected(): iterate rows, check checkbox state
 (table.cellWidget(row, 0).isChecked()), collect well positions from checked rows.
 Falls back to current Qt row-selection if no boxes are checked (backward compat).
 - Update _on_double_click(): column index for well name shifts from 0 to 1.
 - Update column resize modes for the new column layout.

 Column layout (after change)

 [✓] | Well | Cell Line | condition | ... | Images | Timepoints | Labels

 ---
 Step 2: Multi-well stitching in welldata_widget()

 File: packages/omero-screen-napari/src/omero_screen_napari/_welldata_widget.py

 Current logic (single-well only)

 if len(omero_data.well_id_list) == 1 and has_valid_positions(omero_data.image_positions):
     # stitch single well
 else:
     # stack view

 New logic (any number of wells)

 Remove the == 1 restriction. After load_from_cache():

 n_wells = len(omero_data.well_id_list)
 n_per_well = len(omero_data.image_index)

 # Check first well's positions — same plate means same layout
 first_well_pos = omero_data.image_positions[:n_per_well]
 if has_valid_positions(first_well_pos):
     sp = _get_stitch_params()
     stitched_imgs = []
     stitched_lbls = []

     for w in range(n_wells):
         start = w * n_per_well
         end = start + n_per_well
         well_images = omero_data.images[start:end]
         well_positions = omero_data.image_positions[start:end]
         well_labels = omero_data.labels[start:end] if omero_data.labels.size > 0 else None

         stitched_imgs.append(stitch_from_positions(
             well_images, well_positions, omero_data.pixel_size,
             rotation=sp["rotation"], edge=sp["edge"], mode=sp["mode"],
             fallback_overlap=(sp["overlap_x"], sp["overlap_y"]),
         ))
         if well_labels is not None:
             stitched_lbls.append(stitch_labels_from_positions(
                 well_labels, well_positions, omero_data.pixel_size,
                 rotation=sp["rotation"],
                 fallback_overlap=(sp["overlap_x"], sp["overlap_y"]),
             ))

     if n_wells == 1:
         result_img = stitched_imgs[0]          # (Y, X, C)
         result_lbl = stitched_lbls[0] if stitched_lbls else None  # (Y, X, C)
     else:
         result_img = np.stack(stitched_imgs)    # (N_wells, Y, X, C)
         result_lbl = np.stack(stitched_lbls) if stitched_lbls else None

     clear_viewer_layers(viewer)
     _display_stitched(viewer, result_img, result_lbl)
     # ... connect slider with images_per_well_override=1 for multi-well

 How napari handles the extra dimension

 - viewer.add_image(stacked, channel_axis=-1) splits channels
 - Each channel layer gets shape (N_wells, Y, X)
 - Napari creates a slider for dimension 0 → user slides between wells
 - For single-well, shape is (Y, X) → no slider (current behavior preserved)

 ---
 Step 3: Metadata slider for stitched multi-well

 File: _welldata_widget.py

 Problem

 handle_metadata_widget uses images_per_well = len(omero_data.image_index)
 to compute well_index = slider_position // images_per_well. In stitched
 multi-well mode, each slider position = one well, but image_index still
 contains the original per-well indices (e.g. 21), so the division is wrong.

 Solution

 Add an images_per_well_override parameter to handle_metadata_widget:

 def handle_metadata_widget(
     viewer: Viewer,
     slider_position: int,
     images_per_well_override: int | None = None,
 ) -> None:
     images_per_well = images_per_well_override or len(omero_data.image_index)
     ...

 In the stitched multi-well path, connect a callback with override=1:

 def slider_position_change(event):
     pos = event.source.current_step[0]
     handle_metadata_widget(viewer, pos, images_per_well_override=1)

 Also enhance metadata display to show the well position:

 well_metadata = {
     "Well": omero_data.well_pos_list[well_index],
     **omero_data.well_metadata_list[well_index],
 }

 ---
 Step 4: Tests

 File: tests/unit_tests/omero_screen_napari_tests/test_plate_info_dialog.py

 - Test checkbox column exists in table
 - Test _on_load_selected collects checked wells (not row-selection)

 File: tests/unit_tests/omero_screen_napari_tests/test_position_stitching.py

 - No changes needed — existing stitch tests cover the per-well stitching

 ---
 Files Summary
 ┌───────────────────────────┬────────┬────────────────────────────────────────────────────────┐
 │           File            │ Action │                      Description                       │
 ├───────────────────────────┼────────┼────────────────────────────────────────────────────────┤
 │ _plate_info_dialog.py     │ MODIFY │ Add checkbox column, update load/double-click handlers │
 ├───────────────────────────┼────────┼────────────────────────────────────────────────────────┤
 │ _welldata_widget.py       │ MODIFY │ Multi-well stitch loop, metadata slider override       │
 ├───────────────────────────┼────────┼────────────────────────────────────────────────────────┤
 │ test_plate_info_dialog.py │ MODIFY │ Test checkbox selection behavior                       │
 └───────────────────────────┴────────┴────────────────────────────────────────────────────────┘
 ---
 Verification

 1. pytest tests/unit_tests/ — all pass (no regressions)
 2. Manual (single well): Double-click or check one well → stitched view (unchanged)
 3. Manual (multi-well): Check 3+ wells → stitched view with slider between wells,
 metadata widget updates showing well position + condition as user slides
 4. Manual (no valid positions): Falls back to stack view as before
