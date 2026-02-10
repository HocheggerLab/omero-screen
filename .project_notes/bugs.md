# Bug Log

Track recurring bugs, their solutions, and prevention strategies. This helps avoid solving the same problem twice.

## Format

```markdown
### YYYY-MM-DD - Brief Bug Description
- **Issue**: What went wrong
- **Root Cause**: Why it happened
- **Solution**: How it was fixed
- **Prevention**: How to avoid it in the future
```

## Examples

### 2025-01-15 - OMERO Connection Timeout in E2E Tests
- **Issue**: E2E tests fail intermittently with "Connection refused" errors to test OMERO server
- **Root Cause**: Test OMERO server on 127.0.0.2:4064 not started before tests run
- **Solution**: Add pre-test check to ensure OMERO server is running, or start it automatically in test setup
- **Prevention**: Document test server requirements in test README, add startup script

### 2025-01-10 - Cellpose Model Selection Fails for New Cell Lines
- **Issue**: Segmentation fails with KeyError when processing new cell lines not in config
- **Root Cause**: Model selection logic expects all cell lines to be pre-configured
- **Solution**: Add fallback to default Cellpose model (cyto2) when cell line not found
- **Prevention**: Add validation for cell line metadata, warn on unknown cell lines

---

## Active Bugs

(Add new bug entries below this line)

### 2026-02-05 - Database Schema Pollution from Suffixed Columns

- **Issue**: Database imports failed after importing a problematic plate. Error occurred when trying to import new plates via welldata_widget, with errors relating to column mismatches. The measurements table had accumulated 72 duplicate columns with numeric suffixes (`.0`, `.1`, `.2`) that caused INSERT statements to fail because new plates only provided values for clean column names.

- **Root Cause**:
  - Aggregated data files (`agg_data.csv`) sometimes contain duplicate columns from pandas merge operations, which pandas auto-renames with `.0`, `.1` suffixes
  - The `_ensure_intensity_columns_exist()` function in `cellview/importers/measurements.py` dynamically adds missing columns using `ALTER TABLE ADD COLUMN`
  - Once added, these suffixed columns became permanent in the database schema
  - The `clean_up_db()` function only removed orphaned **records** (rows with missing foreign keys), not orphaned **schema columns**
  - Future imports failed because the INSERT statement expected values for ALL table columns, including the spurious suffixed ones

- **Solution**:
  1. **Immediate fix**: Manually dropped all 72 problematic columns using `ALTER TABLE measurements DROP COLUMN "column.0"`
  2. **Permanent fix**: Added `clean_schema_columns()` function to `cellview/db/clean_up.py` that:
     - Detects columns with numeric suffixes using regex pattern `\.\d+$`
     - Automatically drops problematic columns from the measurements table
     - Integrated into both `clean_up_db()` and `deep_clean_db()` functions
     - Runs as first step before record cleanup
  3. Function is now called automatically when:
     - User runs import with `--clean` flag
     - User deletes a plate (cleanup runs automatically)
     - User runs deep cleanup after errors

- **Prevention**:
  - Schema cleaning is now part of the standard cleanup workflow
  - Consider adding pre-import validation in `_clean_agg_data()` (cellview/utils/state.py:365-464) to strip suffixed columns BEFORE they reach the import stage
  - Monitor for recurrence by checking measurements table schema periodically
  - Affected columns included: DAPI, pRb/pRB, gH2AX/γH2AX, CyclinA, Cdk4, EdU (all with various `.0`, `.1`, `.2` suffixes)
  - The `_clean_agg_data()` function already has logic to handle suffixed columns (lines 408-431) but may need strengthening to catch all edge cases

### 2026-02-06 - Direct OMERO Loader: PlateI Has No getWell Attribute

- **Issue**: When trying to load new data via direct OMERO loader, got error: `'PlateI' object has no attribute 'getWell'`
- **Root Cause**: Attempted to use `plate.getWell(row, col)` method which doesn't exist in OMERO API. Plates must be accessed via iterating through wells.
- **Solution**: Changed to iterate through `plate.listChildren()` and match wells by position using `well.getWellPos()` (returns "A1", "B2", etc.)
- **Prevention**: Follow OMERO API patterns - use iteration and wrapper methods, not direct row/col access

### 2026-02-06 - Direct OMERO Loader: Image ID vs Image Index Confusion

- **Issue**: Users entered OMERO image IDs (e.g., 12345) but got "Image not found" errors
- **Root Cause**: Wells access images by **index** (0, 1, 2) not by OMERO ID. The API uses `well.getImage(index)` where index is position within well.
- **Solution**:
  - Updated UI to clarify "Image Index" instead of "Image ID"
  - Added tooltip: "Image index within the well (0 for first image, 1 for second, etc.)"
  - Changed default from 1 to 0
  - Better error messages showing valid index range
- **Prevention**: Document that most wells have single image (use index 0)

### 2026-02-06 - Direct OMERO Loader: CellView Returns Pandas Not Polars

- **Issue**: `AttributeError: 'DataFrame' object has no attribute 'is_empty'` when loading centroids
- **Root Cause**: Code assumed `cellview_load_data()` returns Polars DataFrame (`.is_empty()`, `.filter()`) but actually returns Pandas DataFrame
- **Solution**: Changed all Polars syntax to Pandas:
  - `df.is_empty()` → `df.empty`
  - `df.filter(pl.col("well") == well_id)` → `df[df["well"] == well_id]`
  - `filtered.select(cols)` → `filtered[cols]`
- **Prevention**: Check return types in API documentation

### 2026-02-06 - Direct OMERO Loader: Incorrect Channel Mapping

- **Issue**: Images displayed with wrong colors (BGR instead of RGB)
- **Root Cause**: `fill_missing_channels()` had confusing logic that reversed channel order for 3-channel images
- **Solution**: Simplified channel mapping to straightforward assignment:
  - 1 channel: Grayscale (H, W, 1)
  - 2 channels: RGB with Red=ch0, Green=ch1, Blue=0
  - 3 channels: RGB with Red=ch0, Green=ch1, Blue=ch2
- **Prevention**: Keep channel mapping logic simple and well-documented

### 2026-02-06 - Training Widget: Keybinding "W" Conflict After Loading Classifier

- **Issue**: Get keybinding warning "W already used" when loading the classifier after loading the plate
- **Root Cause**: `setup_key_bindings()` in `TrainingWidget.__init__` called `viewer.bind_key("w")` without `overwrite=True`. If the Training Widget is opened multiple times or "w" was already bound, napari warns about the conflict.
- **Solution**: Added `overwrite=True` to both `viewer.bind_key("w")` and `viewer.bind_key("q")` calls in `_training_widget.py:452-459`.
- **Prevention**: Always use `overwrite=True` when binding viewer keys in napari widgets that may be re-instantiated.

### 2026-02-06 - Training Widget: Grayscale and Color Management Not Working

- **Issue**: Grayscale and color management not functioning correctly in training widget. Single-channel images appeared red instead of grayscale.
- **Root Cause**: Two issues:
  1. `_add_rgb_image()` in `ImageNavigator` called `viewer.add_image(image)` without `rgb=True`. Napari interpreted the 3rd dimension as separate image layers with its own colormapping.
  2. `session_utils.py:parse_npy_file` tried `int("DAPI")` to convert channel names to indices, which failed. The fallback used ALL channels (e.g., `[0,1,2,3]`), so `fill_missing_channels` returned (H,W,3) even for single-channel configs. Only channel 0 (Red) had data → image appeared red.
- **Solution**:
  1. Added `rgb=True` to `viewer.add_image()` in `_add_rgb_image()` at `_training_widget.py:141`.
  2. Fixed `session_utils.py:78-97` to look up channel names via `omero_data.channel_data.get(ch)` (matching `gallery_api.py` approach) before falling back to all channels.
- **Prevention**: Always use `omero_data.channel_data` for channel name→index conversion. Never assume channel names are numeric strings.

### 2026-02-06 - Training Widget: Incorrect Number of Cells Displayed

- **Issue**: Number of cells shown is incorrect
- **Root Cause**: `_parse_data()` used `RandomImageParser` which respects `user_data.rows` and `user_data.columns` to determine sample size. If these were non-zero (carried over from gallery widget settings persisted in `metadata.json`), only `rows * columns` cells were randomly selected instead of ALL crops. Training widget needs every cell for classification.
- **Solution**: Explicitly set `self.user_data.rows = 0` and `self.user_data.columns = 0` in `_parse_data()` before calling `RandomImageParser`, ensuring all crops are used (`_training_widget.py:421-424`).
- **Prevention**: When `RandomImageParser` is used for training (not gallery display), always ensure rows/columns are 0 to select all images.

### 2026-02-09 - Initial Session Save: Gallery Crops Not Persisted

- **Issue**: After creating a new classifier, gallery crop data was not saved as a session. Session manager showed no sessions.
- **Root Cause**: `MetaDataSaver.save_data()` only saved metadata.json, not the crop data as an NPY file or DB session.
- **Solution**: Added `_save_initial_session()` to `MetaDataSaver` that re-parses with `classifier=True`, saves NPY, creates DB session with annotations.
- **Prevention**: Any workflow that produces crop data should persist it immediately.

### 2026-02-09 - Session Loader: Well Position Always Lost (well_id vs well)

- **Issue**: Loading a session from session manager always showed well as "N/A". Re-saving created duplicate session with empty well.
- **Root Cause**: `session_data_loader.py` used `session.get("well_id")` but the DB column/key is `"well"`. Lookup always returned None.
- **Solution**: Changed all three occurrences of `"well_id"` to `"well"` in `session_data_loader.py`.
- **Prevention**: Always check DB schema column names match dictionary key lookups.

### 2026-02-09 - Session Loader: TrainingDataSaver Not Initialized

- **Issue**: After loading a session from session manager, clicking save silently failed ("Training data saver not initialized" printed to console with no UI feedback).
- **Root Cause**: `_on_session_loaded()` never initialized `TrainingDataSaver`, unlike `_on_direct_load()` which did.
- **Solution**: Added `TrainingDataSaver` initialization in `_on_session_loaded()`, matching the pattern in `_on_direct_load()`.
- **Prevention**: Any callback that loads data for annotation must ensure `TrainingDataSaver` is initialized.

### 2026-02-09 - Direct OMERO Loader: Yellow Screen (Channel Name Resolution)

- **Issue**: Data loaded via "Add New Data" appeared as yellow screen instead of grayscale.
- **Root Cause**: `direct_omero_loader.py` only tried `int(ch)` to resolve channel names like "DAPI". This failed, triggering a fallback that used ALL channels. With 2 channels, `fill_missing_channels` produced R+G = Yellow.
- **Solution**: Added channel name lookup via `metadata["channel_data"]` dict, matching the approach in `session_utils.py`. Added proper grayscale fallback for single-channel requests.
- **Prevention**: All three channel resolution paths (`session_utils.py`, `direct_omero_loader.py`, `gallery_api.py`) must follow the same pattern: try int → look up in channel_data → fallback.

### 2026-02-09 - Annotation Count Mismatch (cell_index Collisions)

- **Issue**: Session manager showed fewer annotations than actual crops (e.g., 100 nuclei but 70 unassigned).
- **Root Cause**: `np.max(label_mask)` was used as `cell_index` in annotations. When neighbouring cells bleed into multiple crops, they share the same max label. DB `UNIQUE(session_id, cell_index)` with `INSERT OR REPLACE` silently dropped duplicates.
- **Solution**: Changed to sequential enumerate index for `cell_index` in both `_save_initial_session()` and `TrainingDataSaver._save_to_database()`.
- **Prevention**: Use sequential crop index for DB cell_index, not segmentation label values.
