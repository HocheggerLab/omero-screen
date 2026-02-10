# Work Log

Track completed work, ongoing issues, and ticket references. Helps maintain context across sessions.

## Format

```markdown
### YYYY-MM-DD - TICKET-ID: Brief Description
- **Status**: Completed / In Progress / Blocked
- **Description**: 1-2 line summary
- **URL**: https://your-issue-tracker.com/browse/TICKET-ID
- **Notes**: Any important context or follow-up needed
```

## Examples

### 2025-01-15 - OMERO-123: Implement Flatfield Correction
- **Status**: Completed
- **Description**: Added flatfield correction module for microscopy images using pre-calculated masks
- **URL**: https://github.com/your-org/omero-screen/issues/123
- **Notes**: Correction masks stored per-channel, see `flatfield_corr.py`

### 2025-01-10 - OMERO-118: Add Cell Cycle Classification
- **Status**: Completed
- **Description**: Integrated ML model for cell cycle phase prediction from nuclear features
- **URL**: https://github.com/your-org/omero-screen/issues/118
- **Notes**: Uses intensity and morphology features, see `cellcycle_analysis.py`

### 2025-01-08 - OMERO-115: Set Up E2E Test Infrastructure
- **Status**: In Progress
- **Description**: Configure parallel OMERO test server and integration test suite
- **URL**: https://github.com/your-org/omero-screen/issues/115
- **Notes**: Test server running on 127.0.0.2:4064, need to add more test plates

---

## Recent Work

(Add new work log entries below this line, most recent first)

### 2026-02-06 - Training Workflow Redesign: Direct OMERO Loading

- **Status**: In Progress (1 known bug remaining)
- **Description**: Implemented streamlined training workflow that allows loading cell crops directly from OMERO without pre-loading entire wells via welldata_widget. Includes session management dashboard, direct data loader, and initial session saving when creating a new classifier.
- **Components Created**:
  - `direct_omero_loader.py`: Load crops from OMERO using plate/well/image index
  - `session_data_loader.py`: Load saved sessions from NPY files
  - `session_utils.py`: NPY parsing and mask application utilities
  - `_session_manager_widget.py`: Dashboard showing all annotation sessions for a classifier
  - `_direct_load_dialog.py`: UI for selecting plate/well/image to annotate
  - Updated `_classifier_selector.py`: Added "Manage Sessions" button
  - Updated `_setup_training_widget.py`: Added `_save_initial_session()` to persist gallery data
  - Updated `_training_widget.py`: Added callbacks for session loading and direct loading
- **Bugs Fixed (2026-02-06)**:
  - PlateI has no getWell attribute (use iteration instead)
  - Image ID vs Index confusion (clarified UI to use index)
  - CellView returns Pandas not Polars (fixed DataFrame syntax)
  - Incorrect channel mapping (simplified RGB assignment)
  - Session manager annotation count error (fixed column name)
  - Keybinding "W" conflict warning (added overwrite=True)
  - Grayscale/color management (channel name→index resolution via channel_data)
- **Bugs Fixed (2026-02-09)**:
  - Initial session not saved when creating classifier (added `_save_initial_session`)
  - Gallery crops not persisted (re-parse with classifier=True)
  - no_background not respected on load (conditional mask application)
  - Session data loading order (restore UserData before parse_npy_file)
  - image_input overwritten by session loader (derive from filename)
  - Yellow screen on direct load (channel name resolution in direct_omero_loader.py)
  - Annotation count mismatch (sequential cell_index instead of np.max)
  - Well lost on session load (wrong dict key "well_id" vs "well")
  - TrainingDataSaver not initialized after session load from manager
- **Bugs Fixed (2026-02-09, continued)**:
  - NPY files saved all crops instead of gallery-sized subset. Added `n_crops` to classifier metadata. `direct_omero_loader.py` now randomly selects `n_crops` from all available crops before saving. `_save_initial_session` already respected gallery rows/columns via `RandomImageParser`.
- **Notes**: Direct loader uses well **image index** (0, 1, 2) not OMERO image ID. Most wells have single image (use index 0).
