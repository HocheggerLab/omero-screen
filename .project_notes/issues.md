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

### 2026-02-10 - Training Widget Session Management Improvements
- **Status**: Completed
- **Description**: Major overhaul of training data session management. Replaced Image ID with multi-image input strings across UI. Added session deletion with NPY cleanup. Fixed TrainingDataSaver staleness bug. Removed redundant load functionality. Added 38 new unit tests. Cleanly rebased onto Alex's `diskcache`/caching changes.
- **Branch**: `training-widget` (commits `3019379`, `efe0063`)
- **Notes**: Key files changed: `direct_omero_loader.py`, `_direct_load_dialog.py`, `_session_manager_widget.py`, `_training_widget.py`, `_setup_training_widget.py`, `trainingdata_db/cli.py`. New test file: `test_direct_omero_loader.py`.

### 2026-02-06 - Training Workflow Redesign: Direct OMERO Loading
- **Status**: Completed
- **Description**: Implemented streamlined training workflow that allows loading cell crops directly from OMERO without pre-loading entire wells via welldata_widget. Includes session management dashboard, direct data loader, and initial session saving when creating a new classifier.
- **Notes**: Direct loader uses well **image input** format ("All", "0", "0, 1, 2", "3-5") not OMERO image ID.

### 2026-04-15 - Terminal Progress UX + Documentation Improvements
- **Status**: Completed
- **Description**: Replaced flooding tqdm bars with a single Rich live progress panel (`ScreenProgress` in `progress.py`). Updated docs landing page with architecture SVG and pipeline chapter with loop SVG. Added `--cp4`, `--model`, `--benchmark` flags to sbatch script. Fixed matplotlib Agg backend to prevent display window overflow. Fixed `CELLVIEW_EDITOR` test isolation and explore VS Code path bug. Fixed mypy type errors in `progress.py` and `benchmarks/accuracy/load_image.py`.
- **Notes**: `docs/figures` removed from `.gitignore` — SVGs are now tracked. Figures need `width="100%" height="auto"` patched into SVG tag for responsive display in Sphinx RTD theme.
