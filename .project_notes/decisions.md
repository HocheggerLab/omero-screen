# Architectural Decision Records (ADRs)

Document key architectural and technical decisions with context and trade-offs.

## Format

```markdown
### ADR-XXX: Decision Title (YYYY-MM-DD)

**Context:**
- Why the decision was needed
- What problem it solves

**Decision:**
- What was chosen

**Alternatives Considered:**
- Option 1 -> Why rejected
- Option 2 -> Why rejected

**Consequences:**
- Benefits
- Trade-offs
```

## Examples

### ADR-001: Use Cellpose for Cell Segmentation (2024-12-01)

**Context:**
- Need robust, automated cell and nucleus segmentation for high-content screening
- Must handle various cell lines and imaging conditions
- Team lacks expertise to train segmentation models from scratch

**Decision:**
- Use Cellpose pre-trained models (nucleus and cyto2)
- Select models automatically based on cell line and magnification metadata

**Alternatives Considered:**
- StarDist -> Good for nuclei, but limited cell segmentation
- Custom U-Net -> Requires extensive training data and expertise
- CellProfiler -> Less robust for varied imaging conditions

**Consequences:**
- Benefits: High-quality segmentation out of box, actively maintained, supports custom models
- Trade-offs: Dependency on external library, slower than simpler threshold-based methods

### ADR-002: Use DuckDB for Single-Cell Data Storage (2024-11-15)

**Context:**
- Need fast local querying of millions of single-cell measurements
- CSV files become unwieldy for large screening campaigns
- Don't want overhead of PostgreSQL for local analysis

**Decision:**
- Use DuckDB embedded database for cellview package
- Organize by project → experiment → plate → condition hierarchy

**Alternatives Considered:**
- SQLite -> Slower for analytical queries, less optimized for large datasets
- PostgreSQL -> Overkill for local storage, requires server management
- Parquet files -> Good for storage, but need query layer on top

**Consequences:**
- Benefits: Fast analytical queries, SQL interface, single file database, no server needed
- Trade-offs: Newer technology with smaller ecosystem than SQLite/Postgres

---

## Active Decisions

(Add new ADRs below this line, incrementing the ADR number)

### ADR-003: Prioritize Critical Unit Testing for Core Pipeline (2026-01-20)

**Context:**
- Preparing for Nature Communications submission (Target: May 2026)
- Audit revealed critical gaps in unit testing for `image_analysis.py` (core segmentation) and `flatfield_corr.py`
- Reviewers require robust validation of core algorithms, not just integration tests

**Decision:**
- Complete comprehensive unit tests for `image_analysis.py` and `flatfield_corr.py`
- **Target Date for Completion:** 2026-01-23

**Alternatives Considered:**
- Postpone testing -> High risk of rejection for lack of technical validation
- Rely on E2E tests -> Validates workflow but not algorithmic correctness or edge cases

**Consequences:**
- Benefits: Ensures scientific reproducibility, satisfies peer review requirements, increases confidence in screening results
- Trade-offs: Delays advanced feature work by ~1 week

### ADR-004: Direct OMERO Loading for Training Workflow (2026-02-06)

**Context:**
- Original training workflow required repetitive steps for each annotation session:
  1. Load welldata_widget for entire plate
  2. Generate gallery from loaded data
  3. Setup classifier metadata
  4. Annotate cells in training_widget
  5. Repeat all steps for each new well/image to annotate
- Users needed to reload welldata_widget every time they wanted to annotate a different image
- Memory inefficient (loads entire well when only need one image)
- No visibility into what's already been annotated across sessions

**Decision:**
- Implement direct OMERO loading that bypasses welldata_widget for annotation sessions
- Create new components:
  - `direct_omero_loader.py`: Loads crops directly from OMERO using plate/well/image index
  - `_session_manager_widget.py`: Dashboard showing all annotation sessions for a classifier
  - `_direct_load_dialog.py`: UI for selecting plate/well/image to annotate
- Integrate with existing `_classifier_selector.py` and `_training_widget.py`
- Data loading process:
  1. Load classifier metadata from saved JSON
  2. Fetch specific image from OMERO by well and index (not OMERO ID)
  3. Load segmentation masks from dataset
  4. Query CellView for centroids
  5. Apply flatfield correction
  6. Generate crops on-demand
  7. Populate omero_data singleton for display

**Alternatives Considered:**
- Keep existing workflow -> Users find it too tedious for multiple sessions
- Pre-load all images in project -> Memory intensive, slow startup
- Build caching layer -> Adds complexity, still requires initial welldata load

**Consequences:**
- Benefits:
  - One-time welldata loading for initial setup
  - Direct image loading for subsequent annotation sessions
  - Session management dashboard shows annotation progress
  - Can add new plates/wells without restarting workflow
  - Memory efficient (loads only needed images)
  - NOT a breaking change (old flow still works)
- Trade-offs:
  - More complex architecture with additional components
  - Requires CellView database for centroids
  - Image index confusion (uses well index 0,1,2 not OMERO image ID)

### ADR-005: Extract CyclicIF Data Cleaning as Standalone Utility (2026-02-05)

**Context:**
- `agg_data.csv` files from cyclicIF experiments contain duplicate columns with numeric suffixes (`.0`, `.1`) from pandas merge operations
- CLI import pathway had cleaning logic in `CellViewStateCore._clean_agg_data()` method
- Napari widget import pathway bypassed this cleaning, reading CSV with raw `pd.read_csv()`
- This caused database schema pollution and import failures when problematic plates were imported via napari

**Decision:**
- Extract `_clean_agg_data()` as standalone `clean_agg_data()` function at module level in `cellview/utils/state.py`
- Apply cleaning in both import pathways:
  - CLI: via `CellViewStateCore._clean_agg_data()` wrapper (backward compatible)
  - Napari: directly in `welldata_api._perform_import()` when detecting `agg_data.csv`
- Function automatically detects and handles:
  - Redundant columns with suffixes (drops if base exists)
  - Unique measurements with suffixes (renames to clean base name)
  - Metadata columns, empty rows/columns, data type validation

**Alternatives Considered:**
- Keep as class method -> Would require creating state object just for cleaning
- Duplicate logic in napari -> Code duplication, maintenance burden
- Skip cleaning in napari -> Continues to allow problematic data into database

**Consequences:**
- Benefits: Single source of truth for data cleaning, prevents schema pollution, works for both CLI and GUI imports
- Trade-offs: None significant - adds minimal complexity, improves robustness

### ADR-005: Replace Image ID with Image Input String in Training Session Management (2026-02-10)

**Context:**
- The session manager displayed numeric OMERO image IDs (e.g., 693056) which are meaningless to users
- The "Add New Data" dialog only accepted a single integer image index via QSpinBox
- The welldata widget already supported flexible image selection formats: `"All"`, `"0"`, `"0, 1, 2"`, `"3-5"`
- Users needed to load crops from multiple images per well in a single session

**Decision:**
- Replace Image ID display with image input string throughout session management
- Store `image_input` string in session metadata JSON for display
- Keep DB `image_id` INTEGER column for backward-compatible lookups (set to first resolved image ID)
- Add `_parse_image_input()` helper in `direct_omero_loader.py` matching welldata widget's regex
- Restructure `load_crops_from_omero()` to loop over parsed indices, accumulating crops across images

**Alternatives Considered:**
- Keep numeric image ID -> Inconsistent with gallery widget, confusing for users
- Store image_input in a new DB column -> Schema migration needed, metadata JSON is simpler

**Consequences:**
- Benefits: Consistent UI across widgets, multi-image sessions, human-readable session descriptions
- Trade-offs: Old sessions without metadata fall back to displaying numeric image_id

### ADR-006: Remove Load Functionality from Training Widget (2026-02-10)

**Context:**
- Training widget had its own "Load Images" button and `load_image()` method for loading classifier data
- Session manager widget was introduced to handle session browsing, loading, and deletion
- Having two ways to load data (widget button + session manager) was confusing
- Setup widget also had a ClassifierSelector dropdown that was confusing since it's only for creating new classifiers

**Decision:**
- Remove `load_image_widget` and all helper methods (`_set_paths`, `_parse_classified_data`, `_parse_saved_imagedata`, `_parse_metadata`, `_check_metadata`, `_parse_data`) from `TrainingWidget`
- Remove `ClassifierSelector` from `SetupTrainingWidget` — only keep it in `TrainingWidget`
- Sessions are loaded exclusively via the ClassifierSelector's session manager in the training widget

**Alternatives Considered:**
- Keep both load paths -> Confusing UX, duplicate code paths
- Move load into session manager only -> ClassifierSelector still needed for classifier selection context

**Consequences:**
- Benefits: Single load path, cleaner code (~190 lines removed), less confusing UI
- Trade-offs: Users must select a classifier before loading sessions (enforced by ClassifierSelector)
