# OMERO-Screen Project Overview

## Project Summary
OMERO-Screen is a comprehensive mono-repo for high-content image analysis of immunofluorescence microscopy data. It provides an end-to-end pipeline from OMERO server image acquisition through Cellpose-based segmentation, feature extraction, cell cycle analysis, local database storage, and publication-ready visualization.

**Version**: 0.2.6 | **Python**: 3.12 | **License**: MIT

## Architecture Overview

This is a monorepo workspace managed by `uv` containing 5 interconnected packages:
1. **omero-screen** (main) - Analysis pipeline orchestration
2. **omero-utils** - OMERO server interaction utilities
3. **cellview** - DuckDB-based data storage
4. **omero-screen-plots** - Statistical analysis and visualization
5. **omero-screen-napari** - Interactive Napari widgets

### Data Flow
```
OMERO Server (Plates)
    ↓ [metadata_parser.py - Parse Excel or annotations]
    ↓ [flatfield_corr.py - Generate/retrieve correction masks]
    ↓ [loops.py - Orchestrate well/image processing]
    ↓ [image_analysis.py - Cellpose segmentation + feature extraction]
    ↓ [cellcycle_analysis.py - Classify cell cycle phases]
    ↓ [CSV export to OMERO + local files]
    ↓ [cellview - Import to DuckDB database]
    ↓ [omero-screen-plots - Generate publication figures]
    ↓ [Results attached back to OMERO]
```

### Main Pipeline (`src/omero_screen/`)
**Core functionality**: End-to-end high-content screening analysis pipeline

**Key modules**:
- `loops.py`: Main orchestration - `plate_loop()` coordinates the entire workflow
- `config.py`: Environment management, logging setup, project root detection
- `metadata_parser.py`: Extracts experimental metadata from Excel files or OMERO annotations
- `flatfield_corr.py`: Generates/retrieves per-channel illumination correction masks
- `image_analysis.py`: Cellpose segmentation (nucleus, cell, cytoplasm) + regionprops feature extraction
- `cellcycle_analysis.py`: Multi-nucleate aggregation, normalization, and cell cycle phase assignment
- `aggregator.py`: Image-level feature aggregation using median/mean/std
- `quality_control.py`: Generates QC metrics and visualization figures
- `image_classifier.py`: Optional ML-based image classification with gallery generation
- `plate_dataset.py`: Manages OMERO dataset creation for segmentation masks
- `general_functions.py`: Border filtering, image scaling utilities

**Entry point**: `bin/run_omero_screen.py` provides CLI with arguments for plate IDs, environment selection, inference models, and segmentation-only mode

### Package Structure

#### 1. `omero-utils` (`packages/omero-utils/`)
**Purpose**: Clean abstraction layer for OMERO server operations

**Key modules**:
- `omero_connect.py`: Decorator-based connection management with automatic cleanup
- `attachments.py`: Upload/download CSV, Excel, PDF, and image files to/from OMERO objects
- `map_anns.py`: Create, parse, and delete key-value pair annotations
- `images.py`: Image download, MIP (maximum intensity projection) parsing, mask upload
- `omero_plate.py`: Plate-specific operations and well iteration
- `message.py`: Custom exception classes with rich formatting

**Design pattern**: Functions accept `BlitzGateway` connections and OMERO wrapper objects

#### 2. `cellview` (`packages/cellview/`)
**Purpose**: Local DuckDB database for organizing and querying single-cell measurements

**Database schema**:
```
projects (project_id, name, description)
    ↓
experiments (experiment_id, project_id, name, description)
    ↓
plates (plate_id, experiment_id, omero_plate_id, name, date)
    ↓
conditions (condition_id, plate_id, name)
    ↓
repeats (repeat_id, condition_id, repeat_number)
    ↓
measurements (measurement columns, repeat_id foreign key)
```

**Key modules**:
- `api.py`: `cellview_load_data()` - Main API for loading plates or experiments into pandas DataFrames
- `cli.py`: Rich-formatted CLI commands for display, import, export, cleanup
- `db/db.py`: `CellViewDB` class - DuckDB connection and table management
- `importers/`: CSV import, OMERO plate import with interactive project/experiment selection
- `exporters/`: Export to pandas/polars DataFrames, CSV, Excel
- `utils/state.py`: `CellViewState` dataclass for managing import workflows

**Access patterns**:
- CLI: `cellview display projects`, `cellview import-csv <file>`, `cellview export <plate_id>`
- Python API: `from cellview.api import cellview_load_data; df, vars = cellview_load_data(plate_id)`

#### 3. `omero-screen-plots` (`packages/omero-screen-plots/`)
**Purpose**: Publication-ready statistical plots with consistent styling

**Architecture** (v0.1.3+ refactored):
- Factory pattern: `*_factory.py` modules contain plot generation logic
- API modules: `*_api.py` provide clean public interfaces
- Single-class design for better performance and maintainability
- Comprehensive documentation at hocheggerlab.github.io/omero-screen/

**Plot types**:
- `combplot_api.py`: Multi-panel combined plots (`combplot_feature`, `combplot_cellcycle`)
- `cellcycleplot_api.py`: Cell cycle phase quantification in 2×2 subplot grid
- `cellcyclestacked_api.py`: Stacked barplots of cell cycle proportions
- `featureplot_api.py`: Box/violin plots for feature comparison across conditions
- `featureplot_norm_api.py`: Normalized feature plots with threshold analysis
- `countplot_api.py`: Cell count analysis (normalized or absolute)
- `histogramplot_api.py`: Distribution histograms with log scale and KDE overlays
- `scatterplot_api.py`: Scatter plots with cell cycle coloring and thresholds
- `classificationplot_api.py`: Categorical classification results visualization

**Data normalization** (`normalise.py`):
- `normalize_by_mode()`: Sets intensity peak (mode) to 1.0 using histogram analysis
- `find_intensity_mode()`: Gaussian-smoothed histogram peak detection
- Supports per-plate or per-condition normalization

**Statistical analysis** (`stats.py`):
- Plate-level aggregation with median/mean/std
- Built-in statistical testing when ≥3 replicates
- Significance marking on plots

**Style**: Custom matplotlib style (`hhlab_style01.mplstyle`) for consistent publication figures

#### 4. `omero-screen-napari` (`packages/omero-screen-napari/`)
**Purpose**: Interactive Napari widgets for data exploration and ML training data generation

**Key widgets**:
- `_welldata_widget.py`: Browse and visualize well images from OMERO
- `_gallery_widget.py`: Display cell galleries for classification
- `_training_widget.py`: Generate training datasets by manually classifying cells
- `_classifier_selector.py`: Select and apply trained classifiers
- `_aligned_plate_widget.py`: View aligned plate layouts
- `_setup_training_widget.py`: Configure training data generation workflows

**Database** (`trainingdata_db/`):
- SQLite database for storing user annotations and training labels
- Schema migration support via `migrator.py`
- CLI access via `cli.py` for database management

**Data management**:
- `omero_data.py` / `omero_data_singleton.py`: OMERO connection and image caching
- `gallery_userdata.py` / `gallery_userdata_singleton.py`: User annotation persistence
- `gallery_api.py`: Gallery image generation and selection logic
- `welldata_api.py`: Well-level data retrieval and processing

**Napari integration**: Registered via `napari.yaml` plugin manifest

## Key Technical Details

### Segmentation Strategy
**Cellpose-based two-channel segmentation**:

1. **Nucleus segmentation**:
   - Model: Cellpose built-in `nuclei` model or custom nucleus model
   - Input: Nuclear channel (typically DAPI/Hoechst)
   - Output: Nucleus masks with unique labels

2. **Cell segmentation**:
   - Model: Cellpose `cyto2` or custom cell line-specific models
   - Input: Cytoplasm channel (e.g., Tubulin) + nuclear channel
   - Output: Cell masks with unique labels
   - Border filtering: Removes cells touching image edges (configurable border width)

3. **Cytoplasm masks**:
   - Calculated as: `cytoplasm_mask = cell_mask - nucleus_mask`
   - Handles multi-nucleate cells by aggregating measurements

**Model selection logic** (`image_analysis.py`):
- Cell line extracted from well annotations
- Model mapping defined in `default_config.MODEL_DICT` (e.g., "RPE" → "RPE-1_Tub_Hoechst")
- Configurable via `OMERO_SCREEN_CONFIG` environment variable pointing to JSON file
- Example: `src/data/omero_screen_config.json`

**GPU/CPU detection**: Automatic via `torch.py` module

### Metadata Management
**Two sources** (priority: Excel > Annotations):

1. **Excel file attachment** (preferred):
   - Uploaded to OMERO plate object
   - Contains well layout, conditions, cell lines, timepoints
   - Parsed by `metadata_parser.py`
   - After parsing, converted to OMERO annotations and Excel deleted

2. **OMERO annotations** (fallback):
   - Key-value pairs on plate and well objects
   - Channel metadata: `{"DAPI": "0", "EdU": "1", "H3P": "2", "Tub": "3"}`
   - Well metadata: `{"cell_line": "RPE", "condition": "control", "timepoint": "24h"}`

**Validation**:
- Ensures required channels exist (nucleus channel mandatory)
- Validates cell line has corresponding Cellpose model
- Rich-formatted console tables for user verification

### Flatfield Correction
**Purpose**: Correct for uneven illumination across microscopy images

**Process** (`flatfield_corr.py`):
1. Check if correction masks already exist in dataset
2. If not, randomly sample 100 images across all wells
3. Aggregate samples per channel using median (robust to outliers)
4. Generate correction mask = `median_image / median(median_image)`
5. Upload as multi-channel TIFF to OMERO dataset
6. Save example corrected images as PDF for QC

**Application**: Each raw image divided by corresponding channel mask before segmentation

### Feature Extraction
**Per-region measurements** using `skimage.measure.regionprops_table`:

Default features (configurable via `FEATURELIST`):
- `label`: Unique object ID
- `area`: Region area in pixels
- `intensity_max`, `intensity_min`, `intensity_mean`: Intensity statistics per channel
- `centroid`: Object center coordinates

**Computed for**:
- Nucleus mask + all channels → `*_nucleus` columns
- Cell mask + all channels → `*_cell` columns
- Cytoplasm mask + all channels → `*_cytoplasm` columns

**Additional computed features**:
- `integrated_int_DAPI`: Sum of DAPI intensity (proxy for DNA content)
- Multi-nucleate aggregation: Cells sharing same cell_id get summed nucleus area and DAPI
- Normalized intensities: Background-subtracted (`value - min + 1`)

### Cell Cycle Analysis
**Pipeline** (`cellcycle_analysis.py`):

1. **Multi-nucleate aggregation**: Sum nucleus area and DAPI for cells with multiple nuclei
2. **Duplicate filtering**: Remove duplicate nuclei within same cell
3. **Per-cell-line normalization**: Normalize `integrated_int_DAPI` and `intensity_mean_EdU_nucleus` to mode
4. **Phase assignment**:
   - SubG1: DNA < 0.75
   - G1: DNA < 1.5 and EdU < threshold
   - S phase: EdU > threshold
   - G2/M: DNA ≥ 1.5 and EdU < threshold
   - Polyploid: DNA > 2.5

**Optional markers**:
- H3P (phospho-Histone H3): Identifies mitotic cells within G2/M
- Cytoplasm: Required for multi-nucleate detection

**Output**: New columns `cell_cycle` (G1/S/G2/Polyploid/SubG1) and `cell_cycle_detailed` (adds mitotic classification)

### Quality Control
**Metrics tracked** (`quality_control.py`):
- Number of segmented nuclei and cells per image
- Mean/median intensities per channel
- Segmentation success rate
- Well-level statistics

**Outputs**:
- `quality_ctr.csv`: Plate-level QC metrics
- `quality_ctr.png`: Visualization figure attached to plate
- Per-well CSV files during processing (deleted after plate completion)

**Segmentation masks**:
- Uploaded to dedicated OMERO dataset as multi-channel TIFFs
- Naming: `{original_image_id}_segmentation`
- Enables visual inspection in OMERO.web/OMERO.insight

### Image Classification (Optional)
**Trigger**: Set `OMERO_SCREEN_INFERENCE_MODEL` environment variable

**Process** (`image_classifier.py`):
- Batch classification using PyTorch models
- Generates example galleries (N×N grid) for each predicted class
- Configurable gallery size and batch size via environment variables
- Galleries attached as PNG figures to OMERO plate

## Environment Configuration

### Environment Files
System uses `.env.{ENV}` pattern with fallback to `.env`:
- `.env.development`: Local testing (localhost OMERO, console logging enabled)
- `.env.production`: Production server (file logging only, production credentials)
- `.env.e2etest`: E2E test environment (localhost, file logging only)

**Environment selection**:
1. Set `ENV` environment variable (defaults to "development")
2. System loads `.env.{ENV}` if exists, else `.env`
3. Project root auto-detected via git repo, `pyproject.toml`, or `OMERO_SCREEN_PROJECT_ROOT` override

### Required Variables
**OMERO connection**:
- `USERNAME`, `PASSWORD`, `HOST`: OMERO server credentials

**Logging** (configured in `config.py`):
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `LOG_FORMAT`: Python logging format string
- `ENABLE_CONSOLE_LOGGING`: true/false
- `ENABLE_FILE_LOGGING`: true/false
- `LOG_FILE_PATH`: Path relative to project root (default: logs/app.log)
- `LOG_MAX_BYTES`: Log rotation size (default: 1MB)
- `LOG_BACKUP_COUNT`: Number of backup logs (default: 5)

**CellView database**:
- `TEST_DATABASE`: true for test database, false for production
- `DATABASE_PATH`: Path to DuckDB file (e.g., ~/cellview_data/cellview.db)

**Optional configuration**:
- `OMERO_SCREEN_CONFIG`: Path to JSON file overriding default model/feature config
- `OMERO_SCREEN_INFERENCE_MODEL`: Colon-separated model names for classification
- `OMERO_SCREEN_INFERENCE_GALLERY_WIDTH`: Gallery grid size (default: 10)
- `OMERO_SCREEN_INFERENCE_BATCH_SIZE`: Batch size for inference (default: 100)
- `OMERO_SCREEN_CLEAR_BORDER`: Border width for filtering edge cells (default: 5)

### Logging System
**Smart mode detection** (`config.py`):
- **Standalone mode**: Configures root logger when no existing handlers
- **Plugin mode** (e.g., Napari): Configures package-specific loggers, disables propagation to avoid console spam

**Suppressed loggers**: omero, cellpose, matplotlib, numba, fontTools (set to WARNING)

**Module-aware logging**: Automatically converts `__main__` to proper module path for cleaner logs

### Test Server Setup
**Parallel OMERO instance** for safe testing:
- Host: 127.0.0.2 (loopback alias, doesn't conflict with main 127.0.0.1 server)
- Port: 4064
- Credentials: root/omero
- Management: `scripts/manage_test_server.sh` (start/stop/status/restart)
- Docker-based: `docker-compose.test.yml`

## Testing Strategy

### Unit Tests (`tests/unit_tests/`)
**Structure**:
- `omero_screen_tests/`: Main pipeline component tests
- `omero_utils_tests/`: OMERO utility function tests
- `config_tests/`: Configuration and logging tests

**Framework**: pytest with fixtures in `conftest.py`

**Key fixtures**:
- Mock OMERO connections
- Temporary file cleanup
- Environment variable isolation

**Run**: `pytest -v` (excludes e2e tests by default)

### End-to-End Tests (`tests/e2e_tests/`)
**Purpose**: Full pipeline validation against real OMERO test server

**Test modules**:
- `e2e_connection.py`: OMERO connectivity validation
- `e2e_excel.py`: Excel metadata parsing workflow
- `e2e_pixelsize.py`: Pixel size extraction
- `e2e_plate_dataset.py`: Dataset creation
- `e2e_flatfield_corr.py`: Flatfield correction pipeline
- `e2e_omero_screen.py`: Complete analysis pipeline
- `e2e_cellview/`: CellView database import/export tests

**Environment**: Uses `.env.e2etest` configuration

**Execution**: `omero-integration-test <test_name>` (manual trigger, not part of CI)

**Requirements**:
- Test OMERO server running on 127.0.0.2:4064
- Test data plates loaded (use `scripts/load_plates.sh`)
- Isolated test database for CellView

### Test Data
**Examples directory** includes:
- 2D single-timepoint plates
- 3D z-stack plates
- Timeseries plates
- Various cell lines (RPE, HeLa, U2OS)

**Loading test data**: `scripts/load_plates.sh -d /path/to/plates -x`

## Commands & Workflows

### Main Analysis Pipeline
```bash
# Basic usage - analyze plate(s)
omero-screen <plate_id> [<plate_id2> ...]

# Select environment
omero-screen <plate_id> --env production

# Run only segmentation (skip feature extraction)
omero-screen <plate_id> --segmentation

# Enable image classification with inference models
omero-screen <plate_id> --inference model1.pth model2.pth

# Customize classification parameters
omero-screen <plate_id> --inference model.pth --gallery 15 --batch 32
```

**Process**:
1. Connects to OMERO server
2. Parses metadata (Excel or annotations)
3. Generates/retrieves flatfield correction masks
4. Iterates through all wells and images
5. Segments nucleus and cell channels
6. Extracts features using regionprops
7. Performs cell cycle analysis (if EdU channel present)
8. Uploads results (CSV, figures) back to OMERO
9. Saves intermediate results per well, final results on plate

### CellView Database
```bash
# Display available data
cellview display projects
cellview display experiments
cellview display plates

# Import from CSV file
cellview import-csv /path/to/final_data_cc.csv

# Import from OMERO plate (interactive)
cellview import-plate <plate_id>

# Export data
cellview export <plate_id> --format csv
cellview export <plate_id> --format excel

# Database maintenance
cellview cleanup --dry-run  # Preview orphaned records
cellview cleanup            # Remove orphaned records
```

**Python API**:
```python
from cellview.api import cellview_load_data

# Load specific plates
df, variable_names = cellview_load_data(12345, 67890)

# Load entire experiment
df, variable_names = cellview_load_data(experiment="palb_washout")
df, variable_names = cellview_load_data(experiment=6)  # by ID
```

### Plotting
```python
from omero_screen_plots import (
    combplot_cellcycle,
    combplot_feature,
    cellcycle_plot,
    feature_plot,
    count_plot,
)

# Comprehensive cell cycle analysis
fig, axes = combplot_cellcycle(
    df=df,
    conditions=['ctrl', 'drug1', 'drug2'],
    selector_col="cell_line",
    selector_val="RPE",
    save=True,
    file_format="svg"
)

# Feature analysis with threshold
fig, axes = combplot_feature(
    df=df,
    conditions=['ctrl', 'treatment'],
    feature="intensity_mean_p21_nucleus",
    threshold=5000,
    cell_number=3000,
    save=True
)

# Individual plot types
fig, ax = feature_plot(df, feature="area_nucleus", conditions=['ctrl', 'drug'])
fig, ax = count_plot(df, norm_control="ctrl", conditions=['ctrl', 'drug'])
```

### Testing
```bash
# Unit tests
pytest -v                              # All unit tests
pytest tests/unit_tests/omero_screen   # Specific module

# E2E tests (requires test server running)
omero-integration-test e2e_connection
omero-integration-test e2e_excel
omero-integration-test e2e_omero_screen
```

### Test Server Management
```bash
# Start test OMERO server on 127.0.0.2:4064
./scripts/manage_test_server.sh start

# Check status
./scripts/manage_test_server.sh status

# Stop server
./scripts/manage_test_server.sh stop

# Load test data
./scripts/load_plates.sh -d /path/to/test/plates -x
```

## Development Workflow

### Environment Setup
```bash
# Clone repository
git clone https://github.com/Helfrid/omero-screen.git
cd omero-screen

# Install dependencies (uv manages workspace)
uv sync --dev

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools
- **Formatter/Linter**: `ruff check .` and `ruff format .`
- **Type checking**: `mypy .` (strict mode)
- **Pre-commit**: Runs ruff, mypy, nbstripout automatically on commit

### Versioning (Commitizen)
```bash
# Commit with conventional commit format
cz commit

# Version bump (CI handles this automatically)
cz bump

# Commit message scopes trigger package-specific version bumps:
# - No scope: omero-screen (main package)
# - feat(cellview): cellview package
# - feat(omero-utils): omero-utils package
# - etc.
```

**Version files** (all updated by commitizen):
- All `pyproject.toml` files
- All `__init__.py` version strings
- All README.md version badges
- Root `CHANGELOG.md`

### Package Dependencies
**Workspace structure** (`pyproject.toml`):
```toml
[tool.uv.workspace]
members = [
    ".",
    "packages/omero-utils",
    "packages/omero-screen-napari",
    "packages/omero-screen-plots",
    "packages/cellview",
]
```

**Dependency management**:
- Main package depends on all sub-packages
- Sub-packages have minimal interdependencies
- Use `uv add <package>` to add dependencies (automatically detects workspace location)
- Use `uv sync` to update lockfile and install

### Project Structure
```
omero-screen/
├── .env.development, .env.production, .env.e2etest
├── pyproject.toml              # Root workspace config
├── uv.lock                     # Lockfile for all packages
├── src/omero_screen/           # Main analysis pipeline
├── packages/
│   ├── omero-utils/            # OMERO server utilities
│   ├── cellview/               # DuckDB database
│   ├── omero-screen-plots/     # Plotting library
│   └── omero-screen-napari/    # Napari widgets
├── tests/
│   ├── unit_tests/             # Pytest unit tests
│   └── e2e_tests/              # Integration tests
├── bin/                        # CLI entry points
├── scripts/                    # Utility scripts
├── examples/                   # Example data and notebooks
└── .project_notes/             # Institutional memory
```

## Important Design Patterns

### Decorator-Based OMERO Connections
All OMERO operations use `@omero_connect` decorator:
```python
from omero_utils.omero_connect import omero_connect

@omero_connect
def my_function(plate_id: int, conn: BlitzGateway | None = None):
    assert conn is not None
    # Use conn...
    # Connection automatically closed on exit
```

### Configuration Override Pattern
Default configuration can be overridden:
1. `default_config` in `src/omero_screen/__init__.py`
2. Override via JSON file specified in `OMERO_SCREEN_CONFIG` env var
3. Allows partial updates (only specified keys are replaced)

### Incremental Well Processing
Pipeline saves results per well during processing:
- Enables resuming after failures
- Wells with existing results are loaded instead of reprocessed
- Final plate processing removes intermediate well attachments

## Common Issues & Solutions

### GPU Not Detected for Cellpose
Check `omero_screen.torch.get_device()` output in logs. Ensure PyTorch with CUDA support is installed if GPU available.

### Flatfield Correction Takes Too Long
First run generates masks (100 images per channel). Subsequent runs load cached masks from dataset.

### Missing Cell Line Model
Add mapping to `MODEL_DICT` in config JSON:
```json
{
  "MODEL_DICT": {
    "NEWCELL": "cytoplasm_model_name"
  }
}
```

### CellView Import Fails
Ensure CSV has required columns: `plate_id`, `cell_line`, `condition`, and measurement columns.

### Logging Not Working in Napari
System detects plugin mode and configures package-specific loggers. Check `LOG_FILE_PATH` instead of console.

## Project Memory System

This project maintains institutional knowledge in `.project_notes/` for consistency across sessions.

### Memory Files

- **bugs.md** - Bug log with dates, solutions, and prevention notes
- **decisions.md** - Architectural Decision Records (ADRs) with context and trade-offs
- **key_facts.md** - Project configuration, credentials, ports, important URLs
- **issues.md** - Work log with ticket IDs, descriptions, and URLs

### Memory-Aware Protocols

**Before proposing architectural changes:**
- Check `.project_notes/decisions.md` for existing decisions
- Verify the proposed approach doesn't conflict with past choices
- If it does conflict, acknowledge the existing decision and explain why a change is warranted

**When encountering errors or bugs:**
- Search `.project_notes/bugs.md` for similar issues
- Apply known solutions if found
- Document new bugs and solutions when resolved

**When looking up project configuration:**
- Check `.project_notes/key_facts.md` for credentials, ports, URLs, service accounts
- Prefer documented facts over assumptions

**When completing work on tickets:**
- Log completed work in `.project_notes/issues.md`
- Include ticket ID, date, brief description, and URL

**When user requests memory updates:**
- Update the appropriate memory file (bugs, decisions, key_facts, or issues)
- Follow the established format and style (bullet lists, dates, concise entries)

### Style Guidelines for Memory Files

- **Prefer bullet lists over tables** for simplicity and ease of editing
- **Keep entries concise** (1-3 lines for descriptions)
- **Always include dates** for temporal context
- **Include URLs** for tickets, documentation, monitoring dashboards
- **Manual cleanup** of old entries is expected (not automated)
