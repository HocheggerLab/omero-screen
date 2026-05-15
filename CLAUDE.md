# OMERO-Screen Project Overview

## Project Summary
OMERO-Screen is a monorepo for high-content IF microscopy analysis: OMERO server → Cellpose segmentation → feature extraction → cell cycle analysis → DuckDB storage → publication figures.

**Version**: 0.3.4 | **Python**: 3.12 | **License**: MIT

GitHub: https://github.com/Helfrid/omero-screen | Docs: https://hocheggerlab.github.io/omero-screen/

## Architecture Overview

`uv` workspace with 6 packages. For package-level architecture, consult the Obsidian vault sub-MOCs (loaded on demand via `.claude/CLAUDE.md` session protocol).

| Package | Path | Vault MOC |
|---|---|---|
| omero-screen (core) | `src/omero_screen/` | `&OmeroScreenCore` |
| omero-utils | `packages/omero-utils/` | `&OmeroUtils` |
| cellview | `packages/cellview/` | `&Cellview` |
| omero-screen-plots | `packages/omero-screen-plots/` | `&OmeroScreenPlots` |
| omero-screen-napari | `packages/omero-screen-napari/` | `&OmeroScreenNapari` |
| cellclass | `packages/cellclass/` | `&CellClass` |

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
`.env.{ENV}` pattern (defaults to `.env`):
- `.env.development`: Local testing — localhost OMERO, console logging on
- `.env.production`: Production server — file logging only
- `.env.e2etest`: E2E test environment — localhost, file logging only

Set `ENV` environment variable to select (default: `"development"`). Project root auto-detected via git / `pyproject.toml` / `OMERO_SCREEN_PROJECT_ROOT`.

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
Smart mode detection in `config.py`:
- **Standalone**: configures root logger when no existing handlers
- **Plugin mode** (napari): configures package-specific loggers, disables propagation

Suppressed loggers (WARNING): omero, cellpose, matplotlib, numba, fontTools

### Test Server
Parallel OMERO instance at 127.0.0.2:4064 (credentials: root/omero). Docker-based via `docker-compose.test.yml`. Managed by `scripts/manage_test_server.sh`.

## Commands & Workflows

### Main Pipeline
```bash
omero-screen <plate_id> [<plate_id2> ...]
omero-screen <plate_id> --env production
omero-screen <plate_id> --segmentation          # skip feature extraction
omero-screen <plate_id> --inference model.pth --gallery 15 --batch 32
```

### CellView Database
```bash
cellview display projects|experiments|plates
cellview import-csv /path/to/final_data_cc.csv
cellview import-plate <plate_id>
cellview export <plate_id> --format csv|excel
cellview cleanup [--dry-run]
```

```python
from cellview.api import cellview_load_data
df, variable_names = cellview_load_data(12345, 67890)
df, variable_names = cellview_load_data(experiment="palb_washout")
```

### Plotting
```python
from omero_screen_plots import combplot_cellcycle, combplot_feature, feature_plot, count_plot

fig, axes = combplot_cellcycle(df=df, conditions=['ctrl', 'drug1'],
                               selector_col="cell_line", selector_val="RPE", save=True)
fig, axes = combplot_feature(df=df, conditions=['ctrl', 'treatment'],
                             feature="intensity_mean_p21_nucleus", threshold=5000, save=True)
```

### Testing
```bash
pytest -v                                      # all unit tests
pytest tests/unit_tests/omero_screen           # specific module
omero-integration-test e2e_connection          # e2e (requires test server)
omero-integration-test e2e_omero_screen
```

### Test Server Management
```bash
./scripts/manage_test_server.sh start|stop|status|restart
./scripts/load_plates.sh -d /path/to/plates -x
```

## Development Workflow

```bash
uv sync --dev && source .venv/bin/activate
pre-commit install
ruff check . && ruff format .
mypy .
cz commit   # conventional commit format; scope triggers package version bump
```

**Version files updated by commitizen**: all `pyproject.toml`, `__init__.py`, `README.md`, `CHANGELOG.md`

**Workspace members** (`pyproject.toml`):
```toml
[tool.uv.workspace]
members = [".", "packages/omero-utils", "packages/omero-screen-napari",
           "packages/omero-screen-plots", "packages/cellview", "packages/cellclass"]
```

## Important Design Patterns

### Decorator-Based OMERO Connections
```python
from omero_utils.omero_connect import omero_connect

@omero_connect
def my_function(plate_id: int, conn: BlitzGateway | None = None):
    assert conn is not None
    # conn opened and closed automatically
```

### Configuration Override
1. `default_config` in `src/omero_screen/__init__.py`
2. Override via JSON at `OMERO_SCREEN_CONFIG` — partial updates only replace specified keys
3. Example: add `{"MODEL_DICT": {"NEWCELL": "cytoplasm_model_name"}}`

### Incremental Well Processing
`loops.py` saves per-well CSV during processing. Wells with existing results are loaded rather than reprocessed. Intermediate files removed after plate completes.

## Common Issues & Solutions

| Issue | Solution |
|---|---|
| GPU not detected | Check `omero_screen.torch.get_device()` in logs; verify PyTorch+CUDA |
| Flatfield correction slow | First run samples 100 images; subsequent runs load cached masks from dataset |
| Missing cell line model | Add to `MODEL_DICT` in config JSON via `OMERO_SCREEN_CONFIG` |
| CellView import fails | CSV needs `plate_id`, `cell_line`, `condition`, and measurement columns |
| Logging missing in napari | Plugin mode uses file log — check `LOG_FILE_PATH` |

## Project Memory System

Institutional knowledge lives in two places:

**`.project_notes/`** (in-repo, git-tracked):
- `bugs.md` — bug log with dates and solutions
- `decisions.md` — architectural decision records
- `key_facts.md` — credentials, ports, URLs
- `issues.md` — work log with dates and branch names

**Obsidian vault** (loaded on demand via `.claude/CLAUDE.md` session protocol):
- `&OmeroScreenCore` — pipeline architecture, segmentation strategy, cell cycle phases
- `&OmeroUtils` — OMERO decorator pattern, attachment ops
- `&Cellview` — DuckDB schema, CLI architecture
- `&OmeroScreenPlots` — plot types, normalisation approach
- `&OmeroScreenNapari` — widget architecture, training data pipeline
- `&CellClass` — CNN training workflow, inference integration
- `@OmeroScreen_progresslog` — session records, key decisions

Before proposing architectural changes: check `.project_notes/decisions.md` and the relevant vault sub-MOC.
