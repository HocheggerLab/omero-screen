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

### Project Structure
```
omero-screen/
├── .env.development, .env.production, .env.e2etest
├── pyproject.toml              # Root workspace config
├── uv.lock
├── src/omero_screen/           # Core pipeline
├── packages/
│   ├── omero-utils/
│   ├── cellview/
│   ├── omero-screen-plots/
│   ├── omero-screen-napari/
│   └── cellclass/
├── tests/
│   ├── unit_tests/             # pytest unit tests
│   └── e2e_tests/              # integration tests
├── bin/                        # CLI entry points
├── scripts/                    # manage_test_server.sh, load_plates.sh
└── .project_notes/             # bugs.md, decisions.md, key_facts.md, issues.md
```

## Environment Configuration

### Environment Files
`.env.{ENV}` pattern (defaults to `.env`):
- `.env.development`: Local testing — localhost OMERO, console logging on
- `.env.production`: Production server — file logging only
- `.env.e2etest`: E2E test environment — localhost, file logging only

Set `ENV` environment variable to select (default: `"development"`). Project root auto-detected via git / `pyproject.toml` / `OMERO_SCREEN_PROJECT_ROOT`.

### Required Variables
```
# OMERO connection
USERNAME, PASSWORD, HOST, PROJECT_ID

# Logging (config.py)
LOG_LEVEL, LOG_FORMAT, ENABLE_CONSOLE_LOGGING, ENABLE_FILE_LOGGING
LOG_FILE_PATH, LOG_MAX_BYTES, LOG_BACKUP_COUNT

# CellView database
TEST_DATABASE, DATABASE_PATH

# Optional pipeline config
OMERO_SCREEN_CONFIG           # Path to JSON overriding MODEL_DICT / FEATURELIST
OMERO_SCREEN_INFERENCE_MODEL  # Colon-separated .pth model names for classification
OMERO_SCREEN_INFERENCE_GALLERY_WIDTH  # default: 10
OMERO_SCREEN_INFERENCE_BATCH_SIZE     # default: 100
OMERO_SCREEN_CLEAR_BORDER             # Border width pixels (default: 5)
```

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
