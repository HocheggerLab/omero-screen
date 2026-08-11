# Environment Setup and Configuration

Full setup guide for the omero-screen monorepo: installation, environment files, dependencies, and tooling.

---

## First-Time Installation

```bash
# Clone the repository
git clone https://github.com/Helfrid/omero-screen.git
cd omero-screen

# Install all packages (uv manages the workspace)
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

**Requirements:** Python 3.12+, `uv` package manager, Docker (for OMERO server).

Install `uv` if not present:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Environment Files

The system uses `.env.{ENV}` files with fallback to `.env`:

| File | Purpose |
|---|---|
| `.env.development` | Local dev — localhost OMERO, console logging, test database |
| `.env.production` | Production HPC server — file logging only, production credentials |
| `.env.e2etest` | E2E tests — test server at 127.0.0.2:4064 |

**Selecting an environment:**
```bash
# Via shell variable
ENV=production omero-screen 1234

# Or export for the session
export ENV=production
omero-screen 1234
```

Default is `development` if `ENV` is not set.

### Creating a new environment file

Copy an existing one and edit:
```bash
cp .env.development .env.myenv
# Edit .env.myenv with your credentials
ENV=myenv omero-screen 1234
```

---

## Required Environment Variables

```bash
# OMERO server connection
USERNAME=root
PASSWORD=omero
HOST=127.0.0.1
PROJECT_ID=1

# Logging
LOG_LEVEL=DEBUG           # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=false
LOG_FILE_PATH=logs/app.log
LOG_MAX_BYTES=1048576     # 1MB
LOG_BACKUP_COUNT=5

# CellView database
TEST_DATABASE=false
DATABASE_PATH=~/cellview_data/cellview.db
```

---

## Optional Environment Variables

```bash
# Override the default MODEL_DICT and FEATURELIST
OMERO_SCREEN_CONFIG=/path/to/my_config.json
# Override the default plate stitching parameters
OMERO_SCREEN_STITCH_CONFIG=/path/to/my_stitch_config.json

# Classifier inference
OMERO_SCREEN_INFERENCE_MODEL=model1.pth:model2.pth  # colon-separated
OMERO_SCREEN_INFERENCE_GALLERY_WIDTH=10              # N for NxN gallery
OMERO_SCREEN_INFERENCE_BATCH_SIZE=16

# Segmentation
OMERO_SCREEN_CLEAR_BORDER=5   # border width in pixels for edge-cell removal
```

---

## Config JSON Override

Override model assignments without editing code:

```json
{
  "MODEL_DICT": {
    "RPE": "RPE-1_Tub_Hoechst",
    "HELA": "cp4:cpsam",
    "U2OS": "cp4:cpsam",
    "NEWCELL": "my_custom_model_name"
  },
  "FEATURELIST": [
    "label", "area", "intensity_mean", "intensity_max", "centroid"
  ]
}
```

```bash
export OMERO_SCREEN_CONFIG=/path/to/config.json
omero-screen 1234
```

Partial updates are supported — only the specified keys are replaced.

---

## Workspace Structure

The monorepo is a `uv` workspace:

```toml
# pyproject.toml (root)
[tool.uv.workspace]
members = [
    ".",
    "packages/omero-utils",
    "packages/omero-screen-napari",
    "packages/omero-screen-plots",
    "packages/cellview",
    "packages/cellclass",
]
```

### Adding dependencies

```bash
# To the main omero-screen package
uv add numpy

# To a specific sub-package
uv add --package cellview polars

# Development dependency
uv add --dev pytest-cov

# Sync after editing pyproject.toml manually
uv sync --dev
```

---

## Code Quality

```bash
# Format and lint
ruff format .
ruff check .
ruff check . --fix   # auto-fix safe issues

# Type check
mypy .

# All at once (pre-commit runs these on commit)
pre-commit run --all-files
```

---

## Versioning

Uses [Commitizen](https://commitizen-tools.github.io/commitizen/) with conventional commits:

```bash
# Interactive commit (guides you through format)
cz commit

# Version bump (usually done by CI)
cz bump
```

Commit message scopes trigger package-specific version bumps:
- No scope → `omero-screen` (main package)
- `feat(cellview): ...` → `cellview` package
- `feat(omero-utils): ...` → `omero-utils` package
- `feat(napari): ...` → `omero-screen-napari` package

Version strings updated automatically in all `pyproject.toml`, `__init__.py`, `README.md`, and `CHANGELOG.md` files.

---

## Running Tests

```bash
# All unit tests
pytest -v

# Specific package
pytest tests/unit_tests/omero_screen -v
pytest tests/unit_tests/cellview -v

# With coverage
pytest --cov=src --cov=packages -v

# E2E tests (requires running test server)
./scripts/manage_test_server.sh start
omero-integration-test e2e_connection
omero-integration-test e2e_omero_screen
```

---

## GPU Setup for Cellpose

```bash
# Check if GPU is detected
python -c "from omero_screen.torch import get_device; print(get_device())"

# Or use the dedicated script
python bin/torch-test.py
```

For CUDA: install PyTorch with CUDA support matching your driver version. See https://pytorch.org/get-started/locally/

```bash
# Example: CUDA 12.1
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## HPC / Remote Usage

Running on the HPC uses the sbatch wrapper (in the HPC repo):

```bash
./sbatch-omero-screen.py <plate_id> --env production
./sbatch-omero-screen.py --inference micronuclei_densenet -e omero-screen-infer 1821
```

HPC instructions (Alex): https://gist.github.com/aherbert/a2c0ba5242ba68918f5f109d40680312

Logging: on HPC, `ENABLE_CONSOLE_LOGGING=false` and `ENABLE_FILE_LOGGING=true`. Check `LOG_FILE_PATH` for output.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Import errors after `uv sync` | Check workspace members in `pyproject.toml`; try `uv sync --reinstall` |
| `.env` not loading | Check `ENV` variable is set correctly; project root detection needs git or `pyproject.toml` |
| `pre-commit` hook fails | Fix the underlying issue (ruff/mypy error) — never use `--no-verify` |
| mypy strict errors in new code | Add type hints to all function signatures; use `from __future__ import annotations` for forward refs |
