# Environment Setup and Configuration

Installing, configuring and checking omero-screen. Full guide:
https://hocheggerlab.github.io/omero-screen/ (Install & configure).

---

## Install

**Users** (macOS / Linux), one line in Terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/HocheggerLab/omero-screen/main/install.sh | sh
```

It installs uv if needed, downloads the latest release (a git tag
`omero-screen-vX.Y.Z`), installs the locked environment with napari into
`~/.local/share/omero-screen/<version>` (`current` links to it), links the
commands into `~/.local/bin`, then runs `omero-screen setup` and
`omero-screen doctor`. Rerunning it updates the install. Options as environment
variables: `OMERO_SCREEN_VERSION`, `OMERO_SCREEN_BRANCH` (testing),
`OMERO_SCREEN_HOME`, `OMERO_SCREEN_BIN`, `OMERO_SCREEN_NO_SETUP=1`.

**Developers:**

```bash
git clone https://github.com/HocheggerLab/omero-screen.git
cd omero-screen
uv sync --dev
uv run omero-screen setup
pre-commit install
```

uv only, never pip: zeroc-ice comes from the lab's wheel index
(hocheggerlab.github.io/ice-wheels), configured in `pyproject.toml`.

---

## Configure and check

```bash
omero-screen setup         # site profile, OMERO user, group; password -> keychain
omero-screen doctor        # versions, ice, config, login + connection, CellView, napari, device, disk
omero-screen config show   # effective settings and where each came from (password masked)
```

**Precedence**, highest first: CLI flags > environment variables > `.env`
file (developer checkout) > user config `~/.config/omero-screen/config.toml`
> site profile > built-in defaults.

**User config** (`setup` writes the first two lines):

```toml
site = "sussex"                 # shipped profile name, or a path to a .toml

[omero]
username = "ab123"
# group = "lab-group"           # default: the user's OMERO default group
# password_file = "~/.omero-password"   # headless machines only (chmod 600)

[segmentation]
# model_set = "hocheggerlab"    # a site model set (omero-screen models pull <set>)

[segmentation.models]           # per role / cell line, overrides the model set
# nuclei = "Nuclei_Hoechst"
# RPE = "RPE-1_Tub_Hoechst"

[features]                      # FEATURELIST override
# intensity = ["intensity_mean", "intensity_max"]
# morphology = ["area"]

[paths]
# cache = "~/omero-cache"
# cellview_db = "~/.cellview/cellview.duckdb"
# training_db = "..."
# cellpose_models = "~/.cellpose/models"

[env]                           # any OMERO_SCREEN_* variable
# OMERO_SCREEN_USE_GPU = "0"
```

**Site profile** (`src/omero_screen/sites/<name>.toml`): `[omero]` host/port,
`[stitching.calibrations.<objective>]` (pixel_size_um, overlap_x/y,
translate_x/y; chosen by pixel size), `[segmentation.model_sets.<name>]`
(models, sha256, download url).

**Password:** never in a config file. Looked up as `PASSWORD` env var, then the
system keychain (service `omero-screen`, account `user@host`), then
`[omero] password_file`.

**Developer `.env` files** (`.env.development`, `.env.production` with
`ENV=production`, `.env.e2etest`) at the repo root still work and take
precedence: `USERNAME`, `PASSWORD`, `HOST`, optionally `DATABASE_PATH`.

---

## Segmentation models

- **Default with nothing configured:** Cellpose 4 (`cp4:cpsam`) on an NVIDIA
  GPU; stock Cellpose 3 (`cp3:nuclei`, `cp3:cyto3`) on a Mac or CPU, with a
  one-time hint to train a custom model.
- `--cp4` or `--model NAME` override every role for one run.
- Lab models: `omero-screen models pull hocheggerlab`, then
  `[segmentation] model_set = "hocheggerlab"`.
- Each run records versions, flags, device and models on the plate (map
  annotation, namespace `omero-screen/provenance`).

## Classifiers

`omero-screen models publish NAME.pt` uploads the `.pt` and its `.json` sidecar
(from `cellclass extract`) to the user's `Classifiers` project in OMERO;
`--replace` overwrites. The pipeline then uses `--inference NAME` (no `.pt`).

---

## Other environment variables

```bash
OMERO_SCREEN_CONFIG=/path/to/config.json         # JSON MODEL_DICT/FEATURELIST/CHANNEL_SEG_PROFILES (over the TOML)
OMERO_SCREEN_STITCH_CONFIG=/path/to/stitch.json  # one fixed stitch setting (disables per-objective choice)
OMERO_SCREEN_INFERENCE_MODEL=name1:name2         # set by --inference
OMERO_SCREEN_LOG_LEVEL=DEBUG
OMERO_SCREEN_LOG_FILE=logs/app.log               # "none" disables the file log
OMERO_SCREEN_USE_GPU=0                           # force CPU
OMERO_SCREEN_CLEAR_BORDER=5
OMERO_SCREEN_CACHE_PATH=~/omero-cache
```

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

`omero-screen doctor` also reports the device. The locked torch wheel from
PyPI bundles CUDA on Linux x86_64; no separate index is needed.

---

## HPC / Remote Usage

Running on the HPC uses the sbatch wrapper (in the HPC repo):

```bash
./sbatch-omero-screen.py <plate_id> --env production
./sbatch-omero-screen.py --inference micronuclei_densenet -e omero-screen-infer 1821
```

HPC instructions (Alex): https://gist.github.com/aherbert/a2c0ba5242ba68918f5f109d40680312

Logging: the file log is on by default (`logs/app.log`, or `OMERO_SCREEN_LOG_FILE`). On the HPC there is no keychain: use `PASSWORD` or `[omero] password_file`.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Import errors after `uv sync` | Check workspace members in `pyproject.toml`; try `uv sync --reinstall` |
| Settings not as expected | `omero-screen config show` lists each value and its source |
| Anything fails | `omero-screen doctor` |
| `pre-commit` hook fails | Fix the underlying issue (ruff/mypy error) — never use `--no-verify` |
| mypy strict errors in new code | Add type hints to all function signatures; use `from __future__ import annotations` for forward refs |
