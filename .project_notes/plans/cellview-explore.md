## CellView Explore — Implementation Plan

### Summary

`cellview --explore` launches a template Jupyter notebook for analysing CellView plate data,
optionally alongside a standalone napari viewer. The notebook contains pre-built cells that load
data via `cellview_load_data()` and produce publication-ready plots via `omero_screen_plots`.

Napari and the notebook are **separate processes** — no in-notebook plotly↔napari linking.
The user works in the notebook for analysis/figures and in napari for image inspection.

### CLI Usage

```bash
# Single plate
cellview --explore 12345

# Multiple plates
cellview --explore 12345 12378 12390

# All plates from an experiment (by name or ID)
cellview --explore-experiment palb_washout
cellview --explore-experiment 6

# Flags
cellview --explore 12345 --fresh        # Regenerate notebook even if it exists
cellview --explore 12345 --no-napari    # Skip launching napari
```

### IDE Selection

Set `CELLVIEW_EDITOR` env var:
- `jupyter` (default) → launches `jupyter lab <notebook>`
- `vscode` → launches `code <notebook>` (kernel auto-detected from active venv)

### Notebook Storage

Deterministic paths in `~/.cellview/explore/`:
- Single plate: `explore_plate_12345.ipynb`
- Multiple plates (sorted): `explore_plates_12345_12378_12390.ipynb`
- Experiment: `explore_exp_6.ipynb`

If notebook exists, it's reopened (preserving user work). `--fresh` forces regeneration.

### Notebook Content

Generated programmatically via `nbformat` (no static template to maintain). Cells:
1. Title with plate/experiment info
2. Setup: `cellview_load_data()` call, print summary
3. Configuration: editable conditions, cell_line, feature, threshold
4. Combined cell cycle plot (`combplot_cellcycle`)
5. Combined feature plot (`combplot_feature`)
6. Count plot (`count_plot`)
7. Individual feature plot (`feature_plot`)
8. Histogram (`histogram_plot`)
9. Scatter plot (`scatter_plot`)
10. Tips markdown

### Display Integration

`cellview --experiment <id>` shows a `Notebooks` column in the plates table,
listing which explore notebooks reference each plate. Experiment-level notebook
shown separately below the table.

### Files

#### Created
- `packages/cellview/src/cellview/explore/__init__.py` — package marker
- `packages/cellview/src/cellview/explore/_registry.py` — filesystem scan for notebooks
- `packages/cellview/src/cellview/explore/_notebook_builder.py` — programmatic notebook generation
- `packages/cellview/src/cellview/explore/_cli.py` — `launch_explore()` entry point
- `tests/unit_tests/cellview_tests/explore_tests/test_cli.py` — CLI arg parsing tests
- `tests/unit_tests/cellview_tests/explore_tests/test_notebook_builder.py` — notebook validation tests
- `tests/unit_tests/cellview_tests/explore_tests/test_registry.py` — registry/filesystem tests

#### Modified
- `packages/cellview/src/cellview/cli.py` — added `--explore`, `--explore-experiment`, `--fresh`, `--no-napari`
- `packages/cellview/src/cellview/main.py` — explore dispatch before DB setup
- `packages/cellview/src/cellview/db/display.py` — notebooks column in experiment display
- `packages/cellview/pyproject.toml` — `explore` optional deps
- `pyproject.toml` — mypy overrides for nbformat

### Future Work

- **bridge.py**: Interactive `ExploreSession` class linking plotly scatter selection
  to napari point layers (deferred — adds complexity without clear user demand)
- Napari integration within notebook via `%gui qt` (deferred for same reason)
- `cellview display notebooks` subcommand for listing all explore notebooks
