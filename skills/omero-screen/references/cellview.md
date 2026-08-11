# CellView — Database CLI and Python API

CellView is the DuckDB-backed local database for organising single-cell IF measurements from omero-screen plates.

Package: `packages/cellview/`
Default DB location: `~/.cellview/cellview.duckdb`

---

## CLI Command Reference

All commands accept `--db /path/to/db` to override the default database path.

### Display

```bash
# List all projects with experiment counts
cellview projects

# Show experiments and plates for a project
cellview project <id>

# Show plates, channels, and variables for an experiment
cellview experiment <id>

# Show summary, conditions, and measurements for a plate
cellview plate <id>
```

### Import

```bash
# Import from a CSV file (interactive: assigns to project/experiment)
cellview import csv /path/to/final_data_cc.csv

# Import one or more plates from OMERO by plate ID
cellview import plate <id> [<id2> ...]
cellview import plate <id> --interactive    # force project/experiment selection dialog

# Import all plates from an OMERO screen
cellview import screen <id>
cellview import screen <id> --interactive

# Skip the project/experiment prompts by naming the target up front.
# Works on every import route; --experiment implies its parent project.
# Saves answering the same prompt once per plate on multi-plate imports.
cellview import plate <id> <id> --experiment <experiment_id>
cellview import plate <id> --project <project_id>
```

### Edit

```bash
# Edit project name and description (interactive prompts)
cellview edit project <id>

# Edit experiment name and description (interactive prompts)
cellview edit experiment <id>
```

### Export

```bash
# Export plate data (format is interactive or configured in DB settings)
cellview export <id>
```

### Delete

```bash
# Delete a plate and all its associated data
cellview delete plate <id>

# Delete several plates in one go (deleted in the order given)
cellview delete plate <id> <id> <id>
```

### Clean

```bash
# Remove orphaned records (conditions/measurements with no parent)
cellview clean
```

### Explore (Jupyter notebook launcher)

```bash
# Launch a Jupyter notebook for one or more plates
cellview explore <plate_id> [<plate_id2> ...]

# Explore all plates from an experiment
cellview explore --experiment "palb_washout"
cellview explore --experiment 6              # by experiment ID

# Use a specific analysis template (default: cellcycle)
cellview explore <plate_id> --template cellcycle
cellview explore <plate_id> --template feature

# Regenerate even if notebook already exists
cellview explore <plate_id> --fresh

# Open in VS Code instead of JupyterLab
cellview explore <plate_id> --code

# Skip launching napari alongside the notebook
cellview explore <plate_id> --no-napari

# Print JSON context snapshot to stdout (used by agentic tools)
cellview explore <plate_id> --json
```

### Template management

```bash
# List registered analysis templates
cellview template list

# Register a new template notebook
cellview template add /path/to/template.ipynb
cellview template add /path/to/template.ipynb --name my_template --description "My analysis"

# Remove a template (does not delete the file)
cellview template remove my_template

# Show details for a template
cellview template show cellcycle

# Scan filesystem and register all discovered templates
cellview template sync
```

---

## Python API

```python
from cellview.api import cellview_load_data

# Load one or more plates by OMERO plate ID
df, variable_names = cellview_load_data(12345)
df, variable_names = cellview_load_data(12345, 67890)

# Load an entire experiment by name
df, variable_names = cellview_load_data(experiment="palb_washout")

# Load an entire experiment by ID
df, variable_names = cellview_load_data(experiment=6)

# Returns:
# df             — pandas DataFrame with all single-cell measurements
# variable_names — list of experimental variable names (e.g. ["palb", "gwli"])
```

### Working with the DataFrame

```python
# Filter by cell line
rpe_df = df[df["cell_line"] == "RPE"]

# Filter by condition
ctrl = df[df["condition"] == "control"]

# Cell cycle distribution
cc_counts = df.groupby(["condition", "cell_cycle"]).size().unstack()

# Access single-cell intensity features
dapi = df["integrated_int_DAPI"]
edu  = df["intensity_mean_EdU_nucleus"]
```

---

## Database Schema

```
projects
  └── experiments
        └── plates          (one per OMERO plate)
              └── conditions  (one per well)
                    └── condition_variables  (key-value: drug, dose, siRNA...)
                          └── measurements  (one row per cell)
```

---

## Environment Variables

```bash
# Set in .env file
DATABASE_PATH=~/.cellview/cellview.duckdb
TEST_DATABASE=false    # true → uses a separate test database
```

---

## Typical Analysis Session

```python
from cellview.api import cellview_load_data
import pandas as pd

# 1. Load experiment
df, vars = cellview_load_data(experiment="my_experiment")

# 2. Inspect content
print(df["cell_line"].unique())
print(df["condition"].unique())
print(df["cell_cycle"].value_counts())

# 3. Filter and plot
df_rpe = df[df["cell_line"] == "RPE"]

from omero_screen_plots import combplot_cellcycle
fig, axes = combplot_cellcycle(
    df=df_rpe,
    conditions=df_rpe["condition"].unique().tolist(),
    save=True
)
```

Or use the CLI explore command to get a ready-made notebook:

```bash
cellview explore 12345 --template cellcycle
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `plate not found` | Plate not yet imported — run `cellview import csv` or `cellview import plate` |
| Interactive prompts don't appear | Add `--interactive` flag to `import plate` or `import screen` |
| DB at wrong path | Pass `--db /correct/path.duckdb` or set `DATABASE_PATH` in `.env` |
| `cellview clean` removes too much | Run `cellview plate <id>` first to verify what is orphaned |
| Export produces no output | Check that the plate has measurements — `cellview plate <id>` to inspect |
