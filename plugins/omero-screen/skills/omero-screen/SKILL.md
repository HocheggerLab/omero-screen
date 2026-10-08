---
name: omero-screen
description: Answer questions and guide workflows for the omero-screen monorepo — running the analysis pipeline, OMERO server setup, CellView database, classifier training, plotting, and napari widgets. Use when a user asks "how do I..." about any omero-screen tool or workflow.
---

# OMERO-Screen Skill

This skill covers all user-facing workflows in the **omero-screen** monorepo: a Python pipeline for high-content immunofluorescence microscopy analysis connecting OMERO server, Cellpose segmentation, DuckDB storage, napari visualisation, and CNN classifiers.

**Docs:** https://hocheggerlab.github.io/omero-screen/
**GitHub:** https://github.com/HocheggerLab/omero-screen

**Two kinds of install:**
- **User install** (`install.sh`): the commands (`omero-screen`, `cellview`,
  `cellclass`, `napari`, …) are on the `PATH` in `~/.local/bin`; the code is in
  `~/.local/share/omero-screen/current`. There is no git checkout, so run
  commands directly, without `uv run`.
- **Developer install**: a clone (usually `~/code/omero-screen`); run commands
  with `uv run` from it.

Settings live in `~/.config/omero-screen/config.toml` (written by
`omero-screen setup`); the password is in the system keychain.

---

## Working with someone who does not code

Many users are biologists who know chat assistants but not the terminal.

- **Check first, explain after.** Start with `omero-screen doctor`. When
  anything fails, run it again and fix what it reports before trying
  something else. Most problems are the VPN (OMERO unreachable) or a missing
  `omero-screen setup`.
- **Say what you are about to do, in plain words**, before each command
  ("I'll look up plate 1234 in OMERO"), and what the result means after it.
  Don't paste long logs; summarise them.
- **Never show or ask for the password in the chat.** `omero-screen setup` asks
  for it itself, in the terminal; tell the user to run it there. Don't read or
  print `~/.config/omero-screen/`, `.env` files or the keychain.
- **Ask before anything that changes OMERO**: running the pipeline on a plate
  (it writes masks and results), `omero-screen models publish`, deleting
  data. Reading is fine without asking.
- **napari is a window on the user's screen.** Start it for them
  (`napari &`), then say exactly which menu to open and what to click.
- **Results go into CellView.** For analysis and figures, load data with
  `cellview_load_data` and the omero-screen-plots functions; save figures as
  PDF and say where.

---

## Dispatch Table

Read the relevant reference file before answering questions in these areas:

| User asks about... | Read |
|---|---|
| Running the pipeline, plate IDs, segmentation, inference, HPC | `references/pipeline.md` |
| OMERO server setup, Docker, test server, loading data | `references/docker-setup.md` |
| CellView database, import CSV/plate, export, Python API | `references/cellview.md` |
| Classifier training, generating training data, labelling crops, inference | `references/classifier-training.md` |
| Plots, cell cycle figures, feature plots, normalisation | `references/plotting.md` |
| Napari widgets, browsing images, gallery, training sessions | `references/napari.md` |
| Images to support a quantitative finding (galleries, well overviews, close-ups) without napari, `omero-screen-images`, figure panels from a notebook | `references/images.md` |
| Environment setup, uv install, .env files, config, dependencies | `references/environment.md` |

For questions spanning multiple areas, read both reference files before answering.

---

## Quick Reference

### Run the pipeline
```bash
omero-screen <plate_id>                          # basic run
omero-screen 1234 1235 --env production          # multiple plates, production env
omero-screen 1234 --segmentation                 # segmentation only, no feature extraction
omero-screen 1234 --inference micronuclei       # classifier by name (published with `omero-screen models publish`)
omero-screen 1234 --cp4                          # use Cellpose 4 (cpsam) for everything
omero-screen 1234 --model cp4:cpsam              # override all models explicitly
omero-screen 1234 --benchmark                    # record per-image timing JSON
```

### CellView quick commands
```bash
cellview projects                              # list all projects
cellview project <id>                          # show project detail
cellview import csv /path/to/final_data_cc.csv
cellview import plate <plate_id> [<id2> ...]
cellview import screen <screen_id>
cellview export <plate_id>
cellview explore <plate_id> --template cellcycle  # launch Jupyter notebook
cellview clean                                 # remove orphaned records
```

```python
from cellview.api import cellview_load_data
df, vars = cellview_load_data(12345)                         # by plate ID
df, vars = cellview_load_data(experiment="palb_washout")    # by experiment name
```

### Test server
```bash
./scripts/manage_test_server.sh start|stop|status
./scripts/load_plates.sh -d /path/to/plates -x
```

### Setup and checks
```bash
omero-screen setup         # server, user name, password (keychain)
omero-screen doctor        # check everything; says how to fix failures
omero-screen config show   # settings in effect and where each came from
omero-screen --version
omero-screen models pull hocheggerlab   # the lab's custom Cellpose models
omero-screen models publish model.pt    # upload a classifier for --inference
```

---

## Package Map

```
omero-screen/
├── src/omero_screen/        # Core pipeline (loops, segmentation, cell cycle, QC)
├── packages/
│   ├── omero-utils/         # OMERO connection decorator, attachments, annotations
│   ├── cellview/            # DuckDB database, CLI, Python API
│   ├── omero-screen-plots/  # Publication-ready statistical plots
│   ├── omero-screen-napari/ # Napari widgets for browsing and classifier training
│   └── cellclass/           # CNN classifier training pipeline
├── bin/                     # run_omero_screen.py, aggregate_plates.py, seg-samples.py
├── scripts/                 # manage_test_server.sh, load_plates.sh
└── tests/unit_tests/ + e2e_tests/
```

---

## Common Issues (quick answers)

| Problem | Fix |
|---|---|
| Anything fails | Run `omero-screen doctor` first and fix what it reports |
| Cannot connect to OMERO | VPN off, or login not set up: `omero-screen setup` |
| GPU not detected | `omero-screen doctor` shows the device; on a Mac it is MPS, not CUDA |
| Flatfield correction slow | First run generates masks from 100 images — subsequent runs load cached masks |
| Which model segmented a plate? | Read the `omero-screen/provenance` map annotation on the plate |
| Custom model per cell line | `[segmentation.models]` in the user config (`CELLLINE = "model_name"`), or `[segmentation] model_set = "..."` |
| CellView import fails | CSV needs `plate_id`, `cell_line`, `condition` columns plus measurement columns |
| Logging missing in napari | Plugin mode writes to a file: `logs/app.log`, or `OMERO_SCREEN_LOG_FILE` |
| Default segmentation model | With no models configured: Cellpose 4 (cpsam) on an NVIDIA GPU, stock Cellpose 3 (nuclei, cyto3) on a Mac or CPU. `--cp4` / `--model` override it |
