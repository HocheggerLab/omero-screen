---
name: omero-screen
description: Answer questions and guide workflows for the omero-screen monorepo — running the analysis pipeline, OMERO server setup, CellView database, classifier training, plotting, and napari widgets. Use when a user asks "how do I..." about any omero-screen tool or workflow.
---

# OMERO-Screen Skill

This skill covers all user-facing workflows in the **omero-screen** monorepo: a Python pipeline for high-content immunofluorescence microscopy analysis connecting OMERO server, Cellpose segmentation, DuckDB storage, napari visualisation, and CNN classifiers.

**Docs:** https://hocheggerlab.github.io/omero-screen/
**GitHub:** https://github.com/Helfrid/omero-screen
**Project root:** `/Users/hh65/code/omero-screen`

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
| Environment setup, uv install, .env files, config, dependencies | `references/environment.md` |

For questions spanning multiple areas, read both reference files before answering.

---

## Quick Reference

### Run the pipeline
```bash
omero-screen <plate_id>                          # basic run
omero-screen 1234 1235 --env production          # multiple plates, production env
omero-screen 1234 --segmentation                 # segmentation only, no feature extraction
omero-screen 1234 --inference micronuclei.pth    # with classifier inference
omero-screen 1234 --cp4                          # use Cellpose 4 (cpsam) models
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

### Environment
```bash
uv sync --dev && source .venv/bin/activate
ENV=production omero-screen 1234   # select env inline
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
| GPU not detected | Run `omero_screen.torch.get_device()` in Python; check PyTorch+CUDA install |
| Flatfield correction slow | First run generates masks from 100 images — subsequent runs load cached masks |
| Missing cell line model | Add `{"MODEL_DICT": {"CELLLINE": "model_name"}}` to config JSON at `OMERO_SCREEN_CONFIG` |
| CellView import fails | CSV needs `plate_id`, `cell_line`, `condition` columns plus measurement columns |
| Logging missing in napari | Plugin mode writes to file — check `LOG_FILE_PATH` in `.env` |
| `--cp4` vs default | Default uses Cellpose 3 models from `MODEL_DICT`; `--cp4` uses Cellpose 4 (cpsam) for all cell lines |
