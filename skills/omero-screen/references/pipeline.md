# Running the OMERO-Screen Analysis Pipeline

Docs: https://hocheggerlab.github.io/omero-screen/

## Prerequisites
- OMERO server running and credentials in `.env.{ENV}`
- Plate uploaded to OMERO with metadata (Excel attachment or key-value annotations)
- Cell line has a Cellpose model entry (see Model Configuration below)
- Environment activated: `source .venv/bin/activate`

---

## CLI Reference

```
omero-screen <plate_id> [plate_id2 ...] [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--env NAME` | `development` | Load `.env.NAME` config file |
| `--segmentation` | off | Skip feature extraction — only segment and upload masks |
| `--inference MODEL...` | none | One or more `.pth` classifier filenames |
| `--gallery N` | 10 | Gallery grid size N×N for inference output |
| `--batch N` | 16 | Batch size for classifier inference |
| `--cp4` | off | Use Cellpose 4 (cpsam) for all cell lines instead of MODEL_DICT entries |
| `--model NAME` | none | Override all segmentation models with one model (e.g. `cp4:cpsam`, `cp3:cyto3`) |
| `--benchmark` | off | Write per-image timing JSON to disk |

### Examples

```bash
# Standard run — development environment
omero-screen 1821

# Multiple plates at once
omero-screen 1821 1822 1823

# Production server
omero-screen 1821 --env production

# Segmentation only (inspect masks before full analysis)
omero-screen 1821 --segmentation

# Apply a pre-trained classifier
omero-screen 1821 --inference micronuclei_densenet.pth

# Multiple classifiers
omero-screen 1821 --inference micronuclei.pth mitotic_index.pth --gallery 15 --batch 32

# Use Cellpose 4 models (better for crowded fields)
omero-screen 1821 --cp4

# Override model explicitly (e.g. test a specific version)
omero-screen 1821 --model cp4:cpsam

# Benchmark timing (writes JSON report)
omero-screen 1821 --benchmark

# HPC (sbatch wrapper)
./sbatch-omero-screen.py --inference micronuclei_densenet -e omero-screen-infer 1821
```

---

## What Happens Step by Step

1. **Connect** to OMERO using credentials from `.env.{ENV}`
2. **Parse metadata** — Excel attachment takes priority over key-value annotations; Excel deleted after conversion
3. **Flatfield correction** — check dataset for cached masks; if absent, sample 100 images/channel, generate median masks, upload
4. **Well/image loop** (`loops.py`) — iterates wells then images; saves per-well CSV for resumability
5. **Segmentation** (`image_analysis.py`) — Cellpose nucleus model → cell model (cyto2/custom); cytoplasm = cell − nucleus; border cells filtered
6. **Feature extraction** — `skimage.measure.regionprops_table` per nucleus/cell/cytoplasm mask; all channels
7. **Cell cycle analysis** (`cellcycle_analysis.py`) — multi-nucleate aggregation → normalise to mode → assign G1/S/G2/Polyploid/SubG1
8. **QC** (`quality_control.py`) — metrics CSV + figure attached to plate
9. **Classification** (optional) — batch inference with PyTorch; class gallery PNGs attached
10. **Results upload** — final CSV attached to plate; per-well intermediates deleted

---

## Metadata Format

### Excel file (preferred)
Upload to the OMERO plate object. Required columns: well ID, cell line, condition. Optional: timepoint, antibody. After parsing, system converts to OMERO annotations and deletes the Excel.

### OMERO key-value annotations (fallback)
Plate-level channels: `{"DAPI": "0", "EdU": "1", "H3P": "2", "Tub": "3"}`
Well-level: `{"cell_line": "RPE", "condition": "control", "timepoint": "24h"}`

---

## Model Configuration

Models map cell line names to Cellpose model filenames. Default config in `src/omero_screen/__init__.py`. Override with a JSON file:

```bash
export OMERO_SCREEN_CONFIG=/path/to/my_config.json
```

```json
{
  "MODEL_DICT": {
    "RPE": "RPE-1_Tub_Hoechst",
    "HELA": "cp4:cpsam",
    "NEWCELL": "my_custom_model"
  }
}
```

Model name prefixes: `cp4:` = Cellpose 4, `cp3:` = Cellpose 3, no prefix = custom model file in Cellpose models dir.

Built-in Cellpose 4 model: `cp4:cpsam` (recommended for most cell lines)

---

## Additional CLI Tools

```bash
# Aggregate features across images for a plate
omero-screen-aggregate <plate_id>

# Generate segmentation sample images (QC check before full run)
seg-samples <plate_id> [--env NAME] [--n-wells N] [--n-images N]

# Test GPU availability
torch-test
```

---

## Cell Cycle Output

When an `EdU` (or equivalent S-phase marker) channel is present, the pipeline adds:
- `cell_cycle` column: `G1` | `S` | `G2` | `Polyploid` | `SubG1`
- `cell_cycle_detailed` column: adds `Mitotic` subdivision within G2

Phase thresholds (default, per normalised integrated DAPI):
- SubG1: DNA < 0.75
- G1: DNA < 1.5 AND EdU < threshold
- S: EdU ≥ threshold
- G2/M: DNA ≥ 1.5 AND EdU < threshold
- Polyploid: DNA > 2.5

H3P channel (if present): identifies mitotic cells within G2/M for `cell_cycle_detailed`.

---

## Results Location

All results are attached back to the OMERO plate object:
- `final_data_cc.csv` — single-cell measurements with cell cycle annotations
- `quality_ctr.csv` + `quality_ctr.png` — QC metrics
- Segmentation masks dataset — multi-channel TIFFs named `{image_id}_segmentation`
- Classifier gallery PNGs — one per predicted class (if inference run)

Local intermediate files in working directory are cleaned up after plate completion.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: cell_line` | Missing metadata | Check plate has Excel attachment or well annotations |
| `Model not found: CELLLINE` | Cell line not in MODEL_DICT | Add entry to config JSON; or use `--cp4` flag |
| Segmentation produces empty masks | Wrong channel index | Verify channel metadata (`{"DAPI": "0"}` etc.) |
| Pipeline restarts from scratch | No cached per-well CSV | Normal on first run; subsequent runs resume from last well |
| OOM during Cellpose | Image too large or batch too big | Reduce batch size; check GPU memory with `torch-test` |
