# Running the OMERO-Screen pipeline

Docs: https://hocheggerlab.github.io/omero-screen/ (Analyse a plate, Stitching,
Running on a SLURM cluster).

## Before a run

- `omero-screen doctor` passes (login, connection; on a laptop, the VPN).
- The plate is in OMERO with its metadata: channels on the plate, and cell
  line + condition per well (an Excel attachment or key-value annotations).
- Ask the user before running: a run writes masks and results to OMERO.

## The command

```bash
omero-screen <plate_id> [<plate_id> ...] [OPTIONS]
omero-screen --help          # the current options; check here before using one
```

The CLI is Click: write option names in full (`--segmentation`, never `--seg`).
Leave out `--env` unless the user has a developer `.env.<name>` file; an
installed user's settings come from `~/.config/omero-screen/config.toml`.

The options you will use most:

| Option | What it does |
|---|---|
| `--segmentation` | Segment and upload masks only, no measurements |
| `--delete` | Delete the plate's stored masks and segment again. Without it, a re-run reuses the masks and only re-measures. Use after changing segmentation settings, or when stored masks are wrong |
| `--inference NAME [NAME ...]` | Classify cells with classifiers published to OMERO (`omero-screen models publish`), by name, without `.pt`; `--gallery N` sets the example gallery size |
| `--cp4` / `--model NAME` | Cellpose 4 (cpsam) for every role / one model for every role (`cp4:cpsam`, `cp3:cyto3`, or a custom model name) |
| `--stitch` | Stitch all fields of a well into one canvas and segment that (timelapse, or objects crossing field borders) |
| `--track [MODEL]` | Track nuclei over time with Trackastra (needs `--stitch` and a timelapse); `--track-correct` repairs the first pass. Memory is managed automatically; see `omero-screen --help` for `--track-window`, `--track-device` |
| `--benchmark` | Write per-image timings as JSON |

```bash
omero-screen 1821                                   # standard run
omero-screen 1821 1822 1823                         # several plates
omero-screen 1821 --segmentation                    # check masks first
omero-screen 1821 --delete                          # segment again from scratch
omero-screen 1821 --inference micronuclei           # with a published classifier
omero-screen 5054 --stitch --track --track-correct  # timelapse, tracked
```

## On an HPC cluster

Never run the pipeline on a login node. Submit it as a job from the user's
work space (not from inside a source checkout):

```bash
omero-screen-sbatch <plate_id> [<plate_id> ...] [pipeline options]
omero-screen-sbatch 1821 --no-submit         # write the job script only, to check it
omero-screen-sbatch 1821 1822 --stitch --track
squeue -u $USER
```

One job per plate by default. Job options: `--hours N` (24), `-m/--memory N`
(32 GB), `--gpu/--no-gpu`, `--class` (partition, `gpu`), `-u/--username`
(also the e-mail address `<username>@sussex.ac.uk`). The job checks the GPU,
runs the pipeline, logs to `os<plate>.<pid>.<jobid>.out` in the current
folder and e-mails at the end. `omero-screen-sbatch -h` lists everything. The
templates are set up for the Sussex HPC.

## What a run does

1. Connects to OMERO and reads the plate metadata (an Excel attachment is
   converted to annotations).
2. Flatfield correction: reuses the masks stored with the plate, or builds
   them from a sample of images and uploads them.
3. For each well and image (or each stitched well): segments nuclei and cells
   with Cellpose (cytoplasm = cell − nucleus), measures every channel per
   compartment.
4. With an EdU channel: normalises DNA content and EdU, assigns cell-cycle
   phases.
5. Quality control, then optional classification (`--inference`), then
   attaches the results to the plate.

Results attached to the plate in OMERO:
- `final_data_cc.csv` (or `final_data.csv` without cell cycle): one row per cell
- quality-control metrics and a QC figure
- the segmentation masks (a dataset of mask images)
- classifier galleries, one per class (with `--inference`)
- a provenance map annotation (namespace `omero-screen/provenance`): version,
  options, device and models used

Import the results into CellView for analysis: `cellview import plate <id>`.

## Segmentation models

With nothing configured: Cellpose 4 (`cp4:cpsam`) on an NVIDIA GPU, stock
Cellpose 3 (`nuclei`, `cyto3`) on a Mac or CPU. To use the lab's models:
`omero-screen models pull hocheggerlab`, then in the user config

```toml
[segmentation]
model_set = "hocheggerlab"

[segmentation.models]        # optional: per cell line, over the model set
RPE = "RPE-1_Tub_Hoechst"
```

Model names: `cp4:` = Cellpose 4, `cp3:` = Cellpose 3, no prefix = a custom
model file in `~/.cellpose/models`.

## Stitching

Tiles are placed by grid arithmetic, so the overlap and offset must match the
objective. They are chosen automatically from the plate's pixel size: 10x
(1.2 µm) and 20x (0.6 µm) are calibrated; an unknown pixel size falls back to
the 10x values with a warning in the log. `--stitch-config FILE` (JSON) fixes
one setting instead. Duplicated objects along tile seams mean the calibration
is wrong for this objective: tell the user, don't guess new values.

## Cell-cycle phases

`cell_cycle`: `Sub-G1`, `G1`, `S`, `G2/M`, `Polyploid` (`Unassigned` if outside
every gate). With a mitosis marker (H3P), G2 and `M` are separate.
`cell_cycle_detailed` also splits `Early S`/`Late S` and polyploid
(non-)replicating. Gates use `integrated_int_DAPI_norm` (G1 peak = 2) and
`intensity_mean_EdU_nucleus_norm`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Can't connect / login incomplete | `omero-screen doctor`; VPN; `omero-screen setup` |
| `KeyError` on `cell_line` or `condition` | The wells lack metadata: check the Excel attachment or well annotations |
| Empty or wrong masks | Check the channel annotation on the plate (`DAPI` → index); rerun with `--delete` |
| A re-run didn't change the segmentation | Masks are reused; add `--delete` |
| Out of GPU memory | Tracking: `--track-batch-size` lower, `--track-window N`, or `--track-device cpu`; segmentation: run with fewer plates per job |
| No cell-cycle columns | The plate has no EdU channel (or it is named differently in the channel annotation) |
