# omero-screen-images — headless galleries and well overviews

Use when a user (or an agent working from an analysis notebook) wants cell
galleries or whole-well images **without napari**: visual QC of classifier
calls, treatment effects, figure panels.

User guide: `packages/omero-screen-napari/user_guide/images_cli.qmd`.
Options: `omero-screen-images <command> --help` (always current).

## Commands

```bash
# Galleries: classifier class, seeded, JSON manifest on stdout
omero-screen-images --env production gallery 5108 --wells E2,G5 \
    --classifier-column classifier_nuclei4 --class micronuclei \
    --channels DAPI --grid 5x5 --seed 1 --out qc/mn --json

# Galleries from cells gated downstream (image_id, label[, timepoint])
omero-screen-images gallery 5108 --cells gate.csv --channels DAPI,EdU --json

# Whole-well overviews, same contrast across wells
omero-screen-images --env production well 5108 --wells B2,E2,B5,E5 --json

# Zoomed with nuclei outlines
omero-screen-images well 5108 --wells G5 --layers DAPI,nuclei_masks --zoom 8

# Plan file: plate_id,well,render  (render = well | gallery | gallery:<class>)
omero-screen-images --env production batch plan.csv --channels DAPI \
    --classifier-column classifier_nuclei4 --out qc/ --json
```

## What to know

- `--env` goes **before** the command (it is a group option).
- Zarr-cached plates read from the cache (fast, no download). Per-field plates
  download through the plate disk cache, one well at a time. A stitched plate
  without a zarr cache is refused.
- Galleries need the plate in CellView; overviews do not.
- Display limits are shared by every well of one call (pooled 0.1/99.9
  percentiles, zeros ignored). Pin them with `--limits CHANNEL=LO:HI` to
  compare across calls.
- Galleries keep the background by default; `--blank-background` blanks it
  (hides micronuclei).
- With `--json`, stdout is the manifest only — parse it. Key fields:
  - gallery: `wells.<W>.exported`, `n_available`, `n_in_gallery`, `file`,
    `intensities`, `seed`, `settings`
  - well: `wells.<W>.exported`, `file`, `region_yx`, `downsample`,
    `pixel_size_um`, `limits`, `limits_source`
  - batch: `runs[].{plate_id, render, out, manifest, written, error}`
- Exit status 1 when nothing was written; 2 for a usage error.

## Agent workflow: analysis gate → gallery

1. Export the plate from CellView (`cellview export` or the Python API) and
   gate the cells in the notebook.
2. Write the gate: `df.select("image_id", "label").write_csv("gate.csv")`.
3. `omero-screen-images gallery <plate> --cells gate.csv --channels ... --json`
4. Read the manifest: check `n_available` per well before interpreting a
   gallery (a gallery of 3 cells is not a representative sample).
