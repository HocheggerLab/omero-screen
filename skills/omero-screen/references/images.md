# omero-screen-images — images that support a quantitative finding

Use when a user (or an agent working from an analysis notebook) wants cell
galleries or whole-well images **without napari**: visual QC of classifier
calls, treatment effects, figure panels.

User guide: `packages/omero-screen-napari/user_guide/images_cli.qmd`.
Options: `omero-screen-images <command> --help` (always current).

## Purpose: show the finding, not the plate

Images are the **qualitative support for a quantitative result**. The numbers
come first (CellView export → notebook analysis). The images then let a reader
*see* what the numbers say — and let you check that the numbers are not an
artefact (segmentation errors, classifier mistakes, empty or out-of-focus
wells, edge effects).

So be **selective**. Do not render every well, every condition and every
channel. For each finding, decide:

| Decide | Guided by the quantitative result |
|---|---|
| **Which wells** | The contrast the finding rests on: typically one control and one or two treated wells (the strongest effect, or a dose series' ends). One representative replicate, not all of them. |
| **Which channels** | The channel(s) the measurement is about, plus at most one for context (e.g. DAPI for nuclei). Not every channel by default. |
| **Which output** | A *population* change (fewer cells, altered density, morphology across the well) → `well` overview. A *per-cell* phenotype (micronuclei, mitotic figures, a marker's localisation) → `gallery` of the relevant class or gate. |
| **Which cells** | The class or gate the number counts: `--class` for a classifier call, `--cells` for a gate defined in the notebook (e.g. EdU− 2N). Show the cells that carry the effect, and the matching control cells. |
| **How close** | `--zoom 1` for density or plate-level effects; zoom in (4–8x) when the phenotype is only visible at cellular resolution. Add `nuclei_masks` / `cell_masks` when the claim depends on segmentation. |

A good set of images for one finding is usually 2–6 files, chosen so that each
one makes a point. State the point next to each image.

### Examples

- *"C604 raises micronuclei from 8% to 40% (control E2 vs 48 h G5)."*
  → `gallery` of `--class micronuclei` in G5 and of `--class normal` in E2,
  DAPI only, the same `--limits` for both; optionally one zoomed `well` view of
  G5 with `DAPI,nuclei_masks` to show that the calls follow real micronuclei.
- *"Cell number drops 60% at the top dose."* → `well` overviews of the control
  and the top-dose well, DAPI (+ Tub), `--zoom 1`, the same limits.
- *"EdU− 2N cells accumulate after PALB."* → gate in the notebook, write
  `gate.csv`, `gallery --cells gate.csv --channels DAPI,EdU` for treated wells,
  next to an ungated control gallery.
- *"A well is an outlier."* → one `well` overview at `--zoom 1` first, to see
  whether it is biology or a technical failure (empty, debris, focus), before
  interpreting its numbers.

### Rules that keep images honest

- **Same limits for compared images.** Wells in one call share limits. Across
  calls (e.g. a control gallery and a treated gallery), pin them with
  `--limits CHANNEL=LO:HI`, taken from the first call's manifest.
- **Keep the background** (the default). `--blank-background` hides objects
  outside the mask and makes crops look processed.
- **Fixed seed**, so the gallery can be regenerated. Never re-roll the seed to
  find a prettier draw; if the draw looks unrepresentative, check
  `n_available` and say so.
- **Report the counts** behind a gallery (`n_available`, `n_in_gallery`): a
  gallery drawn from 4 cells is not a representative sample.
- Look at each image before presenting it. If an image contradicts the number
  (e.g. "micronuclei" calls that are debris), report that — it is a finding.

## Navigating the CLI

The CLI is designed to be explored from its own output. Work in this order:

1. **Discover the commands and options.** `omero-screen-images --help`, then
   `omero-screen-images <command> --help`. `--env <name>` goes **before** the
   command (it is a group option).
2. **Discover the plate.** The quickest inventory is a cheap overview:

   ```bash
   omero-screen-images --env production well 5108 --wells All \
       --size 400 --out /tmp/inv --json
   ```

   The manifest lists every available well, its caption (cell line, condition
   and all other well metadata), the channel colours and the pooled limits.
   For channel names, classifier columns and class values, use the CellView
   export in the notebook — or let the CLI tell you: an unknown name is a usage
   error (exit 2) that **lists the valid choices** (plate channels for
   `--channels` / `--layers` / `--limits`, classifier columns for
   `--classifier-column`, available wells for `--wells`).
3. **Render a small selection**, with `--json`, chosen by the table above.
4. **Read the manifest** (stdout with `--json`; also written next to the
   images):
   - gallery (`gallery_export.json`): `wells.<W>.exported`, `file`,
     `n_available`, `n_in_gallery`, `reason` (when not exported);
     `intensities` (limits by channel index), `seed`, `settings`.
   - well (`well_overview.json`): `wells.<W>.exported`, `file`, `caption`,
     `region_yx`, `downsample`, `pixel_size_um`, `missing_masks`; `limits`
     and `limits_source` (by channel name), `colours`.
   - batch (`batch.json`): `runs[].{plate_id, render, wells, out, manifest,
     written, error}`.
5. **Look at the images**, then refine: zoom in (`--zoom`, `--center Y,X` as
   fractions of the well) where the phenotype is, change the class or gate,
   or pin limits for a comparison.
6. **Batch the final set** with a plan file when it spans several plates or
   outputs, so it can be re-run in one command.

Exit status: 0 = at least one image written; 1 = nothing written (see each
well's `reason` / the run's `error`); 2 = usage error (bad option or name).

## Command reference by example

```bash
# Gallery of one classifier class, seeded
omero-screen-images --env production gallery 5108 --wells G5 \
    --classifier-column classifier_nuclei4 --class micronuclei \
    --channels DAPI --grid 5x5 --seed 1 --out qc/mn --json

# Matching control gallery, same limits as the first call's manifest
omero-screen-images --env production gallery 5108 --wells E2 \
    --classifier-column classifier_nuclei4 --class normal \
    --channels DAPI --grid 5x5 --seed 1 --limits DAPI=422:17676 --out qc/ctrl --json

# Gallery from a notebook gate (columns image_id, label[, timepoint])
omero-screen-images gallery 5108 --cells gate.csv --channels DAPI,EdU --json

# Whole-well overviews, shared limits
omero-screen-images --env production well 5108 --wells E2,G5 \
    --layers DAPI,Tub --json

# Close-up with nuclei outlines
omero-screen-images well 5108 --wells G5 --layers DAPI,nuclei_masks \
    --zoom 8 --center 0.4,0.6 --json

# The final set as a plan (plate_id,well,render; render = well | gallery |
# gallery:<class>)
omero-screen-images --env production batch plan.csv --channels DAPI \
    --classifier-column classifier_nuclei4 --out qc/ --json
```

## Data sources (what can fail)

- Zarr-cached plates read from the cache (fast, no download). Per-field plates
  download through the plate disk cache, one well at a time — slower the first
  time. A stitched plate without a zarr cache is refused.
- Galleries need the plate in CellView; overviews do not.
- An unprocessed or unknown plate is a clean error naming the plate.
