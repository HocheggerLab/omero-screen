# Tracks Widget — Viewing and Exporting Cell Lineages

## What this widget does

The Tracks Widget visualises the temporal tracks produced by the omero-screen
pipeline's `--track` option (see {ref}`temporal-tracking`). Once a live-cell
plate has been tracked, every nucleus carries a stable `track_id` across time
and a `parent_track_id` linking daughters to their mother at divisions. This
widget overlays those tracks on the well image, lets you inspect a single
lineage, export one track's measurements for analysis, and hand a well off to
[Mastodon](https://mastodon.readthedocs.io/) for manual curation.

It does **not** generate tracks — that happens in the pipeline. Here you view,
export, and protect them.

## Before you start

* The plate must have been run with `omero-screen <id> --stitch --track`, and
  imported into CellView (`cellview import-plate <id>`), so the `track_id`
  columns exist.
* Load the well first with the **Welldata Widget** — the Tracks Widget reads
  the well that is currently loaded in the viewer.

## Opening the widget

In Napari, go to **Plugins → Omero Screen Napari → Tracks Widget**. The panel
stacks several sections, each with its own button:

1. **Load tracks**
2. **Export track CSV**
3. **Export well for Mastodon**
4. **Pin plate**
5. **Unpin plate**

If the lower buttons are cut off, drag the dock taller or undock it.

---

## Load tracks

Fills in (or leave **Well** blank to use the currently-loaded well), then click
**Load tracks**.

| Field | What to enter |
|-------|---------------|
| **Well** | Well position, e.g. `B2`. Blank = the loaded well. |
| **Color by** | `track_id` (rainbow) or `cell_cycle` (if cell-cycle analysis ran). |
| **Tail length** | How many past frames trail behind each track head (default 10). |
| **Show divisions (lineage)** | Off by default. When on, the division lineage graph is passed to the Tracks layer so mother→daughter links are drawn. Building the graph for thousands of divisions can briefly freeze the UI, so it is opt-in. |

A napari **Tracks** layer appears, overlaid on the nuclei. The track positions
are scaled to match the image, so they sit directly on the cells. Scrub the
time slider to watch tracks move; a division shows as one track splitting into
two.

```{note}
If you see *"No track data in the loaded plate"*, the plate was not tracked or
not yet imported into CellView. Re-run the pipeline with `--track` (and
`--stitch`), then `cellview import-plate`.
```

## Inspecting a single lineage (napari-arboretum)

For the lineage *tree* of a single track, use the
[napari-arboretum](https://www.napari-hub.org/plugins/napari-arboretum) plugin
(installed as a dependency), which is purpose-built for this:

1. Open **Plugins → arboretum → Arboretum**.
2. Select the **tracks** layer in the layer list.
3. **Double-click a track** in the viewer — Arboretum draws that founder's
   lineage tree, with divisions as branch points.

We delegate the tree view to Arboretum rather than reinventing it — it is made
by the same lab that contributed napari's Tracks layer.

## Export track CSV

To pull one track's full measurement time-course out for downstream analysis
(e.g. picking the longest, cleanest tracks):

1. Enter the **track id** (read it off the Tracks layer or the Arboretum tree).
2. Optionally set the **Well**.
3. Click **Export track CSV** and choose a save location.

The CSV contains every measurement column for that single track across time.

## Export well for Mastodon

Prepares a well for manual curation in Mastodon:

1. Optionally set the **Well**.
2. Click **Export well for Mastodon**.

This writes `tracks.csv` next to the cached well image and a `README.txt` (in
`~/mastodon_exports/plate_<id>_<well>/`) with the exact paths and click-by-click
import steps. No image is copied — Mastodon opens the cached image in place. See
the {doc}`../tracking` overview for the full Mastodon walkthrough.

```{note}
The cache auto-writes `tracks.csv` for every tracked well when the OME-Zarr is
built, so a well is often already Mastodon-ready without this step. The button
also (re)writes the README and refreshes the CSV.
```

## Pin / Unpin plate

The OME-Zarr cache is size-bounded and evicts least-recently-used plates. If you
will curate a well in Mastodon over time (a separate Fiji session that can span
days), protect it first:

* **Pin plate** — exempt the loaded plate from eviction. The pin persists across
  napari restarts.
* **Unpin plate** — release it when you are done curating, so it can be
  reclaimed. The button also reports which plates are still pinned.

Pinning is deliberately a manual choice: caching a plate does not mean you are
curating it, and pinning everything would defeat the cache.
