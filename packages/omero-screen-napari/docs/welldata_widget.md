# Welldata Widget — Loading Images and Stitching

## What this widget does

The Welldata Widget is your starting point. It connects to your OMERO server, downloads images from a specific well on a plate, and displays them in the Napari viewer complete with segmentation masks and experimental metadata. It also caches plate data to your local disk so that future loads are nearly instant.

If your microscope acquired images across multiple overlapping positions (tiled acquisitions), the widget can stitch these positions together into a single, seamless composite image.

## Opening the widget

In Napari, go to **Plugins → Omero Screen Napari → Welldata Widget**. The widget opens as a panel on the right side of the viewer.

---

## Step 1 — Load a well

### What you need to fill in

| Field | What to enter | Example |
|-------|--------------|---------|
| **Plate ID** | The numeric ID of your plate in OMERO | `3869` |
| **Well Position** | One or more well positions, comma-separated | `A1` or `A1, B3, C5` |
| **Images** | Which images to load from the well | `All`, `0`, `0, 1, 2`, or a range `3-5` |
| **Time** | Timepoint index, or `All` for a time series | `0` |
| **Cache** | Tick this to download the full plate to disk in the background | ☑ |

Click **Enter** to start loading.

### What appears in the viewer

Once loading completes:

- **Fluorescence channels** are added as separate image layers with automatic colormaps:
  - DAPI / Hoechst → blue
  - Tubulin → green
  - EdU → red
  - Additional channels → magenta, cyan, yellow
- **Segmentation masks** (nucleus outlines, cell outlines) are added as label layers if they were generated during the analysis pipeline.
- A **metadata panel** appears showing the experimental annotations for the well (cell line, condition, timepoint, etc.). When you navigate between images or timepoints, this panel updates automatically.

### Loading multiple wells

Separate well positions with commas: `A1, A2, A3`. All wells are loaded in sequence and stacked in the viewer.

---

## Step 2 — Browse the Plate Info Dialog

Instead of typing well positions manually, you can open the **Plate Info Dialog** by clicking on the plate dropdown at the top of the widget.

The dialog shows a table of every well in the plate with:

- Well position, cell line, condition, timepoint
- Number of images per well
- Whether segmentation masks are available
- Real-time cache download progress

You can:
- **Double-click any row** to load that well immediately.
- **Select multiple rows** using the checkboxes and load them all at once.
- **Sort and filter** the table to find wells quickly.

---

## Step 3 — Cache the plate for faster access

Downloading images from OMERO every time is slow. The **Cache** feature downloads all wells in a plate to your local disk once, so subsequent loads are instant.

### How to cache

1. Tick the **Cache** checkbox before clicking Enter, **or**
2. Open the **Cached Plates panel** at the top of the widget and click the **Cache** button next to the plate you want.

### The Cached Plates panel

| Control | What it does |
|---------|-------------|
| **Plate dropdown** | Shows all locally cached plates with their status |
| **Cache Size** | Total disk space used by the cache |
| **Cache button** | Downloads the full plate in the background |
| **Delete button** | Removes the cached data for a plate to free disk space |
| **Refresh button** | Rescans the cache folder (useful if you cached from another session) |

The cache persists between Napari sessions — you do not need to re-download when you restart.

> **Tip:** For large plates (>20 wells), start the cache download while you work on the first few wells. By the time you need the later wells, they will already be on disk.

---

## Step 4 — Stitch tiled images (multi-position acquisitions)

Some microscope setups acquire several overlapping fields of view per well, which need to be joined together to see the full well. The stitching panel handles this.

### When you need stitching

If your well contains images named in a way that suggests a grid (e.g. position 1, 2, 3 … 25 for a 5×5 grid), or if the metadata indicates a tiled acquisition, use stitching.

### Stitching parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| **Rotation** | 0.15 | Angle correction in degrees to account for slight camera rotation between positions |
| **Precise Rotation** | off | Uses sub-pixel rotation for higher accuracy (slower) |
| **Overlap X / Y** | 7 % | How much adjacent tiles overlap. Match this to your microscope's acquisition settings |
| **Edge** | 7 px | Pixels to trim from each tile edge to remove illumination artefacts |
| **Mode** | reflect | How to fill any gaps at the image boundary |

After loading your well, fill in the stitching parameters and click **Enter** in the stitching panel. The stitched composite appears as a new layer in the viewer.

> **Tip:** If the stitched image looks misaligned, try adjusting the Overlap X/Y values by 1–2 % in each direction. Most Operetta/Opera systems use 7–10 % overlap.

---

## Tips and common questions

**Q: The images look very dim.**
A: Use napari's contrast controls (the coloured bar on each layer) to adjust brightness. The images are stored at full bit depth — napari's auto-scaling sometimes sets a conservative range.

**Q: No segmentation masks appear.**
A: Masks are only shown if your plate was processed through the omero-screen analysis pipeline. If you only have raw images, the masks will not be available.

**Q: I see "Plate not found" error.**
A: Double-check the Plate ID in OMERO.web. The ID is the number shown in the plate URL or in the left-hand tree.

**Q: Can I load images without an OMERO connection?**
A: Yes, if the plate is already cached locally. Cached plates load without any server connection.
