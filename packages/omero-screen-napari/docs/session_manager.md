# Session Manager & Direct Load Dialog

## Session Manager

### What it does

The Session Manager is a dialog that gives you a complete overview of all annotation sessions belonging to a classifier. You can see how many cells have been labelled in each session, check whether the underlying data files are intact, load a session to continue annotating it, delete sessions you no longer need, and add fresh data from OMERO.

### Opening the Session Manager

In the Training Widget, click **Manage Sessions** on the classifier info panel. The button appears once a classifier with at least one session is selected.

### The Sessions table

Each row in the table represents one annotation session — one batch of crops from a specific plate, well, and set of images.

| Column | What it shows |
|--------|--------------|
| **#** | Row number |
| **Plate ID** | The OMERO plate ID the crops came from |
| **Well** | The well position (e.g. `A1`) |
| **Images** | Which images were used (e.g. `0`, `1, 2`, `All`) |
| **Timepoint** | The timepoint index |
| **Cell Cycle** | The cell cycle filter active when this session was created (e.g. `All`, `G1`, `G2/M`) |
| **Cells** | Total number of crops (annotated + unassigned) |
| **Class Distribution** | How many crops are assigned to each class |
| **Status** | ✓ Valid (green) = data file exists and is readable; ✗ Error (red) = file missing or unreadable |
| **Actions** | **Load** and **Delete** buttons |

### Summary header

Above the table, a summary line shows:
- Total number of sessions
- Total cells annotated across all sessions
- Number of distinct plates contributing data
- When the database was last updated

### Loading a session

Click **Load** on any row to restore that session into the Training Widget. This:
1. Reads the NPY file from disk (image crops and masks).
2. Restores the display settings (contour, background removal, channel mapping) from the classifier's saved metadata.
3. Restores all class assignments you previously made.
4. Displays the first crop in the viewer.

You can then continue annotating and click **Save training data** in the Training Widget when finished.

> If the Status column shows ✗ Error, the data file is missing. You can still delete the database record using the Delete button, but you cannot load the session.

### Deleting a session

Click **Delete** on a row. A confirmation dialog asks you to confirm. Once confirmed:
- The database records (session and all its annotations) are removed.
- The NPY data file is deleted from disk, **unless** another session is using the same file.
- If this was the last session for the classifier, you are asked whether to delete the classifier itself (including its folder and `metadata.json`).

> Deletion is permanent. Use the **omero-train export** command to back up your data before deleting anything important.

### Action buttons

| Button | What it does |
|--------|-------------|
| **Add New Data** | Opens the Direct Load Dialog to fetch crops from a new plate or well |
| **Refresh** | Re-queries the database and updates the table (useful if another user added data) |
| **Close** | Closes the Session Manager without making any changes |

---

## Direct Load Dialog

### What it does

The Direct Load Dialog lets you fetch cell crops directly from OMERO and add them as a new session for the current classifier — without having to go back through the Welldata Widget and Gallery Widget. It uses the classifier's saved settings (crop size, channels, segmentation type, cell cycle filter) automatically, so all sessions are consistent.

This is the fastest way to add more training data from a different plate or well.

### Opening the Direct Load Dialog

In the Session Manager, click **Add New Data**.

### What to fill in

| Field | What to enter | Example |
|-------|--------------|---------|
| **Plate ID** | The OMERO plate ID to load from | `3869` |
| **Well ID** | The well position. A dropdown offers all 96-well positions (A1–H12); type directly for larger formats | `B4` |
| **Images** | Which images to use. Accepts: `All`, a single number, a comma-separated list, or a range | `All`, `0`, `0, 1`, `3-5` |
| **Timepoint** | Timepoint index | `0` |

### The classifier settings preview

The dialog shows the settings that will be applied, read from the classifier's `metadata.json`:
- Crop size
- Channel names and RGB mapping
- Segmentation type (nucleus or cell)
- Cell cycle filter

These are applied automatically. You do not need to re-enter them.

### Loading the data

1. Fill in the Plate ID, Well, Images, and Timepoint fields.
2. Click **Validate** to check the inputs and see a formatted preview.
3. Click **Load Data** to:
   - Connect to OMERO and fetch the plate.
   - Find the well and retrieve the images.
   - Apply flatfield correction (if available).
   - Look up cell centroids from the CellView database.
   - Extract crops using the classifier's crop size, centred on each centroid.
   - Apply the cell cycle filter to select only cells in the relevant phase.
   - Display the first crop in the Training Widget.
4. The dialog closes automatically on success.

### What happens behind the scenes

- Crops are normalised to a consistent brightness range using the 99.9th percentile of each channel — the same approach used by the Gallery Widget, so new sessions look the same as existing ones.
- Only the target cell's segmentation mask is retained per crop (other cells in the field of view are removed), matching the Gallery Widget's behaviour.
- The session is not saved to disk automatically. Annotate in the Training Widget and click **Save training data** when finished.

### CLI equivalent

You cannot load crops from OMERO via the CLI, but you can inspect the sessions that result:

```bash
omero-train stats mitosis-rpe     # see all sessions including newly added ones
```

---

## Tips

- **Add data from multiple plates and wells.** A classifier trained on a single plate will not generalise well to other experiments. Aim for at least 3–5 different wells from different repeats.
- **Use the cell cycle filter consistently.** The filter is stored per session and shown in the Cell Cycle column. If some sessions used `All` and others used `G2/M`, the training data will be inconsistent. Create separate classifiers for different filter settings.
- **Check the Status column regularly.** Red entries mean the data file has moved or been deleted. Clean up orphaned records with the Delete button to keep the database tidy.
- **Back up before deleting.** Run `omero-train export <classifier>` to save a CSV of all annotations before doing any deletions.
