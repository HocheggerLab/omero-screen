# Session Manager & Direct Load Dialog

## Session Manager

### Overview

The Session Manager dialog provides a complete view of all annotation sessions for
a classifier. It shows per-session statistics, validates data integrity, and
supports loading, deleting, and adding sessions.

```{video} _static/session_manager_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing session table, loading a session, deleting a
> session, and adding new data via the Direct Load Dialog.

### Accessing the Session Manager

Open it from the Training Widget by clicking **Manage Sessions** on the classifier
info panel. The button is available whenever a classifier with metadata or existing
sessions is selected.

### UI Elements

#### Summary Header

Displays aggregated statistics for the selected classifier:

- Total number of sessions
- Total cells annotated
- Number of distinct plates
- Last updated timestamp

#### Sessions Table

| Column | Description |
|--------|-------------|
| **#** | Session index |
| **Plate ID** | OMERO plate the crops were extracted from |
| **Well** | Well position |
| **Images** | Image input string (e.g. `0`, `1, 2`, `All`) |
| **Timepoint** | Timepoint index |
| **Cells** | Number of annotated crops in this session |
| **Class Distribution** | Per-class annotation counts |
| **Status** | Data integrity: ✓ Valid (green) or ✗ Error (red) |
| **Actions** | **Load** and **Delete** buttons |

#### Action Buttons

| Button | Description |
|--------|-------------|
| **Add New Data** | Opens the Direct Load Dialog to fetch fresh crops from OMERO |
| **Refresh** | Re-query the database and update the table |
| **Close** | Close the dialog |

### Workflow

1. **Review sessions**: The table shows all sessions with their annotation counts
   and validation status.
2. **Load a session**: Click the **Load** button on any row. The NPY file and
   annotations are loaded into the Training Widget for continued annotation.
3. **Delete a session**: Click **Delete**. A confirmation dialog appears. If the
   NPY file is not shared with other sessions, it is removed from disk. If it is
   the last session, you are prompted to delete the entire classifier.
4. **Add new data**: Click **Add New Data** to open the Direct Load Dialog and
   fetch crops from a different plate or well.

### Data Integrity

The **Status** column validates each session:

- **✓ Valid**: The NPY file exists, is readable, and matches the expected shape.
- **✗ Error**: The NPY file is missing or corrupted. The session can still be
  deleted to clean up orphaned database records.

---

## Direct Load Dialog

### Overview

The Direct Load Dialog allows you to load fresh cell crops from OMERO directly into
the Training Widget, without going through the Welldata → Gallery workflow. It
applies the classifier's saved parameters (crop size, channels, segmentation type)
automatically.

### UI Elements

| Control | Description |
|---------|-------------|
| **Plate ID** | OMERO plate identifier (spin box, range 1–999999) |
| **Well ID** | Well position (editable dropdown, pre-populated with 96-well positions A1–H12) |
| **Images** | Image selection (`All`, `0`, `0,1,2`, or ranges like `3-5`) |
| **Timepoint** | Timepoint index (spin box, range 0–999) |
| **Validate** | Check inputs and show a formatted preview |
| **Load Data** | Connect to OMERO, extract crops, and load into the Training Widget |
| **Cancel** | Close without loading |

### Classifier Metadata Preview

The dialog displays the classifier's saved parameters:

- Crop size
- Channel names and mapping
- Segmentation type (nucleus / cell)
- Cell cycle filter

These are read from `metadata.json` and applied automatically during crop
extraction to ensure consistency across sessions.

### Workflow

1. Enter the **Plate ID** and **Well ID** for the new data source.
2. Optionally adjust **Images** and **Timepoint** filters.
3. Click **Validate** to preview the input parameters.
4. Click **Load Data** to:
   - Connect to OMERO and fetch the plate.
   - Parse well coordinates and retrieve images.
   - Extract crops using the classifier's saved parameters.
   - Load crops into the Training Widget for annotation.
5. The dialog closes on success and the first crop appears in the viewer.

### Tips

- The well dropdown includes all 96-well positions (A1–H12) for quick selection,
  but you can type any position for larger plate formats.
- Image ranges support dash notation: `3-5` expands to images 3, 4, and 5.
- If the plate is already cached locally, loading is significantly faster.
- The classifier metadata ensures that all sessions use identical crop parameters,
  making the training data consistent for model training.
