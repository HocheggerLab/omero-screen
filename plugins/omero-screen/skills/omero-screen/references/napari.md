# Napari Widgets — omero-screen-napari

Interactive Napari plugin for browsing OMERO data, reviewing segmentation, generating training data, and applying classifiers.

Package: `packages/omero-screen-napari/`

---

## Launch

```bash
source .venv/bin/activate
napari
```

Open the plugin: **Plugins → omero-screen-napari**

The plugin connects to OMERO using credentials from the active `.env` file. Ensure the correct environment is set before launching:

```bash
ENV=production napari
```

---

## Widgets Overview

| Widget | Purpose |
|---|---|
| **WellData** | Browse and display images from an OMERO plate well |
| **Gallery** | Display cell crop galleries for visual QC and review |
| **Training** | Manually label cells to generate classifier training data |
| **Setup Training** | Create a new classifier and configure crop settings |
| **Classifier Selector** | Select a trained classifier; view class distribution stats |
| **Session Manager** | Browse, load, and delete training sessions |
| **Direct Load** | Load crops from OMERO directly without a gallery pre-step |
| **Aligned Plate** | View aligned plate layout |

---

## Workflow: Browse Images

**WellData widget:**
1. Enter plate ID (OMERO plate object ID)
2. Enter well coordinates (e.g. `C3`, `B2`)
3. Click **Load Well**
4. Images display in the napari viewer with separate layers per channel
5. Navigate images with the widget controls or viewer timeline

**Display options:**
- Toggle channels on/off in napari layers panel
- Adjust LUT (lookup table) and contrast per channel
- Overlay segmentation masks: click **Show Masks** to load from OMERO

---

## Workflow: Review Segmentation Quality

1. Load a well via WellData widget
2. Click **Show Masks** to overlay nucleus and cell masks
3. Check:
   - Nuclei are correctly segmented (not merged, not split)
   - Cell borders are accurate
   - Border cells are excluded
4. If quality is poor: adjust Cellpose model in config and re-run pipeline

---

## Workflow: View Cell Gallery

**Gallery widget:**
1. Select plate and well
2. Set gallery grid size (rows × columns)
3. Click **Generate Gallery** — displays a grid of randomly selected cell crops
4. Gallery shows all channels as composite RGB

Use this for:
- Quick QC of segmentation quality across many cells
- Identifying potential training examples (phenotypes present in the data)

---

## Workflow: Generate Training Data

See `references/classifier-training.md` for the full end-to-end workflow. Summary:

1. **Setup Training widget** → create classifier, set classes and crop size
2. **Training widget** → select classifier, load image, label crops with number keys
3. **Session Manager** → track progress, resume sessions, manage storage

### Critical behaviours to know

**`no_background` (label isolation):** When enabled, only the target cell's label is shown in the crop — neighbouring cells are erased. Enabled by default. Disable for images where cell context matters.

**Channel mapping:** Crops use the channels configured in Setup Training. For 2-channel images, channels map to R+G (not R+G+B). If crops look yellow, check channel configuration.

**`RandomImageParser` classifier flag:**
- `classifier=True` → populates `selected_crops` (needed for saving)
- `classifier=False` → gallery display only, `selected_crops` is empty
- The training widget always uses `classifier=True`

---

## Workflow: Apply a Trained Classifier in Napari

**Classifier Selector widget:**
1. Dropdown lists all classifiers in the trainingdata DB
2. Shows class distribution stats (how many crops per class)
3. Select a classifier → applies to current well display
4. Cells coloured by predicted class in the viewer

This is for visual review; for bulk classification at scale use `omero-screen --inference model.pth`.

---

## Session Management

**Session Manager widget:**
- Lists all training sessions grouped by classifier
- Shows: plate ID, well, image, number of annotated cells, creation date
- **Load session** → resumes labelling from where you left off
- **Delete session** → removes session + NPY file (warns if other sessions share the file)

### Session file format
Training crops are saved as `.npy` files:
```
{plate_id}_{well}_{image_input}_{timepoint}.npy
```

Session metadata JSON is stored alongside and contains: channel configuration, crop size, `no_background` setting, class labels.

---

## Database CLI (trainingdata DB)

The napari plugin stores annotations in a SQLite database. Manage it via:

```bash
# List all classifiers
omero-train list classifiers

# List sessions for a classifier
omero-train list sessions --classifier micronuclei_v1

# Show class distribution
omero-train stats --classifier micronuclei_v1

# Export training data to NPY files for training
omero-train export --classifier micronuclei_v1 --output /path/to/output/

# Delete a classifier and all its sessions
omero-train delete --classifier old_classifier_v1
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Plugin not listed in Plugins menu | Ensure napari plugin is installed: `uv pip install -e packages/omero-screen-napari` |
| OMERO connection fails | Check `.env` credentials; verify OMERO server running |
| Crops appear yellow (2-channel) | Channel mapping issue — check Setup Training channel selection matches image channels |
| `no_background` not working | Known pattern — target cell label must be looked up at centroid; check `gallery_api.py` |
| Session won't load | `image_input` in filename may not match stored session ID — check `session_utils.py` |
| Keybinding already in use warning | Expected on widget re-instantiation — uses `overwrite=True`, warning is benign |
| Duplicate annotation count | `cell_index` collisions — should use sequential enumerate index, not `np.max(label_mask)` |
