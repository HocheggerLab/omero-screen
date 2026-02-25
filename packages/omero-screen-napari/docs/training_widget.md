# Training Widget

## Overview

The Training Widget is the core annotation interface for building cell classification
training datasets. It lets you navigate through cell crops one at a time, assign
class labels, and save annotated data to disk and database.

```{video} _static/training_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing classifier selection, crop navigation with
> Q/W keys, class assignment, and saving a training session.

## UI Elements

### Classifier Selector

| Control | Description |
|---------|-------------|
| **Classifier dropdown** | Select from classifiers registered in the training database |
| **Info panel** | Displays session count, total annotations, class distribution |
| **Manage Sessions** | Opens the Session Manager dialog |

### Image Navigation

| Control | Description |
|---------|-------------|
| **Previous Image** | Navigate to the previous crop (also: **Q** key) |
| **Next Image** | Navigate to the next crop (also: **W** key) |
| **Class choice** | Radio buttons for assigning a class label to the current crop |

### Data Management

| Control | Description |
|---------|-------------|
| **Save training data** | Persist all crops and annotations to NPY file + database |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Q** | Previous image |
| **W** | Next image |

These shortcuts work when the Napari viewer has focus. They allow rapid navigation
without moving the mouse away from the class radio buttons.

## Workflow

### Starting a New Annotation Session

1. **Select a classifier** from the dropdown (create one first with the
   [Setup Training Widget](setup_training_widget.md) if needed).
2. **Load data** using one of two methods:
   - **From Session Manager**: Click *Manage Sessions* → *Add New Data* to open
     the Direct Load Dialog (see [Session Manager](session_manager.md)) and fetch
     fresh crops from OMERO.
   - **From Gallery**: Generate a gallery in the Gallery Widget, then open the
     Training Widget — the crops transfer automatically.
3. The **first crop** appears in the viewer with the mask overlay.
4. **Assign a class** using the radio buttons (e.g. "healthy", "apoptotic").
5. Press **W** to advance to the next crop. Press **Q** to go back.
6. Continue until all crops are labelled (or as many as desired).
7. Click **Save training data** to persist the session.

### Resuming a Previous Session

1. Select the classifier from the dropdown.
2. Click *Manage Sessions* → select a session → click **Load**.
3. All crops and their existing annotations are restored.
4. Continue annotating from where you left off.
5. Click **Save training data** to update the session.

## Data Storage

Each training session produces:

- **NPY file**: `~/omeroscreen_trainingdata/<classifier>/<plate>_<well>_<images>_<timepoint>.npy`
  containing the image crops and label masks as a numpy array.
- **Database records**: Session metadata and per-crop annotations stored in the
  training SQLite database.

## Tips

- Annotations are stored in memory until you click **Save**. Save frequently to
  avoid losing work.
- The class distribution shown in the info panel updates after each save.
- If you load new data (different plate/well), the widget automatically creates a
  fresh session — previous data is not overwritten.
- You can have multiple sessions per classifier, each from different plates or wells.
  The Session Manager shows all sessions at a glance.
