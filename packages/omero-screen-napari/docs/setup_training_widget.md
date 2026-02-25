# Setup Training Widget

## Overview

The Setup Training Widget is used to create new classifier projects. You define the
class labels, save classifier metadata, and automatically register the first
annotation session from the current gallery.

```{video} _static/setup_training_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing class definition, metadata saving, and the
> resulting classifier appearing in the Training Widget dropdown.

## UI Elements

| Control | Description |
|---------|-------------|
| **Class name input** | Text field for entering a new class label |
| **Enter button** | Add the typed class to the class list |
| **Reset Classes** | Clear all defined classes back to `["unassigned"]` |
| **Class list** | Read-only display of all defined class labels |
| **Filename input** | Name for the classifier (used as folder name and DB entry) |
| **Save metadata** | Create the classifier folder, metadata file, and initial session |

## Workflow

### Prerequisites

Before creating a classifier, you need cropped cell images in memory:

1. Load a well using the **Welldata Widget**.
2. Generate a gallery using the **Gallery Widget**.

### Creating a Classifier

1. Open the Setup Training Widget from the Plugins menu.
2. **Define classes**: Type each class name (e.g. `interphase`, `mitotic`,
   `apoptotic`) and press **Enter** to add it. The class list updates with each
   addition.
3. **Name the classifier**: Enter a descriptive name in the filename field (e.g.
   `hela_mitosis_classifier`).
4. **Click Save metadata**. This performs three actions:
   - Creates `~/omeroscreen_trainingdata/<classifier>/` directory.
   - Writes `metadata.json` containing class options, channel data, gallery
     parameters (crop size, rows, columns, segmentation type), and channel mapping.
   - Saves the current gallery crops as the first NPY session and registers it
     in the training database.
5. The new classifier now appears in the Training Widget's classifier dropdown.

## Metadata File

The saved `metadata.json` records all parameters needed to produce consistent
crops across sessions:

```json
{
    "class_options": ["unassigned", "interphase", "mitotic", "apoptotic"],
    "n_crops": 16,
    "user_data": {
        "rows": 4,
        "columns": 4,
        "crop_size": 50,
        "segmentation": "nucleus",
        "no_background": true,
        "cellcycle": "All",
        "channels": ["DAPI", "Tub", "EdU"]
    },
    "channel_data": {
        "DAPI": "0",
        "Tub": "1",
        "EdU": "2"
    }
}
```

When loading new data for this classifier (via the Direct Load Dialog), these
parameters are applied automatically to ensure all crops are comparable.

## Tips

- Choose class names that are mutually exclusive and cover all expected phenotypes.
- The `unassigned` class is always present as the default label.
- You can reset classes and start over at any point before saving.
- The number of crops (`n_crops`) is determined by the gallery grid size
  (rows x columns) at the time of creation.
- Once saved, class definitions are stored in `metadata.json` and used by the
  Training Widget to populate radio buttons.
