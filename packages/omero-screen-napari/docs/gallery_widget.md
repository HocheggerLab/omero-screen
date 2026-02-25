# Gallery Widget

## Overview

The Gallery Widget extracts individual cell crops from segmented well images and
displays them as a montage grid. It supports filtering by cell cycle phase,
adjustable crop sizes, and contour overlays. Galleries serve as the starting point
for creating training data.

```{video} _static/gallery_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing gallery generation with different crop sizes,
> cell cycle filtering, and contour overlays.

## UI Elements

### Gallery Parameters

| Control | Description |
|---------|-------------|
| **Well** | Well position to crop from (defaults to the currently loaded well) |
| **Segmentation** | Mask type for cropping: `nucleus` or `cell` |
| **Crop size** | Side length in pixels: 20, 30, 50, 100, or 200 |
| **Cell cycle** | Filter crops by phase: All, G1, S, G2/M, G2, M, Polyploid |
| **Timepoint** | Timepoint index (default 0) |
| **Columns** | Number of columns in the montage grid |
| **Rows** | Number of rows in the montage grid |
| **Reload** | Re-extract crops from images (disable to reuse existing crops) |
| **Contour** | Draw segmentation contours on each crop |
| **No Background** | Set background pixels to zero outside the mask |
| **Blue / Green / Red Channel** | Channel names mapped to RGB for display |

### Additional Controls

| Control | Description |
|---------|-------------|
| **Reset button** | Clears all cropped images and labels from memory |
| **Analysis widget** | Generate multiple galleries across wells for batch inspection |

## Workflow

1. **Load well data first** using the Welldata Widget (images and masks must be
   in the viewer).
2. Open the Gallery Widget from the Plugins menu.
3. **Select parameters**:
   - Choose `nucleus` or `cell` segmentation.
   - Set crop size appropriate for your cells (50 px is a good default for most).
   - Filter by cell cycle phase if needed.
   - Adjust the grid size (rows x columns = total crops displayed).
4. **Click Enter** to generate the gallery.
5. The montage appears as an RGB image in the viewer.
6. Crops and their corresponding masks are stored in memory for use by the
   Training Widget or Setup Training Widget.

## Channel Mapping

The gallery constructs an RGB composite from three channels:

| RGB Channel | Default | Description |
|-------------|---------|-------------|
| Blue | DAPI | Typically the DNA stain |
| Green | Tub | Typically tubulin or a cytoplasmic marker |
| Red | EdU | Typically an S-phase or other marker |

Change the channel names to match your experiment. Channels are resolved by name
from the plate's channel metadata. For 2-channel images, only Red and Green are
used. For single-channel images, the crop is displayed in grayscale.

## Analysis Mode

The analysis widget generates multiple galleries across specified wells:

1. Enter comma-separated well positions.
2. Set the number of galleries per well.
3. Click Enter to batch-generate galleries for visual inspection.

## Tips

- A grid of **4 x 4** (16 crops) gives a quick overview; **8 x 8** (64 crops)
  is better for assessing rare phenotypes.
- Enable **No Background** to focus on cell morphology without surrounding signal.
- Use **Contour** overlay to verify that segmentation boundaries look correct.
- The **Reset** button frees memory — use it before generating a new gallery from
  a different well to avoid accumulating crops.
