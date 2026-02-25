# Aligned Plate Widget

## Overview

The Aligned Plate Widget overlays images from multiple plates that have been
spatially aligned. This is useful for comparing the same well position across
different experimental conditions or timepoints acquired on separate plates.

```{video} _static/aligned_plate_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing multi-plate loading with translation offsets
> and channel overlay.

## UI Elements

| Control | Description |
|---------|-------------|
| **Plate ID** | Primary plate identifier |
| **Well Position** | Single well position (e.g. `A1`) |
| **Image index** | Image index within the well (default 0) |
| **Sample Alignments** | Show alignment samples for verification |

## Workflow

1. **Enter the primary Plate ID** and a single well position.
2. **Click Enter** to load images.
3. The widget:
   - Loads the primary plate image for the specified well.
   - Discovers all aligned plates from the alignment CSV data.
   - Applies X/Y translation offsets so images from different plates overlay
     correctly in the viewer.
   - Adds each channel as a separate layer with appropriate colormaps.
4. Duplicate channels (same name across plates) are filtered to avoid redundancy.
5. Use Napari's layer visibility toggles and opacity sliders to compare plates.

## Alignment Data

Plate alignment is defined by a CSV file (typically `agg_data.csv`) or OMERO
annotations that specify translation offsets for each plate relative to a reference.
The pixel size from image metadata is used to scale translations correctly.

## Tips

- Only a **single well** can be loaded at a time (no comma-separated lists).
- Ensure alignment data has been generated before using this widget.
- Use Napari's layer blending modes (additive, translucent) for effective overlay
  comparison.
- Enable **Sample Alignments** to visually verify that the spatial registration
  is correct before interpreting results.
