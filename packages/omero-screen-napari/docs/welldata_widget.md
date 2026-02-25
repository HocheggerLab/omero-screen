# Welldata Widget

## Overview

The Welldata Widget is the primary entry point for loading and viewing microscopy images
from OMERO. It supports on-demand loading, background plate caching, multi-position
image stitching, and live metadata display.

```{video} _static/welldata_demo.mp4
:width: 100%
```

> **TODO**: Record demo video showing plate loading, well selection, caching workflow,
> stitched view, and metadata dock interaction.

## UI Elements

### Cached Plates Panel

| Control | Description |
|---------|-------------|
| **Plate dropdown** | Lists all cached and previously loaded plates with status badges |
| **Cache Size** | Displays total disk usage of the plate cache |
| **Cache button** | Downloads all wells for the selected plate in the background |
| **Delete button** | Removes cached data for the selected plate |
| **Refresh button** | Re-scans the cache directory |

### Well Data Input

| Control | Description |
|---------|-------------|
| **Plate ID** | OMERO plate identifier (numeric) |
| **Well Position** | Comma-separated well positions (e.g. `A1, B3, C5`) |
| **Images** | Image indices to load (`All`, `0`, `0,1,2`) |
| **Time** | Timepoint filter (`All` or specific index) |
| **Cache** | Enable background caching of the entire plate while loading |

### Stitched Data Parameters

Available when the plate contains multi-position (tiled) images:

| Control | Description |
|---------|-------------|
| **Rotation** | Rotation correction angle (default 0.15) |
| **Precise Rotation** | Use sub-pixel rotation for higher accuracy |
| **Overlap X / Y** | Tile overlap in percent (default 7) |
| **Edge** | Edge crop in pixels (default 7) |
| **Mode** | Padding mode for stitching (default `reflect`) |

## Workflow

1. **Enter a Plate ID** and one or more well positions in the input fields.
2. **Click Enter** to load images from OMERO (or the local cache if available).
3. Images appear in the Napari viewer with **automatic channel colormaps**:
   - DAPI/Hoechst → blue
   - Tubulin → green
   - EdU → red
   - Other channels → magenta, cyan, yellow
4. **Segmentation masks** (nucleus, cell) are added as label layers if available.
5. A **metadata dock widget** appears showing well annotations. When scrolling
   through timepoints or images, the metadata updates automatically.
6. For tiled acquisitions, click **Enter** on the stitched data panel to assemble
   a composite image from all positions.

## Plate Caching

Enabling the **Cache** checkbox triggers a background download of the full plate.
This is useful for repeated access — subsequent loads are near-instant from disk.

- Cache uses **Blosc compression** for ~2x size reduction.
- The **Cached Plates** panel shows download progress in real time.
- Individual plates can be evicted via the **Delete** button.

## Plate Info Dialog

Clicking the plate dropdown opens a **Plate Info Dialog** showing:

- All wells with dynamic metadata columns (cell line, condition, timepoint, etc.)
- Image count and label availability per well
- Real-time cache status during background downloads
- **Select All** checkbox for bulk well selection
- Double-click a row to load that well immediately

## Tips

- Load multiple wells at once by separating positions with commas: `A1, A2, A3`.
- Use `All` in the Images field to load every image in the well.
- The cache persists across Napari sessions — no need to re-download.
- For very large plates, cache in the background while you work on individual wells.
