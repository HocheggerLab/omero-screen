# omero-screen-napari

Interactive [Napari](https://napari.org) widgets for exploring high-content microscopy data stored on an OMERO server, generating cell galleries, and building training datasets for machine learning classifiers.

## Status

Version: ![version](https://img.shields.io/badge/version-0.8.8-blue)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What this package does

`omero-screen-napari` adds four widgets to the Napari image viewer that cover the full workflow from raw plate data to annotated training datasets:

| Widget | Purpose |
|--------|---------|
| **Welldata Widget** | Load and visualise well images from an OMERO plate, with caching and stitching support |
| **Gallery Widget** | Extract individual cell crops as a montage grid and define new classifiers |
| **Training Widget** | Annotate crops with class labels and save sessions to disk |
| **Aligned Plate Widget** | Overlay images from multiple spatially registered plates |

Two companion command-line tools work without opening Napari: `omero-train` manages the training database, and `omero-screen-images` renders cell galleries and whole-well overviews (for QC and figure panels) with a JSON manifest of every run.

## Documentation

Full user documentation (including workflow guides for non-technical users) is available at the [omero-screen documentation site](https://hocheggerlab.github.io/omero-screen/).

Quick links:
- [Installation](user-guide/installation.html)
- [Welldata Widget — loading images and stitching](user-guide/welldata_widget.html)
- [Gallery Widget — cell crops and classifier creation](user-guide/gallery_widget.html)
- [Training Widget — annotating cells](user-guide/training_widget.html)
- [Session Manager & Direct Load](user-guide/session_manager.html)
- [omero-train CLI reference](user-guide/cli_reference.html)
- [omero-screen-images — galleries and well overviews without Napari](user-guide/images_cli.html)

## Installation

This package is part of the `omero-screen` monorepo. The recommended way to install it is via the workspace:

```bash
git clone https://github.com/HocheggerLab/omero-screen.git
cd omero-screen
uv sync
```

To install the napari plugin standalone:

```bash
uv pip install omero-screen-napari
```

After installation, start Napari and the four widgets will appear under **Plugins → Omero Screen Napari**.

## OMERO connection

The plugin connects to an OMERO server using credentials from a `.env` file in the project root:

```
USERNAME=your_omero_username
PASSWORD=your_omero_password
HOST=your_omero_server
```

## omero-train CLI

A command-line tool for managing training databases without opening Napari:

```bash
omero-train list                    # show all classifiers
omero-train stats mitosis-rpe       # detailed stats for a classifier
omero-train export mitosis-rpe      # export annotations to CSV
omero-train delete mitosis-rpe      # delete a classifier and its data
```

See [CLI reference](user-guide/cli_reference.html) for full details.

## omero-screen-images CLI

Galleries and whole-well overviews without Napari, reproducible from a script or notebook:

```bash
omero-screen-images gallery 5108 --wells E2,G5 --classifier-column classifier_nuclei4 \
    --class micronuclei --channels DAPI --grid 5x5 --json
omero-screen-images well 5108 --wells B2,E2 --layers DAPI,Tub,nuclei_masks --zoom 4
omero-screen-images batch plan.csv --channels DAPI
```

See [omero-screen-images](user-guide/images_cli.html) for the workflows.

## Authors

Created by Helfrid Hochegger — hh65@sussex.ac.uk

## License

MIT — see [LICENSE](LICENSE) for details.
