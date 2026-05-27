# omero-screen-napari

Interactive [Napari](https://napari.org) widgets for exploring high-content microscopy data stored on an OMERO server, generating cell galleries, and building training datasets for machine learning classifiers.

## Status

Version: ![version](https://img.shields.io/badge/version-0.3.5-blue)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What this package does

`omero-screen-napari` adds four widgets to the Napari image viewer that cover the full workflow from raw plate data to annotated training datasets:

| Widget | Purpose |
|--------|---------|
| **Welldata Widget** | Load and visualise well images from an OMERO plate, with caching and stitching support |
| **Gallery Widget** | Extract individual cell crops as a montage grid and define new classifiers |
| **Training Widget** | Annotate crops with class labels and save sessions to disk |
| **Aligned Plate Widget** | Overlay images from multiple spatially registered plates |

A companion command-line tool, `omero-train`, provides database management, statistics, and data export without opening Napari.

## Documentation

Full user documentation (including workflow guides for non-technical users) is available at the [omero-screen documentation site](https://hocheggerlab.github.io/omero-screen/).

Quick links:
- [Installation](docs/installation.md)
- [Welldata Widget — loading images and stitching](docs/welldata_widget.md)
- [Gallery Widget — cell crops and classifier creation](docs/gallery_widget.md)
- [Training Widget — annotating cells](docs/training_widget.md)
- [Session Manager & Direct Load](docs/session_manager.md)
- [omero-train CLI reference](docs/cli_reference.md)

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

See [CLI reference](docs/cli_reference.md) for full details.

## Authors

Created by Helfrid Hochegger — hh65@sussex.ac.uk

## License

MIT — see [LICENSE](LICENSE) for details.
