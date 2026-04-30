# Installation

## Prerequisites

- **Python 3.12**
- **Napari 0.6.2+** with Qt backend
- Access to an **OMERO server** (5.x) with valid credentials
- Segmented plate data on OMERO (nucleus/cell masks uploaded by the omero-screen pipeline)

## Install from the Monorepo

The recommended way to install is from the omero-screen monorepo, which includes all
packages:

```bash
git clone https://github.com/Helfrid/omero-screen.git
cd omero-screen
uv sync --dev
source .venv/bin/activate
```

This installs `omero-screen-napari` along with all workspace dependencies.

## Standalone Install

If you only need the Napari widgets:

```bash
uv add omero-screen-napari
```

## OMERO Connection

The widgets connect to OMERO using credentials from environment files. Create a
`.env.development` (or `.env`) in the project root:

```ini
USERNAME=your_omero_username
PASSWORD=your_omero_password
HOST=your.omero.server.com
```

Select the environment with the `ENV` variable:

```bash
export ENV=development   # default
export ENV=production    # for production server
```

## Verify Installation

Launch Napari and check that the widgets appear under the **Plugins** menu:

```bash
napari
```

You should see:

- Welldata Widget
- Gallery Widget
- Training Widget
- Aligned Plate Widget

## Troubleshooting

**Widgets not appearing in Plugins menu**
: Ensure the package is installed in the same environment as Napari. Run
  `napari --info` and check that `omero-screen-napari` is listed.

**OMERO connection fails**
: Verify your `.env` file exists and contains valid credentials. Check that the
  OMERO server is reachable from your machine.

**Import errors for zeroc-ice**
: The OMERO Python bindings require `zeroc-ice`. Platform-specific wheels are
  configured in the root `pyproject.toml`. If you encounter issues, check the
  [omero-py documentation](https://pypi.org/project/omero-py/).
