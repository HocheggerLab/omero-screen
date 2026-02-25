.. omero-screen documentation master file

OmeroScreen Documentation
=========================

**OmeroScreen** is an end-to-end high-content image analysis pipeline built on top of the OMERO server. It integrates analysis, data storage, and visualization into a seamless workflow.

Key Components
--------------

*   **omero-screen**: The core analysis pipeline using Cellpose for segmentation.
*   **cellview**: A specialized database for single-cell data storage and metadata management.
*   **omero-screen-plots**: A suite of tools for generating publication-ready visualizations.
*   **omero-screen-napari**: Napari plugins for interactive data exploration.
*   **omero-utils**: Helper functions for interacting with the OMERO API.

Installation
------------

Quick start using ``uv``:

.. code-block:: bash

    # Install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Clone the repository
    git clone https://github.com/Helfrid/omero-screen.git
    cd omero-screen

    # Create and activate virtual environment
    uv sync --dev
    source .venv/bin/activate

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Omero Screen

   cli
   cyclic_if
   configuration
   developer_guide

.. toctree::
   :maxdepth: 2
   :caption: Packages

   Omero Screen Plots <omero-screen-plots/index>
   Cellview <cellview/index>
   Omero Screen Napari <omero-screen-napari/index>
