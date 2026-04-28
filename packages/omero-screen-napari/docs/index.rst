.. omero-screen-napari documentation

Omero Screen Napari
===================

**omero-screen-napari** adds interactive `Napari <https://napari.org>`_ widgets to your image viewer so you can load microscopy plates from an OMERO server, browse individual cells, and build labelled training datasets for machine learning classifiers — all without writing code.

The package registers four widgets accessible from the **Plugins → Omero Screen Napari** menu:

*  **Welldata Widget** — Load well images from an OMERO plate, cache them locally, and stitch tiled acquisitions into a single composite view.
*  **Gallery Widget** — Extract individual cell crops as a montage grid, filter by cell cycle phase, and define new classifiers.
*  **Training Widget** — Navigate crops one by one, assign class labels with keyboard shortcuts, and save annotated sessions.
*  **Aligned Plate Widget** — Overlay images from multiple spatially registered plates for side-by-side comparison.

Supporting dialogs launched from within the widgets:

*  **Session Manager** — Browse all annotation sessions for a classifier, check data integrity, load or delete sessions, and add new data.
*  **Direct Load Dialog** — Fetch fresh cell crops from OMERO directly into the Training Widget without going through the Gallery workflow.
*  **Plate Info Dialog** — Inspect well metadata, cache status, and select wells to load.

A command-line companion tool, **omero-train**, provides the same database operations without opening Napari.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   welldata_widget
   gallery_widget
   training_widget
   session_manager
   aligned_plate_widget
   cli_reference
