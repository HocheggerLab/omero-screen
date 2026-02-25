.. omero-screen-napari documentation

Omero Screen Napari
===================

**omero-screen-napari** provides interactive `Napari <https://napari.org>`_ widgets for exploring OMERO microscopy data, generating cell galleries, and creating training datasets for machine learning classifiers.

Widgets
-------

The package registers five widgets accessible from the **Plugins** menu in Napari:

*  **Welldata Widget** -- Browse and visualize well images with caching, stitching, and metadata display.
*  **Gallery Widget** -- Extract cell crops from segmented images and display them as montage grids.
*  **Training Widget** -- Navigate crops, assign class labels, and save annotated training data.
*  **Setup Training Widget** -- Define classifier classes and create new classifier projects.
*  **Aligned Plate Widget** -- Overlay images from multiple aligned plates with spatial translations.

Supporting dialogs:

*  **Session Manager** -- Browse, load, and delete annotation sessions.
*  **Direct Load Dialog** -- Load new OMERO data directly into a training session.
*  **Plate Info Dialog** -- Inspect well metadata and select wells to load.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   welldata_widget
   gallery_widget
   training_widget
   setup_training_widget
   aligned_plate_widget
   session_manager
