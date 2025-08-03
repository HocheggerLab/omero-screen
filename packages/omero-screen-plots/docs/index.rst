.. OmeroScreen Plots documentation master file

OmeroScreen Plots Documentation
===============================

**OmeroScreen Plots** is a comprehensive visualization and analysis package for high-content screening data from OMERO. It provides standardized, publication-ready plots for immunofluorescence microscopy data with built-in statistical analysis.

.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/Helfrid/omero-screen/blob/main/LICENSE
   :alt: License

Key Features
------------

* **Standardized Plots**: Consistent, publication-ready figures for screening data
* **Multiple Plot Types**: Feature plots, cell cycle analysis, combined plots, and more
* **Statistical Analysis**: Built-in statistical tests and data normalization
* **Flexible Customization**: Extensive options for colors, grouping, and styling
* **Integration**: Works seamlessly with OMERO and cellview databases

Quick Start
-----------

.. code-block:: python

   from omero_screen_plots.featureplot import FeaturePlot

   # Create a simple feature plot
   plot = FeaturePlot(
       data_path="path/to/data.csv",
       y_feature="intensity_mean_dapi_nucleus",
       conditions=["Control", "Treatment"],
       condition_col="condition"
   )
   plot.plot()
   plot.save("output_path")

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   plot_types
   customization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/featureplot
   api/cellcycleplot
   api/combplot
   api/countplot
   api/normalise
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_plots
   examples/advanced_analysis
   examples/custom_styling

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
