.. OmeroScreen Plots documentation master file

OmeroScreen Plots Documentation
===============================

**OmeroScreen Plots** is a comprehensive visualization and analysis package for high-content screening data from OMERO. It provides standardized, publication-ready plots for immunofluorescence microscopy data with built-in statistical analysis.

**Recent Architecture Update**: The package has been refactored from a complex multi-class factory pattern to a simplified single-class architecture, providing better performance, maintainability, and error handling while maintaining full backward compatibility.

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

   import pandas as pd
   from omero_screen_plots.countplot_api import count_plot
   from omero_screen_plots.countplot_factory import PlotType

   # Load your data
   df = pd.read_csv("path/to/data.csv")

   # Create a count plot (simplified architecture)
   fig, ax = count_plot(
       df=df,
       norm_control="DMSO",
       conditions=["DMSO", "Treatment1", "Treatment2"],
       condition_col="condition",
       selector_col="cell_line",
       selector_val="MCF10A",
       title="Cell Count Analysis",
       save=True,
       path="output/"
   )

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
   api/featurenormplot
   api/cellcycleplot
   api/cellcyclestacked
   api/combplot
   api/countplot
   api/normalise
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   count_plot_examples
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
