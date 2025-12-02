User Guide
==========

CLI Commands
------------

.. argparse::
   :module: cellview.cli
   :func: get_parser
   :prog: cellview

(Note: This requires `sphinx-argparse` to auto-document. If you prefer no dependencies, we can write this manually.)

Examples
--------

Import a single plate:

.. code-block:: bash

   cellview --plate-id 123

Import multiple plates (must belong to the same screen):

.. code-block:: bash

   cellview --plate-id 101 102 103

Import all plates from a screen:

.. code-block:: bash

   cellview --screen-id 456
