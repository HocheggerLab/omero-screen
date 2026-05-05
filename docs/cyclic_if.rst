Cyclic Immunofluorescence Tools
=============================

Tools for analyzing Cyclic Immunofluorescence (Cyclic IF) data, where the same plate is washed and re-stained multiple times.

align_plates
------------

Aligns multiple repeat OMERO screen plate experiments. This is used to register images from different rounds of staining.

.. code-block:: bash

    python bin/align-plates.py [ID [ID ...]] [options]

**Arguments:**

*   ``ID``: Two or more OMERO plate IDs. The first ID is the reference plate.

**Options:**

*   ``--seed N``: Random seed for samples.
*   ``--channel NAME``: Alignment channel (default: DAPI).
*   ``-n N``: Number of alignments used to create the average (default: 5).
*   ``--sample-alignments``: Compute per-sample alignments; else the specified number of alignments.
*   ``--threshold N``: Distance threshold for alignments (default: 100).
*   ``--tolerance N``: Distance tolerance for alignments to their centroids (default: 10).
*   ``--gallery N``: Alignment gallery grid size (default: 4).

aggregate_plates
----------------

Combines multiple repeat OMERO screen experiments into a single dataset.

.. code-block:: bash

    python bin/aggregate-plates.py ID [options]

**Arguments:**

*   ``ID``: OMERO plate ID (the reference plate).

**Options:**

*   ``--seed N``: Random seed for samples.
*   ``--threshold N``: Distance threshold for alignment mappings (default: 25).
*   ``--method N``: Mapping method (default: 3).
    *   0: minimum distance
    *   1: KD-Tree minimum distance
    *   2: Greedy nearest neighbour
    *   3: Mask overlap
*   ``--std N``: Number of standard deviations from the mean distance to exclude distance mappings (default: 6).
*   ``--gallery N``: Mapping gallery grid size (default: 4).
*   ``--sample-alignments``: Use per-sample alignments; else per-well alignments.
