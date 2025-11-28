Command Line Interface
======================

This section describes the command-line tools available in ``omero-screen``.

run_omero_screen
----------------

The main entry point for running the analysis pipeline.

.. code-block:: bash

    python bin/run_omero_screen.py [ID [ID ...]] [options]

**Arguments:**

*   ``ID``: One or more OMERO plate IDs to process.

**Options:**

*   ``--env ENV``: Environment name (requires configuration file ``.env.{name}``).
*   ``--inference MODEL [MODEL ...]``: Inference model filename(s).
*   ``--gallery N``: Width N of for the inference gallery NxN (default: 10).
*   ``--batch N``: Classification batch size (default: 16).
*   ``--segmentation``: Only perform image segmentation (flag).

sbatch-omero-screen
-------------------

(Sussex HPC Specific) Script to submit Omero Screen jobs to a SLURM cluster.

.. code-block:: bash

    python bin/sbatch-omero-screen.py [ID [ID ...]] [options]

**Arguments:**

*   ``ID``: One or more OMERO plate IDs to process.

**Job Submission Options:**

*   ``--class CLASS``: Job class (default: gpu).
*   ``-u, --username USER``: Username (default: current user).
*   ``-t, --threads N``: Threads (default: 1). Use when not executing on the GPU.
*   ``--hours N``: Expected maximum job hours (default: 24).
*   ``-m, --memory N``: Memory in Gb (default: 32).
*   ``--gpu``: Use a GPU node (default: True).
*   ``--exec``: Execute script statements (default: True).
*   ``--submit``: Submit using sbatch (default: True).
*   ``--multi-submit``: Submit a single job for each Screen ID (default: True).

**Omero Screen Overrides:**

*   ``--inference MODEL``: Inference model(s).
*   ``--env ENV``: Environment name.
*   ``--segmentation``: Only perform image segmentation.
