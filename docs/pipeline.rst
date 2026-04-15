Image Analysis Pipeline
=======================

This section describes the core analysis pipeline of ``omero-screen``: how it fetches
data from an OMERO server, segments cells, extracts features, and stores results.


Pipeline Overview
-----------------

Each plate is processed by ``plate_loop()``, which orchestrates the following stages in order:

.. figure:: figures/Fig1_loop/loop-vs2.svg
   :align: center
   :alt: OmeroScreen analysis loop
   :width: 85%

   **Analysis loop.** Starting from a plate ID on the OMERO server, the pipeline parses
   experimental metadata, generates per-channel flatfield correction masks, then iterates
   over every non-empty well. Within each well every image is segmented with Cellpose,
   features are extracted with ``skimage.measure.regionprops``, and results are written
   back to OMERO as CSV attachments. After all wells are processed, cell-cycle phase
   assignment and optional image classification are performed, and final summary files
   are attached to the plate object.

The stages are:

1. **Metadata parsing** — reads the experimental layout from an Excel file attached to
   the plate object (well positions, cell lines, conditions, channel names). After
   parsing, the metadata is converted to OMERO map-annotations and the Excel file is
   removed.

2. **Flatfield correction** — generates per-channel illumination correction masks by
   aggregating a random sample of 100 images across the plate. Masks are cached in a
   dedicated OMERO dataset so subsequent runs skip this step.

3. **Well / image loop** — for each non-empty well and each image within it:

   * Raw pixel data are downloaded from OMERO and divided by the flatfield mask.
   * Cellpose segments nuclei (DAPI/Hoechst channel) and cells (cytoplasm channel).
   * Border objects are filtered; cytoplasm masks are derived as cell minus nucleus.
   * ``skimage.measure.regionprops_table`` extracts area, intensity statistics, and
     centroid for nucleus, cell, and cytoplasm regions.
   * Per-well results are written back to OMERO as intermediate CSV attachments,
     enabling crash recovery.

4. **Cell-cycle analysis** — if an EdU channel is present, DNA content (integrated
   DAPI intensity) and EdU signal are normalised per cell line and used to assign each
   nucleus to G1, S, G2/M, sub-G1, or polyploid.

5. **Save results** — final plate-level CSVs and QC figures are attached to the OMERO
   plate object; intermediate per-well files are deleted.


Command Line Interface
----------------------

run_omero_screen
~~~~~~~~~~~~~~~~

The main entry point for running the analysis pipeline against one or more plates.

.. code-block:: bash

    omero-screen ID [ID ...] [options]

**Positional arguments**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``ID``
     - One or more OMERO plate IDs to process (space-separated).

**Options**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--env ENV``
     - development
     - Environment name. Loads ``.env.{ENV}`` for server credentials and logging config.
   * - ``--segmentation``
     - off
     - Run segmentation only — skip feature extraction and cell-cycle analysis. Useful
       for inspecting mask quality before a full run.
   * - ``--cp4``
     - off
     - Use **Cellpose 4** (``cpsam``) for all segmentation models instead of the
       default Cellpose 3 models. ``cpsam`` is the Segment Anything Model (SAM)-based
       model included in Cellpose 4 and generally produces higher-quality masks,
       especially for irregularly shaped cells.
   * - ``--model MODEL``
     - —
     - Override **all** segmentation models with a single model name (e.g.
       ``cp4:cpsam``, ``cp3:cyto3``). Takes precedence over ``--cp4``. Useful for
       testing a new model across an entire plate without editing the config file.
   * - ``--inference MODEL [MODEL ...]``
     - —
     - One or more inference model filenames for post-segmentation image classification.
       Multiple models are applied sequentially to each cell crop.
   * - ``--gallery N``
     - 10
     - Width *N* of the N×N example gallery generated for each predicted class when
       ``--inference`` is active.
   * - ``--batch N``
     - 16
     - Batch size for inference classification.
   * - ``--benchmark``
     - off
     - Record per-image timing data and write a JSON benchmark report at the end of
       the run.

**Examples**

.. code-block:: bash

    # Analyse a single plate in the default (development) environment
    omero-screen 12345

    # Analyse multiple plates in the production environment
    omero-screen 12345 67890 --env production

    # Segment only — no feature extraction (fast QC pass)
    omero-screen 12345 --segmentation

    # Use Cellpose 4 SAM-based models for better mask quality
    omero-screen 12345 --cp4

    # Override all models with a specific Cellpose 3 model
    omero-screen 12345 --model cp3:cyto3

    # Run with an image classifier and generate 15×15 example galleries
    omero-screen 12345 --inference my_classifier.pth --gallery 15

    # Record benchmark timing data
    omero-screen 12345 --benchmark


sbatch-omero-screen
~~~~~~~~~~~~~~~~~~~

*(Sussex HPC specific)* Submits ``omero-screen`` jobs to a SLURM cluster.

.. code-block:: bash

    python bin/sbatch-omero-screen.py ID [ID ...] [options]

**Job submission options**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--class CLASS``
     - gpu
     - SLURM job class.
   * - ``-u / --username USER``
     - current user
     - Cluster username.
   * - ``-t / --threads N``
     - 1
     - CPU threads. Increase when running without a GPU.
   * - ``--hours N``
     - 24
     - Maximum wall-clock hours for the job.
   * - ``-m / --memory N``
     - 32
     - Memory in GB.
   * - ``--gpu``
     - on
     - Request a GPU node.
   * - ``--exec``
     - on
     - Execute the generated script statements.
   * - ``--submit``
     - on
     - Submit via ``sbatch``.
   * - ``--multi-submit``
     - on
     - Submit one job per plate ID.

**OmeroScreen overrides** (passed through to each job)

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--inference MODEL``
     - Inference model(s).
   * - ``--env ENV``
     - Environment name.
   * - ``--segmentation``
     - Only perform image segmentation.
   * - ``--cp4``
     - Use Cellpose 4 (cpsam) models.
   * - ``--model MODEL``
     - Override all segmentation models with a single model name.
   * - ``--cp4``
     - Use Cellpose 4 models.
   * - ``--model MODEL``
     - Override all segmentation models.
