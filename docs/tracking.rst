.. _tracking:

Live-Cell Tracking
==================

OmeroScreen can follow individual cells through time. For a live-cell
time-lapse plate, every segmented nucleus is linked across the time axis so it
carries a stable ``track_id`` for its whole lifetime, and divisions are recorded
as a parent → daughter lineage. Tracks flow into the same CellView database as
every other measurement, can be explored interactively in napari, and can be
hand-curated in `Mastodon <https://mastodon.readthedocs.io/>`_.

This page is the map for that workflow; each stage links to its detailed
reference.


The workflow at a glance
------------------------

.. code-block:: text

   omero-screen --stitch --track          (1) generate tracks
        │  Trackastra links nuclei across time; track_id / parent_track_id
        │  flow into the measurements CSV
        ▼
   cellview import-plate                   (2) tracks land in CellView
        ▼
   napari — Tracks Widget                  (3) view & export
        │  overlay tracks, inspect lineages (arboretum), export per-track CSV
        ▼
   Mastodon (Fiji)                         (4) manual curation
        open the cached OME-Zarr well + tracks.csv, correct by hand


1. Generate tracks
------------------

Run the pipeline with ``--track`` (which requires ``--stitch``):

.. code-block:: bash

   omero-screen 12345 --stitch --track general_2d

Tracking uses `Trackastra <https://github.com/weigertlab/trackastra>`_ to relabel
the stitched nucleus mask so each nucleus's label *is* its ``track_id``; cell and
cytoplasm measurements inherit it automatically. Four columns reach the
measurements table — ``track_id`` / ``parent_track_id`` (curatable) and immutable
``track_id_raw`` / ``parent_track_id_raw`` — and it is a no-op on fixed-cell
(``T == 1``) plates.

See :ref:`temporal-tracking` for the full description, linking modes, and CLI
reference.


2. Tracks in CellView
---------------------

Import the plate as usual (``cellview import-plate 12345``). The track columns
are picked up automatically — no schema change is needed. From here the tracks
are queryable alongside intensities, areas, and cell-cycle phase.


3. View and export in napari
----------------------------

Open the well with the **Welldata Widget**, then the **Tracks Widget** to overlay
the tracks, inspect a single lineage tree (via the bundled *napari-arboretum*
plugin), export one track's full time-course as CSV, or prepare a well for
Mastodon. Pinning protects a plate from cache eviction while you curate.

See :doc:`omero-screen-napari/tracks_widget` for the step-by-step guide.


.. _mastodon-curation:

4. Curate in Mastodon
---------------------

Automated tracking is never perfect, so tracks are hand-corrected in Mastodon, a
mature Fiji track-editing tool. OmeroScreen makes the cached well directly
openable — no file conversion or image copy.

**The cache.** Cached plates live in a visible folder, ``~/omero-cache`` by
default (override with ``OMERO_SCREEN_CACHE_PATH``). For a tracked plate, a
``tracks.csv`` is written automatically next to each well's image when the
cache is built, so a well is ready for Mastodon with no extra step. The cache is
size-bounded and evicts least-recently-used plates — **Pin** a plate in the
Tracks Widget before a long curation session so it is not reclaimed, and
**Unpin** it when finished. See :doc:`caching` for the cache layout, the full
set of environment variables, and the eviction/pinning rules.

**Open the image.** In Fiji:

1. **Plugins → Tracking → Mastodon → "new from OME-NGFF…"**.
2. Paste the path to the well's image group — the multiscale ``0`` group inside
   the plate store::

       ~/omero-cache/zarr/plate_<id>.zarr/<row>/<col>/0

   for example ``~/omero-cache/zarr/plate_4155.zarr/B/2/0``. Do **not** point at
   the plate root or the well folder — open the ``0`` image group directly.
3. Click **Detect datasets**, select the listed row, **OK**, and save the BDV
   XML when prompted.
4. The image opens in BigDataViewer. Press ``P`` for the side panel to set
   per-channel contrast; ``1`` / ``2`` switch channels; ``F`` toggles a fused
   overlay.

**Import the tracks.** In the main Mastodon window: **File → Import → CSV
Importer**, choose the ``tracks.csv`` sitting next to the image group
(``…/<row>/<col>/tracks.csv``), and map the columns:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Mastodon field
     - CSV column
   * - X / Y / Z
     - ``x`` / ``y`` / ``z``
   * - Frame
     - ``frame``
   * - ID / Parent ID
     - ``id`` / ``parent_id``
   * - Radius (column)
     - ``radius``
   * - Label
     - ``label``

Set a default **Radius** of ``10`` (used only if the radius column is blank).

**Link the views.** Click the same group-lock number (e.g. ``1``) in *both* the
BigDataViewer and TrackScheme windows, then double-click a spot to navigate
between the views. Save your work as a ``.mastodon`` project in a stable
location **outside** the cache (e.g. next to the README in
``~/mastodon_exports/``) so it is not lost if the plate is later evicted.

.. note::

   The reverse trip — feeding corrected Mastodon tracks back into CellView — is
   planned but not yet implemented. For now the curated ``.mastodon`` project is
   the record of corrections.
