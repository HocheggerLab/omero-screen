.. _caching:

Local Caching & Interactive Viewing
===================================

Fetching pixels from an OMERO server every time you open a well is slow, and
for a tiled (stitched) plate it also means re-stitching in memory on every
view. OmeroScreen keeps a **local cache** so that, once a plate has been
cached, any well — or the whole plate — loads from disk in a fraction of a
second, with no server connection required.

For plates processed with :ref:`stitched-mode <stitched-mode>` segmentation the
cache is an **OME-Zarr (OME-NGFF) store** holding the pre-stitched whole-well
canvas. This is what makes a full plate of large stitched wells interactive:
napari reads only the chunks needed for the current viewport, so you can pan,
zoom, and flip through every well at full resolution within bounded memory.

.. note::

   The cache is built **on demand by napari**, not by the analysis pipeline.
   Running ``omero-screen --stitch`` writes the stitched segmentation back to
   OMERO; the OME-Zarr cache is created later, when you click **Cache** in the
   napari :doc:`omero-screen-napari/welldata_widget`. The widget chooses the
   OME-Zarr backend automatically for stitched-mode plates (detected by the
   ``_stitched_segmentation`` images in the OMERO dataset) and falls back to the
   legacy per-field disk cache for everything else.


The two cache backends
----------------------

.. list-table::
   :widths: 18 32 50
   :header-rows: 1

   * - Backend
     - Used for
     - What is stored
   * - **Disk cache** (legacy)
     - Per-field plates (not re-run with ``--stitch``)
     - Raw per-image ``TCZYX`` pixels (uint16); stitching, if any, happens in
       memory at view time.
   * - **OME-Zarr cache**
     - Stitched-mode plates (``--stitch``)
     - The pre-stitched whole-well canvas as a multiscale OME-NGFF store. Loads
       lazily, chunk by chunk, straight into napari.

Both can coexist; the napari dispatcher picks the right one per plate. The rest
of this page describes the OME-Zarr cache, which is the path that delivers
whole-plate interactivity.


On-disk layout
--------------

The cache root defaults to ``~/omero-cache`` (a *visible* folder, not hidden
under ``~/.cache``, so the stores can also be opened directly by external tools
such as Fiji/Mastodon). Override it with ``OMERO_SCREEN_CACHE_PATH``.

There is **one store per plate**, ``plate_<id>.zarr``, with wells as
``<row>/<col>`` subgroups — the standard OME-NGFF high-content-screening
hierarchy:

.. code-block:: text

   ~/omero-cache/
   ├── images/                       # legacy disk cache (per-field pixels)
   ├── plates/                       # legacy disk cache (plate metadata)
   └── zarr/                         # OME-Zarr cache
       ├── registry.json             # per-plate size / last-accessed / pin state (LRU)
       └── plate_4155.zarr/
           ├── .zattrs               # OME-NGFF plate metadata + channel names, pixel size
           ├── B/                    # row
           │   └── 2/                # well B2
           │       ├── 0/            # multiscale image group (this is what Mastodon opens)
           │       │   ├── 0/        # pyramid level 0 — full resolution
           │       │   ├── 1/        #   level 1 — 2× downsampled
           │       │   ├── 2/        #   level 2 — 4× downsampled
           │       │   └── labels/
           │       │       ├── nuclei/   # nucleus label mask (uint32), same pyramid
           │       │       └── cells/    # cell label mask (uint32), if available
           │       └── tracks.csv    # written for tracked wells (see below)
           └── ...

Each well image is a ``(T, C, Y, X)`` array. Storage details, all
code-verified:

* **Pyramid** — 3 levels, each a 2× down-sample (full / 2× / 4×), so the viewer
  renders an appropriate level for the current zoom without re-reading
  full-resolution data.
* **Chunking** — ``(T=1, C=1, 256, 256)`` for images and ``(T=1, 256, 256)``
  for label masks: one timepoint and one channel per chunk, in 256-pixel
  spatial tiles. This keeps single-cell crops cheap to read and bounds the work
  per slider step.
* **Compression** — Blosc/zstd (the ome-zarr default).
* **Crash safety** — each well is written to a ``.tmp`` staging directory and
  atomically renamed on completion, so an interrupted build leaves only
  fully-written wells. Re-running **Cache** resumes from where it stopped.


Whole-plate interactive viewing
--------------------------------

Because the canvas is already stitched on disk and stored as a lazy,
multiscale, chunked array, napari never has to hold a whole well — let alone a
whole plate — in memory:

* **Lazy reads.** Well arrays are handed to napari as dask arrays backed by the
  zarr store. Only the chunks intersecting the current viewport, at the current
  pyramid level, are decompressed.
* **Multi-well slider.** Loading several wells (or *All*) stacks them along a
  new leading axis; napari exposes it as a slider so you flip through wells one
  at a time at full resolution, zooming freely, with a metadata HUD per well.
* **Hot caches.** A small chunk cache keeps recently-decoded data resident so
  scrubbing the time/well slider and re-zooming stay smooth.

The practical upshot — the capability this whole subsystem exists for — is that
an entire stitched plate, with full-size wells, can be loaded and browsed
interactively from local disk. See
:doc:`omero-screen-napari/welldata_widget` for the click-by-click UI.


Live-cell timelapse
-------------------

A live-cell plate uses the **same per-plate store** — there is no separate
per-well cache. A timelapse simply has ``n_timepoints > 1``, so each well's
image array carries a real ``T`` axis and the napari time slider scrubs through
frames. To keep long multi-channel timelapses within a laptop's RAM during the
build, wells are stitched and written in blocks of timepoints
(``OMERO_SCREEN_CACHE_BLOCK``, default 4) rather than all at once.

For a **tracked** timelapse (``--stitch --track``; see :ref:`tracking`), a
``tracks.csv`` is written automatically next to each well's image group
(``<row>/<col>/tracks.csv``) when the cache is built, so the well is immediately
ready to open in Mastodon. See :ref:`mastodon-curation` for the curation
walkthrough.


Configuration
-------------

All cache behaviour is controlled by environment variables (set them in your
``.env.{ENV}`` file or the shell):

.. list-table::
   :widths: 38 14 48
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``OMERO_SCREEN_CACHE_PATH``
     - ``~/omero-cache``
     - Root directory for *all* caches. Put it on a fast SSD for best
       interactivity. ``~`` is expanded.
   * - ``OMERO_SCREEN_ZARR_MAX_GB``
     - ``100``
     - Size cap for the OME-Zarr cache, in GiB (integer; floored at 10). The
       least-recently-accessed plate is evicted when a new build would exceed
       the cap.
   * - ``OMERO_SCREEN_IMAGE_CACHE_SIZE_LIMIT``
     - ``20 GiB``
     - Size cap for the *legacy* per-field disk cache (raw bytes).
   * - ``OMERO_SCREEN_CACHE_BLOCK``
     - ``4``
     - Timepoints stitched/written per block during a build. Lower it
       (e.g. ``2``) for long live-cell wells on a 16 GB machine.
   * - ``OMERO_SCREEN_CACHE_WORKERS``
     - ``2``
     - Concurrent dask workers during a build.
   * - ``OMERO_SCREEN_ZARR_DISPLAY_CACHE_MB``
     - ``1024``
     - Chunk-cache size (MiB) layered on the zarr store at view time.
   * - ``OMERO_SCREEN_DASK_CACHE_MB``
     - ``2048``
     - Opportunistic cache (MiB) for decoded results, keeping data hot across
       slider scrubbing and re-zooms.
   * - ``OMERO_SCREEN_NAPARI_ASYNC``
     - ``1``
     - Enable napari async slicing (``0`` to disable).


Eviction and pinning
--------------------

The OME-Zarr cache is **bounded and self-managing**. A ``registry.json`` tracks
each plate's size and last-accessed time; when a new build would exceed
``OMERO_SCREEN_ZARR_MAX_GB`` the least-recently-accessed plate is evicted. If a
*single* plate is larger than the whole cap, the build refuses rather than
evicting everything — raise the cap.

A plate under active curation can span days (a separate Fiji/Mastodon session),
so eviction would be disruptive. **Pin** a plate from the napari
:doc:`omero-screen-napari/tracks_widget` to exempt it (the pin persists across
restarts), and **Unpin** it when you are done. Pinning is deliberately manual:
caching a plate does not mean you are curating it.

.. note::

   The default location moved from the old hidden ``~/.cache/omero_screen`` to
   the visible ``~/omero-cache``. On the first run after upgrading, if the old
   cache exists and the new one does not, a one-time log line points this out;
   rebuild plates in the new location (the old folder can then be deleted), or
   set ``OMERO_SCREEN_CACHE_PATH`` to keep using the old path.

.. tip::

   The cache is always safe to delete — ``rm -rf ~/omero-cache`` — since every
   plate rebuilds on demand from OMERO. Do this if you suspect orphaned files
   after a crash.
