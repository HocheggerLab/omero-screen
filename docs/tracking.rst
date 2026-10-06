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
        │  overlay tracks, inspect lineages (arboretum), export a CTC bundle
        ▼
   Mastodon (Fiji)                         (4) manual curation
        │  open the cached OME-Zarr well, import the CTC bundle, correct by hand
        ▼
   reconcile into CellView                 (5) round-trip (planned)
        corrected res_track.txt → curated track_id / parent_track_id


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
plugin), export one track's full time-course as CSV, or export the well as a
**CTC bundle for Mastodon** (the *Export well for Mastodon (CTC)* button).
Pinning protects a plate from cache eviction while you curate.

See :doc:`omero-screen-napari/tracks_widget` for the step-by-step guide.


.. _mastodon-curation:

4. Curate in Mastodon
---------------------

Automated tracking is never perfect, so tracks are hand-corrected in Mastodon, a
mature Fiji track-editing tool.

.. important::

   **Use the CTC importer, not the CSV importer.** Mastodon's plain CSV importer
   creates spots but **no links**, so it silently drops every track and lineage
   — and its label-image importer cannot create *division* links. Only the
   `Cell Tracking Challenge <https://celltrackingchallenge.net/>`_ importer
   rebuilds full lineages (divisions included), because it reads the parent
   relationships from ``res_track.txt``. That is why OmeroScreen exports a CTC
   bundle.

**Export the bundle.** In the Tracks Widget, click **Export well for Mastodon
(CTC)**. This writes ``~/mastodon_exports/plate_<id>_<well>_ctc/`` containing:

* ``mask000.tif`` … ``mask<T-1>.tif`` — one nucleus label image per frame, read
  straight from the cached OME-Zarr (no OMERO round-trip). Each cell's pixel
  value is its CTC track label.
* ``res_track.txt`` — the lineage table, four integers per track ``L B E P``
  (label, begin frame, end frame, parent label; ``0`` = founder). Labels are
  renumbered ``1..N`` in begin-frame order, and the masks use the same labels.
* ``manifest.json`` — maps each CTC label back to the original CellView
  ``track_id`` plus per-frame centroids, for the round-trip below. Keep it; do
  not edit it.
* ``README.txt`` — the exact import steps with the paths filled in.

The cache (``~/omero-cache`` by default, override with ``OMERO_SCREEN_CACHE_PATH``)
is size-bounded and evicts least-recently-used plates — **Pin** a plate in the
Tracks Widget before a long curation session so it is not reclaimed, and
**Unpin** it when finished. See :doc:`caching` for the cache layout and the
eviction/pinning rules.

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

**Import the tracks.** In the main Mastodon window: **File → Import → Import
from CellTrackingChallenge**. Choose the exported bundle folder, set the
filename pattern to ``mask%03d.tif`` (3-digit, 0-based), and Mastodon reads
``res_track.txt`` for the parent / division links.

**Link the views.** Click the same group-lock number (e.g. ``1``) in *both* the
BigDataViewer and TrackScheme windows, then double-click a spot to navigate
between the views. Save your work as a ``.mastodon`` project in a stable
location **outside** the cache so it is not lost if the plate is later evicted.


.. _ctc-roundtrip:

5. Reconcile corrections into CellView (planned)
------------------------------------------------

When done, export the corrected tracks back to CTC (**File → Export →
CellTrackingChallenge**). CellView already separates the frozen Trackastra
output (``track_id_raw`` / ``parent_track_id_raw``) from the curated "current
best" (``track_id`` / ``parent_track_id``), so the plan is to map each corrected
spot back onto its CellView row — by per-frame centroid, using ``manifest.json``
as the anchor — and write the corrections into the curated columns, leaving the
``*_raw`` columns as the audit trail.

.. note::

   This reverse trip is **not yet implemented**. For now the corrected CTC
   export (kept with its ``manifest.json``) and the ``.mastodon`` project are
   the record of corrections.


6. Repair, review and curate in napari (PIP-FUCCI plates)
---------------------------------------------------------

On plates with a geminin channel (PIP-FUCCI), tracks can be repaired
automatically and then checked by a reviewer, with an agent's help, inside
napari. Nothing edits the tracker's output: corrections are entries in an
edit log, and curated tracks are rebuilt from the automatic repair plus that
log.

**Automatic repair.** Cellpose often splits one nucleus into pieces, so the
tracker reports divisions that never happened. The repair accepts a division
only when both daughters lose geminin (it is degraded at anaphase), merges
touching pieces back, detaches mislinked neighbours and re-joins restarts:

.. code-block:: bash

   cellview repair-tracks 5054 --dry-run      # report only
   cellview repair-tracks 5054                # write track_id / parent_track_id

**Review queue.** Cells present at a start frame are followed to a stop
frame. A random sample is drawn per well; the cells the walker flags (lost,
gaps, fragments, phases out of order, rare outcomes) and an audit fraction of
unflagged ones are queued:

.. code-block:: bash

   cellview review-queue 5054 --well C2 --well C3 --sample 50 \
       --start 72 --stop 168 --audit 0.2 --seed 1 --out review/

**Filmstrips.** A crop window follows the cell through every frame, with its
PIP, geminin and area trace underneath:

.. code-block:: bash

   omero-screen-images --env production track 5054 C2-t72-L524
   omero-screen-images --env production track 5054 --queue review/queue.json --log review/edits.jsonl

**Track Review widget.** Open *Plugins → Track Review Widget* and load
``review/queue.json``. For each cell you can:

* link it to its continuation (click it, or press Shift-1…9 for a numbered
  candidate);
* unlink it, or add or remove a nucleus;
* split a nucleus or draw a missed one;
* mark mitosis, death or slippage, or exclude the cell;
* give a verdict.

*Next break* walks along the track and *Follow cell* keeps it centred.
Changes go to ``review/edits.jsonl`` and verdicts to ``review/decisions.json``.

**Working with an agent.** With napari-mcp (currently the fork
``HocheggerLab/napari-mcp@plugin-tools``), omero-screen adds review tools to
the agent: go to a cell, show its filmstrip, rank candidates, and propose an
edit. A proposal appears in the widget and is applied only when the reviewer
confirms it:

.. code-block:: bash

   uv run --with "napari-mcp @ git+https://github.com/HocheggerLab/napari-mcp@plugin-tools" napari
   claude mcp add --transport http napari http://127.0.0.1:9999/mcp

**Edit log on the command line.**

.. code-block:: bash

   cellview curate show review/edits.jsonl
   cellview curate add review/edits.jsonl link --plate 5054 --well C2 --cell 72:524 --frame 156 --label 3132
   cellview curate undo review/edits.jsonl --plate 5054 --well C2
   cellview curate replay review/edits.jsonl --plate 5054 --write
