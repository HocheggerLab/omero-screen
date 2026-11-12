"""Cell-tracking logic for omero-screen, without a GUI or the pipeline.

Works on tracked timelapse plates stored in CellView: lineage repair on
PIP-FUCCI plates, the reviewer's edit log and its replay, debris, cell fates
and phases, and the review queue. The napari Track Review widget and the
``cellview`` CLI are front ends to this package.

Modules:

- ``repair``: rule-based lineage repair (fragments, mislinks, restarts).
- ``edit``: the append-only review edit log and its replay.
- ``cells``: a well's curated tracks, breaks and continuation candidates.
- ``fate`` / ``phases``: cell fates and PIP-FUCCI cell-cycle phases.
- ``debris``: debris classification.
- ``review_queue``: sampling cells for review.
- ``plate``: plate-level repair and curation over CellView.
- ``corrections``: stored track corrections.

Moved from ``cellview.tracks`` (#75); that path remains as an alias for one
release.
"""

__version__ = "0.10.3"
