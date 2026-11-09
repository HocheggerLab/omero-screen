"""Deprecated alias of :mod:`omero_screen_track` (#75).

The tracking logic moved to the ``omero-screen-track`` package. Importing
``cellview.tracks.<module>`` still works for one release and returns the very
same module as ``omero_screen_track.<module>``; it warns so callers can move.
"""

import importlib
import sys
import warnings

_MODULES = (
    "cells",
    "corrections",
    "debris",
    "edit",
    "fate",
    "phases",
    "plate",
    "repair",
    "review_queue",
)

warnings.warn(
    "cellview.tracks moved to omero_screen_track; import from there. "
    "The alias will be removed in a later release.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in _MODULES:
    _module = importlib.import_module(f"omero_screen_track.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
