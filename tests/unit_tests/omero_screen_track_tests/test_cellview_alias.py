"""``cellview.tracks`` stays importable for one release after the move (#75)."""

import importlib
import sys

import pytest


def test_alias_returns_the_same_modules() -> None:
    """Old and new import paths give the very same module objects."""
    for name in [m for m in list(sys.modules) if m.startswith("cellview.tracks")]:
        del sys.modules[name]
    with pytest.warns(DeprecationWarning, match="omero_screen_track"):
        importlib.import_module("cellview.tracks")
    from cellview.tracks.edit import EditLog as old_edit_log
    from omero_screen_track.edit import EditLog

    import cellview.tracks.repair as old_repair
    import omero_screen_track.repair as repair

    assert old_edit_log is EditLog
    assert old_repair is repair
