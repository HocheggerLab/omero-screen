"""Every plugin widget must actually construct.

magicgui resolves a widget function's annotations at **construction** time, not
at import, so a signature it cannot resolve passes every import-level check and
then fails the moment the user opens the panel. That is how
``from __future__ import annotations`` slipped into ``_phase_montage_widget``:
it turns annotations into strings, magicgui hands ``"Path"`` to ``ForwardRef``,
and the panel dies with ``No module named 'Path'``.

Hence a test per widget that builds the thing. It is cheap, and it is the only
level at which this class of error is visible — note that ruff and mypy are both
*happier* with the future import, so neither will ever warn about it.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance exists for Qt widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _factories():
    """(name, callable) for every widget in the napari manifest."""
    from omero_screen_napari._aligned_plate_widget import (
        aligned_plate_widget_gui,
    )
    from omero_screen_napari._gallery_widget import gallery_gui_widget
    from omero_screen_napari._phase_montage_widget import (
        phase_montage_widget_gui,
    )
    from omero_screen_napari._tracks_widget import tracks_gui_widget
    from omero_screen_napari._welldata_widget import well_widget_combined

    return [
        ("well_widget_combined", well_widget_combined),
        ("aligned_plate_widget", aligned_plate_widget_gui),
        ("gallery_gui_widget", gallery_gui_widget),
        ("tracks_gui_widget", tracks_gui_widget),
        ("phase_montage_widget", phase_montage_widget_gui),
    ]


@pytest.mark.parametrize("name", [n for n, _ in _factories()])
def test_widget_constructs(qapp, name: str) -> None:
    """Building the widget must not raise — this is what opening the panel does."""
    factory = dict(_factories())[name]
    with patch(
        "omero_screen_napari._logging.init_plugin_logging", MagicMock()
    ):
        widget = factory()
    assert widget is not None


def test_manifest_widgets_all_importable() -> None:
    """Every python_name in napari.yaml resolves to a real callable.

    A widget renamed in the source but not in the manifest fails only when napari
    reads the manifest, which no other test does.
    """
    import importlib
    from pathlib import Path

    import yaml

    import omero_screen_napari

    manifest = (
        Path(omero_screen_napari.__file__).parent / "napari.yaml"
    ).read_text()
    contributions = yaml.safe_load(manifest)["contributions"]

    commands = {c["id"]: c["python_name"] for c in contributions["commands"]}
    for widget in contributions["widgets"]:
        python_name = commands[widget["command"]]
        module_name, _, attr = python_name.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), (
            f"{python_name} is not callable"
        )


def test_phase_montage_fields(qapp) -> None:
    """The montage widget exposes the fields the export needs.

    Guards the specific failure that prompted these tests: ``output_dir`` must
    resolve to a real path-picking widget, not an unresolved ForwardRef.
    """
    from pathlib import Path

    from omero_screen_napari._phase_montage_widget import (
        phase_montage_widget_gui,
    )

    with patch(
        "omero_screen_napari._logging.init_plugin_logging", MagicMock()
    ):
        container = phase_montage_widget_gui()
    widget = container[0]
    names = {field.name for field in widget}
    assert {
        "output_dir",
        "plate_id",
        "well",
        "cells_per_phase",
        "seed",
        "include_subg1",
        "mask",
    } <= names
    assert isinstance(widget.output_dir.value, Path)
    assert list(widget.mask.choices) == ["cells", "nuclei"]
