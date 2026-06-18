"""Plugin-side logging activation for the napari widgets.

napari has no ``main()`` we control — the plugin's real entry points are the
widget factory functions invoked when the user opens a widget. Each calls
``init_plugin_logging()`` so logging is configured exactly once per session,
the first time any of our widgets is opened.
"""

from __future__ import annotations

from omero_screen.config import configure_logging_once


def init_plugin_logging() -> None:
    """Configure loguru for the napari plugin (idempotent).

    ``plugin=True`` leaves napari's own stdlib root logger untouched;
    ``console=True`` surfaces our logs in napari's built-in console. An earlier
    explicit ``configure_logging()`` (e.g. from a CLI launch) always wins.
    """
    configure_logging_once(plugin=True, console=True)
