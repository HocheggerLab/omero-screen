"""Keep in-memory overlay layers in step with asynchronously loaded images.

With napari's async slicing on (``zarr_cache.display._ensure_async_slicing``),
image and label layers read a new frame from the zarr cache off the GUI
thread, and keep showing the old frame until it arrives. Layers held in memory
(tracks, points) are sliced at once, so during playback they run ahead of the
images: the tracks for frame *t* appear before the cells of frame *t* (#11).

:func:`hold_until_loaded` makes such a layer wait: when the time changes while
any visible async layer is still loading, the layer keeps its old frame, and
moves to the viewer's current frame as soon as every one of them has loaded.
With async slicing off, or nothing loading, it updates at once as before.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from loguru import logger

#: How often a held layer checks whether the images have loaded (ms).
POLL_MS = 15
#: Release a held layer after this long even if a layer never loads (s).
MAX_HOLD_S = 2.0


class LayerHold:
    """Defers a layer's slicing while other layers of the viewer are loading.

    Wraps the layer's ``_slice_dims`` (napari's per-layer slicing entry
    point). The Qt timer is created by :func:`hold_until_loaded`; the logic
    here is plain Python so it can be tested without a GUI.
    """

    def __init__(
        self,
        viewer: Any,
        layer: Any,
        start_timer: Callable[[], None],
        stop_timer: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Hold ``layer`` in ``viewer``; ``start/stop_timer`` drive :meth:`poll`."""
        self.viewer = viewer
        self.layer = layer
        self._original = layer._slice_dims
        self._start_timer = start_timer
        self._stop_timer = stop_timer
        self._clock = clock
        self._pending = False
        self._since = 0.0
        layer._slice_dims = self._slice_dims

    def loading(self) -> bool:
        """Whether a visible layer other than the held one is still loading."""
        return any(
            not getattr(other, "loaded", True)
            for other in self.viewer.layers
            if other is not self.layer and getattr(other, "visible", True)
        )

    def _slice_dims(self, dims: Any, force: bool = False) -> None:
        if not self.loading():
            self._release_pending()
            self._original(dims=dims, force=force)
            return
        if not self._pending:
            self._since = self._clock()
        self._pending = True
        self._start_timer()

    def poll(self) -> None:
        """Slice the held layer once nothing is loading (or after the timeout)."""
        if not self._pending:
            self._stop_timer()
            return
        timed_out = self._clock() - self._since > MAX_HOLD_S
        if timed_out:
            logger.debug(f"{self.layer.name}: released after {MAX_HOLD_S} s")
        if timed_out or not self.loading():
            self._release_pending()
            self._original(dims=self.viewer.dims, force=False)

    def _release_pending(self) -> None:
        self._pending = False
        self._stop_timer()

    def detach(self) -> None:
        """Restore the layer's own slicing."""
        self._stop_timer()
        self.layer._slice_dims = self._original


def hold_until_loaded(viewer: Any, layer: Any) -> LayerHold | None:
    """Make ``layer`` follow the time only once the viewer's images have loaded.

    Args:
        viewer: The napari viewer.
        layer: An in-memory layer (tracks, points) added to ``viewer``.

    Returns:
        The hold, or ``None`` if this napari has no per-layer slicing hook to
        wrap (the layer then behaves as before).
    """
    if not callable(getattr(layer, "_slice_dims", None)):
        logger.debug("napari has no Layer._slice_dims; layer sync disabled")
        return None
    from qtpy.QtCore import QTimer

    timer = QTimer()
    timer.setInterval(POLL_MS)
    hold = LayerHold(viewer, layer, timer.start, timer.stop)
    timer.timeout.connect(hold.poll)
    # The hold (and its timer) live as long as the layer.
    layer._omero_screen_hold = (hold, timer)

    def _on_removed(event: Any) -> None:
        if event.value is layer:
            hold.detach()
            viewer.layers.events.removed.disconnect(_on_removed)

    viewer.layers.events.removed.connect(_on_removed)
    return hold
