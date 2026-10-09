"""Keep in-memory overlay layers in step with asynchronously loaded images.

With napari's async slicing on (``zarr_cache.display._ensure_async_slicing``),
image and label layers read a new frame from the zarr cache off the GUI
thread, and keep showing the old frame until it arrives. Layers held in memory
(tracks) are sliced at once, so during playback they ran ahead of the images:
the tracks for frame *t* appeared before the cells of frame *t* (#11).

:func:`hold_until_loaded` makes such a layer follow the images rather than the
time slider. While an image is loading, a time change leaves the layer where it
is; each time an image layer draws a frame, the layer is moved to that frame.
During fast playback the slider runs ahead of what is shown, so following the
slider would always be wrong; following the drawn frame is right at any speed.
With async slicing off, or nothing loading, the layer updates at once.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def is_async(layer: Any) -> bool:
    """Whether napari slices ``layer`` off the GUI thread (image, labels, …)."""
    state = getattr(layer, "_slicing_state", None)
    return callable(getattr(state, "_make_slice_request", None))


def displayed_point(layer: Any) -> tuple[float, ...] | None:
    """The world point of the frame ``layer`` is drawing, if napari exposes it.

    For async layers, ``_slice_input`` is replaced only when a loaded slice is
    applied, so it names the frame on screen, not the one requested.
    """
    try:
        return tuple(layer._slice_input.world_slice.point)
    except AttributeError:
        return None


class LayerHold:
    """Moves a layer only to frames that the viewer's images are drawing.

    Wraps the layer's ``_slice_dims`` (napari's per-layer slicing entry point)
    and is told by :meth:`on_drawn` when another layer has drawn a frame. The
    logic is plain Python so it can be tested without a GUI.
    """

    def __init__(self, viewer: Any, layer: Any) -> None:
        """Hold ``layer`` in ``viewer``."""
        self.viewer = viewer
        self.layer = layer
        self._original = layer._slice_dims
        layer._slice_dims = self._slice_dims

    def _sources(self) -> list[Any]:
        return [
            other
            for other in self.viewer.layers
            if other is not self.layer
            and getattr(other, "visible", True)
            and is_async(other)
        ]

    def loading(self) -> bool:
        """Whether a visible async layer is still loading a frame."""
        return any(not getattr(s, "loaded", True) for s in self._sources())

    def _slice_dims(self, dims: Any, force: bool = False) -> None:
        # A time change while the images load: wait for them to draw it.
        if not self.loading():
            self._original(dims=dims, force=force)

    def on_drawn(self, source: Any) -> None:
        """Move the held layer to the frame ``source`` has just drawn."""
        if source is self.layer or not getattr(source, "visible", True):
            return
        point = displayed_point(source)
        dims = self.viewer.dims
        if point is not None and hasattr(dims, "model_copy"):
            dims = dims.model_copy()
            full = list(dims.point)
            n = min(len(point), len(full))
            # Layers with fewer dimensions are aligned to the last axes.
            full[len(full) - n :] = point[len(point) - n :]
            dims.point = tuple(full)
        self._original(dims=dims, force=False)

    def detach(self) -> None:
        """Restore the layer's own slicing."""
        self.layer._slice_dims = self._original


def hold_until_loaded(viewer: Any, layer: Any) -> LayerHold | None:
    """Make ``layer`` show the frame that the viewer's images are showing.

    Args:
        viewer: The napari viewer.
        layer: An in-memory layer (tracks) added to ``viewer``.

    Returns:
        The hold, or ``None`` if this napari has no per-layer slicing hook to
        wrap (the layer then behaves as before).
    """
    # Plugin widgets get napari's PublicOnlyProxy, which warns on every
    # private attribute access; the hold works on the objects themselves.
    viewer = getattr(viewer, "__wrapped__", viewer)
    layer = getattr(layer, "__wrapped__", layer)
    if not callable(getattr(layer, "_slice_dims", None)):
        logger.debug("napari has no Layer._slice_dims; layer sync disabled")
        return None
    hold = LayerHold(viewer, layer)
    connected: dict[int, tuple[Any, Any]] = {}

    def _connect(other: Any) -> None:
        if other is layer or not is_async(other) or id(other) in connected:
            return

        def _drawn(event: Any) -> None:
            hold.on_drawn(other)

        other.events.set_data.connect(_drawn)
        connected[id(other)] = (other, _drawn)

    def _on_inserted(event: Any) -> None:
        _connect(event.value)

    def _on_removed(event: Any) -> None:
        if event.value is layer:
            hold.detach()
            for other, callback in connected.values():
                other.events.set_data.disconnect(callback)
            connected.clear()
            viewer.layers.events.inserted.disconnect(_on_inserted)
            viewer.layers.events.removed.disconnect(_on_removed)
        elif (entry := connected.pop(id(event.value), None)) is not None:
            entry[0].events.set_data.disconnect(entry[1])

    for other in viewer.layers:
        _connect(other)
    viewer.layers.events.inserted.connect(_on_inserted)
    viewer.layers.events.removed.connect(_on_removed)
    # napari holds its callbacks weakly; keep them alive with the layer.
    layer._omero_screen_hold = (hold, _on_inserted, _on_removed, connected)
    return hold
