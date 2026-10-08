"""Tracks wait for asynchronously loaded images (#11)."""

from dataclasses import dataclass, field
from typing import Any

from omero_screen_napari import layer_sync
from omero_screen_napari.layer_sync import LayerHold


@dataclass
class FakeLayer:
    """A layer that records what it was sliced to."""

    name: str
    loaded: bool = True
    visible: bool = True
    sliced: list[Any] = field(default_factory=list)

    def _slice_dims(self, dims: Any, force: bool = False) -> None:
        self.sliced.append(dims)


@dataclass
class FakeViewer:
    """Layers and the current dims."""

    layers: list[FakeLayer]
    dims: Any = "current"


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The current time."""
        return self.now


def make_hold() -> tuple[
    FakeViewer, FakeLayer, FakeLayer, LayerHold, list[str]
]:
    """An image and a held tracks layer; the timer calls are logged."""
    image = FakeLayer("image")
    tracks = FakeLayer("tracks")
    viewer = FakeViewer([image, tracks])
    timer: list[str] = []
    hold = LayerHold(
        viewer,
        tracks,
        lambda: timer.append("start"),
        lambda: timer.append("stop"),
        clock=Clock(),
    )
    return viewer, image, tracks, hold, timer


def test_slices_at_once_when_nothing_loads() -> None:
    """No layer loading: the tracks update as before."""
    _, _, tracks, _, timer = make_hold()
    tracks._slice_dims("t1")
    assert tracks.sliced == ["t1"]
    assert "start" not in timer


def test_waits_for_the_image_then_takes_the_current_frame() -> None:
    """Held while the image loads, then moved to the viewer's frame."""
    viewer, image, tracks, hold, timer = make_hold()
    image.loaded = False
    tracks._slice_dims("t1")
    tracks._slice_dims("t2")
    assert tracks.sliced == []
    assert timer[-1] == "start"
    hold.poll()
    assert tracks.sliced == []
    image.loaded = True
    hold.poll()
    assert tracks.sliced == ["current"]
    assert timer[-1] == "stop"


def test_hidden_layers_are_not_waited_for() -> None:
    """A hidden image does not hold the tracks."""
    _, image, tracks, _, _ = make_hold()
    image.loaded, image.visible = False, False
    tracks._slice_dims("t1")
    assert tracks.sliced == ["t1"]


def test_released_after_the_timeout() -> None:
    """An image that never loads does not freeze the tracks."""
    _, image, tracks, hold, _ = make_hold()
    image.loaded = False
    tracks._slice_dims("t1")
    assert isinstance(hold._clock, Clock)
    hold._clock.now = layer_sync.MAX_HOLD_S + 0.1
    hold.poll()
    assert tracks.sliced == ["current"]


def test_detach_restores_slicing() -> None:
    """Detaching gives the layer its own slicing back."""
    _, image, tracks, hold, _ = make_hold()
    hold.detach()
    image.loaded = False
    tracks._slice_dims("t1")
    assert tracks.sliced == ["t1"]
