"""Tracks follow the frame the asynchronously loaded images draw (#11)."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from omero_screen_napari.layer_sync import LayerHold


class FakeDims:
    """Viewer dims with a copyable point."""

    def __init__(self, point: tuple[float, ...]) -> None:
        """Start at ``point``."""
        self.point = point

    def model_copy(self) -> "FakeDims":
        """A detached copy, as pydantic's ``model_copy``."""
        return FakeDims(self.point)


@dataclass
class FakeLayer:
    """A layer that records the points it was sliced to."""

    name: str
    asynchronous: bool = False
    loaded: bool = True
    visible: bool = True
    shown: tuple[float, ...] = (0.0, 0.0, 0.0)
    sliced: list[tuple[float, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Give async layers napari's async slicing hook and a drawn slice."""
        if self.asynchronous:
            self._slicing_state = SimpleNamespace(
                _make_slice_request=lambda d: d
            )

    @property
    def _slice_input(self) -> Any:
        return SimpleNamespace(world_slice=SimpleNamespace(point=self.shown))

    def _slice_dims(self, dims: Any, force: bool = False) -> None:
        self.sliced.append(dims.point)


def make_hold() -> tuple[SimpleNamespace, FakeLayer, FakeLayer, LayerHold]:
    """An async image and a held tracks layer at time 0."""
    image = FakeLayer("image", asynchronous=True)
    tracks = FakeLayer("tracks")
    viewer = SimpleNamespace(
        layers=[image, tracks], dims=FakeDims((0.0, 0.0, 0.0))
    )
    return viewer, image, tracks, LayerHold(viewer, tracks)


def test_slices_at_once_when_nothing_loads() -> None:
    """No layer loading: the tracks update as before."""
    viewer, _, tracks, _ = make_hold()
    viewer.dims.point = (4.0, 0.0, 0.0)
    tracks._slice_dims(viewer.dims)
    assert tracks.sliced == [(4.0, 0.0, 0.0)]


def test_follow_the_drawn_frame_not_the_slider() -> None:
    """During fast playback the tracks show what the image shows."""
    viewer, image, tracks, hold = make_hold()
    image.loaded = False
    viewer.dims.point = (5.0, 0.0, 0.0)
    tracks._slice_dims(viewer.dims)
    assert tracks.sliced == []
    # The image draws frame 3 while the slider has moved on to 5.
    image.shown = (3.0, 0.0, 0.0)
    hold.on_drawn(image)
    assert tracks.sliced == [(3.0, 0.0, 0.0)]
    assert viewer.dims.point == (5.0, 0.0, 0.0)


def test_lower_dimensional_source_is_aligned_to_the_last_axes() -> None:
    """A (t, y, x) image in a (c, t, y, x) viewer sets the time axis."""
    viewer, image, tracks, hold = make_hold()
    viewer.dims = FakeDims((1.0, 0.0, 0.0, 0.0))
    image.shown = (6.0, 0.0, 0.0)
    hold.on_drawn(image)
    assert tracks.sliced == [(1.0, 6.0, 0.0, 0.0)]


def test_hidden_and_sync_layers_are_ignored() -> None:
    """Only visible async layers hold or move the tracks."""
    viewer, image, tracks, hold = make_hold()
    image.loaded, image.visible = False, False
    other_tracks = FakeLayer("other tracks", loaded=False)
    viewer.layers.append(other_tracks)
    tracks._slice_dims(viewer.dims)
    assert tracks.sliced == [(0.0, 0.0, 0.0)]
    hold.on_drawn(image)
    assert tracks.sliced == [(0.0, 0.0, 0.0)]


def test_detach_restores_slicing() -> None:
    """Detaching gives the layer its own slicing back."""
    viewer, image, tracks, hold = make_hold()
    hold.detach()
    image.loaded = False
    tracks._slice_dims(viewer.dims)
    assert tracks.sliced == [(0.0, 0.0, 0.0)]
