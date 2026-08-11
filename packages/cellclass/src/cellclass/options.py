"""Lightweight enums shared by the CellClass CLI and model applications."""

from __future__ import annotations

from enum import Enum, StrEnum


class Model(Enum):
    """Supported classification model architectures."""

    def __new__(cls, *args: object, **kwds: object) -> Model:
        """Create an enum member, storing an optional input size."""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, size: int = 224) -> None:
        """Store the expected input image size."""
        self._size_ = size

    densenet121 = "densenet121"
    densenet161 = "densenet161"
    densenet169 = "densenet169"
    densenet201 = "densenet201"
    squeezenet1_0 = "squeezenet1_0"
    shufflenet2x1_0 = "shufflenet2x1_0"
    shufflenet2x1_5 = "shufflenet2x1_5"
    shufflenet2x2_0 = "shufflenet2x2_0"
    efficientnetb3 = "efficientnetb3", 300
    efficientnetb4 = "efficientnetb4", 380
    efficientnetb3s = "efficientnetb3s", 300
    efficientnetb4s = "efficientnetb4s", 380
    resenext50_32x4d = "resenext50_32x4d"
    alexnet = "alexnet"
    alexnet_att = "alexnet_att"
    googlenet = "googlenet"
    googlenet_att = "googlenet_att"
    simplegooglenet = "simplegooglenet"
    simplegooglenet_att = "simplegooglenet_att"
    simplegooglenet_ch_att = "simplegooglenet_ch_att"
    simplegooglenet_ch_sp_att = "simplegooglenet_ch_sp_att"

    def __str__(self) -> str:
        """Return the command-line value."""
        return str(self.value)

    @property
    def size(self) -> int:
        """Return the expected square input size in pixels."""
        return int(self._size_)  # type: ignore[attr-defined]


class Existing(StrEnum):
    """Behaviour when a checkpoint file already exists."""

    overwrite = "overwrite"
    load = "load"
    error = "error"


class LossFunction(StrEnum):
    """Supported loss functions for training."""

    focal_loss = "focal_loss"
    cross_entropy = "cross_entropy"


class LrScheduler(StrEnum):
    """Supported learning-rate schedulers."""

    step = "step"
    plateau = "plateau"
