"""cellclass: cell image classification package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cellclass.datasets import ROIDataset
    from cellclass.models import create_model
    from cellclass.options import Model
    from cellclass.testing import test_epoch

__all__ = [
    "ROIDataset",
    "Model",
    "create_model",
    "test_epoch",
    "prepare_dataset",
]

_LAZY_EXPORTS = {
    "ROIDataset": ("cellclass.datasets", "ROIDataset"),
    "Model": ("cellclass.options", "Model"),
    "create_model": ("cellclass.models", "create_model"),
    "test_epoch": ("cellclass.testing", "test_epoch"),
}


def __getattr__(name: str) -> Any:
    """Load Torch-dependent public exports only when first requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))


def prepare_dataset(
    data_dir: str,
    output: str,
    channels: list[str] | None = None,
    ignore: list[str] | None = None,
    single_label: bool = True,
) -> None:
    """Prepare an .npz dataset from a directory of .npy cell image files.

    Args:
        data_dir: Directory containing .npy files and optionally metadata.json.
        output: Full path for the output .npz file (e.g. '/path/to/rois.npz').
        channels: Channel names. If None, read from metadata.json in data_dir.
        ignore: Labels to exclude (default: ['unassigned']).
        single_label: Skip masks with more than one label (default: True).

    """
    import argparse
    import os

    from cellclass.bin.create_dataset import run

    out_dir = os.path.dirname(output) or data_dir
    name = os.path.basename(output)
    if name.endswith(".npz"):
        name = name[:-4]

    args = argparse.Namespace(
        dir=data_dir,
        out=out_dir if out_dir != data_dir else None,
        name=name,
        channels=channels,
        ignore=ignore if ignore is not None else ["unassigned"],
        duplicates=False,
        single_label=single_label,
    )
    run(args)
