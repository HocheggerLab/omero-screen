"""cellclass: cell image classification package."""

from cellclass.datasets import ROIDataset
from cellclass.models import Model, create_model
from cellclass.testing import test_epoch

__all__ = [
    "ROIDataset",
    "Model",
    "create_model",
    "test_epoch",
    "prepare_dataset",
]


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
