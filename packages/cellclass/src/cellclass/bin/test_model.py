#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""Evaluate a trained cellclass model on a held-out test split."""

import argparse

from cellclass.models import Model


def run(args: argparse.Namespace) -> None:
    """Run model evaluation and print a classification report."""
    import logging
    import os
    import time

    import numpy as np
    import torch
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import Compose, Resize, ToDtype

    from cellclass._logging import configure_logging
    from cellclass.datasets import ROIDataset
    from cellclass.models import create_model
    from cellclass.testing import test_epoch

    configure_logging(args.log_level)

    start_time = time.time()

    # Load data data
    data = np.load(args.input)
    X = data["X"]
    y_names = data["y_names"]
    size = len(X)
    logging.info(
        f"Data: {args.input} : {size} images: {X[0].shape}, {X[0].dtype}"
    )
    labels, y, counts = np.unique(
        y_names, return_inverse=True, return_counts=True
    )

    if args.reorder:
        reorder = np.array(args.reorder)
        if len(reorder) != len(labels):
            logging.error(
                f"Incorrect dimensions for re-order of {len(labels)}"
            )
            exit(1)
        labels = labels[reorder]
        y = reorder[y]
        counts = counts[reorder]

    # Note: This data division matches that performed during training.
    # It should allow the metrics to recomputed for a model using the
    # same data.
    # Reduce dataset size
    if args.size > 0 and args.size < size:
        logging.info(f"Data subset: {args.size} images")
        rng = np.random.default_rng(seed=args.data_seed)
        idx = rng.choice(len(X), args.size, replace=False)
    else:
        idx = np.arange(size)

    # Testing data
    _, tt = train_test_split(
        idx, test_size=args.testing_size, random_state=args.data_seed
    )

    logging.info(f"Test size {len(tt)}")

    # Create model
    device = torch.device(args.device)
    num_channels = X[0].shape[0]  # Assume CYX format
    num_classes = len(labels)

    # Recommended option is to load the scripted model with its sidecar metadata JSON.
    # This does not require knowing the model class and network architecture.
    input_shape = None
    if args.script:
        name, _ = os.path.splitext(args.script)
        meta_file = name + ".json"
        if not os.path.exists(meta_file):
            logging.error(f"Missing script metadata: {meta_file}")
            exit(1)
        with open(meta_file) as f:
            import json

            meta = json.load(f)
        # Validate number of channels, classes and input_shape
        model_num_channels = len(meta.get("channels", []))
        if num_channels != model_num_channels:
            logging.error(
                f"Incorrect number of channels {num_channels} != {model_num_channels} in: {meta_file}"
            )
            exit(1)
        model_num_classes = len(meta.get("labels", []))
        if num_classes > model_num_classes:
            logging.error(
                f"Incorrect number of classes {num_classes} > {model_num_classes} in: {meta_file}"
            )
            exit(1)
        input_shape = meta.get("input_shape", [])
        if len(input_shape) != 3:
            logging.error(f'Missing metadata "input_shape" in: {meta_file}')
            exit(1)
        input_shape = tuple(input_shape[-2:])
        # Load the scripted model
        model = torch.jit.load(args.script)  # type: ignore[no-untyped-call]
    else:
        # Create from the supported set of models
        model = create_model(
            args.model, num_classes, num_channels=num_channels
        )
        if model is None:
            logging.error(f"Unsupported model: {args.model}")
            exit(1)
        m = Model(args.model)
        input_shape = (m.size, m.size)

        # Use existing checkpoint
        if args.name:
            checkpoint = torch.load(
                args.name, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(f"Loaded checkpoint: {args.name}")
        elif args.weights:
            state = torch.load(
                args.weights, map_location=device, weights_only=True
            )
            model.load_state_dict(state)
            logging.info(f"Loaded weights: {args.weights}")
        else:
            logging.error("Unable to load model")
            exit(1)

    model = model.to(device)
    model.eval()

    # Transformation for the network
    trans_val = Compose(
        [ToDtype(torch.float32, scale=True), Resize(input_shape)]
    )

    pin_memory = args.pin_memory and device.type == "cuda"
    pin_memory_device = args.device if pin_memory else ""
    testing_loader = DataLoader(
        ROIDataset(X[tt], y[tt], transform=trans_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        shuffle=True,
    )

    logging.info("Testing...")

    _yt, _yp = test_epoch(model, testing_loader, device)
    y_true, y_pred = _yt.numpy(), _yp.numpy()

    # Small data sizes may not observe all the classes so pass in the label indices
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=np.nan,
    )
    logging.info("Classification report\n%s", report)

    t = time.time() - start_time
    logging.info(f"Done (in {t:.6g} seconds)")

    if args.matrix_csv:
        import csv

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        with open(args.matrix_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + list(labels))
            for label, row in zip(labels, cm, strict=False):
                writer.writerow([label] + list(row))
        logging.info(f"Confusion matrix saved to {args.matrix_csv}")

    if args.plot_matrix or args.matrix_file:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=labels
        )
        if args.matrix_file:
            plt.savefig(args.matrix_file)
        if args.plot_matrix:
            plt.show()


def main() -> None:
    """Entry point for direct execution of the test command."""
    from cellclass.cli import test_command

    test_command.main(prog_name="cellclass-test")


if __name__ == "__main__":
    main()
