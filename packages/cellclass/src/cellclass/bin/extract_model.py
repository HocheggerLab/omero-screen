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

"""Extract a TorchScript model and JSON sidecar from a training checkpoint."""

import argparse

from cellclass.bin_utils import file_path
from cellclass.models import Model


def run(args: argparse.Namespace) -> None:
    """Extract the best checkpoint to a TorchScript .pt file with metadata sidecar."""
    import json
    import logging
    import os

    import numpy as np
    import torch

    from cellclass._logging import configure_logging
    from cellclass.models import create_model

    configure_logging(logging.INFO)

    device = torch.device("cpu")

    # Load the training settings
    logging.info(f"Loading training settings: {args.file}")
    with open(args.file) as f:
        training = json.load(f)

    # Load the (best) checkpoint
    filename = training["name"]
    if os.path.exists(filename + ".best"):
        filename += ".best"
    logging.info(f"Loading checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Model metadata
    d = {}

    for k in [
        "epoch",
        "train_loss",
        "train_acc",
        "loss",
        "acc",
        "precision",
        "recall",
        "f1",
        "best_f1",
        "labels",
        "model",
        "channels",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_acc",
    ]:
        if k in checkpoint:
            d[k] = checkpoint[k]
    import contextlib

    for k in ["labels", "channels"]:
        with contextlib.suppress(Exception):
            d[k] = list(d[k])

    s = json.dumps(d, indent=2)
    logging.info(f"Model metadata: {s}")

    if not args.save:
        return

    # Overrides
    if args.model:
        d["model"] = args.model
    if args.channels:
        d["channels"] = args.channels
    if args.labels:
        d["labels"] = args.labels

    logging.info("Collecting model metadata")
    m = d.get("model")
    c = len(d.get("channels", []))
    num_labels = len(d.get("labels", []))

    if m is None:
        logging.error("Unknown model type")
        exit(1)
    if c == 0:
        logging.error("Unknown model input channels")
        exit(1)
    if num_labels == 0:
        logging.error("Unknown model output classes")
        exit(1)

    data = np.load(training["input"])
    img_shape = data["X"][0].shape
    del data
    logging.info(f"Image shape: {img_shape}")

    name = args.name
    if not name:
        name = f"{m}_c{c}_l{num_labels}"

    name, _ = os.path.splitext(name)
    name_pt = name + ".pt"
    name_json = name + ".json"

    if not args.overwrite:
        if os.path.exists(name_pt):
            logging.error(f"Output model file exists: {name_pt}")
            exit(1)
        if os.path.exists(name_json):
            logging.error(f"Output model file exists: {name_json}")
            exit(1)

    logging.info(f"Creating model: {m}")
    model = create_model(m, num_labels, num_channels=c, weights=None)
    if model is None:
        logging.error(f"Failed to create model: {m}")
        exit(1)
    model.load_state_dict(checkpoint["model_state_dict"])

    m = Model(m)
    input_shape = (c, m.size, m.size)
    logging.info(f"Input shape: {input_shape}")

    logging.info(f"Saving model script: {name_pt}")
    script = torch.jit.script(model)
    script.save(name_pt)  # type: ignore[no-untyped-call]
    # load using:
    # model = torch.jit.load('model_scripted.pt')
    # model.eval()

    # Save size of training data tiles and size of model input
    logging.info(f"Saving model metadata: {name_json}")
    d["img_shape"] = img_shape
    d["input_shape"] = input_shape
    with open(name_json, "w") as f:
        json.dump(d, f)


def main() -> None:
    """Entry point for cellclass-extract CLI."""
    parser = argparse.ArgumentParser(
        description="Program to extract model metadata from a training settings file "
        "and optionally save to a model state file."
    )

    parser.add_argument(
        "file",
        metavar="FILE",
        type=file_path,
        help="Training settings file (JSON)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Model file prefix (default uses model metadata)",
    )
    parser.add_argument(
        "--save",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save model files",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing model files",
    )

    group = parser.add_argument_group("Settings Overrides")
    group.add_argument(
        "--model",
        type=Model,
        choices=list(Model),
        help="Model name (overrides settings metadata)",
    )
    group.add_argument(
        "--channels",
        nargs="+",
        type=str,
        help="Input channels (overrides settings metadata)",
    )
    group.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help="Class labels (overrides settings metadata)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
