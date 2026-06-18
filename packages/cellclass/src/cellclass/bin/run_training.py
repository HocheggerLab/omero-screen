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

"""Train a cellclass neural network model on an ROI dataset."""

import argparse
import json
from enum import StrEnum

from cellclass.bin_utils import file_path
from cellclass.models import Model


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


def none_or_str(value: str) -> str | None:
    """Return None if value is the string 'None', otherwise return value."""
    if value == "None":
        return None
    return str(value)


class StoppingCriteria:
    """Early-stopping monitor based on validation loss improvement."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        min_rel_delta: float = 1e-4,
    ) -> None:
        """Initialise StoppingCriteria.

        Args:
            patience: Epochs without improvement before stopping.
            min_delta: Minimum absolute improvement to count as progress.
            min_rel_delta: Minimum relative improvement to count as progress.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_rel_delta = min_rel_delta
        self.counter = 0
        self.min_value = 1e300

    def check(self, value: float) -> bool:
        """Return True if training should stop (no improvement for *patience* epochs).

        Args:
            value: Current validation loss.

        Returns:
            True when the stopping criterion is met.

        """
        obs = value + self.min_delta + self.min_rel_delta * self.min_value
        target = self.min_value
        if obs < target:
            # Improvement
            self.min_value = value
            self.counter = 0
        else:
            # No improvement. Check how many times this has happened.
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def run(args: argparse.Namespace) -> None:
    """Execute one complete training run, saving checkpoints and logging to W&B."""
    import logging
    import os
    import shutil
    import time

    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import Compose, Resize, ToDtype

    from cellclass.datasets import ROIDataset
    from cellclass.models import create_model
    from cellclass.testing import test_epoch
    from cellclass.training import FocalLoss, train_epoch

    if args.wandb:
        import wandb

    from cellclass._logging import configure_logging

    configure_logging(args.log_level)

    start_time = time.time()
    restart = hasattr(args, "restart")
    args.pid = os.getpid()

    # Weights and Biases support
    if args.wandb:
        has_id = hasattr(args, "wandb_id")
        if restart:
            if not has_id:
                raise Exception("Cannot restart without W&B id")
            wandb_id = args.wandb_id
            wandb.init(
                id=wandb_id,
                resume="must",
                entity=args.entity,
                project=args.project,
            )
            logging.info(f"Restarted wandb: {wandb_id}")
        else:
            wandb_id = args.wandb_id if has_id else wandb.util.generate_id()
            # Tag with the dataset (remove the default prefix and suffix)
            tags = args.tags
            tags.append(args.input)
            # Do this early to capture logging to W&B
            wandb.init(
                id=wandb_id,
                resume="allow",
                entity=args.entity,
                project=args.project,
                name=args.run_name,
                tags=tags,
                config=vars(args),
            )
            logging.info(f"Initialised wandb: {wandb_id}")
            args.wandb_id = wandb_id

    logging.info(f"Started process {args.pid}")

    if not restart:
        # Save training state
        with open(args.state, "w") as f:
            json.dump(vars(args), f, default=str)
        if args.wandb:
            # Save the state file required to restart the run
            wandb.save(args.state)

    # Load data data
    data = np.load(args.input)
    X = data["X"]
    y_names = data["y_names"]
    channels = data["channels"]
    size = len(X)
    logging.info(
        f"Training data: {args.input} : {size} images: {X[0].shape}, {X[0].dtype}"
    )
    labels, y, counts = np.unique(
        y_names, return_inverse=True, return_counts=True
    )

    # Reduce dataset size
    if args.size > 0 and args.size < size:
        logging.info(f"Training subset: {args.size} images")
        rng = np.random.default_rng(seed=args.data_seed)
        idx = rng.choice(len(X), args.size, replace=False)
    else:
        idx = np.arange(size)

    # Testing data
    t, tt = train_test_split(
        idx, test_size=args.testing_size, random_state=args.data_seed
    )
    # Training/Validation data
    t, v = train_test_split(
        t, test_size=args.validation_size, random_state=args.data_seed
    )
    logging.info(f"Size train {len(t)} : validation {len(v)} : test {len(tt)}")

    # Create model
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = create_model(
        args.model,
        len(labels),
        num_channels=len(channels),
        weights=args.weights,
        freeze_weights=args.freeze_weights,
        dropout=args.dropout,
    )
    if model is None:
        logging.error(f"Unsupported model: {args.model}")
        exit(1)

    model = model.to(device)

    # Create optimizer
    epoch = 0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        # betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )
    best_f1 = 0

    # Use existing checkpoint
    checkpoint_name = args.name
    if os.path.isfile(checkpoint_name):
        if args.existing == Existing.error:
            logging.error(f"Checkpoint exists: {checkpoint_name}")
            exit(1)
        if args.existing == Existing.load:
            checkpoint = torch.load(
                checkpoint_name, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Work-around for old tensor states not saved on the same device
            from cellclass.torch_utils import optimizer_to

            model = model.to(device)
            optimizer_to(optimizer, device)
            epoch = checkpoint["epoch"]
            best_f1 = checkpoint["best_f1"]
            model.train()
            logging.info(f"Loaded checkpoint: {checkpoint_name}")
        if args.existing == Existing.overwrite:
            logging.info(
                f"Existing checkpoint will be overwritten: {checkpoint_name}"
            )

    # Transformation for the network.
    # Make this the correct size for the model.
    trans = [
        ToDtype(torch.float32, scale=True),
        Resize((args.model.size, args.model.size)),
    ]
    trans_val = Compose(trans)
    # Augmentation on the original image
    if args.translate:
        from torchvision.transforms.v2 import RandomAffine

        trans.insert(
            0,
            RandomAffine(
                degrees=0, translate=(args.translate, args.translate)
            ),
        )
    if args.rotate:
        from torchvision.transforms.v2 import RandomRotation

        trans.insert(0, RandomRotation(args.rotate))
    if args.flip:
        from torchvision.transforms.v2 import (
            RandomHorizontalFlip,
            RandomVerticalFlip,
        )

        if args.flip & 1:
            trans.insert(0, RandomHorizontalFlip())
        if args.flip & 2:
            trans.insert(0, RandomVerticalFlip())
    trans_train = Compose(trans)

    use_gpu = device.type == "cuda"
    pin_memory = args.pin_memory and use_gpu
    pin_memory_device = args.device if pin_memory else ""
    train_loader = DataLoader(
        ROIDataset(X[t], y[t], transform=trans_train),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ROIDataset(X[v], y[v], transform=trans_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    testing_loader = None
    if len(tt):
        testing_loader = DataLoader(
            ROIDataset(X[tt], y[tt], transform=trans_val),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device,
        )
    test_interval = max(1, args.testing_interval)

    # Create training objects

    # Loss function
    # Optional inverse frequency weights
    weight = None
    if args.loss_weights:
        # 1 / (counts / counts.sum())
        weight = counts.sum() / counts
        weight = weight / weight.sum()  # normalise to 1
        weight = torch.tensor(weight, dtype=torch.float32).to(device)
    loss_fn: torch.nn.Module
    if args.loss_function == LossFunction.cross_entropy:
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        # Default to FocalLoss
        loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=weight)

    # Learning rate scheduler
    scheduler: torch.optim.lr_scheduler.LRScheduler
    if args.lr_scheduler == LrScheduler.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plat_factor,
            patience=args.lr_plat_patience,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

    # Training loss is expected to go down. Stop when at an approximate plateau.
    train_stop = StoppingCriteria(patience=1, min_rel_delta=1e-2)
    # Validation loss may increase due to overtraining. This is the main
    # control point over early termination.
    val_stop = StoppingCriteria(
        patience=args.patience,
        min_delta=args.delta,
        min_rel_delta=args.rel_delta,
    )

    if use_gpu and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    checkpoint_name_best = checkpoint_name + ".best"

    stop_file = f"{args.pid}.stop"
    logging.info(f"Stop file: {stop_file}")

    for i in range(args.epochs):
        epoch += 1
        train_loss, val_loss, train_acc, val_acc, _yt, _yp = train_epoch(
            model,
            train_loader,
            validation_loader,
            loss_fn,
            optimizer,
            device,
            classes=True,
        )
        assert _yt is not None and _yp is not None
        y_true, y_pred = _yt.numpy(), _yp.numpy()
        pr, re, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=np.nan, average="weighted"
        )
        better = False
        if f1 > best_f1:
            best_f1 = f1
            better = True
        stop = train_stop.check(train_loss) and val_stop.check(val_loss)
        d = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "loss": val_loss,
            "acc": val_acc,
            "precision": pr,
            "recall": re,
            "f1": f1,
            "best_f1": best_f1,
            "labels": labels,
            "model": str(args.model),
            "channels": channels,
        }
        test_stats = []
        if testing_loader and (i + 1) % test_interval == 0:
            _yt2, _yp2 = test_epoch(model, testing_loader, device)
            y_true, y_pred = _yt2.numpy(), _yp2.numpy()
            *test_stats, _ = precision_recall_fscore_support(
                y_true, y_pred, zero_division=np.nan, average="weighted"
            )
            d["test_precision"], d["test_recall"], d["test_f1"] = test_stats
            test_stats.append((y_true == y_pred).sum() / len(y_true))
            d["test_acc"] = test_stats[-1]
        torch.save(d, checkpoint_name)
        star = ""
        if better:
            shutil.copy2(checkpoint_name, checkpoint_name_best)
            star = " *"
        if test_stats:
            logging.info(
                "[%d] Loss train %.6f : val %.6f : Accuracy train %.6f : val %.6f : test %.6f : F1 val %.6f : test %.6f%s",
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                d["test_acc"],
                d["f1"],
                d["test_f1"],
                star,
            )
        else:
            logging.info(
                "[%d] Loss train %.6f : val %.6f : Accuracy train %.6f : val %.6f : F1 val %.6f%s",
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                d["f1"],
                star,
            )
        if args.wandb:
            d = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "acc": val_acc,
                "precision": pr,
                "recall": re,
                "f1": f1,
            }
            if test_stats:
                (
                    d["test_precision"],
                    d["test_recall"],
                    d["test_f1"],
                    d["test_acc"],
                ) = test_stats
            wandb.log(d)
        if stop:
            logging.info("[%d] Stopping due to no improvement", epoch)
            break
        if os.path.exists(stop_file):
            logging.info("[%d] Stopping due to %s", epoch, stop_file)
            os.remove(stop_file)
            break
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    # Ensure best model has IoU computed
    checkpoint = torch.load(
        checkpoint_name_best, map_location=device, weights_only=False
    )
    if testing_loader and "test_f1" not in checkpoint:
        logging.info("[%d] Computing metrics on best model", epoch)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        _yt3, _yp3 = test_epoch(model, testing_loader, device)
        y_true, y_pred = _yt3.numpy(), _yp3.numpy()
        *test_stats, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=np.nan, average="weighted"
        )
        (
            checkpoint["test_precision"],
            checkpoint["test_recall"],
            checkpoint["test_f1"],
        ) = test_stats
        checkpoint["test_acc"] = (y_true == y_pred).sum() / len(y_true)
        torch.save(checkpoint, checkpoint_name_best)
    logging.info(
        "[%d] Best model : Loss train %.6f : val %.6f : Accuracy train %.6f : val %.6f : test %.6f : F1 val %.6f : test %.6f",
        checkpoint["epoch"],
        checkpoint["train_loss"],
        checkpoint["loss"],
        checkpoint["train_acc"],
        checkpoint["acc"],
        checkpoint.get("test_acc", 0),
        checkpoint["f1"],
        checkpoint.get("test_f1", None),
    )

    if args.wandb:
        # Save large files at the end
        wandb.save(checkpoint_name)
        wandb.save(checkpoint_name_best)
        wandb.finish()
    t = time.time() - start_time
    logging.info(f"Done (in {t:.6g} seconds)")


def main() -> None:
    """Entry point for cellclass-train CLI."""
    parser = argparse.ArgumentParser(
        description="Program to train a model using an ROI dataset."
    )

    parser.add_argument(
        "input",
        metavar="dataset.npz | state.json",
        type=file_path,
        help="Dataset file, or training state file",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Number of images to use (default is all)",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Random seed to select data (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        default="cuda",
        help="Device (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--state",
        dest="state",
        type=str,
        default="training.json",
        help="Training state file (default: %(default)s)",
    )

    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model",
        dest="model",
        type=Model,
        choices=list(Model),
        default=Model.densenet121,
        help="Model name (default: %(default)s)",
    )
    group.add_argument(
        "-n",
        "--name",
        dest="name",
        type=str,
        default="model.pt",
        help="Checkpoint name (default: %(default)s)",
    )
    group.add_argument(
        "--existing",
        type=Existing,
        choices=list(Existing),
        default=Existing.error,
        help="Existing checkpoint option (default: %(default)s)",
    )
    # Different model initialisation
    group.add_argument(
        "--weights",
        type=none_or_str,
        default="DEFAULT",
        help="Model weights, e.g. None (default: %(default)s)",
    )
    parser.add_argument(
        "--freeze-weights",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze pretrained model weights",
    )

    group = parser.add_argument_group("Training")
    group.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Training epochs (default: %(default)s)",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the data loader (default: %(default)s)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for asynchronous data loading (default: %(default)s)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0xDEADBEEF,
        help="Random seed for initial model (default: %(default)s)",
    )
    group.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Size for the validation data (default: %(default)s)",
    )
    group.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="The learning rate (default: %(default)s)",
    )
    group.add_argument(
        "--lr-scheduler",
        type=LrScheduler,
        choices=list(LrScheduler),
        default=LrScheduler.step,
        help="Learning rate scheduler (default: %(default)s)",
    )
    group.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="The learning rate step scheduler gamma (default: %(default)s)",
    )
    group.add_argument(
        "--lr-step",
        dest="lr_step_size",
        type=int,
        default=5,
        help="The learning rate step scheduler step size (default: %(default)s)",
    )
    group.add_argument(
        "--lr-factor",
        dest="lr_plat_factor",
        type=float,
        default=0.1,
        help="The learning rate plateau scheduler reduction factor (default: %(default)s)",
    )
    group.add_argument(
        "--lr-patience",
        dest="lr_plat_patience",
        type=int,
        default=2,
        help="The learning rate plateau scheduler patience (default: %(default)s)",
    )
    group.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of times to allow for no improvement before stopping (default: %(default)s)",
    )
    group.add_argument(
        "--delta",
        type=float,
        default=0,
        help="The minimum absolute change to be counted as improvement (default: %(default)s)",
    )
    group.add_argument(
        "--rel-delta",
        type=float,
        default=1e-4,
        help="The minimum relative change to be counted as improvement (default: %(default)s)",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="The weight decay for Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--flip",
        type=int,
        default=1,
        help="Perform random flips each epoch: 0=None; 1=Horizontal; 2=Vertical; 3=Both (default: %(default)s)",
    )
    group.add_argument(
        "--rotate",
        type=int,
        default=180,
        help="Perform random rotations each epoch in (-degrees, +degrees) (default: %(default)s)",
    )
    group.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Perform random translations each epoch in (-f*imgsize, +f*imgsize) (default: %(default)s)",
    )
    group.add_argument(
        "--loss-function",
        type=LossFunction,
        choices=list(LossFunction),
        default=LossFunction.focal_loss,
        help="Loss function (default: %(default)s)",
    )
    group.add_argument(
        "--focal-gamma",
        type=float,
        default=2,
        help="Gamma value for the focal loss (default: %(default)s)",
    )
    group.add_argument(
        "--loss-weights",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use inverse frequencies for the loss weights.",
    )
    group.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout in final feature classification inputs; "
        "larger input vector can have larger dropout (default: %(default)s)",
    )

    group = parser.add_argument_group("Testing")
    group.add_argument(
        "--testing-size",
        type=float,
        default=0.2,
        help="Testing size (default: %(default)s)",
    )
    group.add_argument(
        "--testing-interval",
        type=int,
        default=5,
        help="Testing interval (default: %(default)s)",
    )

    group = parser.add_argument_group("Misc")
    group.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Log level (default: %(default)s). WARNING=30; INFO=20; DEBUG=10",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run cuDNN autotuner",
    )
    parser.add_argument(
        "--pin-memory",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use CUDA pinned memory in the DataLoader",
    )

    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--wandb",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Log to Weights and Biases."
        " Must be logged in, else works offline.",
    )
    group.add_argument(
        "--entity",
        type=none_or_str,
        default=None,
        help="Weights and Biases team/entity (default: logged-in user's workspace)",
    )
    group.add_argument(
        "--project",
        type=str,
        default="cellclass",
        help="Weights and Biases project (default: %(default)s)",
    )
    group.add_argument(
        "--run-name", type=str, help="Weights and Biases run display name"
    )
    group.add_argument(
        "--tags", nargs="+", default=[], help="Weights and Biases tags"
    )
    group.add_argument(
        "--wandb-id",
        type=str,
        help="Weights and Biases ID (overrides generated/saved state id)",
    )

    args = parser.parse_args()

    # Support save/continue of a training run
    if args.input.endswith(".json"):
        # Continue training
        with open(args.input) as f:
            d = json.load(f)
        # To continue training we must load the existing checkpoint.
        # This avoids have to convert the 'existing' str to an enum.
        d["existing"] = Existing.load
        # Convert enum from string
        d["model"] = Model(d["model"])
        # Merge with script arguments.
        # Allow some arguments to override the previous training state.
        import sys

        saved = {}
        args_d = vars(args)
        for s in [
            "log-level",
            "epochs",
            "wandb",
            "device",
            "patience",
            "delta",
            "rel-delta",
            "testing-size",
            "testing-interval",
        ]:
            if "--" + s in sys.argv:
                s = s.replace("-", "_")
                saved[s] = args_d[s]
        args_d.update(d)
        args_d.update(saved)
        args.restart = True

    if args.wandb_id is None:
        del args.wandb_id

    run(args)


if __name__ == "__main__":
    main()
