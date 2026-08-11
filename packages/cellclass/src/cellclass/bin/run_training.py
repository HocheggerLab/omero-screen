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

from cellclass.options import Existing, LossFunction, LrScheduler


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
    """Entry point for direct execution of the train command."""
    from cellclass.cli import train

    train.main(prog_name="cellclass-train")


if __name__ == "__main__":
    main()
