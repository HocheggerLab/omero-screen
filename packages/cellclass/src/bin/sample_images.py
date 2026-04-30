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

"""Samples images from a training dataset."""

import argparse
import os

from cellclass.bin_utils import dir_path, file_path


def run(args: argparse.Namespace) -> None:
    """Sample and save images from a training dataset."""
    import numpy as np
    from skimage.measure import centroid
    from tifffile import imwrite

    out = args.output if args.output else os.path.dirname(args.dataset)

    data = np.load(args.dataset)
    X = data["X"]
    y_names = data["y_names"]
    print(f"Sampling images: {X[0].shape}, {X[0].dtype}")
    labels, counts = np.unique(y_names, return_counts=True)
    for label, count in zip(labels, counts, strict=False):
        print(f"{label} = {count}")
        s = np.flatnonzero(y_names == label)
        s = np.random.choice(s, size=min(len(s), args.samples))
        samples = [X[i] for i in s]
        if args.crop:
            for i, s in enumerate(samples):
                # Use first channel
                c = centroid(s[0])  # type: ignore[no-untyped-call]
                m0 = max(int(c[0]) - args.crop // 2, 0)
                m1 = max(int(c[1]) - args.crop // 2, 0)
                samples[i] = s[:, m0 : m0 + args.crop, m1 : m1 + args.crop]
        img = np.array(samples)
        # ImageJ format: TZCYX
        img = np.expand_dims(img, axis=1)
        imwrite(
            os.path.join(out, label + ".tif"),
            img,
            photometric="minisblack",
            imagej=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to sample images from a training dataset."
    )

    parser.add_argument("dataset", type=file_path, help="Dataset")
    parser.add_argument(
        "--output",
        type=dir_path,
        help="Output path (default: same directory as dataset)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples (default: %(default)s)",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="Crop NxN dimension (default: full image)",
    )

    args = parser.parse_args()
    run(args)
