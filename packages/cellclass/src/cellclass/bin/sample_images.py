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

import os


def run(
    dataset: str,
    output: str | None = None,
    samples: int = 10,
    crop: int = 0,
) -> None:
    """Sample and save example images from an .npz dataset."""
    import numpy as np
    from skimage.measure import centroid
    from tifffile import imwrite

    out = output if output else os.path.dirname(dataset)

    data = np.load(dataset)
    X = data["X"]
    y_names = data["y_names"]
    print(f"Sampling images: {X[0].shape}, {X[0].dtype}")
    labels, counts = np.unique(y_names, return_counts=True)
    for label, count in zip(labels, counts, strict=False):
        print(f"{label} = {count}")
        s = np.flatnonzero(y_names == label)
        s = np.random.choice(s, size=min(len(s), samples))
        sampled_images = [X[i] for i in s]
        if crop:
            for i, image in enumerate(sampled_images):
                # Use first channel
                c = centroid(image[0])  # type: ignore[no-untyped-call]
                m0 = max(int(c[0]) - crop // 2, 0)
                m1 = max(int(c[1]) - crop // 2, 0)
                sampled_images[i] = image[:, m0 : m0 + crop, m1 : m1 + crop]
        img = np.array(sampled_images)
        # ImageJ format: TZCYX
        img = np.expand_dims(img, axis=1)
        imwrite(
            os.path.join(out, label + ".tif"),
            img,
            photometric="minisblack",
            imagej=True,
        )


def main() -> None:
    """Entry point for direct execution of the sample command."""
    from cellclass.cli import sample

    sample.main(prog_name="cellclass-sample")


if __name__ == "__main__":
    main()
