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

"""Convert cell classification NPY files into a consolidated NPZ dataset."""

import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cellclass.bin_utils import dir_path


def _binary_mask(
    mask: NDArray[Any],
) -> tuple[NDArray[Any], bool] | tuple[None, bool]:
    """Convert to a binary mask using the pixel value at the centre."""
    y, x = mask.shape[-2:]
    cell_id = mask[y // 2, x // 2]
    non_central = False
    if cell_id == 0:
        # Mask object is non-central
        u = np.unique(mask[mask != 0])
        if len(u) == 1:
            non_central = True
            cell_id = u[0]
        else:
            # Ignore for now
            return None, False
    return (mask == cell_id).astype(np.uint8), non_central


def _to_uint8(image: NDArray[Any]) -> NDArray[Any]:
    """Convert image to uint8 using the (1, 99) percentiles."""
    out = np.zeros(image.shape, dtype=np.uint8)
    for c in range(image.shape[-1]):
        data = image[..., c]
        pmin, pmax = np.percentile(data, (1, 99))
        # Rescale 0-1
        data = (data - pmin) / (pmax - pmin)
        # Clip to range
        data[data < 0] = 0
        data[data > 1] = 1
        # Convert to [1, 255] to allow masked regions to be distinct at zero
        out[..., c] = (1 + data * 254).astype(np.uint8)
    return out


def _extract_roi(
    image: NDArray[Any],
    binary_mask: NDArray[Any],
) -> tuple[NDArray[Any], int, int]:
    """Extract the ROI from a multi-channel image using the mask.

    Args:
        image: Multi-channel input image (YXC).
        binary_mask: Mask image (YX).

    Returns:
        Tuple of (roi array CYX, shiftx, shifty).

    """
    # Re-centre using the mask boundary
    coords = np.argwhere(binary_mask > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    s = binary_mask.shape
    wx = x1 - x0
    wy = y1 - y0
    padx = (s[1] - wx) // 2
    pady = (s[0] - wy) // 2
    shiftx = x0 - padx
    shifty = y0 - pady

    # Apply the mask to all channels
    roi = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        im = image[..., channel] * binary_mask
        if shiftx or shifty:
            roi[pady : pady + wy, padx : padx + wx, channel] = im[y0:y1, x0:x1]
        else:
            roi[..., channel] = im
    return roi.transpose((2, 0, 1)), shiftx, shifty


def run(args: argparse.Namespace) -> None:
    """Convert NPY cell image files into a consolidated NPZ dataset."""
    import glob
    import hashlib
    import os

    import tqdm

    # Get input channel metadata
    channels = args.channels
    if not channels:
        metafile = os.path.join(args.dir, "metadata.json")
        if not os.path.exists(metafile):
            print(f"Missing metadata file: {metafile}")
            exit(1)
        with open(metafile) as f:
            import json

            meta = json.load(f)
            try:
                channels = meta["user_data"]["channels"]
            except Exception as e:
                print(
                    f'Missing "/user_data/channels" in metadata file: {metafile}: {e}'
                )
                exit(1)

    print(f"Input channels: {channels}")
    num_channels = len(channels)

    counts: dict[str, int] = {}
    labels = []
    rois = []
    hashes: dict[str, list[NDArray[Any]]] = {}
    dup = 0
    ignored = 0
    no_mask = 0
    non_central = 0
    sx, sy = [], []
    multi = 0

    for f in tqdm.tqdm(glob.glob(os.path.join(args.dir, "*.npy"))):
        d = np.load(f, allow_pickle=True).item()  # type: ignore[arg-type]
        targets = d["target"]
        images = d["data"][0]
        masks = d["data"][1]
        for idx, (label, i, m) in enumerate(
            zip(targets, images, masks, strict=False)
        ):
            if label in args.ignore:
                ignored += 1
                continue
            # Expect images (YXC), masks (YX). Handle single channel images.
            if i.ndim == 2:
                i = np.expand_dims(i, axis=2)
            # Validate channels
            if i.shape[2] != num_channels:
                print(
                    f"Incorrect number of channels {i.shape[2]} != {num_channels}: {f}:{idx}"
                )
                exit(1)

            if args.single_label:
                u = np.unique(m[m != 0])
                if len(u) != 1:
                    multi += 1
                    continue

            m, nc = _binary_mask(m)
            if m is None:
                no_mask += 1
                continue
            if nc:
                non_central += 1

            i = _to_uint8(i)
            roi, shiftx, shifty = _extract_roi(i, m)
            sx.append(shiftx)
            sy.append(shifty)

            # Check for image duplicates
            h_key: str = hashlib.sha256(roi.tobytes()).hexdigest()
            h: str | None = h_key
            im = hashes.get(h_key, [])
            # Check not just a hash collision
            for j in im:
                if np.array_equal(roi, j):
                    h = None
                    if args.duplicates:
                        print(f"Duplicate: {f}:{idx}")
                    break
            if h is None:
                dup += 1
                continue
            im.append(roi)
            hashes[h] = im

            counts[label] = counts.get(label, 0) + 1
            labels.append(label)
            rois.append(np.ascontiguousarray(roi))

    fn = os.path.join(args.out if args.out else args.dir, args.name)
    print("Saving", fn)
    np.savez(fn, X=rois, y_names=labels, channels=channels)

    total = sum(counts.values())
    if args.single_label:
        print(f"multi-label masks = {multi}")
    print(f"ignored class = {ignored}")
    print(f"no mask identified = {no_mask}")
    print(f"non-central mask = {non_central}")
    print(f"duplicates = {dup}/{dup + total} ({dup / (dup + total):.4f})")
    print(f"total = {total}")
    for k, v in counts.items():
        print(f"{k} = {v} ({v / total:.4f})")
    sx_arr, sy_arr = np.array(sx), np.array(sy)
    print(
        f"shift x = {np.mean(sx_arr):.4f} +/- {np.std(sx_arr):.4f} [{np.min(sx_arr)}, {np.max(sx_arr)}]"
    )
    print(
        f"shift y = {np.mean(sy_arr):.4f} +/- {np.std(sy_arr):.4f} [{np.min(sy_arr)}, {np.max(sy_arr)}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
      Program to convert a cell classification images (YXC) to a dataset.
      Cell classification images are pickled numpy data saved as dictionary:
      data: tuple of lists of images and masks;
      target: list of labels
      """
    )

    parser.add_argument(
        "dir", metavar="DIR", type=dir_path, help="Data directory"
    )
    parser.add_argument(
        "--name",
        default="rois",
        help="Output dataset name, .npz suffix is appended (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=dir_path,
        help="Output dataset directory (default is data directory)",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=["unassigned"],
        help="Labels to ignore (default: %(default)s)",
    )
    parser.add_argument(
        "--duplicates",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Log duplicates",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Channel names (default: use metadata.json in data directory)",
    )
    parser.add_argument(
        "--single-label",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Only use single label masks",
    )

    args = parser.parse_args()
    run(args)
