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

"""Convert .npy cell image files into a normalised .npz dataset for training."""

from __future__ import annotations

import argparse

import numpy as np
import numpy.typing as npt


def _binary_mask(
    mask: npt.NDArray[np.integer],  # type: ignore[type-arg]
) -> tuple[npt.NDArray[np.uint8] | None, bool]:
    """Convert to a binary mask using the pixel value at the centre."""
    y, x = mask.shape[-2:]
    centre_id = mask[y // 2, x // 2]
    non_central = False
    if centre_id == 0:
        u = np.unique(mask[mask != 0])
        if len(u) == 1:
            non_central = True
            centre_id = u[0]
        else:
            return None, False
    return (mask == centre_id).astype(np.uint8), non_central


def _to_uint8(
    image: npt.NDArray[np.floating],  # type: ignore[type-arg]
) -> npt.NDArray[np.uint8]:
    """Convert image to uint8 using the (1, 99) percentiles."""
    out = np.zeros(image.shape, dtype=np.uint8)
    for c in range(image.shape[-1]):
        data = image[..., c].astype(float)
        pmin, pmax = np.percentile(data, (1, 99))
        data = (data - pmin) / (pmax - pmin)
        data = np.clip(data, 0, 1)
        # Convert to [1, 255] to allow masked regions to be distinct at zero
        out[..., c] = (1 + data * 254).astype(np.uint8)
    return out


def _extract_roi(
    image: npt.NDArray[np.uint8],
    binary_mask: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], int, int]:
    """Extract the ROI from a multi-channel image using the mask.

    Args:
        image: Multi-channel input image (YXC).
        binary_mask: Binary mask image (YX).

    Returns:
        Tuple of (roi (CYX), shift_x, shift_y).

    """
    coords = np.argwhere(binary_mask > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    s = binary_mask.shape
    wx = x1 - x0
    wy = y1 - y0
    padx = (s[1] - wx) // 2
    pady = (s[0] - wy) // 2
    shiftx = int(x0 - padx)
    shifty = int(y0 - pady)

    roi = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        im = image[..., channel] * binary_mask
        if shiftx or shifty:
            roi[pady : pady + wy, padx : padx + wx, channel] = im[y0:y1, x0:x1]
        else:
            roi[..., channel] = im
    return roi.transpose((2, 0, 1)), shiftx, shifty  # type: ignore[return-value]


def run(args: argparse.Namespace) -> None:
    """Build a .npz dataset from a directory of labelled .npy cell images."""
    import glob
    import hashlib
    import json
    import os

    import tqdm

    channels: list[str] = args.channels
    if not channels:
        metafile = os.path.join(args.dir, "metadata.json")
        if not os.path.exists(metafile):
            print(f"Missing metadata file: {metafile}")
            raise SystemExit(1)
        with open(metafile) as f:
            meta = json.load(f)
        try:
            channels = meta["user_data"]["channels"]
        except Exception as e:
            print(
                f'Missing "/user_data/channels" in metadata file: {metafile}: {e}'
            )
            raise SystemExit(1) from e

    print(f"Input channels: {channels}")
    num_channels = len(channels)

    counts: dict[str, int] = {}
    labels: list[str] = []
    rois: list[npt.NDArray[np.uint8]] = []
    hashes: dict[str, list[npt.NDArray[np.uint8]]] = {}
    dup = 0
    ignored = 0
    no_mask = 0
    non_central = 0
    sx: list[int] = []
    sy: list[int] = []
    multi = 0

    for f in tqdm.tqdm(glob.glob(os.path.join(args.dir, "*.npy"))):
        d = np.load(f, allow_pickle=True).item()  # type: ignore[arg-type]
        targets = d["target"]
        images = d["data"][0]
        masks = d["data"][1]
        for idx, (label, img, mask) in enumerate(
            zip(targets, images, masks, strict=False)
        ):
            if label in args.ignore:
                ignored += 1
                continue
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            if img.shape[2] != num_channels:
                print(
                    f"Incorrect number of channels {img.shape[2]} != {num_channels}: {f}:{idx}"
                )
                raise SystemExit(1)

            if args.single_label:
                u = np.unique(mask[mask != 0])
                if len(u) != 1:
                    multi += 1
                    continue

            bin_mask, nc = _binary_mask(mask)
            if bin_mask is None:
                no_mask += 1
                continue
            if nc:
                non_central += 1

            img = _to_uint8(img)
            roi, shiftx, shifty = _extract_roi(img, bin_mask)
            sx.append(shiftx)
            sy.append(shifty)

            h: str | None = hashlib.sha256(roi.tobytes()).hexdigest()
            im = hashes.get(h, [])  # type: ignore[arg-type]
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
            hashes[h] = im  # type: ignore[index]

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


def main() -> None:
    """Entry point for direct execution of the dataset command."""
    from cellclass.cli import dataset

    dataset.main(prog_name="cellclass-dataset")


if __name__ == "__main__":
    main()
