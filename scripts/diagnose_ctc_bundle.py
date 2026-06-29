#!/usr/bin/env python
"""Diagnose a Mastodon CTC export bundle for label/res_track inconsistencies.

Run on the machine that holds the export, e.g.:

    uv run python scripts/diagnose_ctc_bundle.py ~/mastodon_exports/plate_4365_C4_ctc

Answers, straight from the files Mastodon reads:

  1. Does any label occupy >1 connected component in a single frame?
     (one track_id genuinely painted onto two separate nuclei)
  2. For each label, do its frames-present-in-masks match res_track's [B,E]?
       2a markers present OUTSIDE [B,E]  -> "does not cover image marker"
       2b gap frames INSIDE [B,E] where the label is absent
  3. Per-label largest centroid jump between consecutive present frames
     (drift = real cell; teleport = bad id).

Efficient: one ``find_objects`` per frame gives every label's bounding box, so
component checks run on tiny sub-arrays; centroids are vectorised over all
labels at once. stdlib + numpy + tifffile + scipy.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def load_res_track(path: Path) -> dict[int, tuple[int, int, int]]:
    """Parse a CTC ``res_track.txt`` into ``{label: (begin, end, parent)}``."""
    out: dict[int, tuple[int, int, int]] = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 4:
            L, B, E, P = (int(x) for x in p)
            out[L] = (B, E, P)
    return out


def main(bundle: Path) -> None:
    """Print the connectivity / res_track / drift report for ``bundle``."""
    masks = sorted(bundle.glob("mask*.tif"))
    if not masks:
        sys.exit(f"No mask*.tif in {bundle}")
    res = load_res_track(bundle / "res_track.txt")

    f0 = tifffile.imread(masks[0])
    print(f"{len(masks)} frames, mask shape {f0.shape}, dtype {f0.dtype}")
    print(f"{len(res)} tracks in res_track.txt\n")

    # label -> {t: (n_components, (cy, cx))}
    present: dict[int, dict[int, tuple[int, tuple[float, float]]]] = (
        defaultdict(dict)
    )
    multi = []  # (label, t, n_components)

    for t, mp in enumerate(masks):
        frame = tifffile.imread(mp)
        labels = np.unique(frame)
        labels = labels[labels != 0]
        if labels.size == 0:
            continue
        # Vectorised per-label centroids over the whole frame in one call.
        coms = ndimage.center_of_mass(
            np.ones_like(frame, dtype=bool), frame, index=labels
        )
        # One call: bounding slice per label (index = label value).
        objs = ndimage.find_objects(frame)
        for lbl, com in zip(labels.tolist(), coms, strict=False):
            sl = objs[lbl - 1]
            sub = frame[sl] == lbl
            ncomp = int(ndimage.label(sub)[1])
            present[lbl][t] = (ncomp, (float(com[0]), float(com[1])))
            if ncomp > 1:
                multi.append((lbl, t, ncomp))

    print(f"[1] label-frames with >1 connected component: {len(multi)}")
    aff = sorted({m[0] for m in multi})
    print(f"      distinct labels affected: {len(aff)}")
    for lbl, t, n in multi[:15]:
        print(f"      label {lbl} @ t={t}: {n} components")

    outside, gaps = [], []
    for lbl, frames in present.items():
        ts = sorted(frames)
        if lbl not in res:
            outside.append((lbl, f"frames {ts[0]}..{ts[-1]} NOT IN res_track"))
            continue
        B, E, _ = res[lbl]
        before = [t for t in ts if t < B]
        after = [t for t in ts if t > E]
        if before or after:
            outside.append(
                (lbl, f"declared [{B},{E}] but also at {before + after}")
            )
        missing = [t for t in range(B, E + 1) if t not in frames]
        if missing:
            gaps.append((lbl, B, E, len(missing), missing[:8]))

    print(
        f"\n[2a] labels present OUTSIDE their res_track [B,E]: {len(outside)}"
    )
    for lbl, msg in outside[:15]:
        print(f"      label {lbl}: {msg}")
    print(f"\n[2b] labels with gap frames INSIDE [B,E]: {len(gaps)}")
    for lbl, B, E, n, sample in gaps[:15]:
        print(f"      label {lbl} [{B},{E}]: {n} missing, e.g. {sample}")

    jumps = []
    for lbl, frames in present.items():
        ts = sorted(frames)
        mx, worst = 0.0, None
        for a, b in zip(ts, ts[1:], strict=False):
            (ya, xa), (yb, xb) = frames[a][1], frames[b][1]
            d = float(np.hypot(yb - ya, xb - xa))
            if d > mx:
                mx, worst = d, (a, b, round(d, 1))
        if worst:
            jumps.append((lbl, mx, worst))
    jumps.sort(key=lambda r: r[1], reverse=True)
    print("\n[3] largest consecutive-frame centroid jumps (top 15):")
    for lbl, _, (a, b, d) in jumps[:15]:
        print(f"      label {lbl}: t{a}->t{b} jump {d} px")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: diagnose_ctc_bundle.py <bundle_dir>")
    main(Path(sys.argv[1]).expanduser())
