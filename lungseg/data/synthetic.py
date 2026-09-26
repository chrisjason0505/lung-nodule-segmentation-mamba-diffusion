"""Synthetic stand-in for the LIDC crop cache (same files and format).

Only for smoke tests and CI. It has lung parenchyma, vessels, a chest wall and
lobulated nodules (some juxta-pleural, some touching vessels) with 1-4 slightly
different "reader" masks. Numbers on this data say nothing about LIDC.

    python -m lungseg.data.synthetic --out data/synthetic --n_patients 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import ndimage

from lungseg.data.prepare_lidc import META_FIELDS, make_splits


def _cube(rng, S, spacing, with_nodule=True):
    zz, yy, xx = np.meshgrid(*[np.arange(S) - (S - 1) / 2] * 3, indexing="ij")
    img = rng.normal(-850, 30, (S, S, S)).astype(np.float32)
    # vessels: random tubes
    for _ in range(rng.integers(2, 6)):
        p = rng.uniform(-S / 2, S / 2, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        pts = np.stack([zz - p[0], yy - p[1], xx - p[2]], -1)
        dist = np.linalg.norm(pts - (pts @ d)[..., None] * d, axis=-1)
        r = rng.uniform(1.0, 3.0) / spacing
        img[dist < r] = rng.normal(30, 20)
    # chest wall on one side sometimes
    if rng.random() < 0.3:
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        off = rng.uniform(8, 30)
        img[(zz * n[0] + yy * n[1] + xx * n[2]) > off] = 40
    if not with_nodule:
        return img, None, None
    diam = float(np.exp(rng.uniform(np.log(4), np.log(28))))
    r0 = diam / 2 / spacing
    th = np.arctan2(np.hypot(xx, yy), zz)
    ph = np.arctan2(yy, xx)
    lob = 1 + 0.15 * np.sin(3 * ph + rng.uniform(0, 6)) * np.sin(2 * th) + 0.1 * np.cos(4 * th + rng.uniform(0, 6))
    rr = np.sqrt(zz ** 2 + (yy * rng.uniform(0.8, 1.2)) ** 2 + (xx * rng.uniform(0.8, 1.2)) ** 2)
    field = rr / (r0 * lob)
    solid = rng.random() < 0.8
    nod = field <= 1
    img[nod] = rng.normal(20 if solid else -500, 40, nod.sum())
    img = ndimage.gaussian_filter(img, 0.7)
    readers = []
    for _ in range(int(rng.integers(1, 5))):
        readers.append((field <= rng.uniform(0.85, 1.15)).astype(np.uint8))
    readers = np.stack(readers)
    mask = (readers.mean(0) >= 0.5).astype(np.uint8)
    return img, mask, dict(diameter_mm=diam, n_readers=len(readers), readers=readers)


def make_synthetic(out, n_patients=40, nodules_per_patient=(1, 3), size=96, spacing=0.8, seed=0,
                   unlabeled_per_patient=1):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out, "labeled"), exist_ok=True)
    os.makedirs(os.path.join(out, "unlabeled"), exist_ok=True)
    rows, unl = [], []
    for p in range(n_patients):
        pid = f"SYN-{p:04d}"
        for n in range(int(rng.integers(nodules_per_patient[0], nodules_per_patient[1] + 1))):
            img, mask, info = _cube(rng, size, spacing)
            fname = f"{pid}_s{p:04d}_n{n:02d}.npz"
            np.savez_compressed(os.path.join(out, "labeled", fname), image=img.round().astype(np.int16),
                                mask=mask, readers=info["readers"], spacing=np.float32(spacing))
            r = {k: 0 for k in META_FIELDS}
            r.update(file=f"labeled/{fname}", patient_id=pid, scan_id=p, nodule_idx=n,
                     n_readers=info["n_readers"], diameter_mm=info["diameter_mm"],
                     slice_thickness=1.25, pixel_spacing=0.7, mask_voxels=int(mask.sum()))
            rows.append(r)
        for u in range(unlabeled_per_patient):
            img, _, _ = _cube(rng, size, spacing, with_nodule=False)
            fname = f"{pid}_s{p:04d}_u{u:02d}.npz"
            np.savez_compressed(os.path.join(out, "unlabeled", fname), image=img.round().astype(np.int16),
                                spacing=np.float32(spacing))
            unl.append(dict(file=f"unlabeled/{fname}", patient_id=pid, scan_id=p))
    with open(os.path.join(out, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "unlabeled.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "patient_id", "scan_id"])
        w.writeheader()
        w.writerows(unl)
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(make_splits([r["patient_id"] for r in rows], seed=seed), f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_patients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_synthetic(a.out, a.n_patients, seed=a.seed)
    print("synthetic cache written to", a.out)
