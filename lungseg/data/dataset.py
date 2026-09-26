"""In-memory nodule crop datasets + GPU augmentation.

Every crop is a 96^3 cube (0.8 mm isotropic) centred on the nodule. Training
draws random 64^3 views of it (rotation, scaling, flips, off-centre jitter)
straight on the GPU with ``grid_sample``, so the CPU is never the bottleneck and
small labelled sets still see a lot of variety. Evaluation uses the central 64^3.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

HU_MIN, HU_MAX = -1000.0, 500.0


def normalize_hu(x):
    """HU -> [-1, 1] over the lung window [-1000, 500]."""
    return (x.clamp(HU_MIN, HU_MAX) - (HU_MIN + HU_MAX) / 2) / ((HU_MAX - HU_MIN) / 2)


def read_meta(root):
    with open(os.path.join(root, "meta.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("file", "patient_id"):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def read_splits(root):
    with open(os.path.join(root, "splits.json")) as f:
        return json.load(f)


def subsample_patients(patients, fraction, seed=0):
    """Nested patient subsets: the 10% set is contained in the 25% set, etc."""
    pids = sorted(patients)
    if fraction >= 1.0:
        return pids
    rng = np.random.default_rng(seed + 12345)
    perm = list(rng.permutation(pids))
    k = max(1, int(round(fraction * len(pids))))
    return sorted(perm[:k])


class CropSet:
    """All crops of a split loaded into RAM (int16 images, uint8 masks)."""

    def __init__(self, root, split="train", fraction=1.0, seed=0, max_items=0, load_readers=False,
                 min_readers=1):
        self.root = root
        meta = read_meta(root)
        splits = read_splits(root)
        pids = set(subsample_patients(splits[split], fraction, seed) if split == "train"
                   else splits[split])
        self.rows = [r for r in meta if r["patient_id"] in pids and r["n_readers"] >= min_readers]
        if max_items:
            self.rows = self.rows[:max_items]
        self.patients = sorted({r["patient_id"] for r in self.rows})
        imgs, masks, readers = [], [], []
        for r in self.rows:
            z = np.load(os.path.join(root, r["file"]))
            imgs.append(z["image"])
            masks.append(z["mask"])
            if load_readers:
                readers.append(z["readers"])
        self.images = np.stack(imgs) if imgs else np.zeros((0, 96, 96, 96), np.int16)
        self.masks = np.stack(masks) if masks else np.zeros((0, 96, 96, 96), np.uint8)
        self.readers = readers  # list of (R, D, H, W)
        self.spacing = float(np.load(os.path.join(root, self.rows[0]["file"]))["spacing"]) if self.rows else 0.8

    def __len__(self):
        return len(self.rows)

    def batch(self, idx, device):
        img = torch.from_numpy(self.images[idx]).to(device, non_blocking=True).float().unsqueeze(1)
        msk = torch.from_numpy(self.masks[idx]).to(device, non_blocking=True).float().unsqueeze(1)
        return img, msk


class UnlabeledSet:
    """Images only (for denoising pretraining): labelled crops of the given
    patients plus random lung cubes of every patient not in val/test."""

    def __init__(self, root, exclude_patients=(), include_labeled_patients=None, max_items=0):
        exclude = set(exclude_patients)
        files = []
        for r in read_meta(root):
            if r["patient_id"] in exclude:
                continue
            if include_labeled_patients is not None and r["patient_id"] not in include_labeled_patients:
                continue
            files.append(r["file"])
        unl_csv = os.path.join(root, "unlabeled.csv")
        if os.path.exists(unl_csv):
            with open(unl_csv) as f:
                files += [r["file"] for r in csv.DictReader(f) if r["patient_id"] not in exclude]
        if max_items:
            files = files[:max_items]
        self.files = files
        self.images = np.stack([np.load(os.path.join(root, f))["image"] for f in files])

    def __len__(self):
        return len(self.files)

    def batch(self, idx, device):
        return torch.from_numpy(self.images[idx]).to(device).float().unsqueeze(1)


def center_crop(x, size):
    d = x.shape[-1]
    s = (d - size) // 2
    return x[..., s:s + size, s:s + size, s:s + size]


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 0:   # about x (W axis)
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:   # about y (H axis)
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # about z (D axis): axial rotation


def random_affine_views(img, msk=None, out_size=64, rot_axial=math.pi, rot_tilt=math.radians(15),
                        scale=(0.8, 1.25), jitter_vox=8, p_flip=0.5, generator=None):
    """Random rigid+scale views of (B,1,S,S,S) cubes, output (B,1,o,o,o)."""
    B, _, S = img.shape[0], img.shape[1], img.shape[-1]
    g = generator
    thetas = []
    for _ in range(B):
        r = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi, generator=g))
        R = _rot(2, r(-rot_axial, rot_axial)) @ _rot(0, r(-rot_tilt, rot_tilt)) @ _rot(1, r(-rot_tilt, rot_tilt))
        sc = math.exp(r(math.log(scale[0]), math.log(scale[1])))
        flips = torch.diag(torch.tensor([(-1.0 if r(0, 1) < p_flip else 1.0) for _ in range(3)]))
        A = (out_size / S) * sc * (R.float() @ flips)
        t = torch.tensor([r(-1, 1) for _ in range(3)]) * (2.0 * jitter_vox / S)
        thetas.append(torch.cat([A, t[:, None]], dim=1))
    theta = torch.stack(thetas).to(img.device, img.dtype)
    grid = F.affine_grid(theta, (B, 1, out_size, out_size, out_size), align_corners=False)
    img_o = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if msk is None:
        return img_o
    msk_o = F.grid_sample(msk, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return img_o, msk_o


def intensity_augment(x, generator=None):
    """x normalised to [-1,1]. Contrast/brightness/gamma/noise/blur/low-res simulation."""
    B = x.shape[0]
    dev = x.device
    u = lambda lo, hi: torch.empty(B, 1, 1, 1, 1, device="cpu").uniform_(lo, hi, generator=generator).to(dev)
    coin = lambda p: (torch.rand(B, 1, 1, 1, 1, generator=generator) < p).float().to(dev)
    x = x * (1 + coin(0.5) * (u(0.8, 1.2) - 1)) + coin(0.5) * u(-0.1, 0.1)
    # gamma on [0,1]
    x01 = ((x + 1) / 2).clamp(0, 1)
    gam = 1 + coin(0.3) * (u(0.7, 1.5) - 1)
    x = x01.pow(gam) * 2 - 1
    # simulate thicker slices / lower resolution
    if torch.rand(1, generator=generator).item() < 0.25:
        f = float(torch.empty(1).uniform_(1.0, 2.0, generator=generator))
        size = x.shape[-3:]
        lo = F.interpolate(x, scale_factor=(1 / f, 1, 1), mode="trilinear", align_corners=False)
        x = F.interpolate(lo, size=size, mode="trilinear", align_corners=False)
    if torch.rand(1, generator=generator).item() < 0.2:
        k = torch.tensor([0.25, 0.5, 0.25], device=dev)
        for dim in (2, 3, 4):
            shape = [1, 1, 1, 1, 1]
            shape[dim] = 3
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = 1
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    x = x + coin(0.3) * u(0, 0.05) * torch.randn_like(x)
    return x
