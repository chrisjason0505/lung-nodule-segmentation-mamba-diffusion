"""Qualitative figure: central axial / coronal slices of test nodules with the
consensus (green) and predicted (red) contours.

    python -m lungseg.visualize --data data/lidc_crops --ckpt runs/.../best.pt --out docs/predictions.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from lungseg.data.dataset import CropSet, center_crop
from lungseg.metrics import binary_metrics, keep_central_component
from lungseg.models.mamba_unet import build_model
from lungseg.utils import get_device, load_state, predict_probs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="docs/predictions.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    c = ck["config"]
    model = build_model(c["arch"], widths=tuple(c["widths"]), mamba_levels=tuple(c["mamba_levels"]),
                        mamba_depth=c["mamba_depth"]).to(dev)
    load_state(model, args.ckpt)
    cs = CropSet(args.data, "test")
    rng = np.random.default_rng(args.seed)
    # spread over sizes: sort by diameter and take evenly spaced picks
    order = np.argsort([r["diameter_mm"] for r in cs.rows])
    idx = order[np.linspace(0, len(order) - 1, min(args.n, len(order))).astype(int)]
    probs = predict_probs(model, torch.from_numpy(cs.images[idx]).float().unsqueeze(1).to(dev), size=c["crop"])
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, k in enumerate(idx):
        img = center_crop(cs.images[k], c["crop"]).astype(np.float32)
        gt = center_crop(cs.masks[k], c["crop"])
        pr = keep_central_component((probs[j, 0].numpy() >= 0.5).astype(np.uint8))
        d = binary_metrics(pr, gt, with_hd=False)["dice"]
        m = c["crop"] // 2
        for row, sl in enumerate([(m, slice(None), slice(None)), (slice(None), m, slice(None))]):
            ax = axes[row, j]
            ax.imshow(np.clip(img[sl], -1000, 400), cmap="gray")
            if gt[sl].any():
                ax.contour(gt[sl], [0.5], colors="lime", linewidths=1)
            if pr[sl].any():
                ax.contour(pr[sl], [0.5], colors="red", linewidths=1)
            ax.axis("off")
        axes[0, j].set_title(f"{cs.rows[k]['diameter_mm']:.1f} mm\nDice {d:.2f}", fontsize=8)
    fig.suptitle("green = radiologist consensus, red = prediction (top: axial, bottom: coronal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
