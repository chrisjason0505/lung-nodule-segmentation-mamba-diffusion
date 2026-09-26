"""Per-nodule evaluation on the held-out test patients.

    python -m lungseg.evaluate --data data/lidc_crops --ckpt runs/mamba_ddep_f0.25/best.pt

Reports Dice / IoU / precision / recall / HD95 against the 50% consensus mask,
plus the leave-one-reader-out radiologist agreement on the same nodules.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from lungseg.data.dataset import CropSet, center_crop
from lungseg.metrics import binary_metrics, keep_central_component, reader_agreement, summarize
from lungseg.models.mamba_unet import build_model
from lungseg.utils import get_device, load_state, predict_probs, save_json

SIZE_BINS = [(0, 6), (6, 10), (10, 20), (20, 1000)]


def evaluate_set(model, cs: CropSet, size=64, tta=True, postprocess=True, with_hd=True,
                 with_readers=False, batch_size=8):
    dev = next(model.parameters()).device
    rows = []
    for i in range(0, len(cs), 64):  # chunked to bound GPU memory
        idx = np.arange(i, min(i + 64, len(cs)))
        img = torch.from_numpy(cs.images[idx]).float().unsqueeze(1)
        probs = predict_probs(model, img.to(dev), size=size, tta=tta, batch_size=batch_size)
        for j, k in enumerate(idx):
            pred = (probs[j, 0].numpy() >= 0.5).astype(np.uint8)
            if postprocess:
                pred = keep_central_component(pred)
            gt = center_crop(cs.masks[k], size)
            m = binary_metrics(pred, gt, cs.spacing, with_hd=with_hd)
            meta = cs.rows[k]
            m.update(file=meta["file"], patient_id=meta["patient_id"], diameter_mm=meta["diameter_mm"],
                     n_readers=meta["n_readers"])
            if with_readers and cs.readers:
                ra = reader_agreement(center_crop(cs.readers[k], size), cs.spacing)
                if ra:
                    m["reader_dice"], m["reader_iou"] = ra["dice"], ra["iou"]
            rows.append(m)
    return rows


def full_report(rows):
    rep = {"all": summarize(rows)}
    for lo, hi in SIZE_BINS:
        sub = [r for r in rows if lo <= r["diameter_mm"] < hi]
        if sub:
            rep[f"diam_{lo}-{hi}mm"] = summarize(sub)
    for nr in (2, 3, 4):
        sub = [r for r in rows if r["n_readers"] >= nr]
        if sub:
            rep[f"readers>={nr}"] = summarize(sub)
    hum = [r for r in rows if "reader_dice" in r]
    if hum:
        rep["radiologists_leave_one_out"] = {
            "n": len(hum), "dice": float(np.mean([r["reader_dice"] for r in hum])),
            "iou": float(np.mean([r["reader_iou"] for r in hum]))}
    return rep


def write_rows(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--no_postprocess", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = get_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], widths=tuple(cfg["widths"]), mamba_levels=tuple(cfg["mamba_levels"]),
                        mamba_depth=cfg["mamba_depth"]).to(dev)
    load_state(model, args.ckpt)
    cs = CropSet(args.data, args.split, load_readers=True)
    rows = evaluate_set(model, cs, size=cfg["crop"], tta=not args.no_tta,
                        postprocess=not args.no_postprocess, with_readers=True)
    rep = full_report(rows)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out, exist_ok=True)
    save_json(rep, os.path.join(out, "report.json"))
    write_rows(rows, os.path.join(out, "per_nodule.csv"))
    a = rep["all"]
    print(f"[{args.split}] n={a['n']}  Dice {a['dice']:.4f}±{a['dice_std']:.3f}  IoU {a['iou']:.4f}  "
          f"P {a['precision']:.3f}  R {a['recall']:.3f}  HD95 {a.get('hd95_mm', float('nan')):.2f}mm")
    if "radiologists_leave_one_out" in rep:
        h = rep["radiologists_leave_one_out"]
        print(f"   radiologist leave-one-out on the same nodules: Dice {h['dice']:.4f} IoU {h['iou']:.4f}")


if __name__ == "__main__":
    main()
