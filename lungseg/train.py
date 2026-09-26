"""Supervised training on a (possibly small) fraction of the labelled patients.

    python -m lungseg.train --data data/lidc_crops --arch mamba --fraction 0.25 \
        --init runs/pretrain_mamba/pretrain.pt --out runs/mamba_ddep_f0.25

The whole training set sits in RAM and augmentation runs on the GPU, so an
epoch over 100 nodules and one over 1,500 cost the same. We train for a fixed
number of steps for every fraction, which is the usual protocol for
label-efficiency curves (equal compute, different amounts of labels).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from lungseg.data.dataset import CropSet, intensity_augment, normalize_hu, random_affine_views
from lungseg.evaluate import evaluate_set, full_report, write_rows
from lungseg.metrics import deep_supervised_loss, summarize
from lungseg.models.mamba_unet import build_model, count_parameters
from lungseg.models.ssm import backend_name
from lungseg.utils import EMA, cosine_lr, get_device, load_state, save_json, seed_everything


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="mamba", choices=["mamba", "unet"])
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of TRAIN patients with labels")
    ap.add_argument("--init", default="", help="denoising-pretrained checkpoint (pretrain.py)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 320])
    ap.add_argument("--mamba_levels", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--mamba_depth", type=int, default=1)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0, help="seed of the nested patient subsets")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--checkpointing", default="auto", choices=["auto", "on", "off"],
                    help="gradient checkpointing of Mamba scans (auto: on for the PyTorch backend)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_items", type=int, default=0, help="debug: cap crops per split")
    ap.add_argument("--no_test", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    dev = get_device(args.device)
    amp = (not args.no_amp) and dev.type == "cuda"
    os.makedirs(args.out, exist_ok=True)

    train = CropSet(args.data, "train", fraction=args.fraction, seed=args.split_seed, max_items=args.max_items)
    val = CropSet(args.data, "val", max_items=args.max_items)
    print(f"[data] train: {len(train)} nodules / {len(train.patients)} patients (fraction {args.fraction}); "
          f"val: {len(val)} nodules / {len(val.patients)} patients")

    model = build_model(args.arch, widths=tuple(args.widths), mamba_levels=tuple(args.mamba_levels),
                        mamba_depth=args.mamba_depth, use_checkpoint={'auto': 'auto', 'on': True, 'off': False}[args.checkpointing]).to(dev)
    if args.init:
        _, missing, unexpected = load_state(model, args.init, skip_heads=True)
        print(f"[init] loaded {args.init} (re-initialised: {len(missing)} tensors, ignored: {len(unexpected)})")
    n_params = count_parameters(model)
    print(f"[model] {args.arch}: {n_params / 1e6:.2f}M params; mamba backend: "
          f"{backend_name() if args.arch == 'mamba' else '-'}; device {dev}; amp {amp}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, args.ema)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = dict(vars(args), n_params=n_params, n_train=len(train), n_train_patients=len(train.patients))

    best, hist, t0 = -1.0, [], time.time()
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, args.warmup)
        idx = rng.integers(0, len(train), args.batch)
        img, msk = train.batch(idx, dev)
        with torch.no_grad():
            img, msk = random_affine_views(img, msk, out_size=args.crop, generator=gen)
            x = intensity_augment(normalize_hu(img), generator=gen)
            y = (msk > 0.5).float()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        loss = deep_supervised_loss(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model, step)

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            rows = evaluate_set(ema.model, val, size=args.crop, tta=False, with_hd=False)
            s = summarize(rows, keys=("dice", "iou"))
            hist.append(dict(step=step + 1, loss=loss.item(), val_dice=s["dice"], val_iou=s["iou"],
                             time_s=time.time() - t0))
            flag = ""
            if s["dice"] > best:
                best = s["dice"]
                torch.save({"model": ema.model.state_dict(), "config": config, "step": step + 1,
                            "val_dice": best}, os.path.join(args.out, "best.pt"))
                flag = " *"
            print(f"step {step + 1:6d}  loss {loss.item():.4f}  val Dice {s['dice']:.4f}  IoU {s['iou']:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s{flag}", flush=True)
            save_json(hist, os.path.join(args.out, "history.json"))

    torch.save({"model": ema.model.state_dict(), "config": config, "step": args.steps},
               os.path.join(args.out, "last.pt"))
    result = dict(config=config, best_val_dice=best, train_time_s=time.time() - t0)
    if not args.no_test:
        load_state(model, os.path.join(args.out, "best.pt"))
        test = CropSet(args.data, "test", max_items=args.max_items, load_readers=True)
        rows = evaluate_set(model, test, size=args.crop, tta=True, with_readers=True)
        write_rows(rows, os.path.join(args.out, "test_per_nodule.csv"))
        result["test"] = full_report(rows)
        a = result["test"]["all"]
        print(f"[test] n={a['n']}  Dice {a['dice']:.4f}  IoU {a['iou']:.4f}  "
              f"HD95 {a.get('hd95_mm', float('nan')):.2f} mm")
    save_json(result, os.path.join(args.out, "results.json"))
    return result


if __name__ == "__main__":
    main()
