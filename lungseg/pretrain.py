"""Diffusion-denoising pretraining (DDeP) on unlabelled CT cubes.

The segmentation network (encoder + decoder, Mamba layers included) is
trained as a noise predictor. It is given a DDPM-style noised cube

    x_t = sqrt(a) * x_0 + sqrt(1 - a) * eps,   a ~ U[a_min, a_max]

and must output eps (MSE loss). No labels are needed, so every CT cube from the
training patients can be used, including patients whose nodules we pretend
are unlabelled in the low-label runs, plus random lung cubes. Val/test
patients are always excluded. Then the output head is swapped and the network
is fine-tuned with ``train.py --init``.

This is where the diffusion idea from the original proposal stays useful:
unlike a pure diffusion segmenter it is cheap at inference, and unlike
contrastive or MAE pretraining it pretrains the *decoder* too, which is what a
dense segmentation task needs (Brempong et al., "Denoising Pretraining for
Semantic Segmentation", CVPR-W 2022).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lungseg.data.dataset import UnlabeledSet, normalize_hu, random_affine_views, read_splits
from lungseg.models.mamba_unet import build_model, count_parameters
from lungseg.models.ssm import backend_name
from lungseg.utils import cosine_lr, get_device, save_json, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="mamba", choices=["mamba", "unet"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 320])
    ap.add_argument("--mamba_levels", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--mamba_depth", type=int, default=1)
    ap.add_argument("--alpha_min", type=float, default=0.3, help="min signal level sqrt(a)^2")
    ap.add_argument("--alpha_max", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--checkpointing", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_items", type=int, default=0)
    args, unknown = ap.parse_known_args(argv)  # train-only options passed via experiments --extra
    if unknown:
        print(f"[pretrain] ignoring options meant for train.py: {' '.join(unknown)}")

    seed_everything(args.seed)
    dev = get_device(args.device)
    amp = (not args.no_amp) and dev.type == "cuda"
    os.makedirs(args.out, exist_ok=True)
    splits = read_splits(args.data)
    held_out = set(splits["val"]) | set(splits["test"])
    data = UnlabeledSet(args.data, exclude_patients=held_out, max_items=args.max_items)
    print(f"[pretrain] {len(data)} unlabelled cubes (val/test patients excluded)")

    model = build_model(args.arch, widths=tuple(args.widths), mamba_levels=tuple(args.mamba_levels),
                        mamba_depth=args.mamba_depth, deep_supervision=False,
                        use_checkpoint={'auto': 'auto', 'on': True, 'off': False}[args.checkpointing]).to(dev)
    print(f"[model] {args.arch} {count_parameters(model) / 1e6:.2f}M params, backend "
          f"{backend_name() if args.arch == 'mamba' else '-'}, amp {amp}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    hist, t0, run = [], time.time(), 0.0
    for step in range(args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, 250)
        img = data.batch(rng.integers(0, len(data), args.batch), dev)
        with torch.no_grad():
            x0 = normalize_hu(random_affine_views(img, out_size=args.crop, generator=gen))
            a = torch.empty(x0.shape[0], 1, 1, 1, 1, device=dev).uniform_(args.alpha_min, args.alpha_max)
            eps = torch.randn_like(x0)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            pred = model(xt)
        loss = F.mse_loss(pred.float(), eps)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        lv = loss.item()
        run = 0.98 * run + 0.02 * lv if step else lv
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            hist.append(dict(step=step + 1, loss=run, time_s=time.time() - t0))
            print(f"step {step + 1:6d}  denoise MSE {run:.4f}  {time.time() - t0:.0f}s", flush=True)
    cfg = dict(vars(args))
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(args.out, "pretrain.pt"))
    save_json(hist, os.path.join(args.out, "history.json"))
    print(f"[pretrain] saved {os.path.join(args.out, 'pretrain.pt')}")


if __name__ == "__main__":
    main()
