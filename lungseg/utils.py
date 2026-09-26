from __future__ import annotations

import copy
import json
import math
import os
import random

import numpy as np
import torch

from lungseg.data.dataset import center_crop, normalize_hu


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step=None):
        d = self.decay if step is None else min(self.decay, (1 + step) / (10 + step))
        for pe, pm in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)


def cosine_lr(step, total, base, warmup=200, final=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(t, 1.0)))


FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def predict_probs(model, images_hu, size=64, tta=True, amp=True, batch_size=8):
    """images_hu: (N,1,S,S,S) float HU tensor on device. Returns (N,1,size,size,size) probs (cpu)."""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, images_hu.shape[0], batch_size):
        x = normalize_hu(center_crop(images_hu[i:i + batch_size].to(dev), size))
        acc = 0
        flips = FLIPS if tta else [()]
        for f in flips:
            xi = x.flip(f) if f else x
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                y = model(xi)
            y = torch.sigmoid(y.float())
            acc = acc + (y.flip(f) if f else y)
        outs.append((acc / len(flips)).cpu())
    return torch.cat(outs)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_state(model, path, skip_heads=False, strict=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    if skip_heads:
        sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "ds_heads."))}
        strict = False
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return ck, missing, unexpected


def fit_steps_to_budget(step, elapsed_s, steps, max_minutes, reserve=0.12):
    """Shrink the total number of steps so training ends within the budget
    (with ``reserve`` of it left for validation / testing)."""
    rate = elapsed_s / max(step, 1)
    fit = int((max_minutes * 60 * (1 - reserve)) / rate)
    new = max(min(steps, fit), step + 1)
    print(f"[budget] {rate:.2f} s/step -> {'keeping' if new == steps else 'reducing to'} {new} steps "
          f"to fit {max_minutes:.0f} min", flush=True)
    return new
