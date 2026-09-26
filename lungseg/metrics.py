"""Losses, per-nodule metrics and post-processing."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


# ----------------------------------------------------------------------------- losses
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def seg_loss(logits, target):
    return soft_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits.float(), target)


def deep_supervised_loss(out, target, weights=(1.0, 0.5, 0.25)):
    if isinstance(out, tuple):
        main, aux = out
    else:
        main, aux = out, []
    loss = weights[0] * seg_loss(main, target)
    for w, a in zip(weights[1:], aux):
        t = F.adaptive_avg_pool3d(target, a.shape[-3:])
        loss = loss + w * seg_loss(a, t)
    return loss / sum(weights[:1 + len(aux)])


# ----------------------------------------------------------------------------- post-processing
def keep_central_component(mask: np.ndarray, radius_vox: float = 6.0) -> np.ndarray:
    """Keep the connected component(s) touching a small sphere at the crop centre;
    if none does, keep the component closest to the centre. The crops are
    nodule-centred, so this removes vessel / neighbouring-nodule false positives."""
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 1:
        return mask
    c = (np.array(mask.shape) - 1) / 2.0
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    sphere = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= radius_vox ** 2
    ids = np.unique(lab[sphere & (lab > 0)])
    if len(ids) == 0:
        coms = ndimage.center_of_mass(mask, lab, range(1, n + 1))
        d = [np.sum((np.array(cm) - c) ** 2) for cm in coms]
        ids = [int(np.argmin(d)) + 1]
    return np.isin(lab, ids).astype(mask.dtype)


# ----------------------------------------------------------------------------- metrics
def _surface(m):
    return m ^ ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: float) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def binary_metrics(pred: np.ndarray, gt: np.ndarray, spacing: float = 0.8, with_hd=True) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    out = dict(dice=dice, iou=iou,
               precision=tp / (tp + fp) if tp + fp > 0 else 0.0,
               recall=tp / (tp + fn) if tp + fn > 0 else 0.0,
               vol_pred_mm3=(tp + fp) * spacing ** 3, vol_gt_mm3=(tp + fn) * spacing ** 3)
    if with_hd:
        out["hd95_mm"] = hd95(pred, gt, spacing)
    return out


def reader_agreement(readers: np.ndarray, spacing: float = 0.8) -> dict | None:
    """Leave-one-reader-out agreement: each radiologist vs the 50% consensus of
    the others. This is the human inter-observer reference for the metrics."""
    R = readers.shape[0]
    if R < 2:
        return None
    dices, ious = [], []
    for r in range(R):
        others = np.delete(readers, r, axis=0).mean(0) >= 0.5
        m = binary_metrics(readers[r], others, spacing, with_hd=False)
        dices.append(m["dice"])
        ious.append(m["iou"])
    return dict(dice=float(np.mean(dices)), iou=float(np.mean(ious)))


def summarize(rows: list, keys=("dice", "iou", "precision", "recall", "hd95_mm")) -> dict:
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] == r[k]], float)
        if len(v):
            out[k] = float(v.mean())
            out[k + "_std"] = float(v.std())
            out[k + "_median"] = float(np.median(v))
    return out
