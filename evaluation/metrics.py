import torch

def dice_coefficient(pred, target, smooth=1.0):
    """
    Computes standard Dice Coefficient.
    pred and target are tensors with values in [0, 1].
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def compute_iou(pred, target, threshold=0.5):
    """
    Computes Intersection over Union / Jaccard Index.
    """
    pred_bin = (pred > threshold).float().reshape(-1)
    target_bin = (target > threshold).float().reshape(-1)
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou
