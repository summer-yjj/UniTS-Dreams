# 逐点分割损失：CE + Dice / Focal，支持 class_weight (auto/manual)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _dice_per_class(logits, y, num_classes, smooth=1e-5):
    """logits [B, C, T], y [B, T] long. Returns dice loss (1 - dice) per class then mean."""
    probs = F.softmax(logits, dim=1)
    B, C, T = logits.shape
    loss = 0.0
    for c in range(num_classes):
        pred_c = probs[:, c]
        target_c = (y == c).float()
        inter = (pred_c * target_c).sum(dim=(0, 1))
        union = pred_c.sum(dim=(0, 1)) + target_c.sum(dim=(0, 1)) + smooth
        loss = loss + (1.0 - (2 * inter + smooth) / union)
    return loss / max(num_classes, 1)


def compute_seg_loss(logits, y, cfg):
    """
    logits: [B, num_classes, T], y: [B, T] Long in [0..num_classes-1].
    cfg: dict with keys:
      - seg_loss: 'ce' | 'ce_dice' (default) | 'focal'
      - class_weight: 'auto' | 'manual' | None
      - class_weights: optional list/tensor for manual (length num_classes)
      - num_classes: int
      - focal_gamma: float (default 2.0)
    """
    B, num_classes, T = logits.shape
    device = logits.device
    seg_loss = (cfg.get("seg_loss") or "ce_dice").lower()
    class_weight_mode = (cfg.get("class_weight") or "auto").lower()

    if class_weight_mode == "manual" and "class_weights" in cfg:
        weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        if weights.numel() != num_classes:
            weights = F.pad(weights, (0, num_classes - weights.numel())) if weights.numel() < num_classes else weights[:num_classes]
    elif class_weight_mode == "auto" and "class_weights" in cfg:
        weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
    else:
        weights = None

    logits_flat = logits.permute(0, 2, 1).reshape(-1, num_classes)
    y_flat = y.reshape(-1)

    if weights is None and class_weight_mode == "auto":
        with torch.no_grad():
            counts = torch.bincount(y_flat.clamp(0, num_classes - 1), minlength=num_classes).float()
            counts = counts.clamp(1.0)
            median_freq = counts.median()
            weights = median_freq / counts
            weights = weights.to(device)

    ce = F.cross_entropy(logits_flat, y_flat.clamp(0, num_classes - 1), weight=weights, reduction="mean")

    if seg_loss == "ce":
        return ce
    if seg_loss == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        pt = F.softmax(logits_flat, dim=1)
        pt = pt.gather(1, y_flat.unsqueeze(1).clamp(0, num_classes - 1)).squeeze(1)
        focal_weight = (1 - pt) ** gamma
        ce_per = F.cross_entropy(logits_flat, y_flat.clamp(0, num_classes - 1), weight=weights, reduction="none")
        return (focal_weight * ce_per).mean()
    if seg_loss == "ce_dice":
        dice = _dice_per_class(logits, y, num_classes)
        return ce + dice
    return ce
