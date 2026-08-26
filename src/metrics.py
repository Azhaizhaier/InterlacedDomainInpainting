"""Image-quality metrics for interlaced-domain evaluation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR over all pixels or only pixels where ``mask`` is True."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    pred_f = pred.astype(np.float32)
    target_f = target.astype(np.float32)
    if mask is not None:
        if mask.shape != pred.shape:
            raise ValueError("mask must match pred shape")
        if not mask.any():
            return float("inf")
        pred_f = pred_f[mask]
        target_f = target_f[mask]
    mse = float(np.mean((pred_f - target_f) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Multi-channel SSIM over the full image."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    return float(
        structural_similarity(
            pred,
            target,
            channel_axis=-1,
            data_range=255,
        )
    )


def hole_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def equality_ratio(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.mean(a == b))
    return float(np.mean(a[mask] == b[mask]))
