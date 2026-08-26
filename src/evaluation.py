"""Evaluation helpers for predictions in the interlaced domain."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .luts import deinterlace_to_planes
from .metrics import equality_ratio, hole_ratio, psnr, ssim


@dataclass
class EvalResult:
    full_psnr: float
    hole_psnr: float
    valid_psnr: float
    full_ssim: float
    hole_ratio: float
    valid_equality: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "full_psnr": f"{self.full_psnr:.4f}",
            "hole_psnr": f"{self.hole_psnr:.4f}",
            "valid_psnr": f"{self.valid_psnr:.4f}",
            "full_ssim": f"{self.full_ssim:.6f}",
            "hole_ratio": f"{self.hole_ratio:.6f}",
            "valid_equality": f"{self.valid_equality:.6f}",
        }


def evaluate_prediction(
    pred: np.ndarray,
    gt: np.ndarray,
    source_input: np.ndarray,
    mask: np.ndarray,
) -> EvalResult:
    """Evaluate a filled interlaced image against GT.

    ``pred`` is expected to already be composited as
    ``source_input * (1-mask) + network_output * mask``.
    """
    hole = mask > 0
    valid = ~hole
    return EvalResult(
        full_psnr=psnr(pred, gt),
        hole_psnr=psnr(pred, gt, hole),
        valid_psnr=psnr(pred, gt, valid),
        full_ssim=ssim(pred, gt),
        hole_ratio=hole_ratio(mask),
        valid_equality=equality_ratio(pred, source_input, valid),
    )


def write_results_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_view_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    lut: np.ndarray,
    view_num: int | None = None,
) -> tuple[list[float], float]:
    """Per-view PSNR on sampled subpixel positions.

    De-interlacing only rearranges observed subpixels, so this is a
    view-aware grouping of the same interlaced-domain error.
    """
    pred_planes = deinterlace_to_planes(pred, lut, view_num)
    gt_planes = deinterlace_to_planes(gt, lut, view_num)
    per_view: list[float] = []
    for v in range(pred_planes.shape[0]):
        mask = np.isfinite(pred_planes[v]) & np.isfinite(gt_planes[v])
        if not mask.any():
            per_view.append(float("inf"))
            continue
        mse = float(np.mean((pred_planes[v][mask] - gt_planes[v][mask]) ** 2))
        per_view.append(float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse)))
    finite = [v for v in per_view if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("inf")
    return per_view, mean


def per_view_l1(
    pred: np.ndarray,
    gt: np.ndarray,
    lut: np.ndarray,
    view_num: int | None = None,
) -> tuple[list[float], float]:
    """Per-view mean absolute error on sampled subpixel positions."""
    pred_planes = deinterlace_to_planes(pred, lut, view_num)
    gt_planes = deinterlace_to_planes(gt, lut, view_num)
    per_view: list[float] = []
    for v in range(pred_planes.shape[0]):
        mask = np.isfinite(pred_planes[v]) & np.isfinite(gt_planes[v])
        if not mask.any():
            per_view.append(float("nan"))
            continue
        per_view.append(float(np.mean(np.abs(pred_planes[v][mask] - gt_planes[v][mask]))))
    finite = [v for v in per_view if np.isfinite(v)]
    mean = float(np.mean(finite)) if finite else float("nan")
    return per_view, mean
