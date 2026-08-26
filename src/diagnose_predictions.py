"""Diagnose clipped/highlight/dark pixels in saved inpainting PNGs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .sample import iter_samples, load_interlaced


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    return cv2.dilate(binary.astype(np.uint8), kernel) > 0


def _ratio(condition: np.ndarray, region: np.ndarray) -> float:
    return float(condition[region].mean()) if np.any(region) else 0.0


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    prediction_root = Path(args.prediction_root)
    scenes = {value.strip() for value in args.scenes.split(",") if value.strip()}
    rows: list[dict[str, str]] = []
    for sample in iter_samples(root):
        if sample.scene not in scenes:
            continue
        path = prediction_root / sample.scene / f"{sample.root.name}.png"
        pred = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if pred is None:
            continue
        if pred.ndim != 3 or pred.shape[2] != 3 or pred.dtype != np.uint8:
            raise ValueError(f"expected uint8 3-channel PNG: {path}")
        mask = load_interlaced(sample, "interlaced_mask")
        gt = load_interlaced(sample, "interlaced_gt")
        hole = np.any(mask > 0, axis=2)
        boundary = _dilate(hole, args.boundary_radius) ^ hole
        inner_boundary = hole & _dilate(~hole, args.boundary_radius)
        edge_hole = inner_boundary
        interior_hole = hole & ~inner_boundary
        bright = pred >= args.bright_threshold
        dark = pred <= args.dark_threshold
        gt_bright = gt >= args.bright_threshold
        row = {
            "sample": sample.name,
            "scene": sample.scene,
            "mode": sample.mode,
            "hole_ratio": f"{(mask > 0).mean():.6f}",
            "hole_bright_ratio": f"{_ratio(bright, hole):.6f}",
            "edge_hole_bright_ratio": f"{_ratio(bright, edge_hole):.6f}",
            "interior_hole_bright_ratio": f"{_ratio(bright, interior_hole):.6f}",
            "hole_dark_ratio": f"{_ratio(dark, hole):.6f}",
            "edge_hole_dark_ratio": f"{_ratio(dark, edge_hole):.6f}",
            "gt_hole_bright_ratio": f"{_ratio(gt_bright, hole):.6f}",
            "boundary_mae": f"{np.abs(pred[edge_hole].astype(np.float32) - gt[edge_hole]).mean():.4f}",
        }
        rows.append(row)
    if not rows:
        print("no matching predictions")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} samples)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.diagnose_predictions")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--scenes", default="scene010,scene012")
    parser.add_argument("--bright-threshold", type=int, default=250)
    parser.add_argument("--dark-threshold", type=int, default=5)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
