"""Export tiled predictions and metrics for all frozen test-set samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch

from .eval_unet import infer_full
from .evaluation import evaluate_prediction, per_view_psnr
from .luts import build_lut
from .models import UNet
from .sample import iter_samples, load_interlaced


def _channels(input_mode: str) -> int:
    return {"baseline": 6, "view_id": 9, "neighbor": 33}[input_mode]


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    scenes = tuple(part.strip() for part in args.scenes.split(",") if part.strip())
    samples = [sample for sample in iter_samples(root) if sample.scene in scenes]
    sample_indices = {
        int(part.strip()) for part in args.sample_indices.split(",") if part.strip()
    } if args.sample_indices else None
    if sample_indices is not None:
        samples = [sample for sample in samples if sample.sample_index in sample_indices]
    if not samples:
        print("no test samples found")
        return 1
    samples.sort(key=lambda sample: (sample.scene, sample.sample_index))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = UNet(_channels(args.input_mode), 3, args.base)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    out_dir = Path(args.out_dir)
    prediction_dir = out_dir / "predictions"
    rows: list[dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        image = load_interlaced(sample, "interlaced_input")
        mask = load_interlaced(sample, "interlaced_mask")
        gt = load_interlaced(sample, "interlaced_gt")
        composed = infer_full(
            model, image, mask, args.tile_size, args.overlap, device,
            input_mode=args.input_mode, display_params=sample.display_params,
        )
        pred = (composed * 255.0).clip(0, 255).astype("uint8")
        scene_dir = prediction_dir / sample.scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        pred_path = scene_dir / f"{sample.root.name}.png"
        cv2.imwrite(str(pred_path), pred)

        result = evaluate_prediction(pred, gt, image, mask)
        lut = build_lut(image.shape[0], image.shape[1], sample.display_params)
        per_view, view_mean = per_view_psnr(pred, gt, lut)
        row = {
            "sample": sample.name,
            "scene": sample.scene,
            "sample_index": str(sample.sample_index),
            "mode": sample.mode,
            "method": f"unet_tiled_{args.input_mode}",
            "prediction": str(pred_path),
            "view_psnr_mean": f"{view_mean:.4f}",
        }
        row.update({key: str(value) for key, value in result.as_dict().items()})
        for view, value in enumerate(per_view):
            row[f"view_psnr_{view}"] = f"{value:.4f}" if value == float("inf") else f"{value:.4f}"
        rows.append(row)
        if index % args.log_interval == 0 or index == len(samples):
            print(f"{args.input_mode}: {index}/{len(samples)} {sample.name} hole={result.hole_psnr:.2f}")

    csv_path = out_dir / f"{args.input_mode}_test_eval.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} and {len(rows)} predictions")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.eval_test_set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes", default="scene010,scene012")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-indices", default=None, help="comma-separated indices selected in every scene")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-mode", choices=["baseline", "view_id", "neighbor"], required=True)
    parser.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
