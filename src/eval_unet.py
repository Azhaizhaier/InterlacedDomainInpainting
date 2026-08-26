"""Full-image tiled inference for a trained U-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .evaluation import evaluate_prediction, per_view_psnr, write_results_csv
from .features import build_extra_channels
from .luts import build_lut
from .models import UNet
from .sample import iter_samples, load_interlaced


def _tile_weights(height: int, width: int, overlap: int) -> np.ndarray:
    row_ramp = np.minimum(
        np.minimum(np.arange(height), height - 1 - np.arange(height)),
        overlap,
    ).astype(np.float32) + 1.0
    col_ramp = np.minimum(
        np.minimum(np.arange(width), width - 1 - np.arange(width)),
        overlap,
    ).astype(np.float32) + 1.0
    return row_ramp[:, None] * col_ramp[None, :]


@torch.no_grad()
def infer_full(
    model: torch.nn.Module,
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    input_mode: str = "baseline",
    display_params=None,
) -> np.ndarray:
    """Return normalized (0-1) full prediction composited with input."""
    height, width = image.shape[:2]
    tile_size = min(tile_size, height, width)
    image_norm = image.astype(np.float32) / 255.0
    mask_norm = (mask > 0).astype(np.float32)
    pred_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    y_positions: list[int] = []
    y = 0
    while y < height:
        y_positions.append(min(y, height - tile_size))
        y += tile_size - overlap
    x_positions: list[int] = []
    x = 0
    while x < width:
        x_positions.append(min(x, width - tile_size))
        x += tile_size - overlap

    for y0 in y_positions:
        for x0 in x_positions:
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            tile_rgb = torch.from_numpy(image_norm[y0:y1, x0:x1].transpose(2, 0, 1)).unsqueeze(0)
            tile_mask = torch.from_numpy(mask_norm[y0:y1, x0:x1].transpose(2, 0, 1)).unsqueeze(0)
            parts = [tile_rgb, tile_mask]
            if input_mode != "baseline":
                tile_lut = build_lut(y1 - y0, x1 - x0, display_params, origin=(y0, x0))
                tile_extra_np = build_extra_channels(
                    image[y0:y1, x0:x1], mask[y0:y1, x0:x1], tile_lut, input_mode
                )
                tile_extra = torch.from_numpy(tile_extra_np.transpose(2, 0, 1)).unsqueeze(0)
                parts.append(tile_extra)
            x = torch.cat(parts, dim=1).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(x).squeeze(0).cpu().numpy()
            w = _tile_weights(y1 - y0, x1 - x0, overlap)
            pred_sum[y0:y1, x0:x1] += pred.transpose(1, 2, 0) * w[:, :, None]
            weight_sum[y0:y1, x0:x1] += w

    pred = pred_sum / np.maximum(weight_sum[:, :, None], 1e-6)
    pred = np.clip(pred, 0.0, 1.0)
    composed = image_norm * (1.0 - mask_norm) + pred * mask_norm
    return composed


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    if args.scene:
        root = root / args.scene
    samples = [s for s in iter_samples(root) if s.sample_index == args.sample_index]
    if not samples:
        print(f"sample {args.sample_index} not found")
        return 1
    sample = samples[0]

    image = load_interlaced(sample, "interlaced_input")
    mask = load_interlaced(sample, "interlaced_mask")
    gt = load_interlaced(sample, "interlaced_gt")

    if args.input_mode == "baseline":
        in_channels = 6
    elif args.input_mode == "view_id":
        in_channels = 9
    else:
        in_channels = 6 + 3 + 6 * 4
    model = UNet(in_channels=in_channels, out_channels=3, base=args.base)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()

    composed = infer_full(
        model,
        image,
        mask,
        args.tile_size,
        args.overlap,
        device,
        input_mode=args.input_mode,
        display_params=sample.display_params,
    )
    pred = (composed * 255.0).astype(np.uint8)
    result = evaluate_prediction(pred, gt, image, mask)
    full_lut = build_lut(image.shape[0], image.shape[1], sample.display_params)
    per_view, view_psnr_mean = per_view_psnr(pred, gt, full_lut)
    row = {
        "sample": sample.name,
        "mode": sample.mode,
        "method": f"unet_tiled_{args.input_mode}",
        "view_psnr_mean": f"{view_psnr_mean:.4f}",
    }
    row.update(result.as_dict())
    for v, value in enumerate(per_view):
        row[f"view_psnr_{v}"] = f"{value:.4f}" if np.isfinite(value) else "inf"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(out_dir / f"{args.input_mode}_full_eval.csv", [row])
    print(
        f"{sample.name} {row['method']}: "
        f"full={result.full_psnr:.2f} hole={result.hole_psnr:.2f} "
        f"ssim={result.full_ssim:.4f} view_psnr={view_psnr_mean:.2f}"
    )

    if args.save_image:
        out_path = out_dir / "unet_tiled_sample_0000.png"
        cv2.imwrite(str(out_path), pred)
        print(f"saved {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.eval_unet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="outputs/b2_full")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument(
        "--input-mode",
        choices=["baseline", "view_id", "neighbor"],
        default="baseline",
    )
    parser.set_defaults(func=run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
