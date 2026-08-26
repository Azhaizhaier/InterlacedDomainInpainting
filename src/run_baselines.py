"""Run B0 rule-based and optional B1 LaMa baselines on generated samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .baselines.lama import SimpleLamaBaseline
from .baselines.rule_based import RULE_BASED_METHODS
from .evaluation import evaluate_prediction, per_view_psnr, write_results_csv
from .luts import build_lut
from .sample import iter_samples, load_interlaced


def run(args: argparse.Namespace) -> int:
    if args.scene:
        root = Path(args.dataset_root) / args.scene
    else:
        root = Path(args.dataset_root)
    samples = list(iter_samples(root))
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        print("no samples found")
        return 1

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    lama = SimpleLamaBaseline()
    if "lama" in methods and not lama.available:
        print(f"SimpleLama unavailable: {lama._error}")
        return 1

    rows: list[dict] = []
    for sample in samples:
        input_img = load_interlaced(sample, "interlaced_input")
        mask = load_interlaced(sample, "interlaced_mask")
        gt = load_interlaced(sample, "interlaced_gt")
        for method in methods:
            if method == "lama":
                pred = lama.fill(input_img, mask)
            else:
                pred = RULE_BASED_METHODS[method](input_img, mask)
            result = evaluate_prediction(pred, gt, input_img, mask)
            full_lut = build_lut(input_img.shape[0], input_img.shape[1], sample.display_params)
            per_view, view_psnr_mean = per_view_psnr(pred, gt, full_lut)
            row = {
                "sample": sample.name,
                "mode": sample.mode,
                "method": method,
                "view_psnr_mean": f"{view_psnr_mean:.4f}",
            }
            row.update(result.as_dict())
            for v, value in enumerate(per_view):
                row[f"view_psnr_{v}"] = f"{value:.4f}" if np.isfinite(value) else "inf"
            rows.append(row)
            print(
                f"{sample.name} {method}: "
                f"full={result.full_psnr:.2f} hole={result.hole_psnr:.2f}"
            )
            if args.save_images:
                out_dir = Path(args.out_dir) / method
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{sample.name.replace('/', '_')}.npy", pred)

    out_path = Path(args.out_dir) / "baseline_results.csv"
    write_results_csv(out_path, rows)
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.run_baselines")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="nearest,horizontal,bidirectional,morphology",
    )
    parser.add_argument("--out-dir", default="outputs/b0")
    parser.add_argument("--save-images", action="store_true")
    parser.set_defaults(func=run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
