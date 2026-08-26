"""M1 command-line tools: validation and dataset statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .sample import collect_sample_stats, iter_samples


def _format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def run_validate(args: argparse.Namespace) -> int:
    records = collect_sample_stats(args.dataset_root, max_samples=args.max_samples)
    if not records:
        print("no samples found")
        return 1

    failures = [r for r in records if not r["pass"]]
    for r in records:
        fields = {
            "sample": r["sample"],
            "mode": r["mode"],
            "hole_ratio": _format_percent(r.get("hole_ratio", 0.0)),
            "valid_psnr": f"{r.get('valid_psnr', float('nan')):.2f}dB",
        }
        if "equal_ratio" in r:
            fields["equal_ratio"] = _format_percent(r["equal_ratio"])
        if args.strict:
            fields["valid_equality"] = r.get("valid_equality")
        fields["pass"] = r["pass"]
        print("\t".join(f"{k}={v}" for k, v in fields.items()))

    print(f"validated {len(records)} samples, {len(failures)} failures")
    if failures:
        for r in failures[:20]:
            print(f"FAIL: {r['sample']} {r}")
        return 1
    return 0


def run_stats(args: argparse.Namespace) -> int:
    stats: dict[str, list[float]] = defaultdict(list)
    scene_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in iter_samples(args.dataset_root):
        from .sample import load_interlaced, hole_ratio

        mask = load_interlaced(sample, "interlaced_mask")
        ratio = hole_ratio(mask)
        stats[sample.mode].append(ratio)
        scene_stats[sample.scene][sample.mode].append(ratio)

    print("overall hole ratio by mode:")
    for mode, ratios in sorted(stats.items()):
        if ratios:
            print(f"  {mode}: n={len(ratios)}, mean={_format_percent(sum(ratios) / len(ratios))}")

    print("hole ratio by scene/mode:")
    for scene in sorted(scene_stats):
        for mode, ratios in sorted(scene_stats[scene].items()):
            if ratios:
                print(
                    f"  {scene}/{mode}: n={len(ratios)}, "
                    f"mean={_format_percent(sum(ratios) / len(ratios))}"
                )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate generated samples")
    validate.add_argument("dataset_root", type=Path)
    validate.add_argument("--strict", action="store_true", help="also print exact GT equality diagnostic")
    validate.add_argument("--max-samples", type=int, default=None)
    validate.set_defaults(func=run_validate)

    stats = sub.add_parser("stats", help="report hole ratios by mode and scene")
    stats.add_argument("dataset_root", type=Path)
    stats.set_defaults(func=run_stats)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
