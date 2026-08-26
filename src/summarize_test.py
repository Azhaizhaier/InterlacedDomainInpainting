"""Summarize frozen test CSVs by scene, mode, and hole-ratio bucket."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def run(args: argparse.Namespace) -> int:
    path = Path(args.input)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty evaluation CSV")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ratio = float(row["hole_ratio"])
        bucket = "small_<1%" if ratio < 0.01 else "medium_1-3%" if ratio < 0.03 else "large_>=3%"
        groups[(row["scene"], row["mode"], bucket)].append(row)

    summary = []
    for (scene, mode, bucket), group in sorted(groups.items()):
        summary.append({
            "scene": scene,
            "mode": mode,
            "hole_bucket": bucket,
            "samples": str(len(group)),
            "hole_ratio_mean": f"{_mean(group, 'hole_ratio'):.6f}",
            "full_psnr_mean": f"{_mean(group, 'full_psnr'):.4f}",
            "hole_psnr_mean": f"{_mean(group, 'hole_psnr'):.4f}",
            "valid_psnr_mean": f"{_mean(group, 'valid_psnr'):.4f}",
            "ssim_mean": f"{_mean(group, 'full_ssim'):.6f}",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {output} ({len(summary)} groups)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
