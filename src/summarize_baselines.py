"""Combine B0/B1/B2 evaluation CSVs into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    rows: list[dict] = []
    for path in args.inputs:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        print("no rows")
        return 1

    fieldnames = list(rows[0].keys())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'method':<16}{'full_psnr':>10}{'hole_psnr':>10}{'ssim':>10}")
    for row in rows:
        print(
            f"{row.get('method', ''):<16}"
            f"{row.get('full_psnr', ''):>10}"
            f"{row.get('hole_psnr', ''):>10}"
            f"{row.get('full_ssim', ''):>10}"
        )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.summarize_baselines")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/baseline_summary.csv")
    parser.set_defaults(func=run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
