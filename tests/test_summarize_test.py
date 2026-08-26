from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.summarize_test import run


class SummarizeTestTests(unittest.TestCase):
    def test_groups_large_holes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            output = Path(tmp) / "output.csv"
            fields = ["scene", "mode", "hole_ratio", "full_psnr", "hole_psnr", "valid_psnr", "full_ssim"]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(scene="scene010", mode="extrap", hole_ratio="0.04", full_psnr="20", hole_psnr="10", valid_psnr="30", full_ssim="0.8"))
            run(type("Args", (), {"input": str(source), "output": str(output)})())
            row = next(csv.DictReader(output.open(encoding="utf-8")))
            self.assertEqual(row["hole_bucket"], "large_>=3%")


if __name__ == "__main__":
    unittest.main()
