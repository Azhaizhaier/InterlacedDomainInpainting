from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.sample import Sample, validate_sample

_TEST_TMP = Path(__file__).resolve().parent / ".tmp"


def make_temp_dir() -> tempfile.TemporaryDirectory:
    _TEST_TMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=_TEST_TMP)


def make_sample(root: Path, corrupt: bool = False, corrupt_large: bool = False) -> Sample:
    sample_dir = root / "scene001" / "samples" / "sample_0000"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2, 2, :] = 255
    gt = image.copy()
    if corrupt:
        gt[0, 0, 0] = 200 if corrupt_large else 1

    cv2.imwrite(str(sample_dir / "interlaced_input.png"), image)
    cv2.imwrite(str(sample_dir / "interlaced_mask.png"), mask)
    cv2.imwrite(str(sample_dir / "interlaced_gt.png"), gt)
    cv2.imwrite(str(sample_dir / "interlaced_mask_view.png"), (mask.max(axis=2) > 0).astype(np.uint8) * 255)

    meta = {
        "scene": "scene001",
        "sample_index": 0,
        "mode": "interp",
        "view_num": 8,
        "display": {"theta": 0.166666, "koff": 0, "subpixel": 4.666666},
        "files": {
            "interlaced_input": "interlaced_input.png",
            "interlaced_mask": "interlaced_mask.png",
            "interlaced_mask_view": "interlaced_mask_view.png",
            "interlaced_gt": "interlaced_gt.png",
        },
    }
    (sample_dir / "sample.json").write_text(json.dumps(meta), encoding="utf-8")
    return Sample(sample_dir, meta)


class SampleTests(unittest.TestCase):
    def test_valid_sample_passes(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            result = validate_sample(sample)
            self.assertTrue(result["pass"])
            self.assertEqual(result["valid_psnr"], float("inf"))
            self.assertTrue(result["valid_equality"])
            self.assertAlmostEqual(result["hole_ratio"], 255 / (8 * 8 * 255))

    def test_small_difference_passes_psnr(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp), corrupt=True)
            result = validate_sample(sample)
            self.assertTrue(result["pass"])
            self.assertFalse(result["valid_equality"])
            self.assertGreater(result["valid_psnr"], 30.0)

    def test_large_difference_fails_psnr(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp), corrupt=True, corrupt_large=True)
            result = validate_sample(sample)
            self.assertFalse(result["pass"])
            self.assertLess(result["valid_psnr"], 30.0)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(_TEST_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
