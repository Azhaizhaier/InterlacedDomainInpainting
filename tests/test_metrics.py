from __future__ import annotations

import unittest

import numpy as np

from src.evaluation import evaluate_prediction
from src.metrics import psnr, ssim


class MetricsTests(unittest.TestCase):
    def test_psnr_identical_is_inf(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(psnr(image, image), float("inf"))

    def test_psnr_known_value(self) -> None:
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 255
        expected = 10.0 * np.log10(255.0**2 / (255.0**2 / 16))
        self.assertAlmostEqual(psnr(a, b), expected)

    def test_ssim_runs(self) -> None:
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        b = np.full((16, 16, 3), 10, dtype=np.uint8)
        self.assertLess(ssim(a, b), 1.0)

    def test_evaluation_metrics(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[2, 2, :] = 255
        result = evaluate_prediction(image, image, image, mask)
        self.assertEqual(result.full_psnr, float("inf"))
        self.assertAlmostEqual(result.hole_ratio, 3 / (8 * 8 * 3))


if __name__ == "__main__":
    unittest.main()
