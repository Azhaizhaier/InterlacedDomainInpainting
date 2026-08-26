from __future__ import annotations

import unittest

import numpy as np

from src.luts import (
    DisplayParams,
    build_lut,
    deinterlace_to_planes,
    find_exact_periods,
    reinterlace_from_planes,
    view_statistics,
)


class LutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = DisplayParams(theta=1 / 6, subpixel=14 / 3)
        self.lut = build_lut(64, 64, self.params)

    def test_exact_period_1_6(self) -> None:
        self.assertTrue(np.array_equal(self.lut[6:, 1:], self.lut[:-6, :-1]))

    def test_periods_discovered(self) -> None:
        periods = find_exact_periods(self.lut, max_dx=2, max_dy=6)
        self.assertIn((1, 6), periods)

    def test_each_pixel_has_three_views(self) -> None:
        counts = np.array([len(np.unique(self.lut[y, x, :])) for y in range(64) for x in range(64)])
        self.assertTrue(np.all(counts == 3))

    def test_view_coverage(self) -> None:
        stats = view_statistics(self.lut)
        for coverage in stats["pixel_coverage"]:
            self.assertGreaterEqual(coverage, 0.33)
            self.assertLessEqual(coverage, 0.42)

    def test_deinterlace_reinterlace_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        lut = build_lut(16, 16, self.params)
        planes = deinterlace_to_planes(image, lut)
        restored = reinterlace_from_planes(planes, lut)
        self.assertTrue(np.array_equal(image, restored))


if __name__ == "__main__":
    unittest.main()
