from __future__ import annotations

import unittest

import numpy as np

from src.baselines.rule_based import (
    bidirectional_fill,
    horizontal_interp_fill,
    morphological_fill,
    nearest_fill,
    region_growing_fill,
    vertical_interp_fill,
)


class RuleBasedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image[1:7, 1:7, :] = 255
        self.mask = np.zeros_like(self.image, dtype=np.uint8)
        self.mask[3:5, 3:5, :] = 255

    def test_fillers_remove_holes(self) -> None:
        methods = [
            nearest_fill,
            horizontal_interp_fill,
            vertical_interp_fill,
            bidirectional_fill,
            morphological_fill,
            region_growing_fill,
        ]
        for method in methods:
            with self.subTest(method=method.__name__):
                result = method(self.image, self.mask)
                self.assertTrue(np.all(result[self.mask > 0] > 0))
                self.assertTrue(np.array_equal(result[self.mask == 0], self.image[self.mask == 0]))


if __name__ == "__main__":
    unittest.main()
