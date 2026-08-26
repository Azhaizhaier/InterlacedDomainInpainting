from __future__ import annotations

import unittest

import numpy as np

from src.features import SAME_VIEW_OFFSETS, build_extra_channels, same_view_neighbor_features
from src.luts import DisplayParams, build_lut


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)
        self.image[8, 8, :] = 200
        self.mask = np.zeros_like(self.image, dtype=np.uint8)
        self.mask[8, 8, :] = 255
        self.lut = build_lut(16, 16, DisplayParams(theta=1 / 6, subpixel=14 / 3))

    def test_neighbor_feature_shape(self) -> None:
        features = same_view_neighbor_features(self.image, self.mask)
        self.assertEqual(features.shape, (16, 16, 6 * len(SAME_VIEW_OFFSETS)))

    def test_extra_channels(self) -> None:
        view_extra = build_extra_channels(self.image, self.mask, self.lut, "view_id")
        self.assertEqual(view_extra.shape[-1], 3)
        neighbor_extra = build_extra_channels(self.image, self.mask, self.lut, "neighbor")
        self.assertEqual(neighbor_extra.shape[-1], 3 + 6 * len(SAME_VIEW_OFFSETS))

    def test_origin_shifts_lut(self) -> None:
        params = DisplayParams(theta=1 / 6, subpixel=14 / 3)
        base = build_lut(4, 4, params)
        shifted = build_lut(4, 4, params, origin=(1, 1))
        self.assertFalse(np.array_equal(base, shifted))


if __name__ == "__main__":
    unittest.main()
