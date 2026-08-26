from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import InterlacedCropSampler, random_hole_aware_crop
from tests.test_sample import make_sample, make_temp_dir


class DatasetTests(unittest.TestCase):
    def test_hole_aware_crop_contains_hole(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            rng = np.random.default_rng(0)
            crop = random_hole_aware_crop(sample, 4, rng)
            self.assertEqual(crop["image"].shape, (4, 4, 3))
            self.assertGreater(crop["hole_ratio"], 0.0)

    def test_sampler_yields_crops(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            sampler = InterlacedCropSampler([sample], crop_size=4, seed=1)
            first = next(sampler)
            second = next(sampler)
            self.assertEqual(first["image"].shape, (4, 4, 3))
            self.assertEqual(second["image"].shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
