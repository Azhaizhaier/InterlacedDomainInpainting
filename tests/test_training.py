from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.train_unet import CachedCropDataset, parse_scene_list, select_samples
from tests.test_sample import make_sample, make_temp_dir


class TrainingDatasetTests(unittest.TestCase):
    def test_scene_list_is_trimmed_and_deduplicated(self) -> None:
        self.assertEqual(parse_scene_list("scene001, scene002,scene001"), ("scene001", "scene002"))

    def test_select_samples_rejects_missing_scene(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            with self.assertRaisesRegex(ValueError, "scene009"):
                select_samples([sample], ("scene001", "scene009"), "train")

    def test_cache_is_bounded(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            dataset = CachedCropDataset([sample], crop_size=4, length=2, cache_samples=1)
            self.assertEqual(dataset[0]["x"].shape, (6, 4, 4))
            self.assertEqual(dataset[1]["x"].shape, (6, 4, 4))
            self.assertLessEqual(len(dataset._cache), 1)

    def test_crop_is_deterministic_by_index(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            dataset = CachedCropDataset([sample], crop_size=4, length=2, seed=7)
            self.assertTrue(np.array_equal(dataset[0]["x"].numpy(), dataset[0]["x"].numpy()))

    def test_crop_group_reuses_one_sample(self) -> None:
        with make_temp_dir() as tmp:
            sample = make_sample(Path(tmp))
            dataset = CachedCropDataset(
                [sample], crop_size=4, length=8, cache_samples=1, crops_per_sample=8
            )
            for index in range(8):
                dataset[index]
            self.assertEqual(len(dataset._cache), 1)


if __name__ == "__main__":
    unittest.main()
