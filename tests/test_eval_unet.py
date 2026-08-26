from __future__ import annotations

import unittest

import numpy as np
import torch

from src.eval_unet import _tile_weights, infer_full
from src.luts import DisplayParams
from src.models import UNet


class EvalUnetTests(unittest.TestCase):
    def test_tile_weights_are_positive_at_boundaries(self) -> None:
        weights = _tile_weights(8, 8, 2)
        self.assertTrue(np.all(weights > 0))

    def test_nonbaseline_inference_supports_tiled_features(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros_like(image)
        model = UNet(in_channels=9, out_channels=3, base=4).eval()
        output = infer_full(
            model, image, mask, tile_size=16, overlap=4,
            device=torch.device("cpu"), input_mode="view_id",
            display_params=DisplayParams(),
        )
        self.assertEqual(output.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
