"""Optional SimpleLama baseline wrapper."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class SimpleLamaBaseline:
    """Wraps ``simple_lama_inpainting`` when it is installed."""

    def __init__(self) -> None:
        self._error: str | None = None
        self._model = None
        try:
            from simple_lama_inpainting import SimpleLama

            self._model = SimpleLama()
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._error = str(exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    def fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"SimpleLama is not available: {self._error}")
        mask_view = (mask.max(axis=2) > 0).astype(np.uint8) * 255
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._model(Image.fromarray(rgb), Image.fromarray(mask_view))
        result_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        composited = image.copy()
        composited[mask > 0] = result_bgr[mask > 0]
        return composited
