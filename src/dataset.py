"""Dataset discovery and hole-aware crop sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .sample import Sample, load_interlaced


def _hole_block_grid(mask_view: np.ndarray, block: int = 16) -> np.ndarray:
    """Boolean grid marking blocks that contain at least one hole pixel."""
    height, width = mask_view.shape
    rows = (height + block - 1) // block
    cols = (width + block - 1) // block
    grid = np.zeros((rows, cols), dtype=bool)
    ys, xs = np.where(mask_view > 0)
    if ys.size:
        grid[ys // block, xs // block] = True
    return grid


def _crop_origin_around_block(
    block_y: int,
    block_x: int,
    crop_size: int,
    height: int,
    width: int,
    block: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    lo_y = max(0, block_y * block - crop_size + 1)
    hi_y = min(block_y * block, height - crop_size)
    lo_x = max(0, block_x * block - crop_size + 1)
    hi_x = min(block_x * block, width - crop_size)
    return int(rng.integers(lo_y, hi_y + 1)), int(rng.integers(lo_x, hi_x + 1))


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    gt: np.ndarray,
    mask_view: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
    force_hole: bool = False,
    prefer_large_hole: bool = False,
    large_hole_candidates: int = 8,
) -> dict:
    """Return a random crop with optional hole-aware sampling.

    ``force_hole=True`` biases the origin toward a mask block that contains a
    hole.  If the image has no holes at all, a uniform crop is returned.
    """
    height, width = mask_view.shape
    crop_size = min(crop_size, height, width)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    if force_hole and np.any(mask_view > 0):
        block = min(16, max(4, crop_size // 8))
        grid = _hole_block_grid(mask_view, block=block)
        hole_rows, hole_cols = np.where(grid)
        candidate_count = max(1, large_hole_candidates if prefer_large_hole else 1)
        candidates: list[tuple[float, int, int]] = []
        for _ in range(candidate_count):
            pick = int(rng.integers(0, len(hole_rows)))
            candidate_y, candidate_x = _crop_origin_around_block(
                int(hole_rows[pick]), int(hole_cols[pick]), crop_size,
                height, width, block, rng,
            )
            crop_mask = mask[
                candidate_y : candidate_y + crop_size,
                candidate_x : candidate_x + crop_size,
            ]
            candidates.append((float((crop_mask > 0).mean()), candidate_y, candidate_x))
        _, y0, x0 = max(candidates, key=lambda item: item[0])
    else:
        y0 = int(rng.integers(0, height - crop_size + 1))
        x0 = int(rng.integers(0, width - crop_size + 1))

    return {
        "image": image[y0 : y0 + crop_size, x0 : x0 + crop_size].copy(),
        "mask": mask[y0 : y0 + crop_size, x0 : x0 + crop_size].copy(),
        "gt": gt[y0 : y0 + crop_size, x0 : x0 + crop_size].copy(),
        "mask_view": mask_view[y0 : y0 + crop_size, x0 : x0 + crop_size].copy(),
        "hole_ratio": float((mask[y0 : y0 + crop_size, x0 : x0 + crop_size] == 255).mean()),
        "origin": (y0, x0),
    }


def random_hole_aware_crop(sample: Sample, crop_size: int, rng: np.random.Generator) -> dict:
    """Load one sample and return a hole-aware crop."""
    image = load_interlaced(sample, "interlaced_input")
    mask = load_interlaced(sample, "interlaced_mask")
    gt = load_interlaced(sample, "interlaced_gt")
    mask_view = cv2.imread(str(sample.file_path("interlaced_mask_view")), cv2.IMREAD_UNCHANGED)
    if mask_view is None or mask_view.ndim != 2:
        mask_view = (mask.max(axis=2) > 0).astype(np.uint8) * 255
    return random_crop(image, mask, gt, mask_view, crop_size, rng, force_hole=True)


class InterlacedCropSampler:
    """Infinite iterator that yields hole-aware crops from the dataset."""

    def __init__(
        self,
        samples: list[Sample],
        crop_size: int,
        hole_prob: float = 0.5,
        seed: int | None = None,
    ):
        if not samples:
            raise ValueError("samples must not be empty")
        self.samples = samples
        self.crop_size = crop_size
        self.hole_prob = float(hole_prob)
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> "InterlacedCropSampler":
        return self

    def __next__(self) -> dict:
        sample = self.samples[int(self.rng.integers(0, len(self.samples)))]
        force_hole = self.rng.random() < self.hole_prob
        image = load_interlaced(sample, "interlaced_input")
        mask = load_interlaced(sample, "interlaced_mask")
        gt = load_interlaced(sample, "interlaced_gt")
        mask_view = cv2.imread(str(sample.file_path("interlaced_mask_view")), cv2.IMREAD_UNCHANGED)
        if mask_view is None or mask_view.ndim != 2:
            mask_view = (mask.max(axis=2) > 0).astype(np.uint8) * 255
        return random_crop(image, mask, gt, mask_view, self.crop_size, self.rng, force_hole=force_hole)


def iter_samples(root: str | Path) -> Iterator[Sample]:
    """Re-export for callers that prefer importing from dataset."""
    from .sample import iter_samples as _iter

    return _iter(root)
