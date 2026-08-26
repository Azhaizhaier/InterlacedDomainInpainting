"""Per-channel rule-based fillers for subpixel hole masks."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt


def _as_3d_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.repeat((mask > 0)[:, :, None], image.shape[2], axis=2)
    return mask > 0


def nearest_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy the nearest valid subpixel value in each channel."""
    result = image.copy()
    hole = _as_3d_mask(mask, image)
    for c in range(image.shape[2]):
        channel_hole = hole[:, :, c]
        if not channel_hole.any():
            continue
        _, indices = distance_transform_edt(~channel_hole, return_indices=True)
        result[:, :, c][channel_hole] = image[:, :, c][tuple(indices[:, channel_hole])]
    return result


def horizontal_interp_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation along each row, with nearest values at borders."""
    result = image.copy()
    hole = _as_3d_mask(mask, image)
    for c in range(image.shape[2]):
        channel_hole = hole[:, :, c]
        if not channel_hole.any():
            continue
        for y in range(image.shape[0]):
            valid_idx = np.flatnonzero(~channel_hole[y])
            if valid_idx.size == 0:
                continue
            hole_idx = np.flatnonzero(channel_hole[y])
            if hole_idx.size == 0:
                continue
            values = image[y, valid_idx, c].astype(np.float32)
            filled = np.interp(hole_idx, valid_idx, values)
            result[y, hole_idx, c] = np.clip(np.rint(filled), 0, 255).astype(np.uint8)
    return result


def vertical_interp_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation along each column, with nearest values at borders."""
    result = image.copy()
    hole = _as_3d_mask(mask, image)
    for c in range(image.shape[2]):
        channel_hole = hole[:, :, c]
        if not channel_hole.any():
            continue
        for x in range(image.shape[1]):
            valid_idx = np.flatnonzero(~channel_hole[:, x])
            if valid_idx.size == 0:
                continue
            hole_idx = np.flatnonzero(channel_hole[:, x])
            if hole_idx.size == 0:
                continue
            values = image[valid_idx, x, c].astype(np.float32)
            filled = np.interp(hole_idx, valid_idx, values)
            result[hole_idx, x, c] = np.clip(np.rint(filled), 0, 255).astype(np.uint8)
    return result


def bidirectional_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average horizontal and vertical interpolation on holes."""
    horizontal = horizontal_interp_fill(image, mask)
    vertical = vertical_interp_fill(image, mask)
    result = image.copy()
    hole = _as_3d_mask(mask, image)
    blended = (
        horizontal[hole].astype(np.float32) + vertical[hole].astype(np.float32)
    ) * 0.5
    result[hole] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return result


def morphological_fill(image: np.ndarray, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological closing per channel, applied only to hole subpixels."""
    result = image.copy()
    hole = _as_3d_mask(mask, image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for c in range(image.shape[2]):
        channel_hole = hole[:, :, c]
        if not channel_hole.any():
            continue
        channel = image[:, :, c].copy()
        channel[channel_hole] = 0
        closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
        result[:, :, c][channel_hole] = closed[channel_hole]
    return result


def region_growing_fill(
    image: np.ndarray,
    mask: np.ndarray,
    max_iterations: int = 20,
) -> np.ndarray:
    """Vectorized region growing from valid subpixels inward."""
    result = image.astype(np.float32).copy()
    hole = _as_3d_mask(mask, image)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    for _ in range(max_iterations):
        if not hole.any():
            break
        for c in range(image.shape[2]):
            channel_hole = hole[:, :, c]
            if not channel_hole.any():
                continue
            valid = ~channel_hole
            sums = convolve(result[:, :, c] * valid, kernel, mode="constant", cval=0.0)
            counts = convolve(valid.astype(np.float32), kernel, mode="constant", cval=0.0)
            fillable = channel_hole & (counts > 0)
            if not fillable.any():
                continue
            result[:, :, c][fillable] = sums[fillable] / counts[fillable]
            hole[:, :, c][fillable] = False

    remaining = hole.any()
    if remaining:
        result = nearest_fill(result.astype(np.uint8), mask).astype(np.float32)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


RULE_BASED_METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "nearest": nearest_fill,
    "horizontal": horizontal_interp_fill,
    "vertical": vertical_interp_fill,
    "bidirectional": bidirectional_fill,
    "morphology": morphological_fill,
    "region_growing": region_growing_fill,
}
