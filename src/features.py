"""Display-aware input features built from the lenticular LUT."""

from __future__ import annotations

import numpy as np

from .luts import build_lut

SAME_VIEW_OFFSETS: list[tuple[int, int]] = [
    (1, 6),
    (-1, -6),
    (2, 12),
    (-2, -12),
]


def view_id_map(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Return normalized view id per subpixel (0..1)."""
    view_num = int(lut.max()) + 1
    return lut.astype(np.float32) / max(view_num - 1, 1)


def same_view_neighbor_features(
    image: np.ndarray,
    mask: np.ndarray,
    offsets: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Gather values and validity from same-view neighbors on the LUT lattice.

    Returns a ``HxWx(6*len(offsets))`` float32 array.  For every offset the
    first three channels are neighbor RGB (0-1) and the last three are binary
    validity masks (1 = valid, 0 = hole or outside image).
    """
    if offsets is None:
        offsets = SAME_VIEW_OFFSETS
    height, width, channels = image.shape
    channels_out: list[np.ndarray] = []
    for dx, dy in offsets:
        values = np.zeros((height, width, channels), dtype=np.float32)
        valid = np.zeros((height, width, channels), dtype=np.float32)

        sy0 = max(0, -dy)
        sy1 = min(height, height - dy)
        sx0 = max(0, -dx)
        sx1 = min(width, width - dx)
        if sy0 < sy1 and sx0 < sx1:
            dy0, dy1 = sy0 + dy, sy1 + dy
            dx0, dx1 = sx0 + dx, sx1 + dx
            values[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
            valid[dy0:dy1, dx0:dx1] = (mask[sy0:sy1, sx0:sx1] == 0).astype(np.float32)

        channels_out.append(values)
        channels_out.append(valid)
    return np.concatenate(channels_out, axis=-1)


def build_extra_channels(
    image: np.ndarray,
    mask: np.ndarray,
    lut: np.ndarray,
    input_mode: str,
) -> np.ndarray:
    """Build non-RGB/mask channels for a crop or full image."""
    if input_mode == "baseline":
        return np.zeros((*image.shape[:2], 0), dtype=np.float32)
    if input_mode not in {"view_id", "neighbor"}:
        raise ValueError(f"unknown input_mode: {input_mode}")
    extra = [view_id_map(image, lut)]
    if input_mode == "neighbor":
        extra.append(same_view_neighbor_features(image, mask))
    return np.concatenate(extra, axis=-1)
