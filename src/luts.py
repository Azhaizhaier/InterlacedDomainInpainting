"""Lenticular display-mapping utilities.

The LUT maps each RGB subpixel to the view that supplies it.  De-interlacing
extracts the sparse per-view planes already present in an interlaced image;
re-interlacing puts them back exactly.  Neither operation reconstructs the
channels discarded by the original lossy interlacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DisplayParams:
    theta: float = 0.166666
    koff: float = 0.0
    subpixel: float = 4.666666
    view_num: int = 8

    @classmethod
    def from_sample_meta(cls, meta: dict) -> "DisplayParams":
        display = meta.get("display", {})
        return cls(
            theta=float(display.get("theta", 0.166666)),
            koff=float(display.get("koff", 0.0)),
            subpixel=float(display.get("subpixel", 4.666666)),
            view_num=int(meta.get("view_num", 8)),
        )


def build_lut(
    height: int,
    width: int,
    params: DisplayParams | None = None,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Return ``HxWx3`` int32 LUT with the owning view id of each subpixel."""
    if params is None:
        params = DisplayParams()

    ys = (origin[0] + np.arange(height))[:, None, None]
    xs = (origin[1] + np.arange(width))[None, :, None]
    cs = np.arange(3)[None, None, :]

    r = (xs * 3 + cs + params.koff - 3 * ys * params.theta) % params.subpixel
    view = np.floor(r * params.view_num / params.subpixel).astype(np.int32)
    np.minimum(view, params.view_num - 1, out=view)
    return view


def deinterlace_to_planes(image: np.ndarray, lut: np.ndarray, view_num: int | None = None) -> np.ndarray:
    """Group observed subpixels into sparse ``view_num x H x W`` float32 planes.

    Positions where a view does not own a subpixel are ``NaN``.
    """
    if view_num is None:
        view_num = int(lut.max()) + 1
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if image.shape[:2] != lut.shape[:2]:
        raise ValueError("image and lut must share spatial shape")

    planes = np.full((view_num, image.shape[0], image.shape[1]), np.nan, dtype=np.float32)
    for v in range(view_num):
        for c in range(3):
            mask = lut[:, :, c] == v
            planes[v][mask] = image[:, :, c][mask]
    return planes


def reinterlace_from_planes(planes: np.ndarray, lut: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Reassemble an interlaced image from sparse per-view planes.

    Missing (NaN) entries are replaced with ``fill_value``.
    """
    if planes.ndim != 3:
        raise ValueError("planes must be view_num x H x W")
    view_num, height, width = planes.shape
    if lut.shape[:2] != (height, width):
        raise ValueError("planes and lut must share spatial shape")

    out = np.zeros((height, width, 3), dtype=np.uint8)
    for v in range(view_num):
        values = np.nan_to_num(planes[v], nan=fill_value)
        for c in range(3):
            mask = lut[:, :, c] == v
            out[:, :, c][mask] = np.clip(values[mask], 0, 255).astype(np.uint8)
    return out


def view_statistics(lut: np.ndarray, view_num: int | None = None) -> dict[str, list[float]]:
    """Report per-view subpixel counts and pixel coverage."""
    if view_num is None:
        view_num = int(lut.max()) + 1
    subpixel_counts = [int((lut == v).sum()) for v in range(view_num)]
    pixel_coverage = [float((lut == v).any(axis=-1).mean()) for v in range(view_num)]
    return {
        "subpixel_counts": subpixel_counts,
        "pixel_coverage": pixel_coverage,
    }


def find_exact_periods(lut: np.ndarray, max_dx: int = 8, max_dy: int = 28) -> list[tuple[int, int]]:
    """Find small exact (dx, dy) periods in the view LUT.

    Only translations with ``dx >= 0`` and ``dy >= 0`` are considered.  A
    period means the owning view is identical at both locations.
    """
    height, width = lut.shape[:2]
    periods: list[tuple[int, int]] = []
    for dy in range(1, max_dy + 1):
        if dy >= height:
            continue
        if np.array_equal(lut[dy:, :], lut[:-dy, :]):
            periods.append((0, dy))
        for dx in range(1, max_dx + 1):
            if dx >= width:
                continue
            if np.array_equal(lut[dy:, dx:], lut[:-dy, :-dx]):
                periods.append((dx, dy))
    return periods


def same_view_offsets(
    lut: np.ndarray,
    max_offset: int = 8,
    max_dy: int = 28,
) -> dict[int, list[tuple[int, int]]]:
    """Return offsets that keep a subpixel on the same view plane.

    The result is grouped by view id.  Offsets are limited to a local window
    sized ``2*max_offset + 1`` and are suitable for LUT-guided aggregation.
    """
    height, width = lut.shape[:2]
    view_num = int(lut.max()) + 1
    offsets: dict[int, list[tuple[int, int]]] = {v: [] for v in range(view_num)}
    for v in range(view_num):
        source = lut == v
        for dy in range(-max_dy, max_dy + 1):
            for dx in range(-max_offset, max_offset + 1):
                if dx == 0 and dy == 0:
                    continue
                sy0 = max(0, -dy)
                sy1 = min(height, height - dy)
                sx0 = max(0, -dx)
                sx1 = min(width, width - dx)
                if sy0 >= sy1 or sx0 >= sx1:
                    continue
                if np.array_equal(source[sy0:sy1, sx0:sx1], source[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx]):
                    offsets[v].append((dx, dy))
    return offsets


def iterable_to_numpy(values: Iterable[int]) -> np.ndarray:
    """Small helper for callers that prefer numpy arrays."""
    return np.asarray(list(values), dtype=np.int32)
