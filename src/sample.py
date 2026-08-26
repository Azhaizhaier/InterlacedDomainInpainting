"""Sample metadata loading, image I/O, and dataset validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .luts import DisplayParams

DEFAULT_MIN_VALID_PSNR = 30.0


@dataclass(frozen=True)
class Sample:
    root: Path
    meta: dict

    @property
    def scene(self) -> str:
        return str(self.meta.get("scene", self.root.parent.parent.name))

    @property
    def mode(self) -> str:
        return str(self.meta.get("mode", "unknown"))

    @property
    def sample_index(self) -> int:
        return int(self.meta.get("sample_index", -1))

    @property
    def name(self) -> str:
        return f"{self.scene}/{self.root.name}"

    @property
    def display_params(self) -> DisplayParams:
        return DisplayParams.from_sample_meta(self.meta)

    def file_path(self, key: str) -> Path:
        files = self.meta.get("files", {})
        if key not in files:
            raise KeyError(f"sample.json has no file entry for {key!r}")
        return self.root / files[key]


def _natural_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**18, path.name)


def iter_samples(dataset_root: str | Path) -> Iterator[Sample]:
    """Yield all samples under ``<root>/<scene>/samples/sample_*``."""
    root = Path(dataset_root)
    if not root.is_dir():
        return

    if (root / "samples").is_dir():
        yield from _samples_in_scene(root)
        return

    scene_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    for scene_dir in scene_dirs:
        samples_dir = scene_dir / "samples"
        if not samples_dir.is_dir():
            continue
        yield from _samples_in_scene(scene_dir)


def _samples_in_scene(scene_dir: Path) -> Iterator[Sample]:
    samples_dir = scene_dir / "samples"
    if not samples_dir.is_dir():
        return
    sample_dirs = sorted(samples_dir.glob("sample_*"), key=_natural_sort_key)
    for sample_dir in sample_dirs:
        meta_path = sample_dir / "sample.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        yield Sample(sample_dir, meta)


def load_interlaced(sample: Sample, key: str) -> np.ndarray:
    """Load one interlaced PNG as an HxWx3 uint8 array."""
    path = sample.file_path(key)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def hole_ratio(mask: np.ndarray) -> float:
    """Fraction of subpixels marked as holes (mask value 255)."""
    return float((mask == 255).mean())


def validate_sample(sample: Sample, min_valid_psnr: float = DEFAULT_MIN_VALID_PSNR) -> dict:
    """Validate one generated sample against the dataset contract.

    The contract requires the mask to be binary, ``mask_view`` to match the
    per-channel mask, and valid subpixels to agree with GT at at least
    ``min_valid_psnr`` dB.  Exact input/GT equality is kept as a diagnostic;
    forward warp against Blender re-rendering is not expected to be exact.
    """
    required = ["interlaced_input", "interlaced_mask", "interlaced_gt", "interlaced_mask_view"]
    files_ok = True
    missing: list[str] = []
    for key in required:
        try:
            path = sample.file_path(key)
        except KeyError:
            missing.append(key)
            files_ok = False
            continue
        if not path.is_file():
            missing.append(key)
            files_ok = False

    if not files_ok:
        return {
            "sample": sample.name,
            "scene": sample.scene,
            "mode": sample.mode,
            "files_ok": False,
            "missing": missing,
            "pass": False,
        }

    input_img = load_interlaced(sample, "interlaced_input")
    mask = load_interlaced(sample, "interlaced_mask")
    gt = load_interlaced(sample, "interlaced_gt")
    mask_view = cv2.imread(str(sample.file_path("interlaced_mask_view")), cv2.IMREAD_UNCHANGED)

    if mask_view is None:
        mask_view = np.zeros(input_img.shape[:2], dtype=np.uint8)

    dims = [input_img.shape, mask.shape, gt.shape]
    dims_ok = all(shape == input_img.shape for shape in dims) and mask_view.ndim == 2
    if not dims_ok:
        return {
            "sample": sample.name,
            "scene": sample.scene,
            "mode": sample.mode,
            "files_ok": True,
            "dimensions_ok": False,
            "pass": False,
        }

    mask_values = np.unique(mask)
    mask_binary = bool(np.all(np.isin(mask_values, [0, 255])))
    expected_view = (mask.max(axis=2) > 0).astype(np.uint8) * 255
    mask_view_ok = bool(np.array_equal(expected_view, mask_view))

    valid = mask == 0
    if valid.any():
        valid_mse = float(np.mean((input_img[valid].astype(np.float32) - gt[valid].astype(np.float32)) ** 2))
        valid_psnr = float("inf") if valid_mse == 0.0 else float(10.0 * np.log10(255.0**2 / valid_mse))
    else:
        valid_psnr = float("inf")
    equal_ratio = float(np.mean(input_img[valid] == gt[valid])) if valid.any() else 1.0
    valid_equality = bool(np.all(input_img[valid] == gt[valid])) if valid.any() else True
    ratio = hole_ratio(mask)

    passed = bool(
        files_ok
        and dims_ok
        and mask_binary
        and mask_view_ok
        and valid_psnr >= min_valid_psnr
    )
    return {
        "sample": sample.name,
        "scene": sample.scene,
        "mode": sample.mode,
        "files_ok": True,
        "dimensions_ok": True,
        "mask_binary": mask_binary,
        "mask_view_consistent": mask_view_ok,
        "valid_psnr": valid_psnr,
        "equal_ratio": equal_ratio,
        "valid_equality": valid_equality,
        "hole_ratio": ratio,
        "pass": passed,
    }


def collect_sample_stats(dataset_root: str | Path, max_samples: int | None = None) -> list[dict]:
    """Run validation on every sample and return per-sample records."""
    records = []
    for index, sample in enumerate(iter_samples(dataset_root)):
        if max_samples is not None and index >= max_samples:
            break
        records.append(validate_sample(sample))
    return records
