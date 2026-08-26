"""Train U-Net variants with scene-level splits and bounded image caching."""

from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import random_crop
from .features import build_extra_channels
from .luts import build_lut
from .metrics import psnr
from .models import UNet
from .sample import Sample, iter_samples, load_interlaced

DEFAULT_TRAIN_SCENES = tuple(f"scene{i:03d}" for i in range(1, 9))
DEFAULT_VAL_SCENES = ("scene009",)


def parse_scene_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def select_samples(samples: list[Sample], scenes: tuple[str, ...], label: str) -> list[Sample]:
    selected = [sample for sample in samples if sample.scene in scenes]
    found = {sample.scene for sample in selected}
    missing = sorted(set(scenes) - found)
    if missing:
        raise ValueError(f"{label} scenes not found: {', '.join(missing)}")
    return selected


class CachedCropDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        crop_size: int,
        length: int,
        seed: int = 0,
        input_mode: str = "baseline",
        cache_samples: int = 2,
        index_offset: int = 0,
        hole_prob: float = 0.5,
        large_hole_prob: float = 0.0,
        large_hole_candidates: int = 8,
        crops_per_sample: int = 1,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        self.samples = samples
        self.crop_size = crop_size
        self.length = length
        self.seed = seed
        self.input_mode = input_mode
        self.cache_samples = max(0, cache_samples)
        self.index_offset = index_offset
        self.hole_prob = hole_prob
        self.large_hole_prob = large_hole_prob
        self.large_hole_candidates = large_hole_candidates
        self.crops_per_sample = max(1, crops_per_sample)
        self._cache: OrderedDict[Path, tuple[np.ndarray, ...]] = OrderedDict()

    def _load(self, sample: Sample) -> tuple[np.ndarray, ...]:
        image = load_interlaced(sample, "interlaced_input")
        mask = load_interlaced(sample, "interlaced_mask")
        gt = load_interlaced(sample, "interlaced_gt")
        mask_view = (mask.max(axis=2) > 0).astype(np.uint8) * 255
        return image, mask, gt, mask_view

    def _cached(self, sample: Sample) -> tuple[np.ndarray, ...]:
        if self.cache_samples == 0:
            return self._load(sample)
        key = sample.root
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = self._load(sample)
        self._cache[key] = value
        while len(self._cache) > self.cache_samples:
            self._cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        absolute_index = self.index_offset + index
        sample_group = absolute_index // self.crops_per_sample
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, absolute_index]))
        sample_rng = np.random.default_rng(np.random.SeedSequence([self.seed, sample_group]))
        sample = self.samples[int(sample_rng.integers(0, len(self.samples)))]
        image, mask, gt, mask_view = self._cached(sample)
        crop = random_crop(
            image, mask, gt, mask_view, self.crop_size, rng,
            force_hole=bool(rng.random() < self.hole_prob),
            prefer_large_hole=bool(rng.random() < self.large_hole_prob),
            large_hole_candidates=self.large_hole_candidates,
        )
        rgb = _to_tensor(crop["image"])
        mask_t = _to_tensor(crop["mask"])
        gt_t = _to_tensor(crop["gt"])
        x = torch.cat([rgb, mask_t], dim=0)
        if self.input_mode != "baseline":
            lut = build_lut(
                self.crop_size, self.crop_size, sample.display_params, origin=crop["origin"]
            )
            extra = build_extra_channels(crop["image"], crop["mask"], lut, self.input_mode)
            x = torch.cat([x, torch.from_numpy(extra.transpose(2, 0, 1))], dim=0)
        return {
            "x": x, "y": gt_t, "input": rgb, "mask": mask_t,
            "hole_ratio": torch.tensor(crop["hole_ratio"], dtype=torch.float32),
        }


def input_channels_for_mode(input_mode: str) -> int:
    if input_mode == "baseline":
        return 6
    if input_mode == "view_id":
        return 9
    if input_mode == "neighbor":
        return 6 + 3 + 6 * 4
    raise ValueError(f"unknown input_mode: {input_mode}")


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return nn.functional.max_pool2d(mask, kernel, stride=1, padding=radius)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def _masked_loss(
    pred: torch.Tensor, target: torch.Tensor, source: torch.Tensor, mask: torch.Tensor,
    boundary_weight: float = 0.0, boundary_radius: int = 3,
    gradient_weight: float = 0.0, range_weight: float = 0.0,
) -> torch.Tensor:
    composed = source * (1.0 - mask) + pred * mask
    l1 = nn.functional.l1_loss(composed, target, reduction="none")
    full_loss = l1.mean()
    hole_count = mask.sum()
    hole_loss = (l1 * mask).sum() / hole_count if hole_count > 0 else torch.zeros_like(full_loss)
    hole = mask.amax(dim=1, keepdim=True)
    boundary = (_binary_dilate(hole, boundary_radius) - hole).clamp(0.0, 1.0)
    inner_boundary = hole * _binary_dilate(1.0 - hole, boundary_radius)
    boundary_band = (boundary + inner_boundary).clamp(0.0, 1.0)
    boundary_loss = _weighted_mean(l1, boundary_band)

    grad_loss = torch.zeros_like(full_loss)
    if gradient_weight > 0:
        dx = (composed[:, :, :, 1:] - composed[:, :, :, :-1]) - (
            target[:, :, :, 1:] - target[:, :, :, :-1]
        )
        dy = (composed[:, :, 1:, :] - composed[:, :, :-1, :]) - (
            target[:, :, 1:, :] - target[:, :, :-1, :]
        )
        wx = torch.maximum(boundary_band[:, :, :, 1:], boundary_band[:, :, :, :-1])
        wy = torch.maximum(boundary_band[:, :, 1:, :], boundary_band[:, :, :-1, :])
        grad_loss = _weighted_mean(dx.abs(), wx) + _weighted_mean(dy.abs(), wy)

    range_error = nn.functional.relu(-pred) + nn.functional.relu(pred - 1.0)
    range_loss = _weighted_mean(range_error, hole)
    return (
        full_loss + 2.0 * hole_loss + boundary_weight * boundary_loss
        + gradient_weight * grad_loss + range_weight * range_loss
    )


def _make_dataset(
    samples: list[Sample], args: argparse.Namespace, length: int, seed: int,
    index_offset: int = 0, hole_prob: float = 0.5,
    crops_per_sample: int | None = None,
) -> CachedCropDataset:
    return CachedCropDataset(
        samples, crop_size=args.crop_size, length=length, seed=seed,
        input_mode=args.input_mode, cache_samples=args.cache_samples,
        index_offset=index_offset, hole_prob=hole_prob,
        large_hole_prob=args.large_hole_prob,
        large_hole_candidates=args.large_hole_candidates,
        crops_per_sample=args.crops_per_sample if crops_per_sample is None else crops_per_sample,
    )


def _make_loader(dataset: CachedCropDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"), persistent_workers=args.workers > 0,
    )


@torch.inference_mode()
def evaluate_crops(
    model: nn.Module, dataset: CachedCropDataset, count: int, device: torch.device
) -> tuple[float, float]:
    model.eval()
    full_psnrs: list[float] = []
    hole_psnrs: list[float] = []
    for index in range(count):
        crop = dataset[index]
        x = crop["x"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(x).squeeze(0).cpu()
        composed = crop["input"] * (1.0 - crop["mask"]) + pred * crop["mask"]
        pred_np = (composed.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_np = (crop["y"].clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        full_psnrs.append(psnr(pred_np, gt_np))
        hole = crop["mask"].permute(1, 2, 0).numpy() > 0
        if hole.any():
            hole_psnrs.append(psnr(pred_np, gt_np, hole))
    model.train()
    return float(np.mean(full_psnrs)), float(np.mean(hole_psnrs))


def _save_training_state(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, step: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "step": step, "best_psnr": best_psnr,
        "input_mode": args.input_mode, "base": args.base,
    }, path)


def train(args: argparse.Namespace) -> int:
    all_samples = list(iter_samples(Path(args.dataset_root)))
    if not all_samples:
        print("no samples found")
        return 1
    if args.scene:
        train_scenes = (args.scene,)
        val_scenes: tuple[str, ...] = ()
    else:
        train_scenes = parse_scene_list(args.train_scenes)
        val_scenes = parse_scene_list(args.val_scenes)
    overlap = sorted(set(train_scenes) & set(val_scenes))
    if overlap:
        raise ValueError(f"train/val scene overlap: {', '.join(overlap)}")
    train_samples = select_samples(all_samples, train_scenes, "train")
    val_samples = select_samples(all_samples, val_scenes, "val") if val_scenes else []
    print(
        f"train scenes={','.join(train_scenes)} samples={len(train_samples)}; "
        f"val scenes={','.join(val_scenes) or 'disabled'} samples={len(val_samples)}"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    model = UNet(input_channels_for_mode(args.input_mode), 3, args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_step = 0
    best_psnr = float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("input_mode") != args.input_mode or state.get("base") != args.base:
            raise ValueError("resume checkpoint input_mode/base does not match current arguments")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step = int(state["step"])
        best_psnr = float(state.get("best_psnr", best_psnr))
        print(f"resumed {args.resume} at step={start_step}")
    if start_step >= args.steps:
        print(f"checkpoint already reached requested steps={args.steps}")
        return 0

    train_dataset = _make_dataset(
        train_samples, args, (args.steps - start_step) * args.batch_size,
        args.seed, start_step * args.batch_size,
    )
    loader = _make_loader(train_dataset, args)
    val_dataset = _make_dataset(
        val_samples, args, args.eval_crops, args.seed + 1_000_000, crops_per_sample=1
    ) if val_samples else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"unet_{args.input_mode}.pt"
    last_path = out_dir / f"unet_{args.input_mode}.last.pt"
    if device.type == "cuda":
        print(f"training on {torch.cuda.get_device_name(0)} amp={amp_enabled}")

    model.train()
    step = start_step
    interval_start = time.perf_counter()
    interval_steps = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        source = batch["input"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss = _masked_loss(
                model(x), target, source, mask,
                boundary_weight=args.boundary_weight,
                boundary_radius=args.boundary_radius,
                gradient_weight=args.gradient_weight,
                range_weight=args.range_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        interval_steps += 1

        if step % args.log_interval == 0 or step == start_step + 1:
            elapsed = time.perf_counter() - interval_start
            print(f"step={step}/{args.steps} loss={loss.item():.4f} sec_per_step={elapsed / interval_steps:.3f}")
            interval_start = time.perf_counter()
            interval_steps = 0
        if val_dataset is not None and (step % args.val_interval == 0 or step == args.steps):
            val_full_psnr, val_hole_psnr = evaluate_crops(model, val_dataset, args.eval_crops, device)
            print(
                f"step={step} val_full_psnr={val_full_psnr:.2f} "
                f"val_hole_psnr={val_hole_psnr:.2f}"
            )
            if val_hole_psnr > best_psnr:
                best_psnr = val_hole_psnr
                torch.save(model.state_dict(), weights_path)
                print(f"saved best weights {weights_path}")
        if step % args.save_interval == 0 or step == args.steps:
            _save_training_state(last_path, model, optimizer, scaler, step, best_psnr, args)

    if val_dataset is None:
        torch.save(model.state_dict(), weights_path)
    print(f"saved resume checkpoint {last_path}; best_val_hole_psnr={best_psnr:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.train_unet")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", default=None, help="legacy single-scene training; disables validation")
    parser.add_argument("--train-scenes", default=",".join(DEFAULT_TRAIN_SCENES))
    parser.add_argument("--val-scenes", default=",".join(DEFAULT_VAL_SCENES))
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000, help="total optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-samples", type=int, default=2, help="decoded 4K samples cached per worker")
    parser.add_argument(
        "--crops-per-sample", type=int, default=8,
        help="consecutive crops drawn from one decoded image; use batch size for one image per batch",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-crops", type=int, default=40)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--large-hole-prob", type=float, default=0.0)
    parser.add_argument("--large-hole-candidates", type=int, default=8)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--range-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out-dir", default="outputs/b2")
    parser.add_argument("--input-mode", choices=["baseline", "view_id", "neighbor"], default="baseline")
    parser.set_defaults(func=train)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
