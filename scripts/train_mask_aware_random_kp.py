#!/usr/bin/env python
"""Train Deep JSCC with random-Kp fixed energy-rank masks.

This script finetunes, or trains from scratch, the existing DeepJSCC model while
randomly sampling one Kp per mini-batch. The sampled Kp selects a nested mask
from a fixed energy ranking produced by the innovation importance scripts.

Supported scales:
  block_size=1: element-level HWC flat energy rank
  block_size=2: 2x2 spatial blocks inside each channel
  block_size=4: 4x4 spatial blocks inside each channel
  block_size=8: full 8x8 channel blocks

Example:
  python scripts/train_mask_aware_random_kp.py \
      --data_dir /home/lc/class/yuyi/cifar-10 \
      --checkpoint outputs/jscc/snr_7/best_jscc_snr7.pt \
      --output_dir outputs/mask_aware_b8_snr7 \
      --snr_db 7 \
      --block_size 8 \
      --kp_list 128,256,384,512,640,768,896,1024
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eval_rate_distortion import INPUT_K, LATENT_C, LATENT_H, LATENT_K, LATENT_W, parse_kp_list
from innovation_block_energy_importance import build_blocks, make_block_mask_from_rank
from jscc_lab.channel import awgn, power_normalize, snr_db_to_noise_std
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import count_parameters
from jscc_lab.plotting import save_rate_distortion_curve, save_training_curves
from jscc_lab.utils import TeeLogger, ensure_dir, get_device, save_checkpoint, save_json, seed_everything
from train_jscc import cap_subset, make_loader, maybe_create_tiny_dataset, snr_tag


VALID_BLOCK_SIZES = {1, 2, 4, 8}


def _rate_column(r: float | torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(r, torch.Tensor):
        value = r.to(device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        if value.ndim != 2 or value.shape[1] != 1:
            raise ValueError(f"Expected r as scalar, (N,), or (N,1), got {tuple(value.shape)}.")
        if value.shape[0] == 1 and batch_size != 1:
            value = value.expand(batch_size, 1)
        if value.shape[0] != batch_size:
            raise ValueError(f"Rate batch size mismatch: expected {batch_size}, got {value.shape[0]}.")
        return value
    return torch.full((batch_size, 1), float(r), device=device, dtype=dtype)


class RateConditionedConvEncoder(nn.Module):
    """Encoder E(x, r): append a scalar-rate feature map to the RGB input."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(self.net[0].weight[:, in_channels:, :, :])

    def forward(self, x: torch.Tensor, r: float | torch.Tensor) -> torch.Tensor:
        rate = _rate_column(r, x.shape[0], x.device, x.dtype).view(x.shape[0], 1, 1, 1)
        rate_map = rate.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, rate_map], dim=1))


class RateConditionedConvDecoder(nn.Module):
    """Decoder D(z, r): append a scalar-rate feature map to the latent tensor."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[0].weight[latent_channels:, :, :, :])

    def forward(self, z: torch.Tensor, r: float | torch.Tensor) -> torch.Tensor:
        rate = _rate_column(r, z.shape[0], z.device, z.dtype).view(z.shape[0], 1, 1, 1)
        rate_map = rate.expand(z.shape[0], 1, z.shape[2], z.shape[3])
        return self.net(torch.cat([z, rate_map], dim=1))


class RateConditionedDeepJSCC(nn.Module):
    """Deep JSCC variant with E(x,r) and D(z,r)."""

    def __init__(self, snr_db: float = 7.0, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.snr_db = float(snr_db)
        self.encoder = RateConditionedConvEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = RateConditionedConvDecoder(out_channels=in_channels, latent_channels=latent_channels)

    def encode(self, x: torch.Tensor, r: float | torch.Tensor = 1.0) -> torch.Tensor:
        return self.encoder(x, r)

    def encode_normalized(self, x: torch.Tensor, r: float | torch.Tensor = 1.0) -> torch.Tensor:
        return power_normalize(self.encode(x, r))

    def decode(self, z: torch.Tensor, r: float | torch.Tensor = 1.0) -> torch.Tensor:
        return self.decoder(z, r)

    def forward(self, x: torch.Tensor, r: float | torch.Tensor = 1.0, snr_db: float | None = None, add_noise: bool = True) -> torch.Tensor:
        z = self.encode_normalized(x, r)
        z_noisy = awgn(z, self.snr_db if snr_db is None else float(snr_db), training=add_noise)
        return self.decode(z_noisy, r)


def str2bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {text!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Deep JSCC with random-Kp energy-rank mask-aware finetuning.")
    parser.add_argument("--data_dir", "--data_path", dest="data_dir", default="/home/lc/class/yuyi/cifar-10")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab")
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default=None)
    parser.add_argument(
        "--checkpoint",
        default="outputs/jscc/snr_7/best_jscc_snr7.pt",
        help="Initial model checkpoint. Use --from_scratch or --checkpoint none to skip.",
    )
    parser.add_argument("--resume", default=None, help="Resume full training state, including optimizer when available.")
    parser.add_argument("--from_scratch", action="store_true", help="Ignore --checkpoint and train from random initialization.")
    parser.add_argument("--snr_db", "--train_snr_db", dest="snr_db", type=float, default=7.0)
    parser.add_argument("--kp_list", default="128,256,384,512,640,768,896,1024")
    parser.add_argument("--block_size", type=int, choices=sorted(VALID_BLOCK_SIZES), default=8)
    parser.add_argument("--rank_path", default=None, help="Optional rank .npy path. If omitted, auto-search by block_size.")
    parser.add_argument(
        "--rank_format",
        choices=["auto", "hwc_flat", "block_id", "channel"],
        default="auto",
        help="Rank index interpretation. Use hwc_flat for element energy_rank_indices.npy; block_id for block_energy_rank_indices.",
    )
    parser.add_argument("--kp_rounding", choices=["floor", "ceil"], default="floor")
    parser.add_argument("--noise_only_on_kept", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--full_loss_weight",
        type=float,
        default=0.3,
        help="Weight for auxiliary full-latent loss; set 0 to recover the original mask-only loss.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for validation and final test samples.")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0, help="Save epoch checkpoints every N epochs; 0 disables.")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def resolve_path(path_text: str | Path | None, project_dir: Path) -> Path | None:
    if path_text is None:
        return None
    text = str(path_text).strip()
    if not text or text.lower() in {"none", "null", "false"}:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def find_rank_path(project_dir: Path, block_size: int) -> Path:
    candidates: List[Path]
    if block_size == 1:
        candidates = [
            project_dir / "outputs" / "innovation" / "energy_importance" / "energy_rank_indices.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / "b1" / "block_energy_rank_indices_b1.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / "b1" / "energy_rank_indices_b1.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / "block_size_1_rank_indices.npy",
        ]
    elif block_size in {2, 4}:
        candidates = [
            project_dir
            / "outputs"
            / "innovation"
            / "block_energy_importance"
            / f"b{block_size}"
            / f"block_energy_rank_indices_b{block_size}.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / f"energy_rank_indices_b{block_size}.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / f"block_size_{block_size}_rank_indices.npy",
        ]
    else:
        candidates = [
            project_dir / "outputs" / "innovation" / "channel_importance" / "channel_energy_rank_indices.npy",
            project_dir / "outputs" / "innovation" / "channel_importance" / "channel_energy_rank.npy",
            project_dir / "outputs" / "innovation" / "channel_importance" / "channel_rank.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / "b8" / "block_energy_rank_indices_b8.npy",
            project_dir / "outputs" / "innovation" / "block_energy_importance" / "block_size_8_rank_indices.npy",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find an energy rank file for block_size={block_size}.\n"
        f"Checked:\n{searched}\n"
        "Please pass --rank_path explicitly."
    )


def hwc_flat_to_chw(flat_index: int, C: int = LATENT_C, H: int = LATENT_H, W: int = LATENT_W) -> Tuple[int, int, int]:
    c = int(flat_index) % C
    w = (int(flat_index) // C) % W
    h = int(flat_index) // (W * C)
    if not (0 <= c < C and 0 <= h < H and 0 <= w < W):
        raise ValueError(f"HWC flat index out of range: {flat_index}")
    return c, h, w


def block_id_for_chw(c: int, h: int, w: int, block_size: int) -> int:
    blocks_per_row = LATENT_W // block_size
    blocks_per_channel = (LATENT_H // block_size) * blocks_per_row
    return int(c) * blocks_per_channel + (int(h) // block_size) * blocks_per_row + (int(w) // block_size)


def derive_block_rank_from_hwc_flat(rank_indices: np.ndarray, block_size: int) -> np.ndarray:
    num_blocks = LATENT_C * (LATENT_H // block_size) * (LATENT_W // block_size)
    seen = np.zeros(num_blocks, dtype=bool)
    block_rank: List[int] = []
    for flat_index in rank_indices.tolist():
        c, h, w = hwc_flat_to_chw(int(flat_index))
        block_id = block_id_for_chw(c, h, w, block_size)
        if not seen[block_id]:
            seen[block_id] = True
            block_rank.append(block_id)
            if len(block_rank) == num_blocks:
                break
    if len(block_rank) != num_blocks:
        raise ValueError(
            f"Could not derive all {num_blocks} blocks from HWC flat rank; got {len(block_rank)} unique blocks."
        )
    return np.asarray(block_rank, dtype=np.int64)


def derive_channel_rank_from_hwc_flat(rank_indices: np.ndarray) -> np.ndarray:
    seen = np.zeros(LATENT_C, dtype=bool)
    channel_rank: List[int] = []
    for flat_index in rank_indices.tolist():
        channel = int(flat_index) % LATENT_C
        if not seen[channel]:
            seen[channel] = True
            channel_rank.append(channel)
            if len(channel_rank) == LATENT_C:
                break
    if len(channel_rank) != LATENT_C:
        raise ValueError(f"Could not derive all {LATENT_C} channels from HWC flat rank.")
    return np.asarray(channel_rank, dtype=np.int64)


def infer_rank_format(rank_path: Path, block_size: int, raw_length: int, num_blocks: int, rank_format: str) -> str:
    if rank_format != "auto":
        return rank_format

    path_text = str(rank_path).lower()
    name = rank_path.name.lower()
    if block_size == 8 and raw_length == LATENT_C:
        return "channel"
    if raw_length == LATENT_K:
        if block_size == 1:
            if "block_energy_importance" in path_text or "block_energy_rank" in name or "block_size_1" in name:
                return "block_id"
            return "hwc_flat"
        return "hwc_flat"
    if raw_length == num_blocks:
        return "block_id"
    return "auto"


def load_and_normalize_rank(rank_path: Path, block_size: int, rank_format: str = "auto") -> Tuple[np.ndarray, Dict[str, object]]:
    if not rank_path.is_file():
        raise FileNotFoundError(f"Rank file does not exist: {rank_path}")

    raw_rank = np.asarray(np.load(rank_path), dtype=np.int64).reshape(-1)
    if raw_rank.size == 0:
        raise ValueError(f"Rank file is empty: {rank_path}")

    num_blocks = LATENT_C * (LATENT_H // block_size) * (LATENT_W // block_size)
    metadata: Dict[str, object] = {
        "rank_path": str(rank_path),
        "raw_rank_length": int(raw_rank.shape[0]),
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "latent_shape_chw": [LATENT_C, LATENT_H, LATENT_W],
        "task7_flatten_order": "HWC",
        "requested_rank_format": rank_format,
    }
    resolved_format = infer_rank_format(rank_path, block_size, int(raw_rank.shape[0]), num_blocks, rank_format)
    metadata["resolved_rank_format"] = resolved_format

    if resolved_format == "channel" and block_size == 8 and raw_rank.shape[0] == LATENT_C and np.array_equal(
        np.sort(raw_rank), np.arange(LATENT_C, dtype=np.int64)
    ):
        metadata["rank_input_type"] = "channel_id_permutation"
        block_rank = raw_rank
    elif resolved_format == "block_id" and raw_rank.shape[0] == num_blocks and np.array_equal(np.sort(raw_rank), np.arange(num_blocks, dtype=np.int64)):
        metadata["rank_input_type"] = "block_id_permutation"
        block_rank = raw_rank
    elif resolved_format == "hwc_flat" and raw_rank.shape[0] == LATENT_K and np.array_equal(np.sort(raw_rank), np.arange(LATENT_K, dtype=np.int64)):
        metadata["rank_input_type"] = "hwc_flat_index_permutation"
        if block_size == 8:
            block_rank = derive_channel_rank_from_hwc_flat(raw_rank)
            metadata["rank_conversion"] = "derived channel order by first occurrence in HWC element rank"
        elif block_size == 1:
            block_rank = derive_block_rank_from_hwc_flat(raw_rank, block_size=1)
            metadata["rank_conversion"] = "converted HWC flat indices to CHW singleton block ids"
        else:
            block_rank = derive_block_rank_from_hwc_flat(raw_rank, block_size=block_size)
            metadata["rank_conversion"] = f"derived {block_size}x{block_size} block order by first occurrence"
    else:
        raise ValueError(
            f"Unsupported rank file shape/content for block_size={block_size}: length={raw_rank.shape[0]}, "
            f"resolved_rank_format={resolved_format}. "
            f"Expected {num_blocks} block ids, {LATENT_K} HWC flat indices, or 16 channel ids for block_size=8."
        )

    if block_rank.shape[0] != num_blocks:
        raise ValueError(f"Expected normalized rank length {num_blocks}, got {block_rank.shape[0]}.")
    if not np.array_equal(np.sort(block_rank), np.arange(num_blocks, dtype=np.int64)):
        raise ValueError("Normalized block rank is not a valid block-id permutation.")

    metadata["normalized_rank_length"] = int(block_rank.shape[0])
    metadata["normalized_rank_first20"] = [int(x) for x in block_rank[:20].tolist()]
    return block_rank.astype(np.int64, copy=False), metadata


def effective_keep_blocks(kp: int, elements_per_block: int, num_blocks: int, rounding: str) -> int:
    if kp < 0 or kp > LATENT_K:
        raise ValueError(f"Kp must be in [0, {LATENT_K}], got {kp}.")
    value = float(kp) / float(elements_per_block)
    keep = math.floor(value) if rounding == "floor" else math.ceil(value)
    return max(0, min(int(keep), int(num_blocks)))


def make_mask_from_rank(
    rank_indices: np.ndarray,
    blocks: Sequence[Dict[str, object]],
    kp: int,
    block_size: int,
    rounding: str = "floor",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    elements_per_block = int(block_size * block_size)
    num_blocks = len(blocks)
    if kp % elements_per_block == 0:
        mask_np = make_block_mask_from_rank(rank_indices, blocks, kp, LATENT_C, LATENT_H, LATENT_W)
        effective_kp = int(kp)
        keep_blocks = int(kp // elements_per_block)
    else:
        keep_blocks = effective_keep_blocks(kp, elements_per_block, num_blocks, rounding)
        effective_kp = int(keep_blocks * elements_per_block)
        mask_np = make_block_mask_from_rank(rank_indices, blocks, effective_kp, LATENT_C, LATENT_H, LATENT_W)

    mask = torch.from_numpy(mask_np.astype(np.float32, copy=False)).unsqueeze(0)
    info = {
        "requested_kp": int(kp),
        "effective_kp": int(effective_kp),
        "elements_per_block": int(elements_per_block),
        "num_keep_blocks": int(keep_blocks),
    }
    return mask, info


def build_mask_bank(
    rank_indices: np.ndarray,
    block_size: int,
    kp_values: Sequence[int],
    rounding: str,
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, int]], List[Dict[str, object]]]:
    blocks = build_blocks(LATENT_C, LATENT_H, LATENT_W, block_size)
    masks: Dict[int, torch.Tensor] = {}
    mask_info: Dict[int, Dict[str, int]] = {}
    for kp in kp_values:
        mask, info = make_mask_from_rank(rank_indices, blocks, int(kp), block_size, rounding=rounding)
        masks[int(kp)] = mask.to(device=device)
        mask_info[int(kp)] = info
    return masks, mask_info, blocks


def run_mask_sanity_checks(
    masks: Dict[int, torch.Tensor],
    mask_info: Dict[int, Dict[str, int]],
    block_size: int,
    kp_values: Sequence[int],
) -> List[str]:
    lines: List[str] = []
    for kp in kp_values:
        mask_sum = int(masks[int(kp)].detach().cpu().sum().item())
        expected = int(mask_info[int(kp)]["effective_kp"])
        if mask_sum != expected:
            raise AssertionError(f"Mask sum mismatch for Kp={kp}: expected {expected}, got {mask_sum}.")
        lines.append(f"Kp={kp}: mask.sum()={mask_sum}, keep_blocks={mask_info[int(kp)]['num_keep_blocks']}")

    if 1024 in masks:
        full_mask = masks[1024].detach().cpu()
        if int(full_mask.sum().item()) != LATENT_K or not bool(torch.all(full_mask == 1)):
            raise AssertionError("Kp=1024 mask must be all ones.")
        lines.append("Kp=1024 sanity: mask is all ones.")

    if 128 in masks:
        mask_sum = int(masks[128].detach().cpu().sum().item())
        if mask_sum != 128:
            raise AssertionError(f"Kp=128 mask must contain 128 ones for block_size={block_size}, got {mask_sum}.")
        if block_size == 8 and mask_info[128]["num_keep_blocks"] != 2:
            raise AssertionError("block_size=8, Kp=128 should keep 2 full channels.")
        if block_size == 4 and mask_info[128]["num_keep_blocks"] != 8:
            raise AssertionError("block_size=4, Kp=128 should keep 8 4x4 blocks.")
        if block_size == 2 and mask_info[128]["num_keep_blocks"] != 32:
            raise AssertionError("block_size=2, Kp=128 should keep 32 2x2 blocks.")
        if block_size == 1 and mask_info[128]["num_keep_blocks"] != 128:
            raise AssertionError("block_size=1, Kp=128 should keep 128 elements.")
        lines.append(f"Kp=128 sanity for block_size={block_size}: passed.")
    return lines


def kp_to_rate(kp: int) -> float:
    """Rate-conditioning scalar for the network: retained latent fraction in [0, 1]."""

    return float(kp) / float(LATENT_K)


def mask_aware_reconstruct(
    model: RateConditionedDeepJSCC,
    images: torch.Tensor,
    mask: torch.Tensor,
    r: float | torch.Tensor,
    snr_db: float,
    noise_only_on_kept: bool,
) -> torch.Tensor:
    latent = power_normalize(model.encode(images, r))
    latent_masked = latent * mask
    latent_noisy = awgn(latent_masked, snr_db, training=True)
    if noise_only_on_kept:
        latent_noisy = latent_noisy * mask
    return model.decode(latent_noisy, r)


def mask_aware_losses(
    model: RateConditionedDeepJSCC,
    images: torch.Tensor,
    mask: torch.Tensor,
    r: float | torch.Tensor,
    snr_db: float,
    noise_only_on_kept: bool,
    criterion: nn.Module,
    full_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = power_normalize(model.encode(images, r))

    latent_tx = latent * mask
    latent_rx = awgn(latent_tx, snr_db, training=True)
    if noise_only_on_kept:
        latent_rx = latent_rx * mask
    recon = model.decode(latent_rx, r)
    mask_loss = criterion(recon, images)

    if full_loss_weight > 0:
        full_r = 1.0
        full_latent = power_normalize(model.encode(images, full_r))
        full_latent_noisy = awgn(full_latent, snr_db, training=True)
        full_recon = model.decode(full_latent_noisy, full_r)
        full_loss = criterion(full_recon, images)
        total_loss = mask_loss + float(full_loss_weight) * full_loss
    else:
        full_loss = torch.zeros((), dtype=mask_loss.dtype, device=mask_loss.device)
        total_loss = mask_loss
    return total_loss, mask_loss, full_loss


def run_train_epoch(
    model: RateConditionedDeepJSCC,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    masks: Dict[int, torch.Tensor],
    kp_values: Sequence[int],
    noise_only_on_kept: bool,
    full_loss_weight: float,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float, float, Dict[int, int]]:
    model.train(True)
    total_loss = 0.0
    total_mask_loss = 0.0
    total_full_loss = 0.0
    total_samples = 0
    kp_counts = {int(kp): 0 for kp in kp_values}

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        kp = int(random.choice(list(kp_values)))
        kp_counts[kp] += 1
        mask = masks[kp]
        r = kp_to_rate(kp)

        optimizer.zero_grad(set_to_none=True)
        loss, mask_loss, full_loss = mask_aware_losses(
            model,
            images,
            mask,
            r,
            snr_db,
            noise_only_on_kept,
            criterion,
            full_loss_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_mask_loss += float(mask_loss.item()) * batch_size
        total_full_loss += float(full_loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training dataloader did not produce any samples.")
    return total_loss / total_samples, total_mask_loss / total_samples, total_full_loss / total_samples, kp_counts


@torch.no_grad()
def evaluate_one_kp(
    model: RateConditionedDeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    snr_db: float,
    kp: int,
    mask: torch.Tensor,
    noise_only_on_kept: bool,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        recon = mask_aware_reconstruct(
            model,
            images,
            mask,
            kp_to_rate(kp),
            snr_db,
            noise_only_on_kept,
        ).clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        count = int(images.shape[0])
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += count

    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {
        "Kp": int(kp),
        "R": float(kp) / INPUT_K,
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
    }


@torch.no_grad()
def evaluate_all_kp(
    model: RateConditionedDeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    snr_db: float,
    kp_values: Sequence[int],
    masks: Dict[int, torch.Tensor],
    noise_only_on_kept: bool,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for kp in kp_values:
        torch.manual_seed(seed + int(kp))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + int(kp))
        rows.append(evaluate_one_kp(model, dataloader, device, snr_db, int(kp), masks[int(kp)], noise_only_on_kept))
    return rows


def write_eval_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Kp", "R", "avg_mse", "avg_psnr"])
        writer.writeheader()
        writer.writerows(rows)


def write_mask_metadata(mask_info: Dict[int, Dict[str, int]], path: Path) -> None:
    rows = []
    for kp in sorted(mask_info):
        rows.append(mask_info[kp])
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["requested_kp", "effective_kp", "elements_per_block", "num_keep_blocks"])
        writer.writeheader()
        writer.writerows(rows)


def load_state_with_rate_conditioning(
    model: RateConditionedDeepJSCC,
    state_dict: Dict[str, torch.Tensor],
    strict_rate_conditioned: bool = False,
) -> Dict[str, object]:
    if strict_rate_conditioned:
        model.load_state_dict(state_dict)
        return {"load_type": "strict_rate_conditioned", "missing_keys": [], "shape_mismatches": []}

    model_state = model.state_dict()
    missing_keys: List[str] = []
    shape_mismatches: List[Dict[str, object]] = []
    copied = 0
    transplanted = 0

    for key, target in model_state.items():
        if key not in state_dict:
            missing_keys.append(key)
            continue
        source = state_dict[key]
        if tuple(source.shape) == tuple(target.shape):
            model_state[key] = source.detach().clone()
            copied += 1
            continue

        if key == "encoder.net.0.weight" and source.ndim == 4 and target.ndim == 4:
            if source.shape[0] == target.shape[0] and source.shape[2:] == target.shape[2:] and source.shape[1] + 1 == target.shape[1]:
                new_weight = torch.zeros_like(target)
                new_weight[:, : source.shape[1], :, :] = source
                model_state[key] = new_weight
                transplanted += 1
                continue

        if key == "decoder.net.0.weight" and source.ndim == 4 and target.ndim == 4:
            if source.shape[1:] == target.shape[1:] and source.shape[0] + 1 == target.shape[0]:
                new_weight = torch.zeros_like(target)
                new_weight[: source.shape[0], :, :, :] = source
                model_state[key] = new_weight
                transplanted += 1
                continue

        shape_mismatches.append({"key": key, "source_shape": list(source.shape), "target_shape": list(target.shape)})

    model.load_state_dict(model_state)
    return {
        "load_type": "transplant_from_plain_or_partial_state",
        "copied_tensors": copied,
        "transplanted_tensors": transplanted,
        "missing_keys": missing_keys,
        "shape_mismatches": shape_mismatches,
    }


def load_initial_checkpoint(
    model: RateConditionedDeepJSCC,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | None,
    resume_path: Path | None,
    device: torch.device,
) -> Tuple[int, Dict[str, object]]:
    info: Dict[str, object] = {"mode": "from_scratch", "start_epoch": 1}
    path = resume_path if resume_path is not None else checkpoint_path
    if path is None:
        return 1, info
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        extra = checkpoint.get("extra", {})
        strict_rate_conditioned = bool(extra.get("model_type") == "rate_conditioned_deepjscc")
        load_info = load_state_with_rate_conditioning(
            model,
            checkpoint["model_state"],
            strict_rate_conditioned=strict_rate_conditioned,
        )
        loaded_optimizer = False
        if resume_path is not None and strict_rate_conditioned and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            loaded_optimizer = True
        previous_epoch = int(checkpoint.get("epoch") or 0)
        start_epoch = previous_epoch + 1 if resume_path is not None else 1
        info = {
            "mode": "resume" if resume_path is not None else "checkpoint_init",
            "path": str(path),
            "previous_epoch": previous_epoch,
            "start_epoch": start_epoch,
            "loaded_optimizer_state": loaded_optimizer,
            "checkpoint_extra": extra,
            "model_load_info": load_info,
        }
        return start_epoch, info

    load_info = load_state_with_rate_conditioning(model, checkpoint, strict_rate_conditioned=False)
    info = {"mode": "raw_state_dict_init", "path": str(path), "start_epoch": 1, "model_load_info": load_info}
    return 1, info


def write_summary(
    path: Path,
    args: argparse.Namespace,
    rows: List[Dict[str, float]],
    checkpoint_info: Dict[str, object],
    final_eval_checkpoint: Path,
    rank_metadata: Dict[str, object],
    out_dir: Path,
    num_train: int,
    num_val: int,
    num_test: int,
) -> None:
    lines = [
        "Mask-aware random-Kp Deep JSCC training summary",
        "=" * 54,
        f"Output directory: {out_dir}",
        f"Dataset path: {args.data_dir}",
        f"Checkpoint mode: {checkpoint_info.get('mode')}",
        f"Checkpoint path: {checkpoint_info.get('path', 'none')}",
        f"Final test checkpoint: {final_eval_checkpoint}",
        f"SNR: {args.snr_db:g} dB",
        f"block_size: {args.block_size}",
        f"kp_list: {parse_kp_list(args.kp_list)}",
        f"noise_only_on_kept: {bool(args.noise_only_on_kept)}",
        f"full_loss_weight: {float(args.full_loss_weight):.6f}",
        "Training loss: mse(D(masked_noisy_latent, r), x) + full_loss_weight * mse(D(full_latent_noisy, r=1), x)",
        "Rate conditioning scalar r: Kp / 1024 for masked path, r=1 for auxiliary full-latent path",
        f"kp_rounding: {args.kp_rounding}",
        "Latent shape: PyTorch C,H,W = 16 x 8 x 8; report H,W,C = 8 x 8 x 16",
        "Task7 flatten order for element ranks: HWC, flat_index=h*W*C+w*C+c",
        f"Rank file: {rank_metadata.get('rank_path')}",
        f"Rank format: requested={rank_metadata.get('requested_rank_format')}, resolved={rank_metadata.get('resolved_rank_format')}",
        f"Rank input type: {rank_metadata.get('rank_input_type')}",
        f"Rank conversion: {rank_metadata.get('rank_conversion', 'none')}",
        f"Train/val/test samples used: {num_train}/{num_val}/{num_test}",
        "",
        "Final fixed-Kp evaluation rows:",
    ]
    for row in rows:
        lines.append(
            f"Kp={int(row['Kp'])} | R={row['R']:.6f} | avg_mse={row['avg_mse']:.6f} | "
            f"avg_psnr={row['avg_psnr']:.3f} dB"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_history_row(
    epoch: int,
    train_loss: float,
    train_mask_loss: float,
    train_full_loss: float,
    val_rows: List[Dict[str, float]] | None,
    best_val_loss: float,
    kp_counts: Dict[int, int],
    kp_values: Sequence[int],
) -> Dict[str, float]:
    if val_rows:
        val_loss = float(np.mean([row["avg_mse"] for row in val_rows]))
        val_psnr = float(np.mean([row["avg_psnr"] for row in val_rows]))
    else:
        val_loss = float("nan")
        val_psnr = float("nan")

    row: Dict[str, float] = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_mask_loss": float(train_mask_loss),
        "train_full_loss": float(train_full_loss),
        "val_loss": val_loss,
        "val_psnr": val_psnr,
        "best_val_loss": float(best_val_loss),
    }
    for kp in kp_values:
        row[f"kp_count_{int(kp)}"] = int(kp_counts.get(int(kp), 0))
    return row


def write_mask_history_csv(history: Iterable[Dict[str, float]], path: Path, kp_values: Sequence[int]) -> None:
    rows = list(history)
    fieldnames = ["epoch", "train_loss", "train_mask_loss", "train_full_loss", "val_loss", "val_psnr", "best_val_loss"] + [
        f"kp_count_{int(kp)}" for kp in kp_values
    ]
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.eval_every < 1:
        raise ValueError("--eval_every must be at least 1.")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative.")

    seed_everything(args.seed)
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    tag = snr_tag(args.snr_db)
    out_dir = ensure_dir(resolve_path(args.output_dir, project_dir) or (project_dir / f"outputs/mask_aware_b{args.block_size}_snr{tag}"))
    log = TeeLogger(out_dir / "train.log", mode="w")

    try:
        device = get_device(args.device)
        data_dir = Path(args.data_dir).expanduser()
        maybe_create_tiny_dataset(data_dir, args.seed)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

        kp_values = parse_kp_list(args.kp_list)
        rank_path = resolve_path(args.rank_path, project_dir) if args.rank_path else find_rank_path(project_dir, args.block_size)
        rank_indices, rank_metadata = load_and_normalize_rank(rank_path, args.block_size, args.rank_format)
        sanity_kp_values = sorted(set(int(kp) for kp in kp_values) | {128, 1024})
        sanity_masks, sanity_mask_info, _ = build_mask_bank(
            rank_indices, args.block_size, sanity_kp_values, args.kp_rounding, device
        )
        sanity_lines = run_mask_sanity_checks(sanity_masks, sanity_mask_info, args.block_size, sanity_kp_values)
        masks = {int(kp): sanity_masks[int(kp)] for kp in kp_values}
        mask_info = {int(kp): sanity_mask_info[int(kp)] for kp in kp_values}

        shutil.copy2(rank_path, out_dir / f"source_rank_b{args.block_size}.npy")
        np.save(out_dir / f"normalized_block_rank_indices_b{args.block_size}.npy", rank_indices)
        save_json(rank_metadata, out_dir / "rank_metadata.json")
        write_mask_metadata(mask_info, out_dir / "mask_metadata.csv")

        images, labels = load_cifar_array_dataset(data_dir)
        dataset = CIFARArrayDataset(images, labels)
        train_set, val_set, test_set = make_splits(
            dataset,
            train=args.train_split,
            val=args.val_split,
            test=args.test_split,
            seed=args.seed,
        )
        train_set = cap_subset(train_set, args.max_train_samples)
        val_set = cap_subset(val_set, args.max_eval_samples)
        test_set = cap_subset(test_set, args.max_eval_samples)

        pin_memory = device.type == "cuda"
        train_loader = make_loader(train_set, args.batch_size, True, args.seed, args.num_workers, pin_memory)
        val_loader = make_loader(val_set, args.batch_size, False, args.seed, args.num_workers, pin_memory)
        test_loader = make_loader(test_set, args.batch_size, False, args.seed, args.num_workers, pin_memory)

        model = RateConditionedDeepJSCC(snr_db=args.snr_db).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        resume_path = resolve_path(args.resume, project_dir)
        checkpoint_path = None if args.from_scratch else resolve_path(args.checkpoint, project_dir)
        start_epoch, checkpoint_info = load_initial_checkpoint(model, optimizer, checkpoint_path, resume_path, device)
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint starts at epoch {start_epoch}, but --epochs={args.epochs}. "
                "Increase --epochs or use --checkpoint for weight-only initialization."
            )

        summary_text = "\n".join(
            [
                repr(model),
                "",
                f"Mask-aware training SNR: {float(args.snr_db):.4g} dB",
                f"AWGN noise std after power normalization: {snr_db_to_noise_std(args.snr_db):.8f}",
                "Latent shape CHW: 16 x 8 x 8; report HWC: 8 x 8 x 16",
                f"block_size: {args.block_size}",
                f"elements_per_block: {args.block_size * args.block_size}",
                f"Rank path: {rank_path}",
                f"Rank format: requested={rank_metadata.get('requested_rank_format')} resolved={rank_metadata.get('resolved_rank_format')}",
                f"Rank input type: {rank_metadata.get('rank_input_type')}",
                f"noise_only_on_kept: {bool(args.noise_only_on_kept)}",
                f"full_loss_weight: {float(args.full_loss_weight):.6f}",
                "Model: RateConditionedDeepJSCC with E(x,r) and D(z,r)",
                "Rate conditioning scalar r=Kp/1024 on masked path; auxiliary full path uses r=1.",
                f"Trainable parameters: {count_parameters(model, trainable_only=True):,}",
                f"Total parameters: {count_parameters(model, trainable_only=False):,}",
            ]
        )
        (out_dir / "model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
        log.log(summary_text)
        for line in sanity_lines:
            log.log(f"Sanity: {line}")

        config = vars(args).copy()
        config.update(
            {
                "resolved_output_dir": str(out_dir),
                "resolved_rank_path": str(rank_path),
                "checkpoint_info": checkpoint_info,
                "rank_metadata": rank_metadata,
                "mask_info": {str(kp): info for kp, info in mask_info.items()},
                "model_type": "rate_conditioned_deepjscc",
                "full_loss_weight": float(args.full_loss_weight),
                "rate_conditioning": "r=Kp/1024 for masked path; r=1 for full-latent auxiliary path",
                "rank_format": args.rank_format,
                "device": str(device),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "test_samples": len(test_set),
            }
        )
        save_json(config, out_dir / "config.json")

        history: List[Dict[str, float]] = []
        best_val_loss = float("inf")
        best_path = out_dir / f"best_mask_aware_b{args.block_size}_snr{tag}.pt"
        last_path = out_dir / f"last_mask_aware_b{args.block_size}_snr{tag}.pt"
        extra = {
            "train_snr_db": float(args.snr_db),
            "noise_std": snr_db_to_noise_std(args.snr_db),
            "latent_shape_hwc": [8, 8, 16],
            "data_path": str(data_dir),
            "block_size": int(args.block_size),
            "kp_list": [int(kp) for kp in kp_values],
            "rank_path": str(rank_path),
            "noise_only_on_kept": bool(args.noise_only_on_kept),
            "kp_rounding": args.kp_rounding,
            "model_type": "rate_conditioned_deepjscc",
            "full_loss_weight": float(args.full_loss_weight),
            "rate_conditioning": "r=Kp/1024 for masked path; r=1 for full-latent auxiliary path",
            "rank_format": args.rank_format,
            "mask_info": {str(kp): info for kp, info in mask_info.items()},
        }

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_mask_loss, train_full_loss, kp_counts = run_train_epoch(
                model,
                train_loader,
                criterion,
                device,
                args.snr_db,
                masks,
                kp_values,
                bool(args.noise_only_on_kept),
                float(args.full_loss_weight),
                optimizer,
            )
            should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            val_rows = (
                evaluate_all_kp(model, val_loader, device, args.snr_db, kp_values, masks, bool(args.noise_only_on_kept), args.seed)
                if should_eval
                else None
            )
            val_loss = float(np.mean([row["avg_mse"] for row in val_rows])) if val_rows else float("nan")
            is_best = bool(val_rows) and val_loss <= best_val_loss
            if is_best:
                best_val_loss = val_loss

            row = make_history_row(
                epoch,
                train_loss,
                train_mask_loss,
                train_full_loss,
                val_rows,
                best_val_loss,
                kp_counts,
                kp_values,
            )
            history.append(row)
            log.log(
                f"epoch {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} "
                f"mask_loss={train_mask_loss:.6f} full_loss={train_full_loss:.6f} "
                f"val_loss={row['val_loss']:.6f} val_psnr={row['val_psnr']:.3f} best_val_loss={best_val_loss:.6f} "
                f"kp_counts={kp_counts}"
            )

            if is_best:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    extra=extra,
                )
            if args.save_every and epoch % args.save_every == 0:
                save_checkpoint(
                    out_dir / f"checkpoint_epoch_{epoch:03d}_b{args.block_size}_snr{tag}.pt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    extra=extra,
                )

            write_mask_history_csv(history, out_dir / "history.csv", kp_values)
            save_training_curves(
                {
                    "train_loss": [item["train_loss"] for item in history],
                    "train_mask_loss": [item["train_mask_loss"] for item in history],
                    "train_full_loss": [item["train_full_loss"] for item in history],
                    "val_loss": [item["val_loss"] for item in history if not math.isnan(float(item["val_loss"]))],
                },
                out_dir / "loss_curve.png",
            )

        save_checkpoint(last_path, model, optimizer=optimizer, epoch=args.epochs, metrics=history[-1], extra=extra)
        if not best_path.exists():
            save_checkpoint(best_path, model, optimizer=optimizer, epoch=args.epochs, metrics=history[-1], extra=extra)

        best_model = RateConditionedDeepJSCC(snr_db=args.snr_db).to(device)
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=args.lr)
        _, best_eval_checkpoint_info = load_initial_checkpoint(best_model, best_optimizer, best_path, None, device)
        log.log(f"Final fixed-Kp test uses best checkpoint: {best_path}")
        log.log(f"Best checkpoint load info: {best_eval_checkpoint_info.get('model_load_info')}")

        test_rows = evaluate_all_kp(
            best_model, test_loader, device, args.snr_db, kp_values, masks, bool(args.noise_only_on_kept), args.seed
        )
        eval_csv = out_dir / f"mask_aware_random_kp_eval_block{args.block_size}.csv"
        rd_png = out_dir / f"mask_aware_random_kp_rd_block{args.block_size}.png"
        write_eval_csv(test_rows, eval_csv)
        save_rate_distortion_curve(
            [row["R"] for row in test_rows],
            [row["avg_psnr"] for row in test_rows],
            rd_png,
        )
        write_summary(
            out_dir / "mask_aware_random_kp_summary.txt",
            args,
            test_rows,
            checkpoint_info,
            best_path,
            rank_metadata,
            out_dir,
            len(train_set),
            len(val_set),
            len(test_set),
        )
        log.log(f"Saved best checkpoint: {best_path}")
        log.log(f"Saved last checkpoint: {last_path}")
        log.log(f"Saved final eval CSV: {eval_csv}")
        log.log(f"Saved RD curve: {rd_png}")
    finally:
        log.close()


if __name__ == "__main__":
    main()

# Example b8 channel-level finetune:
# python scripts/train_mask_aware_random_kp.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --checkpoint outputs/jscc/snr_7/best_jscc_snr7.pt \
#   --output_dir outputs/mask_aware_b8_snr7 \
#   --snr_db 7 \
#   --block_size 8 \
#   --kp_list 128,256,384,512,640,768,896,1024
#
# Example b4 block-level finetune:
# python scripts/train_mask_aware_random_kp.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --checkpoint outputs/jscc/snr_7/best_jscc_snr7.pt \
#   --output_dir outputs/mask_aware_b4_snr7 \
#   --snr_db 7 \
#   --block_size 4 \
#   --kp_list 128,256,384,512,640,768,896,1024
