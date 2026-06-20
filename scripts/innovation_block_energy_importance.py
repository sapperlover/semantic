#!/usr/bin/env python
"""Innovation experiment: block-level latent energy importance.

The script rebuilds the same train split convention used by train_jscc.py:
  load_cifar_array_dataset(data_dir) -> make_splits(train=0.8, val=0.1, test=0.1, seed=42)

For each power-normalized latent tensor with PyTorch shape (N,C,H,W)=(N,16,8,8),
it computes energy statistics for spatial blocks inside each channel. Task7's
HWC flatten index is recorded for every block so the saved ranking can be used
to build masks compatible with prefix-Kp rate-distortion code.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from eval_rate_distortion import LATENT_C, LATENT_H, LATENT_K, LATENT_W, load_jscc_checkpoint
from jscc_lab.channel import power_normalize
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.utils import ensure_dir, get_device, seed_everything


KP_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024]
VALID_BLOCK_SIZES = {1, 2, 4, 8}


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
    parser = argparse.ArgumentParser(description="Compute block-level latent energy importance for SNR=7 Deep JSCC.")
    parser.add_argument("--data_dir", default="/home/lc/class/yuyi/cifar-10", help="CIFAR-10 directory or data file.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--checkpoint", default=None, help="Optional SNR=7 checkpoint path. If omitted, auto-search.")
    parser.add_argument("--output_dir", default="outputs/innovation/block_energy_importance")
    parser.add_argument("--block_sizes", default="2,4", help="Comma-separated block sizes, valid values: 1,2,4,8.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Device string; default auto uses CUDA if available, else CPU.")
    parser.add_argument("--pair_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train split samples for debugging.")
    parser.add_argument("--save_per_image_energy", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for rebuilding the train/val/test split.")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def parse_block_sizes(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--block_sizes must contain at least one value.")
    result: List[int] = []
    for value in values:
        if value not in VALID_BLOCK_SIZES:
            raise ValueError(f"Invalid block_size={value}. Valid values are {sorted(VALID_BLOCK_SIZES)}.")
        if LATENT_H % value != 0 or LATENT_W % value != 0:
            raise ValueError(f"block_size={value} must divide latent spatial size {LATENT_H}x{LATENT_W}.")
        if value not in result:
            result.append(value)
    return result


def resolve_relative_to_project(path_text: str | Path, project_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def find_snr7_checkpoint(project_dir: Path) -> Path:
    preferred = [
        project_dir / "outputs" / "jscc" / "snr_7" / "best_jscc_snr7.pt",
        project_dir / "outputs" / "jscc" / "snr_7" / "last_jscc_snr7.pt",
        project_dir / "outputs" / "debug_jscc_sweep" / "snr_7" / "best_jscc_snr7.pt",
        project_dir / "outputs" / "debug_jscc_sweep" / "snr_7" / "last_jscc_snr7.pt",
        project_dir / "outputs" / "debug_jscc" / "snr_7" / "best_jscc_snr7.pt",
        project_dir / "outputs" / "debug_jscc" / "snr_7" / "last_jscc_snr7.pt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    outputs_dir = project_dir / "outputs"
    found = sorted(
        path
        for path in outputs_dir.rglob("*jscc_snr7.pt")
        if path.is_file() and "snr_7" in {part.lower() for part in path.parts}
    )
    if found:
        return found[0]

    searched = "\n".join(f"  - {path}" for path in preferred)
    raise FileNotFoundError(
        "Could not find an SNR=7 Deep JSCC checkpoint. Pass --checkpoint explicitly.\n"
        f"Checked:\n{searched}"
    )


def load_project_train_split_dataset(
    data_dir: Path,
    max_train_samples: int | None,
    train_split: float,
    val_split: float,
    test_split: float,
    split_seed: int,
) -> Tuple[Dataset, str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    images, labels = load_cifar_array_dataset(data_dir)
    dataset = CIFARArrayDataset(images, labels)
    train_set, _, _ = make_splits(dataset, train=train_split, val=val_split, test=test_split, seed=split_seed)
    note = (
        "project train split rebuilt with jscc_lab.data.make_splits; "
        f"loaded_samples={len(dataset)}, train_split={train_split}, val_split={val_split}, "
        f"test_split={test_split}, split_seed={split_seed}; validation/test splits excluded"
    )
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("--max_train_samples must be positive when provided.")
        if max_train_samples < len(train_set):
            train_set = Subset(train_set, list(range(max_train_samples)))
            note += f"; first {max_train_samples} samples from the train split"
    return train_set, note


def build_blocks(C: int, H: int, W: int, block_size: int) -> List[Dict[str, object]]:
    """Build channel-contained spatial blocks and record task7 HWC flat indices."""

    if block_size <= 0 or H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"block_size={block_size} must divide H={H} and W={W}.")

    blocks: List[Dict[str, object]] = []
    block_id = 0
    for channel in range(C):
        for h_start in range(0, H, block_size):
            h_end = h_start + block_size
            for w_start in range(0, W, block_size):
                w_end = w_start + block_size
                flat_hwc: List[int] = []
                flat_chw: List[int] = []
                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        flat_hwc.append(((h * W + w) * C) + channel)
                        flat_chw.append(((channel * H + h) * W) + w)
                blocks.append(
                    {
                        "block_id": block_id,
                        "channel": channel,
                        "h_start": h_start,
                        "h_end": h_end,
                        "w_start": w_start,
                        "w_end": w_end,
                        "block_size": block_size,
                        "num_elements": block_size * block_size,
                        "flat_indices_hwc": flat_hwc,
                        "flat_indices_chw": flat_chw,
                    }
                )
                block_id += 1
    return blocks


def compute_block_energy(latent: torch.Tensor, blocks: Sequence[Dict[str, object]]) -> torch.Tensor:
    """Compute per-image block energy for latent shape (N,C,H,W)."""

    if latent.ndim != 4:
        raise ValueError(f"Expected latent shape (N,C,H,W), got {tuple(latent.shape)}.")
    if not blocks:
        raise ValueError("blocks must be non-empty.")

    _, C, H, W = latent.shape
    block_size = int(blocks[0]["block_size"])
    expected_blocks = C * (H // block_size) * (W // block_size)
    if len(blocks) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} blocks for latent shape {tuple(latent.shape)}, got {len(blocks)}.")

    z2 = latent.pow(2)
    # Block order is channel -> h-block -> w-block, matching build_blocks.
    return (
        z2.reshape(latent.shape[0], C, H // block_size, block_size, W // block_size, block_size)
        .mean(dim=(3, 5))
        .reshape(latent.shape[0], expected_blocks)
    )


def make_block_mask_from_rank(
    rank_indices: Sequence[int],
    blocks: Sequence[Dict[str, object]],
    Kp: int,
    C: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Create a CHW mask that keeps top-ranked whole blocks for a given Kp."""

    if not blocks:
        raise ValueError("blocks must be non-empty.")
    elements_per_block = int(blocks[0]["num_elements"])
    if Kp % elements_per_block != 0:
        raise ValueError(f"Kp={Kp} must be divisible by elements_per_block={elements_per_block}.")
    num_keep_blocks = Kp // elements_per_block
    if num_keep_blocks > len(blocks):
        raise ValueError(f"Kp={Kp} requests {num_keep_blocks} blocks, but only {len(blocks)} exist.")

    mask = np.zeros((C, H, W), dtype=np.float32)
    for block_id in list(rank_indices)[:num_keep_blocks]:
        block = blocks[int(block_id)]
        channel = int(block["channel"])
        mask[channel, int(block["h_start"]) : int(block["h_end"]), int(block["w_start"]) : int(block["w_end"])] = 1.0

    if int(mask.sum()) != int(Kp):
        raise AssertionError(f"Mask should contain {Kp} ones, got {int(mask.sum())}.")
    return mask


def self_test_block_masks(blocks_by_size: Dict[int, List[Dict[str, object]]]) -> None:
    """Small mask-construction self-test for all requested scales and Kp values."""

    for block_size, blocks in blocks_by_size.items():
        rank = np.arange(len(blocks), dtype=np.int64)
        for kp in KP_VALUES:
            mask = make_block_mask_from_rank(rank, blocks, kp, LATENT_C, LATENT_H, LATENT_W)
            assert mask.shape == (LATENT_C, LATENT_H, LATENT_W)
            assert int(mask.sum()) == int(kp)


def inverse_rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    if sorted_indices.ndim == 1:
        positions = np.empty_like(sorted_indices, dtype=np.int32)
        positions[sorted_indices] = np.arange(sorted_indices.shape[0], dtype=np.int32)
        return positions

    positions = np.empty_like(sorted_indices, dtype=np.int32)
    ranks = np.arange(sorted_indices.shape[1], dtype=np.int32)
    rows = np.arange(sorted_indices.shape[0])[:, None]
    positions[rows, sorted_indices] = ranks
    return positions


def spearman_from_rank_positions(rank_a: np.ndarray, rank_b: np.ndarray, num_items: int) -> np.ndarray:
    diff = rank_a.astype(np.int64) - rank_b.astype(np.int64)
    sum_sq = np.sum(diff * diff, axis=-1, dtype=np.float64)
    denominator = float(num_items * (num_items * num_items - 1))
    return 1.0 - (6.0 * sum_sq / denominator)


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def sample_pairs(num_items: int, pair_samples: int, seed: int) -> np.ndarray:
    if pair_samples < 0:
        raise ValueError("--pair_samples must be non-negative.")
    if pair_samples == 0:
        return np.empty((0, 2), dtype=np.int64)
    if num_items < 2:
        raise ValueError("At least two images are required for pairwise analysis.")

    rng = np.random.default_rng(seed)
    pairs = np.empty((pair_samples, 2), dtype=np.int64)
    for idx in range(pair_samples):
        first = int(rng.integers(0, num_items))
        second = int(rng.integers(0, num_items - 1))
        if second >= first:
            second += 1
        pairs[idx] = (first, second)
    return pairs


@torch.no_grad()
def collect_per_image_block_energy(
    model,
    dataloader: DataLoader,
    device: torch.device,
    blocks_by_size: Dict[int, List[Dict[str, object]]],
) -> Tuple[Dict[int, np.ndarray], int]:
    model.eval()
    batches_by_size: Dict[int, List[np.ndarray]] = {block_size: [] for block_size in blocks_by_size}
    total_samples = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        latent = power_normalize(model.encoder(images))
        if tuple(latent.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
            raise ValueError(f"Expected latent shape (N,16,8,8), got {tuple(latent.shape)}.")
        for block_size, blocks in blocks_by_size.items():
            energy = compute_block_energy(latent, blocks)
            batches_by_size[block_size].append(energy.detach().cpu().numpy().astype(np.float32, copy=False))
        total_samples += int(images.shape[0])

    if total_samples == 0:
        raise ValueError("No samples were loaded from the train split.")
    return {block_size: np.concatenate(parts, axis=0) for block_size, parts in batches_by_size.items()}, total_samples


def block_label(block: Dict[str, object]) -> str:
    return (
        f"ch{int(block['channel'])} "
        f"h{int(block['h_start'])}:{int(block['h_end'])} "
        f"w{int(block['w_start'])}:{int(block['w_end'])}"
    )


def top_ratio_from_sorted(energy_sum: np.ndarray, rank_indices: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(rank_indices) * fraction)))
    total = float(np.sum(energy_sum))
    return float(np.sum(energy_sum[rank_indices[:count]]) / total) if total > 0 else 0.0


def write_block_importance_table(
    path: Path,
    blocks: Sequence[Dict[str, object]],
    global_energy: np.ndarray,
    energy_sum: np.ndarray,
    rank_indices: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    total_energy_sum = float(np.sum(energy_sum))
    sorted_energy_sum = energy_sum[rank_indices]
    cumulative = np.cumsum(sorted_energy_sum, dtype=np.float64)
    rank_positions = inverse_rank_positions(rank_indices)

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "block_id",
            "rank",
            "channel",
            "h_start",
            "h_end",
            "w_start",
            "w_end",
            "block_size",
            "num_elements",
            "energy",
            "energy_sum",
            "energy_ratio",
            "cumulative_energy_ratio",
            "flat_indices_hwc",
            "flat_indices_chw",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for block_id in rank_indices.tolist():
            block = blocks[int(block_id)]
            rank = int(rank_positions[int(block_id)]) + 1
            writer.writerow(
                {
                    "block_id": int(block_id),
                    "rank": rank,
                    "channel": int(block["channel"]),
                    "h_start": int(block["h_start"]),
                    "h_end": int(block["h_end"]),
                    "w_start": int(block["w_start"]),
                    "w_end": int(block["w_end"]),
                    "block_size": int(block["block_size"]),
                    "num_elements": int(block["num_elements"]),
                    "energy": float(global_energy[int(block_id)]),
                    "energy_sum": float(energy_sum[int(block_id)]),
                    "energy_ratio": float(energy_sum[int(block_id)] / total_energy_sum) if total_energy_sum > 0 else 0.0,
                    "cumulative_energy_ratio": float(cumulative[rank - 1] / total_energy_sum) if total_energy_sum > 0 else 0.0,
                    "flat_indices_hwc": " ".join(str(x) for x in block["flat_indices_hwc"]),
                    "flat_indices_chw": " ".join(str(x) for x in block["flat_indices_chw"]),
                }
            )


def write_per_image_spearman(path: Path, spearman: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "spearman_with_global_block_rank"])
        writer.writeheader()
        for idx, value in enumerate(spearman.tolist()):
            writer.writerow({"image_index": idx, "spearman_with_global_block_rank": float(value)})


def write_pairwise_spearman(path: Path, pairs: np.ndarray, spearman: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_index_a", "image_index_b", "spearman"])
        writer.writeheader()
        for pair_id, ((idx_a, idx_b), value) in enumerate(zip(pairs.tolist(), spearman.tolist())):
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "image_index_a": int(idx_a),
                    "image_index_b": int(idx_b),
                    "spearman": float(value),
                }
            )


def write_summary_rows(path: Path, rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_overlap_sample(
    path: Path,
    pairs: np.ndarray,
    overlap_by_kp: Dict[int, np.ndarray],
    keep_blocks_by_kp: Dict[int, int],
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_index_a", "image_index_b", "Kp", "num_keep_blocks", "overlap"])
        writer.writeheader()
        for pair_id, (idx_a, idx_b) in enumerate(pairs.tolist()):
            for kp in sorted(overlap_by_kp):
                overlaps = overlap_by_kp[kp]
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "image_index_a": int(idx_a),
                        "image_index_b": int(idx_b),
                        "Kp": int(kp),
                        "num_keep_blocks": int(keep_blocks_by_kp[kp]),
                        "overlap": float(overlaps[pair_id]),
                    }
                )


def save_rank_curve(sorted_energy: np.ndarray, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(np.arange(1, len(sorted_energy) + 1), sorted_energy, linewidth=1.6)
    ax.set_xlabel("Block rank")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_top_bar(blocks: Sequence[Dict[str, object]], global_energy: np.ndarray, rank_indices: np.ndarray, out_path: Path) -> None:
    top = rank_indices[: min(30, len(rank_indices))]
    values = global_energy[top]
    labels = [block_label(blocks[int(block_id)]) for block_id in top]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=220)
    x = np.arange(len(top))
    ax.bar(x, values, color="#2563eb")
    ax.set_ylabel("Global block energy")
    ax.set_xlabel("Top blocks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_spearman_hist(spearman: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.hist(spearman, bins=50, color="#0ea5e9", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Spearman correlation with global block rank")
    ax.set_ylabel("Number of images")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_topkp_overlap_curve(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    kps = np.asarray([row["Kp"] for row in rows], dtype=np.int32)
    means = np.asarray([row["mean"] for row in rows], dtype=np.float64)
    stds = np.asarray([row["std"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(kps, means, marker="o", linewidth=1.8)
    ax.fill_between(kps, np.maximum(0.0, means - stds), np.minimum(1.0, means + stds), alpha=0.18)
    ax.set_xlabel("Kp")
    ax.set_ylabel("Overlap with global top blocks")
    ax.set_xticks(kps)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_block_heatmap_grid(blocks: Sequence[Dict[str, object]], global_energy: np.ndarray, block_size: int, out_path: Path) -> None:
    heatmap = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    for block in blocks:
        block_id = int(block["block_id"])
        channel = int(block["channel"])
        heatmap[channel, int(block["h_start"]) : int(block["h_end"]), int(block["w_start"]) : int(block["w_end"])] = global_energy[block_id]

    vmax = float(np.max(heatmap))
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 7.4), dpi=220)
    images = []
    for channel, ax in enumerate(axes.flat):
        image = ax.imshow(heatmap[channel], cmap="viridis", vmin=0.0, vmax=vmax)
        images.append(image)
        ax.set_title(f"ch {channel}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.84)
    cbar.set_label("Global block energy")
    fig.suptitle(f"Block energy heatmaps, block_size={block_size}", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_stats(prefix: str, stats: Dict[str, float]) -> str:
    return (
        f"{prefix}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, "
        f"max={stats['max']:.6f}, median={stats['median']:.6f}"
    )


def analyze_block_size(
    block_size: int,
    blocks: List[Dict[str, object]],
    per_image_energy: np.ndarray,
    pairs: np.ndarray,
    out_dir: Path,
    save_per_image_energy: bool,
    data_dir: Path,
    dataset_note: str,
    checkpoint_path: Path,
    model_name: str,
    output_root: Path,
) -> Dict[str, object]:
    num_samples, num_blocks = per_image_energy.shape
    elements_per_block = block_size * block_size
    global_energy = per_image_energy.mean(axis=0, dtype=np.float64)
    energy_sum = per_image_energy.sum(axis=0, dtype=np.float64) * float(elements_per_block)
    rank_indices = np.argsort(-global_energy, kind="mergesort").astype(np.int64)
    rank_positions = inverse_rank_positions(rank_indices)
    sorted_energy = global_energy[rank_indices]
    total_energy_sum = float(np.sum(energy_sum))
    cumulative_ratio = np.cumsum(energy_sum[rank_indices], dtype=np.float64) / total_energy_sum if total_energy_sum > 0 else np.zeros(num_blocks)

    image_rank_indices = np.argsort(-per_image_energy, axis=1, kind="mergesort").astype(np.int32)
    image_rank_positions = inverse_rank_positions(image_rank_indices)
    per_image_spearman = spearman_from_rank_positions(image_rank_positions, rank_positions[None, :], num_blocks)
    per_image_stats = summarize(per_image_spearman)

    pairwise_spearman = (
        spearman_from_rank_positions(image_rank_positions[pairs[:, 0]], image_rank_positions[pairs[:, 1]], num_blocks)
        if len(pairs)
        else np.asarray([], dtype=np.float64)
    )
    pairwise_stats = summarize(pairwise_spearman) if len(pairwise_spearman) else summarize(np.asarray([np.nan]))

    topkp_global_rows: List[Dict[str, float]] = []
    topkp_pairwise_rows: List[Dict[str, float]] = []
    pairwise_overlap_by_kp: Dict[int, np.ndarray] = {}
    keep_blocks_by_kp: Dict[int, int] = {}
    for kp in KP_VALUES:
        num_keep_blocks = kp // elements_per_block
        global_top_mask = rank_positions < num_keep_blocks
        image_top_mask = image_rank_positions < num_keep_blocks
        overlap_global = np.sum(image_top_mask & global_top_mask[None, :], axis=1, dtype=np.float64) / float(num_keep_blocks)
        topkp_global_rows.append({"Kp": int(kp), "num_keep_blocks": int(num_keep_blocks), **summarize(overlap_global)})

        overlap_pairwise = (
            np.sum(image_top_mask[pairs[:, 0]] & image_top_mask[pairs[:, 1]], axis=1, dtype=np.float64) / float(num_keep_blocks)
            if len(pairs)
            else np.asarray([], dtype=np.float64)
        )
        pairwise_overlap_by_kp[int(kp)] = overlap_pairwise
        keep_blocks_by_kp[int(kp)] = int(num_keep_blocks)
        topkp_pairwise_rows.append(
            {"Kp": int(kp), "num_keep_blocks": int(num_keep_blocks), **(summarize(overlap_pairwise) if len(overlap_pairwise) else summarize(np.asarray([np.nan])))}
        )

    ensure_dir(out_dir)
    np.save(out_dir / f"global_block_energy_b{block_size}.npy", global_energy)
    np.save(out_dir / f"block_energy_rank_indices_b{block_size}.npy", rank_indices)
    if save_per_image_energy:
        np.save(out_dir / f"per_image_block_energy_b{block_size}.npy", per_image_energy.astype(np.float32, copy=False))

    write_block_importance_table(out_dir / f"block_importance_table_b{block_size}.csv", blocks, global_energy, energy_sum, rank_indices)
    write_per_image_spearman(out_dir / f"per_image_block_spearman_b{block_size}.csv", per_image_spearman)
    write_pairwise_spearman(out_dir / f"pairwise_block_spearman_sample_b{block_size}.csv", pairs, pairwise_spearman)
    write_summary_rows(
        out_dir / f"topkp_block_overlap_with_global_b{block_size}.csv",
        topkp_global_rows,
        ["Kp", "num_keep_blocks", "mean", "std", "min", "max", "median"],
    )
    write_pairwise_overlap_sample(
        out_dir / f"topkp_block_overlap_pairwise_sample_b{block_size}.csv",
        pairs,
        pairwise_overlap_by_kp,
        keep_blocks_by_kp,
    )
    write_summary_rows(
        out_dir / f"topkp_block_overlap_pairwise_summary_b{block_size}.csv",
        topkp_pairwise_rows,
        ["Kp", "num_keep_blocks", "mean", "std", "min", "max", "median"],
    )

    save_rank_curve(sorted_energy, out_dir / f"block_energy_rank_curve_b{block_size}.png", "Global block energy")
    save_rank_curve(cumulative_ratio, out_dir / f"block_cumulative_energy_curve_b{block_size}.png", "Cumulative energy ratio")
    save_top_bar(blocks, global_energy, rank_indices, out_dir / f"block_energy_bar_top_b{block_size}.png")
    save_spearman_hist(per_image_spearman, out_dir / f"per_image_block_spearman_hist_b{block_size}.png")
    save_topkp_overlap_curve(topkp_global_rows, out_dir / f"topkp_block_overlap_curve_b{block_size}.png")
    save_block_heatmap_grid(blocks, global_energy, block_size, out_dir / f"block_energy_heatmap_grid_b{block_size}.png")

    energy_stats = summarize(global_energy)
    energy_cv = float(energy_stats["std"] / max(abs(energy_stats["mean"]), 1e-12))
    top_10 = top_ratio_from_sorted(energy_sum, rank_indices, 0.10)
    top_25 = top_ratio_from_sorted(energy_sum, rank_indices, 0.25)
    top_50 = top_ratio_from_sorted(energy_sum, rank_indices, 0.50)
    global_more_stable = per_image_stats["mean"] > pairwise_stats["mean"]
    layered = top_10 >= 0.18 or energy_cv >= 0.15
    suitable = "yes" if layered and global_more_stable else "maybe"

    summary_lines = [
        f"Block energy summary, block_size={block_size}",
        "=" * 52,
        f"Dataset path: {data_dir}",
        f"Dataset split/source: {dataset_note}",
        f"Checkpoint path: {checkpoint_path}",
        f"Model: {model_name}",
        "PyTorch latent shape (C,H,W): 16 x 8 x 8",
        "Report latent shape (H,W,C): 8 x 8 x 16",
        "Power normalization: yes",
        "Task7 flatten order: HWC; flat_index_hwc=((h*W+w)*C+channel)",
        f"block_size: {block_size}",
        f"num_blocks: {num_blocks}",
        f"elements_per_block: {elements_per_block}",
        f"Training samples analyzed: {num_samples}",
        f"Output directory: {out_dir}",
        "Ranking metric: global_block_energy = mean_x mean_elements(c_i^2)",
        "",
        "Global block energy statistics:",
        f"mean={energy_stats['mean']:.8f}, std={energy_stats['std']:.8f}, min={energy_stats['min']:.8f}, "
        f"max={energy_stats['max']:.8f}, median={energy_stats['median']:.8f}, cv={energy_cv:.8f}",
        f"top 10% / 25% / 50% cumulative energy ratios: {top_10:.6f}, {top_25:.6f}, {top_50:.6f}",
        "",
        "Top 20 blocks:",
    ]
    for rank, block_id in enumerate(rank_indices[:20].tolist(), start=1):
        block = blocks[int(block_id)]
        summary_lines.append(
            f"{rank:02d}. block_id={int(block_id)} {block_label(block)} energy={global_energy[int(block_id)]:.8f} "
            f"flat_indices_hwc={block['flat_indices_hwc']}"
        )

    summary_lines.extend(["", "Spearman statistics:", format_stats("per-image block rank vs global block rank", per_image_stats), format_stats("random image-pair block rank", pairwise_stats), "", "Top-Kp block overlap with global:"])
    for row in topkp_global_rows:
        summary_lines.append(
            f"Kp={int(row['Kp'])} keep_blocks={int(row['num_keep_blocks'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, "
            f"min={row['min']:.6f}, max={row['max']:.6f}, median={row['median']:.6f}"
        )
    summary_lines.extend(["", "Random image-pair Top-Kp block overlap:"])
    for row in topkp_pairwise_rows:
        summary_lines.append(
            f"Kp={int(row['Kp'])} keep_blocks={int(row['num_keep_blocks'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, "
            f"min={row['min']:.6f}, max={row['max']:.6f}, median={row['median']:.6f}"
        )
    summary_lines.extend(
        [
            "",
            "Short conclusion hints:",
            f"该 block_size 下是否存在明显能量分层: {'yes' if layered else 'not obvious'}",
            f"全局 block 排序是否比随机图片对更稳定: {'yes' if global_more_stable else 'no'}",
            f"该尺度是否可能适合作为后续 top-Kp 保留策略: {suitable}",
        ]
    )
    (out_dir / f"block_energy_summary_b{block_size}.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "block_size": block_size,
        "num_blocks": num_blocks,
        "elements_per_block": elements_per_block,
        "energy_mean": energy_stats["mean"],
        "energy_std": energy_stats["std"],
        "energy_min": energy_stats["min"],
        "energy_max": energy_stats["max"],
        "energy_cv": energy_cv,
        "top_10_percent_energy_ratio": top_10,
        "top_25_percent_energy_ratio": top_25,
        "top_50_percent_energy_ratio": top_50,
        "per_image_vs_global_spearman_mean": per_image_stats["mean"],
        "per_image_vs_global_spearman_std": per_image_stats["std"],
        "pairwise_spearman_mean": pairwise_stats["mean"],
        "pairwise_spearman_std": pairwise_stats["std"],
        "topkp_rows": topkp_global_rows,
        "pairwise_rows": topkp_pairwise_rows,
        "rank_indices": rank_indices,
        "global_energy": global_energy,
        "summary_dir": str(out_dir),
    }


def save_scale_energy_cv_comparison(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    sizes = [int(row["block_size"]) for row in rows]
    cvs = [float(row["energy_cv"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(sizes, cvs, marker="o", linewidth=1.8)
    ax.set_xlabel("Block size")
    ax.set_ylabel("Energy coefficient of variation")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scale_spearman_comparison(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    sizes = [int(row["block_size"]) for row in rows]
    per_means = [float(row["per_image_vs_global_spearman_mean"]) for row in rows]
    pair_means = [float(row["pairwise_spearman_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(sizes, per_means, marker="o", linewidth=1.8, label="image vs global")
    ax.plot(sizes, pair_means, marker="s", linewidth=1.8, label="pairwise")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Spearman mean")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scale_topkp_overlap_comparison(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=220)
    for row in rows:
        topkp_rows = row["topkp_rows"]
        ax.plot(
            [int(item["Kp"]) for item in topkp_rows],
            [float(item["mean"]) for item in topkp_rows],
            marker="o",
            linewidth=1.6,
            label=f"b={int(row['block_size'])}",
        )
    ax.set_xlabel("Kp")
    ax.set_ylabel("Overlap with global top blocks")
    ax.set_xticks(KP_VALUES)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_all_scales_outputs(output_dir: Path, comparison_rows: Sequence[Dict[str, object]], block_sizes: Sequence[int]) -> None:
    csv_path = output_dir / "block_scale_comparison.csv"
    fieldnames = [
        "block_size",
        "num_blocks",
        "elements_per_block",
        "energy_mean",
        "energy_std",
        "energy_min",
        "energy_max",
        "energy_cv",
        "top_10_percent_energy_ratio",
        "top_25_percent_energy_ratio",
        "top_50_percent_energy_ratio",
        "per_image_vs_global_spearman_mean",
        "per_image_vs_global_spearman_std",
        "pairwise_spearman_mean",
        "pairwise_spearman_std",
        "topkp_128_overlap_global_mean",
        "topkp_256_overlap_global_mean",
        "topkp_512_overlap_global_mean",
        "topkp_768_overlap_global_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            topkp_by_kp = {int(item["Kp"]): float(item["mean"]) for item in row["topkp_rows"]}
            writer.writerow(
                {
                    "block_size": int(row["block_size"]),
                    "num_blocks": int(row["num_blocks"]),
                    "elements_per_block": int(row["elements_per_block"]),
                    "energy_mean": float(row["energy_mean"]),
                    "energy_std": float(row["energy_std"]),
                    "energy_min": float(row["energy_min"]),
                    "energy_max": float(row["energy_max"]),
                    "energy_cv": float(row["energy_cv"]),
                    "top_10_percent_energy_ratio": float(row["top_10_percent_energy_ratio"]),
                    "top_25_percent_energy_ratio": float(row["top_25_percent_energy_ratio"]),
                    "top_50_percent_energy_ratio": float(row["top_50_percent_energy_ratio"]),
                    "per_image_vs_global_spearman_mean": float(row["per_image_vs_global_spearman_mean"]),
                    "per_image_vs_global_spearman_std": float(row["per_image_vs_global_spearman_std"]),
                    "pairwise_spearman_mean": float(row["pairwise_spearman_mean"]),
                    "pairwise_spearman_std": float(row["pairwise_spearman_std"]),
                    "topkp_128_overlap_global_mean": topkp_by_kp[128],
                    "topkp_256_overlap_global_mean": topkp_by_kp[256],
                    "topkp_512_overlap_global_mean": topkp_by_kp[512],
                    "topkp_768_overlap_global_mean": topkp_by_kp[768],
                }
            )

    save_scale_energy_cv_comparison(comparison_rows, output_dir / "block_scale_energy_cv_comparison.png")
    save_scale_spearman_comparison(comparison_rows, output_dir / "block_scale_spearman_comparison.png")
    save_scale_topkp_overlap_comparison(comparison_rows, output_dir / "block_scale_topkp_overlap_comparison.png")

    best_by_cv = max(comparison_rows, key=lambda row: float(row["energy_cv"]))
    best_by_stability = max(comparison_rows, key=lambda row: float(row["per_image_vs_global_spearman_mean"]) - float(row["pairwise_spearman_mean"]))
    lines = [
        "Block energy all-scales summary",
        "=" * 40,
        f"Run block sizes: {list(block_sizes)}",
        "",
        "Scale comparison:",
    ]
    for row in comparison_rows:
        lines.append(
            f"b={int(row['block_size'])}: num_blocks={int(row['num_blocks'])}, elements_per_block={int(row['elements_per_block'])}, "
            f"energy_cv={float(row['energy_cv']):.6f}, per-image/global Spearman mean={float(row['per_image_vs_global_spearman_mean']):.6f}, "
            f"pairwise Spearman mean={float(row['pairwise_spearman_mean']):.6f}"
        )
        topkp_means = {int(item["Kp"]): float(item["mean"]) for item in row["topkp_rows"]}
        lines.append(
            f"  Top-Kp overlap means: Kp128={topkp_means[128]:.6f}, Kp256={topkp_means[256]:.6f}, "
            f"Kp512={topkp_means[512]:.6f}, Kp768={topkp_means[768]:.6f}"
        )
    lines.extend(
        [
            "",
            f"Highest energy CV scale: b={int(best_by_cv['block_size'])}",
            f"Most stable-vs-pairwise scale: b={int(best_by_stability['block_size'])}",
            "Initial RD-test suggestion: prioritize the scale with both clear energy concentration and stable global ranking; "
            f"from this run, b={int(best_by_stability['block_size'])} is the first candidate.",
        ]
    )
    (output_dir / "block_energy_all_scales_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint_path = (
        resolve_relative_to_project(args.checkpoint, project_dir).resolve()
        if args.checkpoint
        else find_snr7_checkpoint(project_dir).resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    block_sizes = parse_block_sizes(args.block_sizes)
    output_dir = ensure_dir(resolve_relative_to_project(args.output_dir, project_dir))
    device = get_device(args.device)

    train_set, dataset_note = load_project_train_split_dataset(
        data_dir,
        max_train_samples=args.max_train_samples,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        split_seed=args.split_seed,
    )
    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    blocks_by_size = {block_size: build_blocks(LATENT_C, LATENT_H, LATENT_W, block_size) for block_size in block_sizes}
    self_test_block_masks(blocks_by_size)
    model = load_jscc_checkpoint(checkpoint_path, device)

    try:
        per_image_by_size, num_samples = collect_per_image_block_energy(model, dataloader, device, blocks_by_size)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or ("cuda" in str(exc).lower() and "memory" in str(exc).lower()):
            raise RuntimeError("Block energy extraction ran out of memory. Try reducing --batch_size.") from exc
        raise

    pairs = sample_pairs(num_samples, args.pair_samples, args.seed)
    comparison_rows: List[Dict[str, object]] = []
    for block_size in block_sizes:
        row = analyze_block_size(
            block_size=block_size,
            blocks=blocks_by_size[block_size],
            per_image_energy=per_image_by_size[block_size],
            pairs=pairs,
            out_dir=ensure_dir(output_dir / f"b{block_size}"),
            save_per_image_energy=bool(args.save_per_image_energy),
            data_dir=data_dir,
            dataset_note=dataset_note,
            checkpoint_path=checkpoint_path,
            model_name=type(model).__name__,
            output_root=output_dir,
        )
        comparison_rows.append(row)
        print(
            f"b={block_size}: samples={num_samples} blocks={row['num_blocks']} "
            f"energy_cv={float(row['energy_cv']):.6f} "
            f"per/global_spearman={float(row['per_image_vs_global_spearman_mean']):.6f}"
        )

    write_all_scales_outputs(output_dir, comparison_rows, block_sizes)
    print(f"Analyzed {num_samples} train-split samples on {device}.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

# Example full run:
# python scripts/innovation_block_energy_importance.py \
#     --data_dir /home/lc/class/yuyi/cifar-10 \
#     --project_dir /home/lc/class/yuyi/semantic_jscc_lab \
#     --output_dir outputs/innovation/block_energy_importance \
#     --block_sizes 2,4 \
#     --batch_size 256 \
#     --pair_samples 5000
