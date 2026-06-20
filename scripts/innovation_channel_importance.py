#!/usr/bin/env python
"""Innovation experiment: channel-level latent energy importance.

This script rebuilds the same train split convention used by train_jscc.py:
  load_cifar_array_dataset(data_dir) -> make_splits(train=0.8, val=0.1, test=0.1, seed=42)

It uses the task7-compatible Deep JSCC latent pipeline:
  image -> encoder -> power_normalize -> channel statistics

PyTorch latent shape is (C,H,W) = (16,8,8). The report latent shape is
(H,W,C) = (8,8,16). Flattening is not needed for channel statistics, but the
summary records that task7 uses HWC flattening.
"""

from __future__ import annotations

import argparse
import csv
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

from eval_rate_distortion import LATENT_C, LATENT_H, LATENT_W, load_jscc_checkpoint
from jscc_lab.channel import power_normalize
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.utils import ensure_dir, get_device, seed_everything
from train_film_decoder_multirate import (
    ComplexFiLMDeepJSCC,
    FiLMDecoderDeepJSCC,
    MODEL_TYPE as FILM_MODEL_TYPE,
    MODEL_VARIANT_COMPLEX_DECODER,
    MODEL_VARIANT_DECODER,
    MODEL_VARIANT_SNR_RATE_COMPLEX,
    SNRRateFiLMDeepJSCC,
)


TOPM_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
CHANNEL_SPEARMAN_DENOMINATOR = float(LATENT_C * (LATENT_C * LATENT_C - 1))


def load_channel_statistics_model(checkpoint_path: Path, device: torch.device):
    """Load a checkpoint whose encoder can be called without a rate condition."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return load_jscc_checkpoint(checkpoint_path, device)

    extra = dict(checkpoint.get("extra", {}) or {})
    model_type = str(extra.get("model_type", "deepjscc"))
    if model_type != FILM_MODEL_TYPE:
        return load_jscc_checkpoint(checkpoint_path, device)

    variant = str(extra.get("model_variant", MODEL_VARIANT_DECODER))
    train_snr_db = float(extra.get("train_snr_db", extra.get("snr_db", 7.0)))
    film_hidden_dim = int(extra.get("film_hidden_dim", 64))
    cond_dim = int(extra.get("cond_dim", 2))

    if variant == MODEL_VARIANT_COMPLEX_DECODER:
        model = ComplexFiLMDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
    elif variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
        model = SNRRateFiLMDeepJSCC(snr_db=train_snr_db, cond_dim=3, film_hidden_dim=film_hidden_dim)
    elif variant == MODEL_VARIANT_DECODER:
        model = FiLMDecoderDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
    else:
        raise ValueError(
            f"Checkpoint model_variant={variant!r} uses a conditioned encoder. "
            "Channel energy statistics require an encoder callable as E(x)."
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute channel-level latent energy importance for SNR=7 Deep JSCC.")
    parser.add_argument("--data_dir", default="/home/lc/class/yuyi/cifar-10", help="CIFAR-10 directory or data file.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--checkpoint", default=None, help="Optional SNR=7 checkpoint path. If omitted, auto-search.")
    parser.add_argument("--output_dir", default="outputs/innovation/channel_importance")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Device string; default auto uses CUDA if available, else CPU.")
    parser.add_argument("--pair_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026, help="Seed for pair sampling and reproducibility.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train split samples for debugging.")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for rebuilding the train/val/test split.")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def resolve_relative_to_project(path_text: str | Path, project_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def find_snr7_checkpoint(project_dir: Path) -> Path:
    """Find the SNR=7 Deep JSCC checkpoint without requiring a hard-coded run layout."""

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
    """Rebuild the train split used by train_jscc.py and exclude validation/test samples."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    images, labels = load_cifar_array_dataset(data_dir)
    dataset = CIFARArrayDataset(images, labels)
    train_set, _, _ = make_splits(
        dataset,
        train=train_split,
        val=val_split,
        test=test_split,
        seed=split_seed,
    )
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
            note += f"; first {max_train_samples} samples"
    return train_set, note


def inverse_rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Convert argsort output to rank positions where 0 is most important."""

    if sorted_indices.ndim == 1:
        positions = np.empty_like(sorted_indices, dtype=np.int16)
        positions[sorted_indices] = np.arange(sorted_indices.shape[0], dtype=np.int16)
        return positions

    positions = np.empty_like(sorted_indices, dtype=np.int16)
    ranks = np.arange(sorted_indices.shape[1], dtype=np.int16)
    rows = np.arange(sorted_indices.shape[0])[:, None]
    positions[rows, sorted_indices] = ranks
    return positions


def spearman_from_rank_positions(rank_a: np.ndarray, rank_b: np.ndarray) -> np.ndarray:
    """Spearman correlation for complete channel rank vectors without scipy."""

    diff = rank_a.astype(np.int32) - rank_b.astype(np.int32)
    sum_sq = np.sum(diff * diff, axis=-1, dtype=np.float64)
    return 1.0 - (6.0 * sum_sq / CHANNEL_SPEARMAN_DENOMINATOR)


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
def collect_channel_statistics(model, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray | int]:
    """Collect channel statistics from power-normalized latent codes."""

    model.eval()
    sum_z = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    sum_z2 = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    sum_abs = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    per_image_energy_batches: List[np.ndarray] = []
    total_samples = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        if tuple(z.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
            raise ValueError(f"Expected latent shape (N,16,8,8), got {tuple(z.shape)}.")

        z_cpu = z.detach().cpu().float()
        z_np = z_cpu.numpy()
        z2_np = z_cpu.pow(2).numpy()
        abs_np = z_cpu.abs().numpy()

        sum_z += z_np.sum(axis=0, dtype=np.float64)
        sum_z2 += z2_np.sum(axis=0, dtype=np.float64)
        sum_abs += abs_np.sum(axis=0, dtype=np.float64)
        per_image_energy_batches.append(z2_np.mean(axis=(2, 3)).astype(np.float32, copy=False))
        total_samples += int(z.shape[0])

    if total_samples == 0:
        raise ValueError("No samples were loaded from the train split.")

    per_image_channel_energy = np.concatenate(per_image_energy_batches, axis=0)
    channel_energy_sum = sum_z2.sum(axis=(1, 2))
    global_channel_energy = channel_energy_sum / float(total_samples * LATENT_H * LATENT_W)
    mean_map = sum_z / float(total_samples)
    second_moment_map = sum_z2 / float(total_samples)
    variance_map = np.maximum(second_moment_map - mean_map * mean_map, 0.0)
    global_channel_variance = variance_map.mean(axis=(1, 2))

    flat_count = float(total_samples * LATENT_H * LATENT_W)
    global_channel_mean = sum_z.sum(axis=(1, 2)) / flat_count
    global_channel_abs_mean = sum_abs.sum(axis=(1, 2)) / flat_count
    channel_second_moment = sum_z2.sum(axis=(1, 2)) / flat_count
    global_channel_std = np.sqrt(np.maximum(channel_second_moment - global_channel_mean * global_channel_mean, 0.0))
    channel_energy_map = second_moment_map

    return {
        "num_samples": total_samples,
        "per_image_channel_energy": per_image_channel_energy.astype(np.float32),
        "channel_energy_sum": channel_energy_sum.astype(np.float64),
        "global_channel_energy": global_channel_energy.astype(np.float64),
        "global_channel_variance": global_channel_variance.astype(np.float64),
        "global_channel_mean": global_channel_mean.astype(np.float64),
        "global_channel_std": global_channel_std.astype(np.float64),
        "global_channel_abs_mean": global_channel_abs_mean.astype(np.float64),
        "global_channel_l1": global_channel_abs_mean.astype(np.float64),
        "channel_energy_map": channel_energy_map.astype(np.float64),
    }


def write_channel_importance_table(
    path: Path,
    energy: np.ndarray,
    variance: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    abs_mean: np.ndarray,
    energy_sum: np.ndarray,
    energy_rank_indices: np.ndarray,
    variance_rank_indices: np.ndarray,
    abs_mean_rank_indices: np.ndarray,
) -> None:
    total_energy = float(np.sum(energy))
    energy_ratio = energy / total_energy if total_energy > 0 else np.zeros_like(energy)
    energy_rank_positions = inverse_rank_positions(energy_rank_indices)
    variance_rank_positions = inverse_rank_positions(variance_rank_indices)
    abs_rank_positions = inverse_rank_positions(abs_mean_rank_indices)
    sorted_cumulative = np.cumsum(energy[energy_rank_indices], dtype=np.float64)
    cumulative_by_channel = np.zeros_like(energy, dtype=np.float64)
    if total_energy > 0:
        cumulative_by_channel[energy_rank_indices] = sorted_cumulative / total_energy

    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "channel",
            "energy",
            "energy_sum",
            "energy_rank",
            "energy_ratio",
            "energy_cumulative_ratio",
            "variance",
            "variance_rank",
            "mean",
            "std",
            "abs_mean",
            "channel_l1",
            "abs_mean_rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for channel in range(LATENT_C):
            writer.writerow(
                {
                    "channel": channel,
                    "energy": float(energy[channel]),
                    "energy_sum": float(energy_sum[channel]),
                    "energy_rank": int(energy_rank_positions[channel]) + 1,
                    "energy_ratio": float(energy_ratio[channel]),
                    "energy_cumulative_ratio": float(cumulative_by_channel[channel]),
                    "variance": float(variance[channel]),
                    "variance_rank": int(variance_rank_positions[channel]) + 1,
                    "mean": float(mean[channel]),
                    "std": float(std[channel]),
                    "abs_mean": float(abs_mean[channel]),
                    "channel_l1": float(abs_mean[channel]),
                    "abs_mean_rank": int(abs_rank_positions[channel]) + 1,
                }
            )


def write_per_image_spearman(path: Path, spearman: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "spearman_with_global_channel_rank"])
        writer.writeheader()
        for idx, value in enumerate(spearman.tolist()):
            writer.writerow({"image_index": idx, "spearman_with_global_channel_rank": float(value)})


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


def write_pairwise_topm_sample(path: Path, pairs: np.ndarray, overlap_by_m: Dict[int, np.ndarray]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_index_a", "image_index_b", "M", "overlap"])
        writer.writeheader()
        for pair_id, (idx_a, idx_b) in enumerate(pairs.tolist()):
            for top_m, overlaps in overlap_by_m.items():
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "image_index_a": int(idx_a),
                        "image_index_b": int(idx_b),
                        "M": int(top_m),
                        "overlap": float(overlaps[pair_id]),
                    }
                )


def save_channel_energy_bar(energy: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    channels = np.arange(LATENT_C)
    ax.bar(channels, energy, color="#2563eb")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Global channel energy")
    ax.set_xticks(channels)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_channel_energy_rank_bar(energy: np.ndarray, energy_rank_indices: np.ndarray, out_path: Path) -> None:
    sorted_energy = energy[energy_rank_indices]
    ranks = np.arange(1, LATENT_C + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax.bar(ranks, sorted_energy, color="#059669")
    ax.set_xlabel("Energy rank")
    ax.set_ylabel("Global channel energy")
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{rank}\nch{channel}" for rank, channel in zip(ranks, energy_rank_indices)], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_channel_cumulative_energy_curve(energy: np.ndarray, energy_rank_indices: np.ndarray, out_path: Path) -> None:
    sorted_energy = energy[energy_rank_indices]
    total = float(np.sum(sorted_energy))
    cumulative = np.cumsum(sorted_energy, dtype=np.float64) / total if total > 0 else np.zeros_like(sorted_energy)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(np.arange(1, LATENT_C + 1), cumulative, marker="o", linewidth=1.8)
    ax.set_xlabel("Top-M channels")
    ax.set_ylabel("Cumulative energy ratio")
    ax.set_xticks(np.arange(1, LATENT_C + 1))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_channel_variance_bar(variance: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    channels = np.arange(LATENT_C)
    ax.bar(channels, variance, color="#7c3aed")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Channel variance")
    ax.set_xticks(channels)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_energy_vs_variance_scatter(energy: np.ndarray, variance: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=220)
    ax.scatter(energy, variance, s=44, color="#dc2626", alpha=0.85)
    for channel, (x_value, y_value) in enumerate(zip(energy, variance)):
        ax.annotate(str(channel), (float(x_value), float(y_value)), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Global channel energy")
    ax.set_ylabel("Channel variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_spearman_hist(spearman: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    bins = np.linspace(-1.0, 1.0, 33)
    ax.hist(spearman, bins=bins, color="#0ea5e9", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Spearman correlation with global channel rank")
    ax.set_ylabel("Number of images")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_topm_overlap_curve(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    top_m = np.asarray([row["M"] for row in rows], dtype=np.int32)
    means = np.asarray([row["mean"] for row in rows], dtype=np.float64)
    stds = np.asarray([row["std"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(top_m, means, marker="o", linewidth=1.8)
    ax.fill_between(top_m, np.maximum(0.0, means - stds), np.minimum(1.0, means + stds), alpha=0.18)
    ax.set_xlabel("M")
    ax.set_ylabel("Overlap with global top-M channels")
    ax.set_xticks(top_m)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_channel_energy_heatmap_grid(
    channel_energy_map: np.ndarray,
    energy_rank_positions: np.ndarray,
    out_path: Path,
) -> None:
    vmax = float(np.max(channel_energy_map))
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 7.4), dpi=220)
    images = []
    for channel, ax in enumerate(axes.flat):
        image = ax.imshow(channel_energy_map[channel], cmap="viridis", vmin=0.0, vmax=vmax)
        images.append(image)
        ax.set_title(f"ch {channel} | rank {int(energy_rank_positions[channel]) + 1}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.84)
    cbar.set_label("mean_x(c^2)")
    fig.suptitle("Channel energy maps", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_stats(prefix: str, stats: Dict[str, float]) -> str:
    return (
        f"{prefix}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, "
        f"max={stats['max']:.6f}, median={stats['median']:.6f}"
    )


def write_summary(
    path: Path,
    *,
    data_dir: Path,
    dataset_note: str,
    checkpoint_path: Path,
    model_name: str,
    num_samples: int,
    output_dir: Path,
    energy: np.ndarray,
    variance: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    abs_mean: np.ndarray,
    energy_rank_indices: np.ndarray,
    variance_rank_indices: np.ndarray,
    abs_mean_rank_indices: np.ndarray,
    rank_correlations: Dict[str, float],
    per_image_stats: Dict[str, float],
    pairwise_stats: Dict[str, float],
    topm_global_rows: Sequence[Dict[str, float]],
    topm_pairwise_rows: Sequence[Dict[str, float]],
) -> None:
    energy_stats = summarize(energy)
    variance_stats = summarize(variance)
    total_energy = float(np.sum(energy))
    energy_ratio = energy / total_energy if total_energy > 0 else np.zeros_like(energy)
    cumulative_sorted = np.cumsum(energy[energy_rank_indices], dtype=np.float64)
    cumulative_ratio_sorted = cumulative_sorted / total_energy if total_energy > 0 else np.zeros_like(cumulative_sorted)
    top_ratios = {
        top_m: float(cumulative_ratio_sorted[top_m - 1])
        for top_m in [2, 4, 6, 8]
    }
    layering = "yes" if (top_ratios[4] >= 0.35 or top_ratios[8] >= 0.65 or energy.max() >= 2.0 * max(energy.min(), 1e-12)) else "not obvious"
    global_more_stable = per_image_stats["mean"] > pairwise_stats["mean"]

    lines = [
        "Innovation experiment: channel-level latent importance",
        "=" * 60,
        f"Dataset path: {data_dir}",
        f"Dataset split/source: {dataset_note}",
        f"Checkpoint path: {checkpoint_path}",
        f"Model: {model_name}",
        "PyTorch latent shape (C,H,W): 16 x 8 x 8",
        "Report latent shape (H,W,C): 8 x 8 x 16",
        "Task7 flatten order: HWC (CHW -> HWC -> flatten); channel stats do not require flattening",
        "Power normalization: yes, jscc_lab.channel.power_normalize before statistics",
        f"Training samples analyzed: {num_samples}",
        f"Output directory: {output_dir}",
        "",
        "Global channel energy statistics:",
        format_stats("global_channel_energy", energy_stats),
        "",
        "Global channel variance statistics:",
        format_stats("global_channel_variance", variance_stats),
        "",
        "Channel rankings:",
        f"energy rank channels: {energy_rank_indices.tolist()}",
        f"variance rank channels: {variance_rank_indices.tolist()}",
        f"abs_mean rank channels: {abs_mean_rank_indices.tolist()}",
        "",
        "Rank correlation between channel metrics:",
        f"energy vs variance Spearman: {rank_correlations['energy_vs_variance']:.6f}",
        f"energy vs abs_mean Spearman: {rank_correlations['energy_vs_abs_mean']:.6f}",
        f"variance vs abs_mean Spearman: {rank_correlations['variance_vs_abs_mean']:.6f}",
        "",
        "Per-channel values in energy-rank order:",
    ]
    for sorted_pos, channel in enumerate(energy_rank_indices.tolist(), start=1):
        lines.append(
            f"rank={sorted_pos:02d} channel={channel:02d} energy={energy[channel]:.8f} "
            f"energy_ratio={energy_ratio[channel]:.6f} cumulative_energy_ratio={cumulative_ratio_sorted[sorted_pos - 1]:.6f} "
            f"variance={variance[channel]:.8f} mean={mean[channel]:.8f} std={std[channel]:.8f} "
            f"abs_mean={abs_mean[channel]:.8f}"
        )

    lines.extend(
        [
            "",
            "Spearman stability statistics:",
            format_stats("per-image channel rank vs global channel rank", per_image_stats),
            format_stats("random image-pair channel rank", pairwise_stats),
            "",
            "Top-M channel overlap with global:",
        ]
    )
    for row in topm_global_rows:
        lines.append(
            f"M={int(row['M'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(["", "Random image-pair Top-M channel overlap:"])
    for row in topm_pairwise_rows:
        lines.append(
            f"M={int(row['M'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(
        [
            "",
            "Short conclusion hints:",
            f"明显 channel 能量分层: {layering}",
            f"top-2/top-4/top-6/top-8 channel energy ratios: "
            f"{top_ratios[2]:.6f}, {top_ratios[4]:.6f}, {top_ratios[6]:.6f}, {top_ratios[8]:.6f}",
            "channel 级全局排序是否比随机图片对更稳定: "
            f"{'yes' if global_more_stable else 'no'} "
            f"(per-image-vs-global mean={per_image_stats['mean']:.6f}, pairwise mean={pairwise_stats['mean']:.6f})",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    model = load_channel_statistics_model(checkpoint_path, device)
    stats = collect_channel_statistics(model, dataloader, device)
    num_samples = int(stats["num_samples"])
    per_image_channel_energy = stats["per_image_channel_energy"]
    energy = stats["global_channel_energy"]
    variance = stats["global_channel_variance"]
    mean = stats["global_channel_mean"]
    std = stats["global_channel_std"]
    abs_mean = stats["global_channel_abs_mean"]
    channel_l1 = stats["global_channel_l1"]
    energy_sum = stats["channel_energy_sum"]
    channel_energy_map = stats["channel_energy_map"]

    energy_rank_indices = np.argsort(-energy, kind="mergesort").astype(np.int16)
    variance_rank_indices = np.argsort(-variance, kind="mergesort").astype(np.int16)
    abs_mean_rank_indices = np.argsort(-abs_mean, kind="mergesort").astype(np.int16)
    energy_rank_positions = inverse_rank_positions(energy_rank_indices)
    variance_rank_positions = inverse_rank_positions(variance_rank_indices)
    abs_rank_positions = inverse_rank_positions(abs_mean_rank_indices)

    rank_correlations = {
        "energy_vs_variance": float(spearman_from_rank_positions(energy_rank_positions, variance_rank_positions)),
        "energy_vs_abs_mean": float(spearman_from_rank_positions(energy_rank_positions, abs_rank_positions)),
        "variance_vs_abs_mean": float(spearman_from_rank_positions(variance_rank_positions, abs_rank_positions)),
    }

    image_rank_indices = np.argsort(-per_image_channel_energy, axis=1, kind="mergesort").astype(np.int16)
    image_rank_positions = inverse_rank_positions(image_rank_indices)
    per_image_spearman = spearman_from_rank_positions(image_rank_positions, energy_rank_positions[None, :])
    per_image_stats = summarize(per_image_spearman)

    pairs = sample_pairs(num_samples, args.pair_samples, args.seed)
    pairwise_spearman = (
        spearman_from_rank_positions(image_rank_positions[pairs[:, 0]], image_rank_positions[pairs[:, 1]])
        if len(pairs)
        else np.asarray([], dtype=np.float64)
    )
    pairwise_stats = summarize(pairwise_spearman) if len(pairwise_spearman) else summarize(np.asarray([np.nan]))

    topm_global_rows: List[Dict[str, float]] = []
    topm_pairwise_rows: List[Dict[str, float]] = []
    pairwise_overlap_by_m: Dict[int, np.ndarray] = {}
    for top_m in TOPM_VALUES:
        image_topm_mask = image_rank_positions < top_m
        global_topm_mask = energy_rank_positions < top_m
        overlap_global = np.sum(image_topm_mask & global_topm_mask[None, :], axis=1, dtype=np.float64) / float(top_m)
        topm_global_rows.append({"M": int(top_m), **summarize(overlap_global)})

        overlap_pairwise = (
            np.sum(image_topm_mask[pairs[:, 0]] & image_topm_mask[pairs[:, 1]], axis=1, dtype=np.float64) / float(top_m)
            if len(pairs)
            else np.asarray([], dtype=np.float64)
        )
        pairwise_overlap_by_m[int(top_m)] = overlap_pairwise
        topm_pairwise_rows.append(
            {"M": int(top_m), **(summarize(overlap_pairwise) if len(overlap_pairwise) else summarize(np.asarray([np.nan])))}
        )

    np.save(output_dir / "global_channel_energy.npy", energy)
    np.save(output_dir / "global_channel_variance.npy", variance)
    np.save(output_dir / "global_channel_mean.npy", mean)
    np.save(output_dir / "global_channel_std.npy", std)
    np.save(output_dir / "global_channel_abs_mean.npy", abs_mean)
    np.save(output_dir / "global_channel_l1.npy", channel_l1)
    np.save(output_dir / "channel_energy_sum.npy", energy_sum)
    total_energy = float(np.sum(energy))
    np.save(output_dir / "channel_energy_ratio.npy", energy / total_energy if total_energy > 0 else np.zeros_like(energy))
    np.save(output_dir / "channel_energy_rank_indices.npy", energy_rank_indices)
    np.save(output_dir / "channel_variance_rank_indices.npy", variance_rank_indices)
    np.save(output_dir / "channel_abs_mean_rank_indices.npy", abs_mean_rank_indices)
    np.save(output_dir / "per_image_channel_energy.npy", per_image_channel_energy)
    np.save(output_dir / "channel_energy_map.npy", channel_energy_map)

    write_channel_importance_table(
        output_dir / "channel_importance_table.csv",
        energy,
        variance,
        mean,
        std,
        abs_mean,
        energy_sum,
        energy_rank_indices,
        variance_rank_indices,
        abs_mean_rank_indices,
    )
    write_per_image_spearman(output_dir / "per_image_channel_spearman.csv", per_image_spearman)
    write_pairwise_spearman(output_dir / "pairwise_channel_spearman_sample.csv", pairs, pairwise_spearman)
    write_summary_rows(
        output_dir / "topm_channel_overlap_with_global.csv",
        topm_global_rows,
        ["M", "mean", "std", "min", "max", "median"],
    )
    write_pairwise_topm_sample(output_dir / "topm_channel_overlap_pairwise_sample.csv", pairs, pairwise_overlap_by_m)
    write_summary_rows(
        output_dir / "topm_channel_overlap_pairwise_summary.csv",
        topm_pairwise_rows,
        ["M", "mean", "std", "min", "max", "median"],
    )

    save_channel_energy_bar(energy, output_dir / "channel_energy_bar.png")
    save_channel_energy_rank_bar(energy, energy_rank_indices, output_dir / "channel_energy_rank_bar.png")
    save_channel_cumulative_energy_curve(energy, energy_rank_indices, output_dir / "channel_cumulative_energy_curve.png")
    save_channel_variance_bar(variance, output_dir / "channel_variance_bar.png")
    save_energy_vs_variance_scatter(energy, variance, output_dir / "channel_energy_vs_variance_scatter.png")
    save_spearman_hist(per_image_spearman, output_dir / "per_image_channel_spearman_hist.png")
    save_topm_overlap_curve(topm_global_rows, output_dir / "topm_channel_overlap_curve.png")
    save_channel_energy_heatmap_grid(channel_energy_map, energy_rank_positions, output_dir / "channel_energy_heatmap_grid.png")

    write_summary(
        output_dir / "channel_importance_summary.txt",
        data_dir=data_dir,
        dataset_note=dataset_note,
        checkpoint_path=checkpoint_path,
        model_name=type(model).__name__,
        num_samples=num_samples,
        output_dir=output_dir,
        energy=energy,
        variance=variance,
        mean=mean,
        std=std,
        abs_mean=abs_mean,
        energy_rank_indices=energy_rank_indices,
        variance_rank_indices=variance_rank_indices,
        abs_mean_rank_indices=abs_mean_rank_indices,
        rank_correlations=rank_correlations,
        per_image_stats=per_image_stats,
        pairwise_stats=pairwise_stats,
        topm_global_rows=topm_global_rows,
        topm_pairwise_rows=topm_pairwise_rows,
    )

    print(f"Analyzed {num_samples} training-split images on {device}.")
    print(f"Checkpoint: {checkpoint_path}")
    print("PyTorch latent shape: C,H,W = 16,8,8; report latent shape: H,W,C = 8,8,16")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

# Example full run:
# python scripts/innovation_channel_importance.py \
#     --data_dir /home/lc/class/yuyi/cifar-10 \
#     --project_dir /home/lc/class/yuyi/semantic_jscc_lab \
#     --output_dir outputs/innovation/channel_importance \
#     --batch_size 256 \
#     --pair_samples 5000
