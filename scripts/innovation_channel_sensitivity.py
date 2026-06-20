#!/usr/bin/env python
"""Innovation experiment: channel-level latent sensitivity analysis.

The script rebuilds the same train split convention used by train_jscc.py:
  load_cifar_array_dataset(data_dir) -> make_splits(train=0.8, val=0.1, test=0.1, seed=42)

For each image it computes the normalized latent code, reconstructs once with
the full latent, then zeros one latent channel at a time. Channel sensitivity is
the average MSE increase caused by masking that channel.
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
from jscc_lab.channel import power_normalize, snr_db_to_noise_std
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.utils import ensure_dir, get_device, seed_everything


TOPM_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]
CHANNEL_SPEARMAN_DENOMINATOR = float(LATENT_C * (LATENT_C * LATENT_C - 1))


def parse_optional_int(text: str | None) -> int | None:
    if text is None:
        return None
    if str(text).lower() in {"none", "null"}:
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("--max_samples must be positive or None.")
    return value


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
    parser = argparse.ArgumentParser(description="Compute channel-level latent sensitivity for SNR=7 Deep JSCC.")
    parser.add_argument("--data_dir", default="/home/lc/class/yuyi/cifar-10", help="CIFAR-10 directory or data file.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--checkpoint", default=None, help="Optional SNR=7 checkpoint path. If omitted, auto-search.")
    parser.add_argument(
        "--channel_importance_csv",
        default="outputs/innovation/channel_importance/channel_importance_table.csv",
        help="Optional existing channel importance table for energy/variance comparison.",
    )
    parser.add_argument("--output_dir", default="outputs/innovation/channel_sensitivity")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Device string; default auto uses CUDA if available, else CPU.")
    parser.add_argument("--test_snr_db", type=float, default=7.0)
    parser.add_argument("--pair_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_samples", type=parse_optional_int, default=10000)
    parser.add_argument("--shared_noise", type=str2bool, nargs="?", const=True, default=True)
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
    max_samples: int | None,
    train_split: float,
    val_split: float,
    test_split: float,
    split_seed: int,
) -> Tuple[Dataset, str]:
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

    if max_samples is not None and max_samples < len(train_set):
        train_set = Subset(train_set, list(range(max_samples)))
        note += f"; first {max_samples} samples from the train split"
    return train_set, note


def inverse_rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
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


def load_channel_importance_table(path: Path) -> Dict[str, np.ndarray] | None:
    if not path.is_file():
        return None

    energy = np.full(LATENT_C, np.nan, dtype=np.float64)
    variance = np.full(LATENT_C, np.nan, dtype=np.float64)
    abs_mean = np.full(LATENT_C, np.nan, dtype=np.float64)
    energy_rank_pos = np.full(LATENT_C, -1, dtype=np.int16)
    variance_rank_pos = np.full(LATENT_C, -1, dtype=np.int16)
    abs_rank_pos = np.full(LATENT_C, -1, dtype=np.int16)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = int(row["channel"])
            energy[channel] = float(row["energy"])
            variance[channel] = float(row["variance"])
            abs_mean[channel] = float(row["abs_mean"])
            energy_rank_pos[channel] = int(row["energy_rank"]) - 1
            variance_rank_pos[channel] = int(row["variance_rank"]) - 1
            abs_rank_pos[channel] = int(row["abs_mean_rank"]) - 1

    if np.any(~np.isfinite(energy)) or np.any(energy_rank_pos < 0):
        raise ValueError(f"Invalid or incomplete channel importance CSV: {path}")

    return {
        "source": np.asarray([str(path)], dtype=object),
        "energy": energy,
        "variance": variance,
        "abs_mean": abs_mean,
        "energy_rank_positions": energy_rank_pos,
        "variance_rank_positions": variance_rank_pos,
        "abs_mean_rank_positions": abs_rank_pos,
    }


def fallback_importance_from_sums(sum_z: np.ndarray, sum_z2: np.ndarray, sum_abs: np.ndarray, num_samples: int) -> Dict[str, np.ndarray]:
    flat_count = float(num_samples * LATENT_H * LATENT_W)
    energy = sum_z2.sum(axis=(1, 2)) / flat_count
    mean_map = sum_z / float(num_samples)
    second_moment_map = sum_z2 / float(num_samples)
    variance = np.maximum(second_moment_map - mean_map * mean_map, 0.0).mean(axis=(1, 2))
    abs_mean = sum_abs.sum(axis=(1, 2)) / flat_count

    energy_rank_indices = np.argsort(-energy, kind="mergesort").astype(np.int16)
    variance_rank_indices = np.argsort(-variance, kind="mergesort").astype(np.int16)
    abs_rank_indices = np.argsort(-abs_mean, kind="mergesort").astype(np.int16)
    return {
        "source": np.asarray(["recomputed from sensitivity sample"], dtype=object),
        "energy": energy,
        "variance": variance,
        "abs_mean": abs_mean,
        "energy_rank_positions": inverse_rank_positions(energy_rank_indices),
        "variance_rank_positions": inverse_rank_positions(variance_rank_indices),
        "abs_mean_rank_positions": inverse_rank_positions(abs_rank_indices),
    }


@torch.no_grad()
def collect_sensitivity_statistics(
    model,
    dataloader: DataLoader,
    device: torch.device,
    test_snr_db: float,
    shared_noise: bool,
) -> Dict[str, np.ndarray | float | int]:
    model.eval()
    noise_std = snr_db_to_noise_std(test_snr_db)

    sensitivity_sum = np.zeros(LATENT_C, dtype=np.float64)
    sensitivity_abs_sum = np.zeros(LATENT_C, dtype=np.float64)
    psnr_drop_sum = np.zeros(LATENT_C, dtype=np.float64)
    masked_mse_sum = np.zeros(LATENT_C, dtype=np.float64)
    masked_psnr_sum = np.zeros(LATENT_C, dtype=np.float64)
    negative_counts = np.zeros(LATENT_C, dtype=np.int64)
    per_image_batches: List[np.ndarray] = []

    sum_z = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    sum_z2 = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    sum_abs = np.zeros((LATENT_C, LATENT_H, LATENT_W), dtype=np.float64)
    base_mse_sum = 0.0
    base_psnr_sum = 0.0
    total_samples = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        if tuple(z.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
            raise ValueError(f"Expected latent shape (N,16,8,8), got {tuple(z.shape)}.")

        z_cpu = z.detach().cpu().float()
        sum_z += z_cpu.numpy().sum(axis=0, dtype=np.float64)
        z2_np = z_cpu.pow(2).numpy()
        sum_z2 += z2_np.sum(axis=0, dtype=np.float64)
        sum_abs += z_cpu.abs().numpy().sum(axis=0, dtype=np.float64)

        base_noise = torch.randn_like(z) * noise_std if shared_noise else None
        y_full = z + base_noise if shared_noise else z + torch.randn_like(z) * noise_std
        recon_full = model.decoder(y_full).clamp(0.0, 1.0)
        base_mse = batch_mse(recon_full, images)
        base_psnr = batch_psnr(recon_full, images)
        base_mse_sum += float(base_mse.sum().item())
        base_psnr_sum += float(base_psnr.sum().item())

        batch_delta = torch.empty((z.shape[0], LATENT_C), dtype=torch.float32)
        for channel in range(LATENT_C):
            z_masked = z.clone()
            z_masked[:, channel, :, :] = 0.0
            y_masked = z_masked + base_noise if shared_noise else z_masked + torch.randn_like(z_masked) * noise_std
            recon_masked = model.decoder(y_masked).clamp(0.0, 1.0)
            masked_mse = batch_mse(recon_masked, images)
            masked_psnr = batch_psnr(recon_masked, images)
            delta = masked_mse - base_mse

            delta_cpu = delta.detach().cpu()
            batch_delta[:, channel] = delta_cpu.float()
            sensitivity_sum[channel] += float(delta_cpu.sum().item())
            sensitivity_abs_sum[channel] += float(delta_cpu.abs().sum().item())
            psnr_drop_sum[channel] += float((base_psnr - masked_psnr).detach().cpu().sum().item())
            masked_mse_sum[channel] += float(masked_mse.detach().cpu().sum().item())
            masked_psnr_sum[channel] += float(masked_psnr.detach().cpu().sum().item())
            negative_counts[channel] += int((delta_cpu < 0).sum().item())

        per_image_batches.append(batch_delta.numpy())
        total_samples += int(z.shape[0])

    if total_samples == 0:
        raise ValueError("No samples were evaluated.")

    per_image_sensitivity = np.concatenate(per_image_batches, axis=0).astype(np.float32)
    return {
        "num_samples": total_samples,
        "base_avg_mse": base_mse_sum / total_samples,
        "base_avg_psnr": base_psnr_sum / total_samples,
        "global_channel_sensitivity": sensitivity_sum / total_samples,
        "global_channel_sensitivity_abs": sensitivity_abs_sum / total_samples,
        "global_channel_psnr_drop": psnr_drop_sum / total_samples,
        "masked_avg_mse": masked_mse_sum / total_samples,
        "masked_avg_psnr": masked_psnr_sum / total_samples,
        "negative_delta_ratio": negative_counts.astype(np.float64) / float(total_samples),
        "per_image_channel_sensitivity": per_image_sensitivity,
        "sum_z": sum_z,
        "sum_z2": sum_z2,
        "sum_abs": sum_abs,
    }


def write_sensitivity_table(
    path: Path,
    sensitivity: np.ndarray,
    sensitivity_abs: np.ndarray,
    psnr_drop: np.ndarray,
    masked_mse: np.ndarray,
    masked_psnr: np.ndarray,
    negative_ratio: np.ndarray,
    sensitivity_rank_indices: np.ndarray,
    importance: Dict[str, np.ndarray],
) -> None:
    sens_rank_positions = inverse_rank_positions(sensitivity_rank_indices)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "channel",
            "sensitivity_mse_increase",
            "sensitivity_rank",
            "sensitivity_abs_mse_change",
            "sensitivity_psnr_drop",
            "masked_avg_mse",
            "masked_avg_psnr",
            "negative_delta_ratio",
            "energy",
            "energy_rank",
            "variance",
            "variance_rank",
            "abs_mean",
            "abs_mean_rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for channel in range(LATENT_C):
            writer.writerow(
                {
                    "channel": channel,
                    "sensitivity_mse_increase": float(sensitivity[channel]),
                    "sensitivity_rank": int(sens_rank_positions[channel]) + 1,
                    "sensitivity_abs_mse_change": float(sensitivity_abs[channel]),
                    "sensitivity_psnr_drop": float(psnr_drop[channel]),
                    "masked_avg_mse": float(masked_mse[channel]),
                    "masked_avg_psnr": float(masked_psnr[channel]),
                    "negative_delta_ratio": float(negative_ratio[channel]),
                    "energy": float(importance["energy"][channel]),
                    "energy_rank": int(importance["energy_rank_positions"][channel]) + 1,
                    "variance": float(importance["variance"][channel]),
                    "variance_rank": int(importance["variance_rank_positions"][channel]) + 1,
                    "abs_mean": float(importance["abs_mean"][channel]),
                    "abs_mean_rank": int(importance["abs_mean_rank_positions"][channel]) + 1,
                }
            )


def write_per_image_spearman(path: Path, spearman: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "spearman_with_global_sensitivity_rank"])
        writer.writeheader()
        for idx, value in enumerate(spearman.tolist()):
            writer.writerow({"image_index": idx, "spearman_with_global_sensitivity_rank": float(value)})


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


def save_bar(values: np.ndarray, out_path: Path, ylabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    channels = np.arange(LATENT_C)
    ax.bar(channels, values, color=color)
    ax.set_xlabel("Channel index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(channels)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_sensitivity_rank_bar(sensitivity: np.ndarray, rank_indices: np.ndarray, out_path: Path) -> None:
    ranks = np.arange(1, LATENT_C + 1)
    sorted_values = sensitivity[rank_indices]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax.bar(ranks, sorted_values, color="#059669")
    ax.set_xlabel("Sensitivity rank")
    ax.set_ylabel("MSE increase")
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"{rank}\nch{channel}" for rank, channel in zip(ranks, rank_indices)], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_scatter(x_values: np.ndarray, y_values: np.ndarray, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=220)
    ax.scatter(x_values, y_values, s=44, color="#dc2626", alpha=0.85)
    for channel, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        ax.annotate(str(channel), (float(x_value), float(y_value)), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_spearman_hist(spearman: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    bins = np.linspace(-1.0, 1.0, 33)
    ax.hist(spearman, bins=bins, color="#0ea5e9", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Spearman correlation with global sensitivity rank")
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
    ax.set_ylabel("Overlap with global sensitivity top-M")
    ax.set_xticks(top_m)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_rank_comparison(
    sensitivity_rank_pos: np.ndarray,
    importance: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    channels = np.arange(LATENT_C)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    ax.plot(channels, sensitivity_rank_pos + 1, marker="o", linewidth=1.6, label="sensitivity")
    ax.plot(channels, importance["energy_rank_positions"] + 1, marker="s", linewidth=1.4, label="energy")
    ax.plot(channels, importance["variance_rank_positions"] + 1, marker="^", linewidth=1.4, label="variance")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Rank (1 is highest)")
    ax.set_xticks(channels)
    ax.set_yticks(np.arange(1, LATENT_C + 1))
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
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
    output_dir: Path,
    test_snr_db: float,
    shared_noise: bool,
    num_samples: int,
    base_avg_mse: float,
    base_avg_psnr: float,
    sensitivity: np.ndarray,
    sensitivity_abs: np.ndarray,
    psnr_drop: np.ndarray,
    masked_mse: np.ndarray,
    masked_psnr: np.ndarray,
    negative_ratio: np.ndarray,
    sensitivity_rank_indices: np.ndarray,
    importance: Dict[str, np.ndarray],
    rank_correlations: Dict[str, float],
    per_image_stats: Dict[str, float],
    pairwise_stats: Dict[str, float],
    topm_global_rows: Sequence[Dict[str, float]],
    topm_pairwise_rows: Sequence[Dict[str, float]],
) -> None:
    sens_stats = summarize(sensitivity)
    sens_rank_pos = inverse_rank_positions(sensitivity_rank_indices)
    sensitivity_cv = float(np.std(sensitivity) / max(abs(float(np.mean(sensitivity))), 1e-12))
    energy_cv = float(np.std(importance["energy"]) / max(abs(float(np.mean(importance["energy"]))), 1e-12))
    top_sens = int(sensitivity_rank_indices[0])
    top_energy = int(np.argmin(importance["energy_rank_positions"]))
    same_top = top_sens == top_energy
    global_more_stable = per_image_stats["mean"] > pairwise_stats["mean"]
    recommend = "yes" if (not same_top or rank_correlations["sensitivity_vs_energy"] < 0.9) else "yes, as a confirmation ablation"

    lines = [
        "Innovation experiment: channel-level latent sensitivity",
        "=" * 62,
        f"Dataset path: {data_dir}",
        f"Dataset split/source: {dataset_note}",
        f"Checkpoint path: {checkpoint_path}",
        f"Model: {model_name}",
        "PyTorch latent shape (C,H,W): 16 x 8 x 8",
        "Report latent shape (H,W,C): 8 x 8 x 16",
        "Task7 flatten order: HWC (CHW -> HWC -> flatten); channel masking does not require flattening",
        "Power normalization: yes, jscc_lab.channel.power_normalize before masking",
        f"test_snr_db: {test_snr_db:g}",
        f"shared_noise: {shared_noise}",
        f"Samples analyzed: {num_samples}",
        f"Output directory: {output_dir}",
        f"Channel importance source: {importance['source'][0]}",
        "",
        f"base_avg_mse: {base_avg_mse:.8f}",
        f"base_avg_psnr: {base_avg_psnr:.6f}",
        "",
        "Global channel sensitivity statistics:",
        format_stats("global_channel_sensitivity", sens_stats),
        "",
        f"sensitivity rank channels: {sensitivity_rank_indices.tolist()}",
        "",
        "Rank correlation between channel metrics:",
        f"sensitivity vs energy Spearman: {rank_correlations['sensitivity_vs_energy']:.6f}",
        f"sensitivity vs variance Spearman: {rank_correlations['sensitivity_vs_variance']:.6f}",
        f"sensitivity vs abs_mean Spearman: {rank_correlations['sensitivity_vs_abs_mean']:.6f}",
        f"energy vs variance Spearman: {rank_correlations['energy_vs_variance']:.6f}",
        "",
        "Per-channel sensitivity values in sensitivity-rank order:",
    ]
    for rank, channel in enumerate(sensitivity_rank_indices.tolist(), start=1):
        lines.append(
            f"rank={rank:02d} channel={channel:02d} sensitivity_mse_increase={sensitivity[channel]:.8f} "
            f"sensitivity_psnr_drop={psnr_drop[channel]:.6f} masked_avg_mse={masked_mse[channel]:.8f} "
            f"masked_avg_psnr={masked_psnr[channel]:.6f} negative_delta_ratio={negative_ratio[channel]:.6f} "
            f"energy={importance['energy'][channel]:.8f} energy_rank={int(importance['energy_rank_positions'][channel]) + 1} "
            f"variance={importance['variance'][channel]:.8f} variance_rank={int(importance['variance_rank_positions'][channel]) + 1}"
        )

    lines.extend(
        [
            "",
            "Spearman stability statistics:",
            format_stats("per-image sensitivity rank vs global sensitivity rank", per_image_stats),
            format_stats("random image-pair sensitivity rank", pairwise_stats),
            "",
            "Top-M sensitivity overlap with global:",
        ]
    )
    for row in topm_global_rows:
        lines.append(
            f"M={int(row['M'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(["", "Random image-pair Top-M sensitivity overlap:"])
    for row in topm_pairwise_rows:
        lines.append(
            f"M={int(row['M'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(
        [
            "",
            "Short conclusion hints:",
            f"敏感性最高的 channel 是否与能量最高的 channel 一致: {'yes' if same_top else 'no'} "
            f"(top_sensitivity={top_sens}, top_energy={top_energy})",
            f"channel sensitivity 是否比 channel energy 更分散或更集中: "
            f"{'more dispersed' if sensitivity_cv > energy_cv else 'more concentrated'} "
            f"(sensitivity_cv={sensitivity_cv:.6f}, energy_cv={energy_cv:.6f})",
            "channel 级全局 sensitivity 排序是否比随机图片对更稳定: "
            f"{'yes' if global_more_stable else 'no'} "
            f"(per-image-vs-global mean={per_image_stats['mean']:.6f}, pairwise mean={pairwise_stats['mean']:.6f})",
            f"是否建议后续做 channel-sensitivity top-Kp 率失真测试: {recommend}",
            "",
            "Sanity check ranks by channel:",
        ]
    )
    for channel in range(LATENT_C):
        lines.append(
            f"channel={channel:02d} sensitivity_rank={int(sens_rank_pos[channel]) + 1} "
            f"energy_rank={int(importance['energy_rank_positions'][channel]) + 1} "
            f"variance_rank={int(importance['variance_rank_positions'][channel]) + 1} "
            f"abs_mean_rank={int(importance['abs_mean_rank_positions'][channel]) + 1}"
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

    channel_importance_csv = resolve_relative_to_project(args.channel_importance_csv, project_dir).resolve()
    output_dir = ensure_dir(resolve_relative_to_project(args.output_dir, project_dir))
    device = get_device(args.device)

    train_set, dataset_note = load_project_train_split_dataset(
        data_dir,
        max_samples=args.max_samples,
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

    model = load_jscc_checkpoint(checkpoint_path, device)
    try:
        stats = collect_sensitivity_statistics(model, dataloader, device, args.test_snr_db, args.shared_noise)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower():
            raise RuntimeError(
                "Sensitivity evaluation ran out of memory. Try reducing --batch_size, for example --batch_size 32."
            ) from exc
        raise

    num_samples = int(stats["num_samples"])
    sensitivity = stats["global_channel_sensitivity"]
    sensitivity_abs = stats["global_channel_sensitivity_abs"]
    psnr_drop = stats["global_channel_psnr_drop"]
    masked_mse = stats["masked_avg_mse"]
    masked_psnr = stats["masked_avg_psnr"]
    negative_ratio = stats["negative_delta_ratio"]
    per_image_sensitivity = stats["per_image_channel_sensitivity"]

    importance = load_channel_importance_table(channel_importance_csv)
    recomputed_importance = False
    if importance is None:
        importance = fallback_importance_from_sums(stats["sum_z"], stats["sum_z2"], stats["sum_abs"], num_samples)
        recomputed_importance = True

    sensitivity_rank_indices = np.argsort(-sensitivity, kind="mergesort").astype(np.int16)
    sensitivity_rank_positions = inverse_rank_positions(sensitivity_rank_indices)

    rank_correlations = {
        "sensitivity_vs_energy": float(
            spearman_from_rank_positions(sensitivity_rank_positions, importance["energy_rank_positions"])
        ),
        "sensitivity_vs_variance": float(
            spearman_from_rank_positions(sensitivity_rank_positions, importance["variance_rank_positions"])
        ),
        "sensitivity_vs_abs_mean": float(
            spearman_from_rank_positions(sensitivity_rank_positions, importance["abs_mean_rank_positions"])
        ),
        "energy_vs_variance": float(
            spearman_from_rank_positions(importance["energy_rank_positions"], importance["variance_rank_positions"])
        ),
    }

    image_rank_indices = np.argsort(-per_image_sensitivity, axis=1, kind="mergesort").astype(np.int16)
    image_rank_positions = inverse_rank_positions(image_rank_indices)
    per_image_spearman = spearman_from_rank_positions(image_rank_positions, sensitivity_rank_positions[None, :])
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
        global_topm_mask = sensitivity_rank_positions < top_m
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

    np.save(output_dir / "global_channel_sensitivity.npy", sensitivity)
    np.save(output_dir / "global_channel_sensitivity_abs.npy", sensitivity_abs)
    np.save(output_dir / "global_channel_sensitivity_psnr_drop.npy", psnr_drop)
    np.save(output_dir / "channel_sensitivity_rank_indices.npy", sensitivity_rank_indices)
    np.save(output_dir / "per_image_channel_sensitivity.npy", per_image_sensitivity)
    np.save(output_dir / "masked_avg_mse.npy", masked_mse)
    np.save(output_dir / "masked_avg_psnr.npy", masked_psnr)
    np.save(output_dir / "negative_delta_ratio.npy", negative_ratio)
    if recomputed_importance:
        np.save(output_dir / "global_channel_energy.npy", importance["energy"])
        np.save(output_dir / "global_channel_variance.npy", importance["variance"])
        np.save(output_dir / "global_channel_abs_mean.npy", importance["abs_mean"])

    write_sensitivity_table(
        output_dir / "channel_sensitivity_table.csv",
        sensitivity,
        sensitivity_abs,
        psnr_drop,
        masked_mse,
        masked_psnr,
        negative_ratio,
        sensitivity_rank_indices,
        importance,
    )
    write_per_image_spearman(output_dir / "per_image_channel_sensitivity_spearman.csv", per_image_spearman)
    write_pairwise_spearman(
        output_dir / "pairwise_channel_sensitivity_spearman_sample.csv",
        pairs,
        pairwise_spearman,
    )
    write_summary_rows(
        output_dir / "topm_channel_sensitivity_overlap_with_global.csv",
        topm_global_rows,
        ["M", "mean", "std", "min", "max", "median"],
    )
    write_pairwise_topm_sample(
        output_dir / "topm_channel_sensitivity_overlap_pairwise_sample.csv",
        pairs,
        pairwise_overlap_by_m,
    )
    write_summary_rows(
        output_dir / "topm_channel_sensitivity_overlap_pairwise_summary.csv",
        topm_pairwise_rows,
        ["M", "mean", "std", "min", "max", "median"],
    )

    save_bar(sensitivity, output_dir / "channel_sensitivity_bar.png", "MSE increase after channel masking", "#2563eb")
    save_sensitivity_rank_bar(sensitivity, sensitivity_rank_indices, output_dir / "channel_sensitivity_rank_bar.png")
    save_bar(psnr_drop, output_dir / "channel_sensitivity_psnr_drop_bar.png", "Average PSNR drop", "#7c3aed")
    save_scatter(
        importance["energy"],
        sensitivity,
        output_dir / "sensitivity_vs_energy_scatter.png",
        "Channel energy",
        "Channel sensitivity (MSE increase)",
    )
    save_scatter(
        importance["variance"],
        sensitivity,
        output_dir / "sensitivity_vs_variance_scatter.png",
        "Channel variance",
        "Channel sensitivity (MSE increase)",
    )
    save_spearman_hist(per_image_spearman, output_dir / "per_image_channel_sensitivity_spearman_hist.png")
    save_topm_overlap_curve(topm_global_rows, output_dir / "topm_channel_sensitivity_overlap_curve.png")
    save_rank_comparison(sensitivity_rank_positions, importance, output_dir / "channel_rank_comparison.png")

    write_summary(
        output_dir / "channel_sensitivity_summary.txt",
        data_dir=data_dir,
        dataset_note=dataset_note,
        checkpoint_path=checkpoint_path,
        model_name=type(model).__name__,
        output_dir=output_dir,
        test_snr_db=args.test_snr_db,
        shared_noise=bool(args.shared_noise),
        num_samples=num_samples,
        base_avg_mse=float(stats["base_avg_mse"]),
        base_avg_psnr=float(stats["base_avg_psnr"]),
        sensitivity=sensitivity,
        sensitivity_abs=sensitivity_abs,
        psnr_drop=psnr_drop,
        masked_mse=masked_mse,
        masked_psnr=masked_psnr,
        negative_ratio=negative_ratio,
        sensitivity_rank_indices=sensitivity_rank_indices,
        importance=importance,
        rank_correlations=rank_correlations,
        per_image_stats=per_image_stats,
        pairwise_stats=pairwise_stats,
        topm_global_rows=topm_global_rows,
        topm_pairwise_rows=topm_pairwise_rows,
    )

    print(f"Analyzed {num_samples} train-split samples on {device}.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base avg MSE={float(stats['base_avg_mse']):.8f}, PSNR={float(stats['base_avg_psnr']):.4f} dB")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()

# Example full run:
# python scripts/innovation_channel_sensitivity.py \
#     --data_dir /home/lc/class/yuyi/cifar-10 \
#     --project_dir /home/lc/class/yuyi/semantic_jscc_lab \
#     --output_dir outputs/innovation/channel_sensitivity \
#     --batch_size 128 \
#     --test_snr_db 7 \
#     --max_samples 10000 \
#     --pair_samples 5000
