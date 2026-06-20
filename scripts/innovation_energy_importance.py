#!/usr/bin/env python
"""Innovation experiment 1: latent energy importance for the SNR=7 Deep JSCC model.

By default this script rebuilds the same train split used by train_jscc.py:
  load_cifar_array_dataset(data_dir) -> make_splits(train=0.8, val=0.1, test=0.1, seed=42)

The task (7) baseline uses HWC flattening:
  PyTorch latent (N, 16, 8, 8) -> (N, 8, 8, 16) -> flatten length 1024.

Example:
  python scripts/innovation_energy_importance.py \
      --data_dir /home/lc/class/yuyi/cifar-10 \
      --project_dir /home/lc/class/yuyi/semantic_jscc_lab \
      --output_dir outputs/innovation/energy_importance \
      --batch_size 256 \
      --pair_samples 5000
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
from torch.utils.data import DataLoader, Subset

from eval_rate_distortion import LATENT_C, LATENT_H, LATENT_K, LATENT_W, flatten_latent_hwc, load_jscc_checkpoint
from jscc_lab.channel import power_normalize
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.utils import ensure_dir, get_device, seed_everything


KP_VALUES = [128, 256, 384, 512, 640, 768, 896, 1024]
SPEARMAN_DENOMINATOR = float(LATENT_K * (LATENT_K * LATENT_K - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute global latent energy importance and cross-image ranking correlations."
    )
    parser.add_argument("--data_dir", default="/home/lc/class/yuyi/cifar-10", help="CIFAR-10 directory or data file.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--checkpoint", default=None, help="Optional SNR=7 checkpoint path. If omitted, auto-search.")
    parser.add_argument("--output_dir", default="outputs/innovation/energy_importance")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Device string; default auto uses CUDA if available, else CPU.")
    parser.add_argument("--pair_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for rebuilding the train/val/test split.")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for debugging.")
    parser.add_argument(
        "--flatten_order",
        default="task7",
        choices=["task7", "hwc", "chw"],
        help="Default 'task7' reuses task7 HWC flattening. CHW is provided only for explicit ablations.",
    )
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

    found = sorted(
        path
        for path in (project_dir / "outputs").rglob("*jscc_snr7.pt")
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
    max_train_samples: int | None = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    split_seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, str]:
    """Rebuild the exact train split convention used by train_jscc.py."""

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
    source_note = (
        "project train split rebuilt with jscc_lab.data.make_splits; "
        f"loaded_samples={len(dataset)}, train_split={train_split}, val_split={val_split}, "
        f"test_split={test_split}, split_seed={split_seed}; validation/test splits excluded"
    )
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("--max_train_samples must be positive when provided.")
        if max_train_samples < len(train_set):
            train_set = Subset(train_set, list(range(max_train_samples)))
            source_note += f"; first {max_train_samples} samples"
    return train_set, source_note


def flatten_latent(z: torch.Tensor, flatten_order: str) -> torch.Tensor:
    """Flatten normalized latent codes in task7-compatible HWC order unless explicitly overridden."""

    if flatten_order == "task7":
        flatten_order = "hwc"
    if flatten_order == "hwc":
        flat, _ = flatten_latent_hwc(z)
        return flat
    if flatten_order == "chw":
        if z.ndim != 4 or tuple(z.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
            raise ValueError(f"Expected latent shape (N,16,8,8), got {tuple(z.shape)}.")
        return z.contiguous().reshape(z.shape[0], LATENT_K)
    raise ValueError(f"Unsupported flatten order: {flatten_order}")


def canonical_flatten_order(flatten_order: str) -> str:
    return "hwc" if flatten_order == "task7" else flatten_order


def flat_index_to_coords(flat_index: int, flatten_order: str) -> Tuple[int, int, int]:
    """Return (h, w, c) for a flat latent index."""

    if flatten_order == "hwc":
        h = flat_index // (LATENT_W * LATENT_C)
        rem = flat_index % (LATENT_W * LATENT_C)
        w = rem // LATENT_C
        c = rem % LATENT_C
        return int(h), int(w), int(c)

    c = flat_index // (LATENT_H * LATENT_W)
    rem = flat_index % (LATENT_H * LATENT_W)
    h = rem // LATENT_W
    w = rem % LATENT_W
    return int(h), int(w), int(c)


def flat_to_hwc(values: np.ndarray, flatten_order: str) -> np.ndarray:
    if flatten_order == "hwc":
        return values.reshape(LATENT_H, LATENT_W, LATENT_C)
    return values.reshape(LATENT_C, LATENT_H, LATENT_W).transpose(1, 2, 0)


@torch.no_grad()
def collect_normalized_energy(
    model,
    dataloader: DataLoader,
    device: torch.device,
    flatten_order: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode all images, apply task7 power normalization, and collect c_i^2."""

    model.eval()
    energy_batches: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for images, batch_labels in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        flat = flatten_latent(z, flatten_order)
        energy_batches.append(flat.pow(2).detach().cpu().numpy().astype(np.float32, copy=False))
        labels.append(batch_labels.detach().cpu().numpy().astype(np.int64, copy=False))

    if not energy_batches:
        raise ValueError("No samples were loaded from the training set.")
    return np.concatenate(energy_batches, axis=0), np.concatenate(labels, axis=0)


def inverse_rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Convert argsort output to per-flat-index rank positions, where 0 is most important."""

    positions = np.empty_like(sorted_indices, dtype=np.int16)
    ranks = np.arange(sorted_indices.shape[1], dtype=np.int16)
    rows = np.arange(sorted_indices.shape[0])[:, None]
    positions[rows, sorted_indices] = ranks
    return positions


def inverse_single_rank(sorted_indices: np.ndarray) -> np.ndarray:
    positions = np.empty_like(sorted_indices, dtype=np.int16)
    positions[sorted_indices] = np.arange(sorted_indices.shape[0], dtype=np.int16)
    return positions


def spearman_from_rank_positions(rank_a: np.ndarray, rank_b: np.ndarray) -> np.ndarray:
    """Spearman correlation for complete rank vectors with deterministic argsort tie-breaking."""

    diff = rank_a.astype(np.int32) - rank_b.astype(np.int32)
    sum_sq = np.sum(diff * diff, axis=-1, dtype=np.float64)
    return 1.0 - (6.0 * sum_sq / SPEARMAN_DENOMINATOR)


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


def write_energy_rank_table(
    path: Path,
    global_energy: np.ndarray,
    energy_rank_indices: np.ndarray,
    flatten_order: str,
) -> None:
    total_energy = float(np.sum(global_energy))
    sorted_energy = global_energy[energy_rank_indices]
    cumulative = np.cumsum(sorted_energy, dtype=np.float64)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "flat_index",
            "h",
            "w",
            "channel",
            "energy",
            "normalized_energy",
            "cumulative_energy_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, flat_index in enumerate(energy_rank_indices.tolist(), start=1):
            h, w, channel = flat_index_to_coords(flat_index, flatten_order)
            energy = float(global_energy[flat_index])
            writer.writerow(
                {
                    "rank": rank,
                    "flat_index": int(flat_index),
                    "h": h,
                    "w": w,
                    "channel": channel,
                    "energy": energy,
                    "normalized_energy": energy / total_energy if total_energy > 0 else 0.0,
                    "cumulative_energy_ratio": float(cumulative[rank - 1] / total_energy) if total_energy > 0 else 0.0,
                }
            )


def write_per_image_spearman(path: Path, spearman: np.ndarray, labels: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "label", "spearman_with_global"])
        writer.writeheader()
        for idx, value in enumerate(spearman.tolist()):
            writer.writerow({"image_index": idx, "label": int(labels[idx]), "spearman_with_global": float(value)})


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


def write_pairwise_topk_sample(path: Path, pairs: np.ndarray, overlap_by_k: Dict[int, np.ndarray]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_index_a", "image_index_b", "K", "overlap"])
        writer.writeheader()
        for pair_id, (idx_a, idx_b) in enumerate(pairs.tolist()):
            for k, overlaps in overlap_by_k.items():
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "image_index_a": int(idx_a),
                        "image_index_b": int(idx_b),
                        "K": int(k),
                        "overlap": float(overlaps[pair_id]),
                    }
                )


def save_global_energy_curve(sorted_energy: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(np.arange(1, len(sorted_energy) + 1), sorted_energy, linewidth=1.6)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Global energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_cumulative_energy_curve(sorted_energy: np.ndarray, out_path: Path) -> None:
    total = float(np.sum(sorted_energy))
    cumulative = np.cumsum(sorted_energy, dtype=np.float64) / total if total > 0 else np.zeros_like(sorted_energy)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(np.arange(1, len(sorted_energy) + 1), cumulative, linewidth=1.6)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative energy ratio")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_global_energy_heatmap(global_energy: np.ndarray, flatten_order: str, out_path: Path) -> None:
    energy_hwc = flat_to_hwc(global_energy, flatten_order)
    vmax = float(np.max(energy_hwc))
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 7.4), dpi=220)
    images = []
    for channel, ax in enumerate(axes.flat):
        image = ax.imshow(energy_hwc[:, :, channel], cmap="viridis", vmin=0.0, vmax=vmax)
        images.append(image)
        ax.set_title(f"ch {channel}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.84)
    cbar.set_label("Global energy")
    fig.suptitle("Global latent energy by channel", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_spearman_hist(spearman: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.hist(spearman, bins=50, color="#3b82f6", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Spearman correlation with global ranking")
    ax.set_ylabel("Number of images")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_topk_overlap_curve(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    ks = np.asarray([row["K"] for row in rows], dtype=np.int32)
    means = np.asarray([row["mean"] for row in rows], dtype=np.float64)
    stds = np.asarray([row["std"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(ks, means, marker="o", linewidth=1.8)
    ax.fill_between(ks, np.maximum(0.0, means - stds), np.minimum(1.0, means + stds), alpha=0.18)
    ax.set_xlabel("Kp")
    ax.set_ylabel("Overlap with global top-K")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def format_stats(prefix: str, stats: Dict[str, float]) -> str:
    return (
        f"{prefix}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, "
        f"max={stats['max']:.6f}, median={stats['median']:.6f}"
    )


def write_text_summary(
    path: Path,
    *,
    data_dir: Path,
    dataset_note: str,
    checkpoint_path: Path,
    model_name: str,
    latent_shape: Tuple[int, int, int],
    flatten_order: str,
    num_samples: int,
    global_energy: np.ndarray,
    energy_rank_indices: np.ndarray,
    per_image_spearman_stats: Dict[str, float],
    pairwise_spearman_stats: Dict[str, float],
    topk_global_rows: Sequence[Dict[str, float]],
    topk_pairwise_rows: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    global_stats = {
        "mean": float(np.mean(global_energy)),
        "std": float(np.std(global_energy)),
        "min": float(np.min(global_energy)),
        "max": float(np.max(global_energy)),
    }
    lines = [
        "Innovation experiment 1: latent energy importance",
        "=" * 56,
        f"Dataset path: {data_dir}",
        f"Dataset split/source: {dataset_note}",
        f"Checkpoint path: {checkpoint_path}",
        f"Model: {model_name}",
        f"Latent shape (PyTorch C,H,W): {latent_shape}",
        "Report latent shape (H,W,C): 8 x 8 x 16",
        f"Flatten order: {flatten_order.upper()} (task7-compatible CHW -> HWC -> flatten)" if flatten_order == "hwc" else f"Flatten order: {flatten_order.upper()}",
        "Power normalization: yes, jscc_lab.channel.power_normalize before energy computation",
        f"Training samples analyzed: {num_samples}",
        f"Output directory: {output_dir}",
        "",
        "Global energy statistics:",
        f"mean={global_stats['mean']:.8f}, std={global_stats['std']:.8f}, min={global_stats['min']:.8f}, max={global_stats['max']:.8f}",
        "",
        "Spearman correlation statistics:",
        format_stats("per-image energy rank vs global energy rank", per_image_spearman_stats),
        format_stats("random image-pair energy rank", pairwise_spearman_stats),
        "",
        "Top-K overlap with global top-K:",
    ]
    for row in topk_global_rows:
        lines.append(
            f"K={int(row['K'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(["", "Random image-pair Top-K overlap:"])
    for row in topk_pairwise_rows:
        lines.append(
            f"K={int(row['K'])}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, "
            f"max={row['max']:.6f}, median={row['median']:.6f}"
        )

    lines.extend(["", "Top 20 latent positions by global energy:"])
    for rank, flat_index in enumerate(energy_rank_indices[:20].tolist(), start=1):
        h, w, channel = flat_index_to_coords(flat_index, flatten_order)
        lines.append(
            f"{rank:02d}. flat_index={int(flat_index)}, h={h}, w={w}, channel={channel}, "
            f"energy={float(global_energy[flat_index]):.8f}"
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
    flatten_order = canonical_flatten_order(args.flatten_order)

    dataset, dataset_note = load_project_train_split_dataset(
        data_dir,
        max_train_samples=args.max_train_samples,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        split_seed=args.split_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_jscc_checkpoint(checkpoint_path, device)
    energy, labels = collect_normalized_energy(model, dataloader, device, flatten_order)
    num_samples = int(energy.shape[0])
    if energy.shape[1] != LATENT_K:
        raise ValueError(f"Expected flattened latent length {LATENT_K}, got {energy.shape[1]}.")

    global_energy = energy.mean(axis=0, dtype=np.float64).astype(np.float32)
    energy_rank_indices = np.argsort(-global_energy, kind="mergesort").astype(np.int32)
    sorted_energy = global_energy[energy_rank_indices]

    np.save(output_dir / "global_energy.npy", global_energy)
    np.save(output_dir / "energy_rank_indices.npy", energy_rank_indices)
    write_energy_rank_table(output_dir / "energy_rank_table.csv", global_energy, energy_rank_indices, flatten_order)

    image_rank_indices = np.argsort(-energy, axis=1, kind="mergesort").astype(np.int16)
    image_rank_positions = inverse_rank_positions(image_rank_indices)
    global_rank_positions = inverse_single_rank(energy_rank_indices).astype(np.int16)
    del energy, image_rank_indices

    per_image_spearman = spearman_from_rank_positions(image_rank_positions, global_rank_positions[None, :])
    per_image_spearman_stats = summarize(per_image_spearman)
    write_per_image_spearman(output_dir / "per_image_spearman.csv", per_image_spearman, labels)

    pairs = sample_pairs(num_samples, args.pair_samples, args.seed)
    pairwise_spearman = spearman_from_rank_positions(image_rank_positions[pairs[:, 0]], image_rank_positions[pairs[:, 1]])
    pairwise_spearman_stats = summarize(pairwise_spearman) if len(pairwise_spearman) else summarize(np.array([np.nan]))
    write_pairwise_spearman(output_dir / "pairwise_spearman_sample.csv", pairs, pairwise_spearman)

    topk_global_rows: List[Dict[str, float]] = []
    topk_pairwise_rows: List[Dict[str, float]] = []
    pairwise_overlap_by_k: Dict[int, np.ndarray] = {}
    for k in KP_VALUES:
        image_topk_mask = image_rank_positions < k
        global_topk_mask = global_rank_positions < k
        overlap_global = np.sum(image_topk_mask & global_topk_mask[None, :], axis=1, dtype=np.float64) / float(k)
        stats = summarize(overlap_global)
        topk_global_rows.append({"K": int(k), **stats})

        overlap_pairwise = (
            np.sum(image_topk_mask[pairs[:, 0]] & image_topk_mask[pairs[:, 1]], axis=1, dtype=np.float64) / float(k)
            if len(pairs)
            else np.asarray([], dtype=np.float64)
        )
        pairwise_overlap_by_k[int(k)] = overlap_pairwise
        pair_stats = summarize(overlap_pairwise) if len(overlap_pairwise) else summarize(np.array([np.nan]))
        topk_pairwise_rows.append({"K": int(k), **pair_stats})

    write_summary_rows(
        output_dir / "topk_overlap_with_global.csv",
        topk_global_rows,
        ["K", "mean", "std", "min", "max", "median"],
    )
    write_pairwise_topk_sample(output_dir / "topk_overlap_pairwise_sample.csv", pairs, pairwise_overlap_by_k)
    write_summary_rows(
        output_dir / "topk_overlap_pairwise_summary.csv",
        topk_pairwise_rows,
        ["K", "mean", "std", "min", "max", "median"],
    )

    save_global_energy_curve(sorted_energy, output_dir / "global_energy_curve.png")
    save_cumulative_energy_curve(sorted_energy, output_dir / "cumulative_energy_curve.png")
    save_global_energy_heatmap(global_energy, flatten_order, output_dir / "global_energy_heatmap.png")
    save_spearman_hist(per_image_spearman, output_dir / "spearman_hist.png")
    save_topk_overlap_curve(topk_global_rows, output_dir / "topk_overlap_curve.png")

    write_text_summary(
        output_dir / "energy_importance_summary.txt",
        data_dir=data_dir,
        dataset_note=dataset_note,
        checkpoint_path=checkpoint_path,
        model_name=type(model).__name__,
        latent_shape=(LATENT_C, LATENT_H, LATENT_W),
        flatten_order=flatten_order,
        num_samples=num_samples,
        global_energy=global_energy,
        energy_rank_indices=energy_rank_indices,
        per_image_spearman_stats=per_image_spearman_stats,
        pairwise_spearman_stats=pairwise_spearman_stats,
        topk_global_rows=topk_global_rows,
        topk_pairwise_rows=topk_pairwise_rows,
        output_dir=output_dir,
    )

    print(f"Analyzed {num_samples} CIFAR-10 training images on {device}.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Flatten order: {flatten_order.upper()}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
