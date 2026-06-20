#!/usr/bin/env python
"""Task (7) variant: keep latent spatial blocks by block-energy rank.

This mirrors scripts/eval_rate_distortion.py, but replaces task7's HWC prefix-Kp
mask with a block-level mask built from
outputs/innovation/block_energy_importance/b{b}/block_energy_rank_indices_b{b}.npy.

For block_size=b, each retained block contributes b*b Kp elements. Blocks are
defined inside one latent channel in PyTorch CHW space, while each block table
also records task7-compatible HWC flat indices.

Example:
  python scripts/innovation_block_ranked_rate_distortion.py \
      --data_path /home/lc/class/yuyi/cifar-10 \
      --project_dir /home/lc/class/yuyi/semantic_jscc_lab \
      --block_sizes 2,4 \
      --block_rank_root outputs/innovation/block_energy_importance \
      --out_dir outputs/innovation/block_energy_ranked_task7 \
      --batch_size 256 \
      --num_images 500
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
from torch.utils.data import DataLoader, TensorDataset

from eval_rate_distortion import INPUT_K, LATENT_C, LATENT_H, LATENT_K, LATENT_W, load_jscc_checkpoint, parse_kp_list
from innovation_block_energy_importance import build_blocks, make_block_mask_from_rank, parse_block_sizes
from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices
from jscc_lab.channel import awgn, power_normalize
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate task7 rate-distortion with block-energy-ranked latent retention."
    )
    parser.add_argument("--data_path", default="/home/lc/class/yuyi/cifar-10", help="Path to CIFAR data.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--ckpt", default="outputs/jscc/snr_7/best_jscc_snr7.pt", help="Path to the SNR=7 checkpoint.")
    parser.add_argument(
        "--block_rank_root",
        default="outputs/innovation/block_energy_importance",
        help="Root directory produced by innovation_block_energy_importance.py.",
    )
    parser.add_argument("--out_dir", default="outputs/innovation/block_energy_ranked_task7")
    parser.add_argument("--block_sizes", default="2,4", help="Comma-separated block sizes, e.g. 1,2,4,8.")
    parser.add_argument("--kp_list", default="128,256,384,512,640,768,896,1024")
    parser.add_argument("--test_snr_db", type=float, default=7.0)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def resolve_path(path_text: str | Path, project_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def block_rank_path(block_rank_root: Path, block_size: int) -> Path:
    return block_rank_root / f"b{block_size}" / f"block_energy_rank_indices_b{block_size}.npy"


def load_block_rank_indices(path: Path, num_blocks: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            "Block rank file not found. Run scripts/innovation_block_energy_importance.py first, "
            f"or pass --block_rank_root explicitly. Missing: {path}"
        )
    rank_indices = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if rank_indices.shape[0] != num_blocks:
        raise ValueError(f"Expected {num_blocks} block rank indices, got {rank_indices.shape[0]} from {path}.")
    if not np.array_equal(np.sort(rank_indices), np.arange(num_blocks, dtype=np.int64)):
        raise ValueError(f"{path} is not a valid permutation of block ids [0, {num_blocks - 1}].")
    return rank_indices


def validate_block_kp_values(kp_values: Sequence[int], block_size: int, num_blocks: int) -> None:
    elements_per_block = block_size * block_size
    for kp in kp_values:
        if kp < 0 or kp > LATENT_K:
            raise ValueError(f"Kp must be in [0, {LATENT_K}], got {kp}.")
        if kp % elements_per_block != 0:
            raise ValueError(
                f"Kp={kp} is not divisible by elements_per_block={elements_per_block} for block_size={block_size}."
            )
        keep_blocks = kp // elements_per_block
        if keep_blocks > num_blocks:
            raise ValueError(f"Kp={kp} keeps {keep_blocks} blocks, but block_size={block_size} has only {num_blocks}.")


def build_mask_tensors(
    rank_indices: np.ndarray,
    blocks: Sequence[Dict[str, object]],
    kp_values: Sequence[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    masks: Dict[int, torch.Tensor] = {}
    for kp in kp_values:
        mask_np = make_block_mask_from_rank(rank_indices, blocks, kp, LATENT_C, LATENT_H, LATENT_W)
        masks[int(kp)] = torch.from_numpy(mask_np).to(device=device).unsqueeze(0)
    return masks


@torch.no_grad()
def evaluate_kp_block_ranked(
    model: DeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    kp: int,
    test_snr_db: float,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate one Kp using normalized latent, block-energy top-block mask, AWGN, and decoder."""

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        z_masked = z * mask
        z_noisy = awgn(z_masked, test_snr_db, training=True)
        recon = model.decoder(z_noisy).clamp(0.0, 1.0)

        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += int(images.shape[0])

    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {
        "Kp": int(kp),
        "R": float(kp) / INPUT_K,
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
    }


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Kp", "R", "avg_mse", "avg_psnr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Kp": row["Kp"],
                    "R": row["R"],
                    "avg_mse": row["avg_mse"],
                    "avg_psnr": row["avg_psnr"],
                }
            )


def save_rate_distortion_curve(rows: List[Dict[str, float]], out_path: Path, block_size: int) -> None:
    rates = [row["R"] for row in rows]
    psnrs = [row["avg_psnr"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(rates, psnrs, marker="o", linewidth=1.8)
    ax.set_xlabel("Approximate rate R = Kp / 3072")
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_title(f"Block-energy ranked task7, block_size={block_size}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_comparison_curve(rows: List[Dict[str, float]], out_path: Path) -> None:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["block_size"]), []).append(row)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=220)
    for block_size in sorted(grouped):
        items = sorted(grouped[block_size], key=lambda item: int(item["Kp"]))
        ax.plot(
            [item["R"] for item in items],
            [item["avg_psnr"] for item in items],
            marker="o",
            linewidth=1.7,
            label=f"b={block_size}",
        )
    ax.set_xlabel("Approximate rate R = Kp / 3072")
    ax.set_ylabel("Average PSNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_comparison_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["block_size", "elements_per_block", "num_keep_blocks", "Kp", "R", "avg_mse", "avg_psnr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_summary(
    out_path: Path,
    rows: List[Dict[str, float]],
    num_images: int,
    test_snr_db: float,
    ckpt_path: Path,
    rank_path: Path,
    rank_indices: np.ndarray,
    block_size: int,
) -> None:
    elements_per_block = block_size * block_size
    lines = [
        f"Task (7) block-energy-ranked rate-distortion summary, block_size={block_size}",
        "=" * 78,
        f"Checkpoint: {ckpt_path}",
        f"Block energy rank indices: {rank_path}",
        f"Top-ranked block ids, first 20: {rank_indices[:20].tolist()}",
        f"Shared sampled test images: {num_images}",
        f"Test SNR: {test_snr_db:g} dB",
        f"Mask strategy: keep whole blocks by global block energy rank; 1 block = {elements_per_block} Kp elements",
        "Latent shape: PyTorch C,H,W = 16 x 8 x 8; report H,W,C = 8 x 8 x 16",
        "Task7 compatibility: block table uses HWC flat indices; mask is applied in CHW latent tensor by block coordinates",
        "Power normalization: yes, before block masking and AWGN",
        "",
        "Measured rows:",
    ]
    for row in rows:
        keep_blocks = int(row["num_keep_blocks"])
        kept_preview = rank_indices[: min(20, keep_blocks)].tolist()
        lines.append(
            f"Kp={int(row['Kp'])} | keep_blocks={keep_blocks} | kept_block_ids_first20={kept_preview} | "
            f"R={row['R']:.6f} | avg_mse={row['avg_mse']:.6f} | avg_psnr={row['avg_psnr']:.3f} dB"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_scales_summary(
    out_path: Path,
    comparison_rows: List[Dict[str, float]],
    block_sizes: Sequence[int],
    num_images: int,
    test_snr_db: float,
    ckpt_path: Path,
    block_rank_root: Path,
) -> None:
    lines = [
        "Task (7) block-energy-ranked all-scales summary",
        "=" * 56,
        f"Checkpoint: {ckpt_path}",
        f"Block rank root: {block_rank_root}",
        f"Block sizes: {list(block_sizes)}",
        f"Shared sampled test images: {num_images}",
        f"Test SNR: {test_snr_db:g} dB",
        "Each block_size subdirectory contains task7_rate_distortion.csv, task7_rate_distortion_curve.png, and task7_summary.txt.",
        "",
        "Best PSNR by Kp among evaluated block sizes:",
    ]

    by_kp: Dict[int, List[Dict[str, float]]] = {}
    for row in comparison_rows:
        by_kp.setdefault(int(row["Kp"]), []).append(row)
    for kp in sorted(by_kp):
        best = max(by_kp[kp], key=lambda row: float(row["avg_psnr"]))
        lines.append(
            f"Kp={kp}: best_b={int(best['block_size'])}, avg_psnr={best['avg_psnr']:.3f} dB, "
            f"avg_mse={best['avg_mse']:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    data_path = resolve_path(args.data_path, project_dir)
    ckpt_path = resolve_path(args.ckpt, project_dir)
    block_rank_root = resolve_path(args.block_rank_root, project_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir, project_dir))
    device = get_device(args.device)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    if not block_rank_root.is_dir():
        raise FileNotFoundError(
            f"Block rank root does not exist: {block_rank_root}. "
            "Run scripts/innovation_block_energy_importance.py first."
        )

    block_sizes = parse_block_sizes(args.block_sizes)
    kp_values = parse_kp_list(args.kp_list)
    num_images = int(args.max_eval_samples if args.max_eval_samples is not None else args.num_images)
    if num_images <= 0:
        raise ValueError("--num_images/--max_eval_samples must be positive.")

    _, test_set = load_test_split(
        data_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    samples = sample_test_items(test_set, num_images, seed=args.seed)
    if len(samples.images) < num_images:
        print(f"Requested {num_images} images, but test split has {len(samples.images)}. Using all available samples.")
    save_selected_indices(out_dir / "task7_selected_indices.txt", samples)

    dataloader = DataLoader(
        TensorDataset(samples.images, samples.labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_jscc_checkpoint(ckpt_path, device)

    comparison_rows: List[Dict[str, float]] = []
    for block_size in block_sizes:
        blocks = build_blocks(LATENT_C, LATENT_H, LATENT_W, block_size)
        rank_path = block_rank_path(block_rank_root, block_size)
        rank_np = load_block_rank_indices(rank_path, len(blocks))
        validate_block_kp_values(kp_values, block_size, len(blocks))
        masks = build_mask_tensors(rank_np, blocks, kp_values, device)

        subdir = ensure_dir(out_dir / f"b{block_size}")
        save_selected_indices(subdir / "task7_selected_indices.txt", samples)

        rows: List[Dict[str, float]] = []
        for kp in kp_values:
            # Match task7 baseline determinism: same sampled images, deterministic AWGN per Kp.
            torch.manual_seed(args.seed + int(kp))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + int(kp))
            metrics = evaluate_kp_block_ranked(model, dataloader, device, kp, args.test_snr_db, masks[int(kp)])
            metrics["block_size"] = int(block_size)
            metrics["elements_per_block"] = int(block_size * block_size)
            metrics["num_keep_blocks"] = int(kp // (block_size * block_size))
            rows.append(metrics)
            comparison_rows.append(dict(metrics))
            print(
                f"b={block_size} Kp={kp} keep_blocks={metrics['num_keep_blocks']} "
                f"R={metrics['R']:.6f} avg_mse={metrics['avg_mse']:.6f} avg_psnr={metrics['avg_psnr']:.3f}"
            )

        csv_path = subdir / "task7_rate_distortion.csv"
        curve_path = subdir / "task7_rate_distortion_curve.png"
        summary_path = subdir / "task7_summary.txt"
        write_csv(rows, csv_path)
        save_rate_distortion_curve(rows, curve_path, block_size)
        write_summary(summary_path, rows, len(samples.images), args.test_snr_db, ckpt_path, rank_path, rank_np, block_size)
        print(f"Saved {csv_path}")
        print(f"Saved {curve_path}")
        print(f"Saved {summary_path}")

    comparison_csv = out_dir / "block_ranked_task7_comparison.csv"
    comparison_curve = out_dir / "block_ranked_task7_comparison_curve.png"
    all_summary = out_dir / "block_ranked_task7_summary.txt"
    write_comparison_csv(comparison_rows, comparison_csv)
    save_comparison_curve(comparison_rows, comparison_curve)
    write_all_scales_summary(
        all_summary,
        comparison_rows,
        block_sizes,
        len(samples.images),
        args.test_snr_db,
        ckpt_path,
        block_rank_root,
    )
    print(f"Saved {comparison_csv}")
    print(f"Saved {comparison_curve}")
    print(f"Saved {all_summary}")


if __name__ == "__main__":
    main()
