#!/usr/bin/env python
"""Task (7) variant: keep latent positions by global energy importance.

This script mirrors scripts/eval_rate_distortion.py, but replaces the prefix-Kp
mask with a top-Kp mask from outputs/innovation/energy_importance/energy_rank_indices.npy.
The flatten/unflatten order is the task7 HWC order:
  (N,16,8,8) -> (N,8,8,16) -> flatten length 1024.

Example:
  python scripts/innovation_energy_ranked_rate_distortion.py \
      --data_path /home/lc/class/yuyi/cifar-10 \
      --ckpt outputs/jscc/snr_7/best_jscc_snr7.pt \
      --rank_indices outputs/innovation/energy_importance/energy_rank_indices.npy \
      --out_dir outputs/innovation/energy_ranked_task7 \
      --num_images 500 \
      --batch_size 256
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

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

from eval_rate_distortion import (
    INPUT_K,
    LATENT_K,
    flatten_latent_hwc,
    load_jscc_checkpoint,
    parse_kp_list,
    unflatten_latent_hwc,
)
from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices
from jscc_lab.channel import awgn, power_normalize
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate task7 rate-distortion with global-energy-ranked latent retention."
    )
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or CIFAR directory.")
    parser.add_argument("--ckpt", default="outputs/jscc/snr_7/best_jscc_snr7.pt", help="Path to the SNR=7 checkpoint.")
    parser.add_argument(
        "--rank_indices",
        default="outputs/innovation/energy_importance/energy_rank_indices.npy",
        help="Path to energy_rank_indices.npy produced by innovation_energy_importance.py.",
    )
    parser.add_argument("--out_dir", default="outputs/innovation/energy_ranked_task7")
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


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def load_energy_rank_indices(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            "Energy rank file not found. Run scripts/innovation_energy_importance.py first, "
            f"or pass --rank_indices explicitly. Missing: {path}"
        )
    rank_indices = np.load(path)
    rank_indices = np.asarray(rank_indices, dtype=np.int64).reshape(-1)
    if rank_indices.shape[0] != LATENT_K:
        raise ValueError(f"Expected {LATENT_K} rank indices, got {rank_indices.shape[0]} from {path}.")
    if not np.array_equal(np.sort(rank_indices), np.arange(LATENT_K, dtype=np.int64)):
        raise ValueError(f"{path} is not a valid permutation of latent flat indices [0, {LATENT_K - 1}].")
    return rank_indices


def apply_energy_rank_kp_mask(z: torch.Tensor, kp: int, rank_indices: torch.Tensor) -> torch.Tensor:
    """Keep top-Kp globally important HWC-flat latent positions and zero the rest."""

    if kp < 0 or kp > LATENT_K:
        raise ValueError(f"kp must be in [0, {LATENT_K}], got {kp}.")
    flat, single = flatten_latent_hwc(z)
    masked = torch.zeros_like(flat)
    if kp > 0:
        keep = rank_indices[:kp].to(device=flat.device)
        masked[:, keep] = flat[:, keep]
    return unflatten_latent_hwc(masked, single=single)


@torch.no_grad()
def evaluate_kp_energy_ranked(
    model: DeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    kp: int,
    test_snr_db: float,
    rank_indices: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate one Kp using normalized latent, global energy top-K mask, AWGN, and decoder."""

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        z_masked = apply_energy_rank_kp_mask(z, kp, rank_indices)
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
        writer.writerows(rows)


def save_rate_distortion_curve(rows: List[Dict[str, float]], out_path: Path) -> None:
    rates = [row["R"] for row in rows]
    psnrs = [row["avg_psnr"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(rates, psnrs, marker="o", linewidth=1.8)
    ax.set_xlabel("Approximate rate R = Kp / 3072")
    ax.set_ylabel("Average PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(
    out_path: Path,
    rows: List[Dict[str, float]],
    num_images: int,
    test_snr_db: float,
    ckpt_path: Path,
    rank_indices_path: Path,
) -> None:
    lines = [
        "Task (7) global-energy-ranked rate-distortion summary",
        "=" * 62,
        f"Checkpoint: {ckpt_path}",
        f"Energy rank indices: {rank_indices_path}",
        f"Shared sampled test images: {num_images}",
        f"Test SNR: {test_snr_db:g} dB",
        "Mask strategy: keep global-energy-ranked top-Kp latent positions",
        "Flatten order: HWC, same as task7 prefix baseline",
        "Power normalization: yes, before masking and AWGN",
        "",
        "Measured rows:",
    ]
    for row in rows:
        lines.append(
            f"Kp={int(row['Kp'])} | R={row['R']:.6f} | avg_mse={row['avg_mse']:.6f} | "
            f"avg_psnr={row['avg_psnr']:.3f} dB"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    data_path = resolve_path(args.data_path)
    ckpt_path = resolve_path(args.ckpt)
    rank_indices_path = resolve_path(args.rank_indices)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    device = get_device(args.device)
    kp_values = parse_kp_list(args.kp_list)
    rank_indices_np = load_energy_rank_indices(rank_indices_path)
    rank_indices = torch.from_numpy(rank_indices_np).long()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
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
    rows: List[Dict[str, float]] = []
    for kp in kp_values:
        # Match task7 baseline determinism: same sampled images, deterministic AWGN per Kp.
        torch.manual_seed(args.seed + int(kp))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + int(kp))
        metrics = evaluate_kp_energy_ranked(model, dataloader, device, kp, args.test_snr_db, rank_indices)
        rows.append(metrics)
        print(
            f"Kp={kp} R={metrics['R']:.6f} avg_mse={metrics['avg_mse']:.6f} "
            f"avg_psnr={metrics['avg_psnr']:.3f}"
        )

    csv_path = out_dir / "task7_rate_distortion.csv"
    curve_path = out_dir / "task7_rate_distortion_curve.png"
    summary_path = out_dir / "task7_summary.txt"
    write_csv(rows, csv_path)
    save_rate_distortion_curve(rows, curve_path)
    write_summary(summary_path, rows, len(samples.images), args.test_snr_db, ckpt_path, rank_indices_path)

    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
