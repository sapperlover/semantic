#!/usr/bin/env python
"""Task (7) variant: keep whole latent channels by channel energy importance.

This mirrors scripts/eval_rate_distortion.py, but replaces the prefix-Kp mask
with a channel-level mask. One latent channel contains 8*8 = 64 Kp elements, so
Kp=128 keeps the top 2 channels, Kp=256 keeps the top 4 channels, ..., and
Kp=1024 keeps all 16 channels.

Example:
  python scripts/innovation_channel_ranked_rate_distortion.py \
      --data_path /home/lc/class/yuyi/cifar-10 \
      --ckpt outputs/jscc/snr_7/best_jscc_snr7.pt \
      --channel_rank_indices outputs/innovation/channel_importance/channel_energy_rank_indices.npy \
      --out_dir outputs/innovation/channel_ranked_task7 \
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

from eval_rate_distortion import INPUT_K, LATENT_C, LATENT_H, LATENT_K, LATENT_W, load_jscc_checkpoint, parse_kp_list
from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices
from jscc_lab.channel import awgn, power_normalize
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, seed_everything


ELEMENTS_PER_CHANNEL = LATENT_H * LATENT_W


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate task7 rate-distortion with channel-energy-ranked whole-channel retention."
    )
    parser.add_argument("--data_path", default="/home/lc/class/yuyi/cifar-10", help="Path to CIFAR data.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab", help="Project root.")
    parser.add_argument("--ckpt", default="outputs/jscc/snr_7/best_jscc_snr7.pt", help="Path to the SNR=7 checkpoint.")
    parser.add_argument(
        "--channel_rank_indices",
        default="outputs/innovation/channel_importance/channel_energy_rank_indices.npy",
        help="Path to channel_energy_rank_indices.npy produced by innovation_channel_importance.py.",
    )
    parser.add_argument("--out_dir", default="outputs/innovation/channel_ranked_task7")
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


def load_channel_rank_indices(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            "Channel rank file not found. Run scripts/innovation_channel_importance.py first, "
            f"or pass --channel_rank_indices explicitly. Missing: {path}"
        )
    rank_indices = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if rank_indices.shape[0] != LATENT_C:
        raise ValueError(f"Expected {LATENT_C} channel rank indices, got {rank_indices.shape[0]} from {path}.")
    if not np.array_equal(np.sort(rank_indices), np.arange(LATENT_C, dtype=np.int64)):
        raise ValueError(f"{path} is not a valid permutation of channel indices [0, {LATENT_C - 1}].")
    return rank_indices


def validate_channel_kp_values(kp_values: List[int]) -> None:
    for kp in kp_values:
        if kp % ELEMENTS_PER_CHANNEL != 0:
            raise ValueError(
                f"Kp={kp} is not divisible by {ELEMENTS_PER_CHANNEL}. "
                "Channel-ranked masking keeps whole channels, so use Kp values like 128,256,...,1024."
            )
        channels = kp // ELEMENTS_PER_CHANNEL
        if channels < 0 or channels > LATENT_C:
            raise ValueError(f"Kp={kp} maps to {channels} channels, but valid channel count is [0, {LATENT_C}].")


def apply_channel_rank_kp_mask(z: torch.Tensor, kp: int, channel_rank_indices: torch.Tensor) -> torch.Tensor:
    """Keep top-M whole channels, where M = Kp / 64, and zero the other channels."""

    if z.ndim != 4 or tuple(z.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
        raise ValueError(f"Expected latent shape (N,16,8,8), got {tuple(z.shape)}.")
    if kp % ELEMENTS_PER_CHANNEL != 0:
        raise ValueError(f"Kp must be divisible by {ELEMENTS_PER_CHANNEL}, got {kp}.")

    keep_channels = kp // ELEMENTS_PER_CHANNEL
    masked = torch.zeros_like(z)
    if keep_channels > 0:
        keep = channel_rank_indices[:keep_channels].to(device=z.device)
        masked[:, keep, :, :] = z[:, keep, :, :]
    return masked


@torch.no_grad()
def evaluate_kp_channel_ranked(
    model: DeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    kp: int,
    test_snr_db: float,
    channel_rank_indices: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate one Kp using normalized latent, channel top-M mask, AWGN, and decoder."""

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        z_masked = apply_channel_rank_kp_mask(z, kp, channel_rank_indices)
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
    channel_rank_path: Path,
    channel_rank_indices: np.ndarray,
) -> None:
    lines = [
        "Task (7) channel-energy-ranked rate-distortion summary",
        "=" * 62,
        f"Checkpoint: {ckpt_path}",
        f"Channel rank indices: {channel_rank_path}",
        f"Energy-ranked channel order: {channel_rank_indices.tolist()}",
        f"Shared sampled test images: {num_images}",
        f"Test SNR: {test_snr_db:g} dB",
        f"Mask strategy: keep whole channels by channel energy rank; 1 channel = {ELEMENTS_PER_CHANNEL} Kp elements",
        "Latent shape: PyTorch C,H,W = 16 x 8 x 8; report H,W,C = 8 x 8 x 16",
        "Power normalization: yes, before channel masking and AWGN",
        "",
        "Measured rows:",
    ]
    for row in rows:
        channels = int(row["Kp"]) // ELEMENTS_PER_CHANNEL
        kept = channel_rank_indices[:channels].tolist()
        lines.append(
            f"Kp={int(row['Kp'])} | channels={channels} | kept_channels={kept} | "
            f"R={row['R']:.6f} | avg_mse={row['avg_mse']:.6f} | avg_psnr={row['avg_psnr']:.3f} dB"
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
    channel_rank_path = resolve_path(args.channel_rank_indices, project_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir, project_dir))
    device = get_device(args.device)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    kp_values = parse_kp_list(args.kp_list)
    validate_channel_kp_values(kp_values)
    channel_rank_np = load_channel_rank_indices(channel_rank_path)
    channel_rank = torch.from_numpy(channel_rank_np).long()

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
        metrics = evaluate_kp_channel_ranked(model, dataloader, device, kp, args.test_snr_db, channel_rank)
        rows.append(metrics)
        channels = kp // ELEMENTS_PER_CHANNEL
        print(
            f"Kp={kp} channels={channels} R={metrics['R']:.6f} "
            f"avg_mse={metrics['avg_mse']:.6f} avg_psnr={metrics['avg_psnr']:.3f}"
        )

    csv_path = out_dir / "task7_rate_distortion.csv"
    curve_path = out_dir / "task7_rate_distortion_curve.png"
    summary_path = out_dir / "task7_summary.txt"
    write_csv(rows, csv_path)
    save_rate_distortion_curve(rows, curve_path)
    write_summary(summary_path, rows, len(samples.images), args.test_snr_db, ckpt_path, channel_rank_path, channel_rank_np)

    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
