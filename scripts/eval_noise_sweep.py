#!/usr/bin/env python
"""Task (6): evaluate one SNR=7 dB Deep JSCC model under multiple test SNRs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices
from jscc_lab.channel import awgn, power_normalize
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SNR=7 Deep JSCC model under a test-noise sweep.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or CIFAR directory.")
    parser.add_argument("--ckpt", default="outputs/jscc/snr_7/best_jscc_snr7.pt", help="Path to the SNR=7 checkpoint.")
    parser.add_argument("--out_dir", default="outputs/task6", help="Directory for task (6) outputs.")
    parser.add_argument("--test_snr_list", default="1,4,7,13,19", help="Comma-separated test SNR values in dB.")
    parser.add_argument("--num_images", type=int, default=500, help="Number of fixed test images to sample.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Alias to reduce evaluation samples for debugging.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def parse_snr_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--test_snr_list must contain at least one SNR value.")
    return values


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_jscc_checkpoint(ckpt_path: Path, device: torch.device) -> DeepJSCC:
    """Load a DeepJSCC checkpoint produced by scripts/train_jscc.py."""

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    train_snr = 7.0
    if isinstance(checkpoint, dict):
        train_snr = float(checkpoint.get("extra", {}).get("train_snr_db", train_snr))
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model = DeepJSCC(snr_db=train_snr)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def warmup_model(model: DeepJSCC, dataloader: DataLoader, device: torch.device, test_snr_db: float, warmup_batches: int) -> None:
    """Warm up kernels before timing; data loading and plotting are outside timing."""

    if warmup_batches <= 0:
        return
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        z_noisy = awgn(z, test_snr_db, training=True)
        _ = model.decoder(z_noisy)
    sync_if_cuda(device)


@torch.no_grad()
def evaluate_one_snr(model: DeepJSCC, dataloader: DataLoader, device: torch.device, test_snr_db: float) -> Dict[str, float]:
    """Evaluate metrics and forward times for one test SNR.

    Encoder timing includes `encoder + power_normalize`. Decoder timing only
    includes `decoder forward`; AWGN sampling is intentionally outside both.
    """

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0
    encoder_seconds = 0.0
    decoder_seconds = 0.0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        batch_size = int(images.shape[0])

        sync_if_cuda(device)
        start = time.perf_counter()
        z = power_normalize(model.encoder(images))
        sync_if_cuda(device)
        encoder_seconds += time.perf_counter() - start

        z_noisy = awgn(z, test_snr_db, training=True)

        sync_if_cuda(device)
        start = time.perf_counter()
        recon = model.decoder(z_noisy)
        sync_if_cuda(device)
        decoder_seconds += time.perf_counter() - start

        recon = recon.clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += batch_size

    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {
        "test_snr_db": float(test_snr_db),
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
        "encoder_ms_per_image": 1000.0 * encoder_seconds / total_images,
        "decoder_ms_per_image": 1000.0 * decoder_seconds / total_images,
    }


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_snr_db",
                "avg_mse",
                "avg_psnr",
                "encoder_ms_per_image",
                "decoder_ms_per_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_psnr_curve(rows: List[Dict[str, float]], out_path: Path) -> None:
    snrs = [row["test_snr_db"] for row in rows]
    psnrs = [row["avg_psnr"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(snrs, psnrs, marker="o", linewidth=1.8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_xticks(snrs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(out_path: Path, rows: List[Dict[str, float]], num_images: int, ckpt_path: Path) -> None:
    lines = [
        "Task (6) noise sweep summary",
        "=" * 36,
        f"Checkpoint: {ckpt_path}",
        f"Shared sampled test images: {num_images}",
        "",
        "简短分析模板：",
        "当 SNR 低于 7dB 时，测试信道比训练信道更差，噪声更强，重建质量和 PSNR 通常下降。",
        "当 SNR 高于 7dB 时，测试信道更好，噪声更弱，重建质量通常提高，但提升可能逐渐饱和。",
        "",
        "Measured rows:",
    ]
    for row in rows:
        lines.append(
            f"SNR={row['test_snr_db']:g} dB | avg_mse={row['avg_mse']:.6f} | "
            f"avg_psnr={row['avg_psnr']:.3f} dB | encoder={row['encoder_ms_per_image']:.4f} ms/image | "
            f"decoder={row['decoder_ms_per_image']:.4f} ms/image"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    data_path = Path(args.data_path).expanduser()
    ckpt_path = Path(args.ckpt).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    device = get_device(args.device)
    test_snrs = parse_snr_list(args.test_snr_list)
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
    save_selected_indices(out_dir / "task6_selected_indices.txt", samples)

    dataloader = DataLoader(
        TensorDataset(samples.images, samples.labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_jscc_checkpoint(ckpt_path, device)
    rows: List[Dict[str, float]] = []
    for snr in test_snrs:
        # Keep AWGN draws reproducible while all SNR values share the same images.
        torch.manual_seed(args.seed + int(round(float(snr) * 1000)))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + int(round(float(snr) * 1000)))
        warmup_model(model, dataloader, device, snr, args.warmup_batches)
        metrics = evaluate_one_snr(model, dataloader, device, snr)
        rows.append(metrics)
        print(
            f"SNR={snr:g} dB avg_mse={metrics['avg_mse']:.6f} avg_psnr={metrics['avg_psnr']:.3f} "
            f"encoder={metrics['encoder_ms_per_image']:.4f} ms/image decoder={metrics['decoder_ms_per_image']:.4f} ms/image"
        )

    csv_path = out_dir / "task6_noise_sweep.csv"
    curve_path = out_dir / "task6_psnr_vs_snr.png"
    summary_path = out_dir / "task6_summary.txt"
    write_csv(rows, csv_path)
    save_psnr_curve(rows, curve_path)
    write_summary(summary_path, rows, len(samples.images), ckpt_path)

    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
