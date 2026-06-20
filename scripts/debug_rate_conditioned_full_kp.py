#!/usr/bin/env python
"""Debug full-Kp loading for the rate-conditioned Deep JSCC model.

This script checks the physically full-rate path:
  Kp=1024, r=1.0, full mask, power normalization, AWGN at SNR=7, decoder.

When loading the original SNR=7 DeepJSCC checkpoint into RateConditionedDeepJSCC,
the PSNR should be close to the task7 Kp=1024 baseline, about 27.8 dB on the
default 500 sampled test images.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader, TensorDataset

from eval_rate_distortion import LATENT_C, LATENT_H, LATENT_K, LATENT_W
from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices
from jscc_lab.channel import awgn, power_normalize
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.utils import ensure_dir, get_device, save_json, seed_everything
from train_mask_aware_random_kp import RateConditionedDeepJSCC, load_state_with_rate_conditioning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug Kp=1024, r=1 full-mask PSNR for rate-conditioned model loading.")
    parser.add_argument("--data_dir", "--data_path", dest="data_dir", default="/home/lc/class/yuyi/cifar-10")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab")
    parser.add_argument("--checkpoint", default="outputs/jscc/snr_7/best_jscc_snr7.pt")
    parser.add_argument("--output_dir", default="outputs/debug_rate_conditioned_full_kp")
    parser.add_argument("--test_snr_db", type=float, default=7.0)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--expected_psnr", type=float, default=27.8)
    parser.add_argument("--tolerance_db", type=float, default=0.7)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def resolve_path(path_text: str | Path, project_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def load_rate_conditioned_model(checkpoint_path: Path, device: torch.device) -> tuple[RateConditionedDeepJSCC, Dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    train_snr = 7.0
    extra: Dict[str, object] = {}
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        extra = checkpoint.get("extra", {})
        train_snr = float(extra.get("train_snr_db", train_snr))
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model = RateConditionedDeepJSCC(snr_db=train_snr).to(device)
    strict_rate_conditioned = bool(extra.get("model_type") == "rate_conditioned_deepjscc")
    load_info = load_state_with_rate_conditioning(model, state_dict, strict_rate_conditioned=strict_rate_conditioned)
    model.eval()
    return model, {"checkpoint_extra": extra, "load_info": load_info, "model_snr_db": train_snr}


@torch.no_grad()
def evaluate_full_kp(
    model: RateConditionedDeepJSCC,
    dataloader: DataLoader,
    device: torch.device,
    test_snr_db: float,
    seed: int,
) -> Dict[str, float]:
    full_mask = torch.ones((1, LATENT_C, LATENT_H, LATENT_W), dtype=torch.float32, device=device)
    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    # Match task7's deterministic AWGN for Kp=1024.
    torch.manual_seed(seed + LATENT_K)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + LATENT_K)

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        full_r = 1.0
        latent = power_normalize(model.encode(images, full_r))
        latent_tx = latent * full_mask
        latent_rx = awgn(latent_tx, test_snr_db, training=True)
        latent_rx = latent_rx * full_mask
        recon = model.decode(latent_rx, full_r).clamp(0.0, 1.0)

        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        count = int(images.shape[0])
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += count

    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {
        "Kp": LATENT_K,
        "r": 1.0,
        "mask_sum": LATENT_K,
        "test_snr_db": float(test_snr_db),
        "num_images": int(total_images),
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
    }


def write_csv(row: Dict[str, float], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = ["Kp", "r", "mask_sum", "test_snr_db", "num_images", "avg_mse", "avg_psnr"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    project_dir = Path(args.project_dir).expanduser().resolve()
    data_dir = resolve_path(args.data_dir, project_dir)
    checkpoint_path = resolve_path(args.checkpoint, project_dir)
    out_dir = ensure_dir(resolve_path(args.output_dir, project_dir))
    device = get_device(args.device)

    num_images = int(args.max_eval_samples if args.max_eval_samples is not None else args.num_images)
    if num_images <= 0:
        raise ValueError("--num_images/--max_eval_samples must be positive.")

    _, test_set = load_test_split(
        data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    samples = sample_test_items(test_set, num_images, seed=args.seed)
    save_selected_indices(out_dir / "debug_selected_indices.txt", samples)

    dataloader = DataLoader(
        TensorDataset(samples.images, samples.labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, load_info = load_rate_conditioned_model(checkpoint_path, device)
    row = evaluate_full_kp(model, dataloader, device, args.test_snr_db, args.seed)
    lower = float(args.expected_psnr) - float(args.tolerance_db)
    upper = float(args.expected_psnr) + float(args.tolerance_db)
    passed = lower <= float(row["avg_psnr"]) <= upper

    csv_path = out_dir / "debug_full_kp_metrics.csv"
    summary_path = out_dir / "debug_full_kp_summary.txt"
    write_csv(row, csv_path)
    save_json({"checkpoint": str(checkpoint_path), **load_info, "metrics": row, "passed": passed}, out_dir / "debug_full_kp.json")

    lines = [
        "Rate-conditioned full-Kp debug summary",
        "=" * 42,
        f"Checkpoint: {checkpoint_path}",
        f"Data: {data_dir}",
        "Path: Kp=1024, r=1.0, full mask, power_normalize, AWGN, decoder",
        f"Test SNR: {args.test_snr_db:g} dB",
        f"Images: {int(row['num_images'])}",
        f"avg_mse: {row['avg_mse']:.8f}",
        f"avg_psnr: {row['avg_psnr']:.4f} dB",
        f"Expected PSNR window: [{lower:.3f}, {upper:.3f}] dB",
        f"Load check passed: {passed}",
        f"Load info: {load_info}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"avg_mse={row['avg_mse']:.8f} avg_psnr={row['avg_psnr']:.4f} dB passed={passed}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

# Example:
# python scripts/debug_rate_conditioned_full_kp.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --checkpoint outputs/jscc/snr_7/best_jscc_snr7.pt \
#   --output_dir outputs/debug_rate_conditioned_full_kp \
#   --test_snr_db 7 \
#   --num_images 500
