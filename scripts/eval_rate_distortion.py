#!/usr/bin/env python
"""Task (7): evaluate prefix-Kp latent masking rate-distortion performance."""

from __future__ import annotations

import argparse
import csv
import os
import sys
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


LATENT_C = 16
LATENT_H = 8
LATENT_W = 8
LATENT_K = LATENT_H * LATENT_W * LATENT_C
INPUT_K = 32 * 32 * 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prefix-Kp rate-distortion for an SNR=7 Deep JSCC model.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or CIFAR directory.")
    parser.add_argument("--ckpt", default="outputs/jscc/snr_7/best_jscc_snr7.pt", help="Path to the SNR=7 checkpoint.")
    parser.add_argument("--out_dir", default="outputs/task7", help="Directory for task (7) outputs.")
    parser.add_argument("--kp_list", default="128,256,384,512,640,768,896,1024", help="Comma-separated Kp values.")
    parser.add_argument("--test_snr_db", type=float, default=7.0)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Alias to reduce evaluation samples for debugging.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_strategy", default="prefix", help="Currently only 'prefix' is implemented.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def parse_kp_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--kp_list must contain at least one Kp value.")
    for kp in values:
        if kp < 0 or kp > LATENT_K:
            raise ValueError(f"Kp must be in [0, {LATENT_K}], got {kp}.")
    return values


def _as_batched_latent(z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Accept either (16,8,8) or (N,16,8,8) latent tensors."""

    if z.ndim == 3:
        z = z.unsqueeze(0)
        single = True
    elif z.ndim == 4:
        single = False
    else:
        raise ValueError(f"Expected latent shape (16,8,8) or (N,16,8,8), got {tuple(z.shape)}.")

    if tuple(z.shape[1:]) != (LATENT_C, LATENT_H, LATENT_W):
        raise ValueError(f"Expected latent shape (*,16,8,8), got {tuple(z.shape)}.")
    return z, single


def flatten_latent_hwc(z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Flatten latent using the assignment order: CHW -> HWC -> length 1024."""

    z, single = _as_batched_latent(z)
    flat = z.permute(0, 2, 3, 1).contiguous().reshape(z.shape[0], LATENT_K)
    return flat, single


def unflatten_latent_hwc(flat: torch.Tensor, single: bool = False) -> torch.Tensor:
    """Undo HWC flattening back to PyTorch CHW latent order."""

    if flat.ndim != 2 or flat.shape[1] != LATENT_K:
        raise ValueError(f"Expected flat latent shape (N,{LATENT_K}), got {tuple(flat.shape)}.")
    z = flat.reshape(flat.shape[0], LATENT_H, LATENT_W, LATENT_C).permute(0, 3, 1, 2).contiguous()
    return z[0] if single else z


def apply_prefix_kp_mask(z: torch.Tensor, kp: int) -> torch.Tensor:
    """Keep the first Kp elements in HWC-flattened latent order and zero the rest.

    The assignment defines latent size as 8 x 8 x 16. PyTorch stores it as
    (16,8,8), so the exact masking order is:
      (16,8,8) -> permute to (8,8,16) -> flatten -> prefix mask -> unflatten.
    """

    if kp < 0 or kp > LATENT_K:
        raise ValueError(f"kp must be in [0, {LATENT_K}], got {kp}.")
    flat, single = flatten_latent_hwc(z)
    masked = torch.zeros_like(flat)
    if kp > 0:
        masked[:, :kp] = flat[:, :kp]
    return unflatten_latent_hwc(masked, single=single)


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
def evaluate_kp(model: DeepJSCC, dataloader: DataLoader, device: torch.device, kp: int, test_snr_db: float) -> Dict[str, float]:
    """Evaluate one Kp using normalized latent, prefix mask, AWGN, and decoder."""

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        z = power_normalize(model.encoder(images))
        z_masked = apply_prefix_kp_mask(z, kp)
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


def write_summary(out_path: Path, rows: List[Dict[str, float]], num_images: int, test_snr_db: float, ckpt_path: Path) -> None:
    lines = [
        "Task (7) rate-distortion summary",
        "=" * 38,
        f"Checkpoint: {ckpt_path}",
        f"Shared sampled test images: {num_images}",
        f"Test SNR: {test_snr_db:g} dB",
        "Mask strategy: prefix baseline only",
        "",
        "简短趋势分析模板：",
        "通常 R 增大时保留 latent 信息更多，PSNR 上升；低 R 区域可能提升较快，高 R 区域可能趋于饱和或边际收益下降。",
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
    if args.mask_strategy != "prefix":
        raise NotImplementedError("Only --mask_strategy prefix is implemented for the task (7) baseline.")

    seed_everything(args.seed)
    data_path = Path(args.data_path).expanduser()
    ckpt_path = Path(args.ckpt).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    device = get_device(args.device)
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
    rows: List[Dict[str, float]] = []
    for kp in kp_values:
        # Keep AWGN draws deterministic while all Kp values share the same images.
        torch.manual_seed(args.seed + int(kp))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + int(kp))
        metrics = evaluate_kp(model, dataloader, device, kp, args.test_snr_db)
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
    write_summary(summary_path, rows, len(samples.images), args.test_snr_db, ckpt_path)

    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
