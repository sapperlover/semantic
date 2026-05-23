#!/usr/bin/env python
"""Perturb AE latent codes with Gaussian noise for homework task (4)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from jscc_lab.analysis import (
    load_autoencoder,
    load_test_split,
    read_selected_indices,
    sample_test_items,
    save_selected_indices,
)
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perturb AE latent codes and visualize reconstructions.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", default="outputs/ae/best_ae.pt")
    parser.add_argument("--out_dir", default="outputs/ae/task4_perturb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--selected_indices", default=None, help="Optional selected_indices.txt from task (2).")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Gaussian noise std when --snr_db is not set.")
    parser.add_argument("--snr_db", type=float, default=None, help="Optional latent SNR in dB; overrides --noise_std.")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def to_nhwc(images: torch.Tensor) -> np.ndarray:
    return np.clip(images.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0)


def compute_noise_std(latent: torch.Tensor, noise_std: float, snr_db: float | None) -> float:
    """Use explicit std or derive std from latent power and requested SNR."""

    if snr_db is None:
        return float(noise_std)
    power = torch.mean(latent**2).item()
    return float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))


def save_perturb_grid(images, recon, perturbed, labels, indices, out_path: Path, caption: str) -> None:
    originals = to_nhwc(images)
    clean = to_nhwc(recon)
    noisy = to_nhwc(perturbed)
    rows = len(originals)
    fig, axes = plt.subplots(rows, 3, figsize=(7.8, 2.0 * rows), dpi=220)
    if rows == 1:
        axes = np.asarray([axes])

    for i in range(rows):
        columns = [
            ("Original", originals[i]),
            ("AE recon", clean[i]),
            ("Perturbed recon", noisy[i]),
        ]
        for j, (title, image) in enumerate(columns):
            ax = axes[i, j]
            ax.imshow(image)
            ax.set_title(f"{title}\nidx {indices[i]} | label {int(labels[i])}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    data_path = Path(args.data_path).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())

    _, test_set = load_test_split(
        data_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    selected = read_selected_indices(Path(args.selected_indices).expanduser()) if args.selected_indices else None
    samples = sample_test_items(test_set, args.num_images, seed=args.seed, selected_original_indices=selected)
    save_selected_indices(out_dir / "perturb_selected_indices.txt", samples)

    model = load_autoencoder(checkpoint_path, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 404)
    with torch.no_grad():
        images = samples.images.to(device)
        latent = model.encode(images)
        recon = model.decode(latent).clamp(0.0, 1.0)
        std = compute_noise_std(latent, args.noise_std, args.snr_db)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype) * std
        perturbed = model.decode(latent + noise).clamp(0.0, 1.0)

    if args.snr_db is None:
        caption = f"Latent Gaussian perturbation: noise_std={std:.4f}"
    else:
        caption = f"Latent Gaussian perturbation: SNR={args.snr_db:.2f} dB, noise_std={std:.4f}"
    save_perturb_grid(
        samples.images,
        recon.cpu(),
        perturbed.cpu(),
        samples.labels,
        samples.original_indices,
        out_dir / "perturb_recon_10.png",
        caption,
    )
    (out_dir / "perturb_settings.txt").write_text(caption + "\n", encoding="utf-8")

    print(f"Saved task (4) outputs to {out_dir}")


if __name__ == "__main__":
    main()
