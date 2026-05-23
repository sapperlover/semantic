#!/usr/bin/env python
"""Visualize AE reconstructions and latent heatmaps for homework task (2)."""

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

from jscc_lab.analysis import load_autoencoder, load_test_split, sample_test_items, save_selected_indices
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize 10 AE reconstructions and latent codes.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", default="outputs/ae/best_ae.pt")
    parser.add_argument("--out_dir", default="outputs/ae/task2_latent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def to_nhwc(images: torch.Tensor) -> np.ndarray:
    return np.clip(images.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0)


def save_reconstruction_pairs(images, recons, labels, indices, out_path: Path) -> None:
    """Save a report-ready 10x2 original/reconstruction comparison."""

    originals = to_nhwc(images)
    reconstructions = to_nhwc(recons)
    rows = len(originals)
    fig, axes = plt.subplots(rows, 2, figsize=(5.2, 2.0 * rows), dpi=220)
    if rows == 1:
        axes = np.asarray([axes])

    for i in range(rows):
        titles = [
            f"idx {indices[i]} | label {int(labels[i])}\nOriginal",
            f"idx {indices[i]} | label {int(labels[i])}\nAE recon",
        ]
        for j, image in enumerate([originals[i], reconstructions[i]]):
            ax = axes[i, j]
            ax.imshow(image)
            ax.set_title(titles[j], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def save_latent_heatmap(latent_chw: np.ndarray, label: int, original_index: int, out_path: Path) -> None:
    """Plot one image's 16 latent channels as a compact 4x4 heatmap grid."""

    vmin = float(np.percentile(latent_chw, 2))
    vmax = float(np.percentile(latent_chw, 98))
    if np.isclose(vmin, vmax):
        vmin, vmax = float(latent_chw.min()), float(latent_chw.max())

    fig, axes = plt.subplots(4, 4, figsize=(6.2, 6.2), dpi=220)
    for channel, ax in enumerate(axes.flat):
        im = ax.imshow(latent_chw[channel], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"ch {channel:02d}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Latent heatmaps | idx {original_index} | label {label}", fontsize=10)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.96])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
    fig.savefig(out_path, bbox_inches="tight")
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
    samples = sample_test_items(test_set, args.num_images, seed=args.seed)
    save_selected_indices(out_dir / "selected_indices.txt", samples)

    model = load_autoencoder(checkpoint_path, device=device)
    with torch.no_grad():
        images = samples.images.to(device)
        latents = model.encode(images)
        recons = model.decode(latents).clamp(0.0, 1.0)

    save_reconstruction_pairs(
        samples.images,
        recons.cpu(),
        samples.labels,
        samples.original_indices,
        out_dir / "recon_10.png",
    )

    heatmap_paths = []
    latents_np = latents.detach().cpu().numpy()
    for i, latent in enumerate(latents_np):
        path = out_dir / f"latent_heatmap_img_{i:02d}_idx_{samples.original_indices[i]}_label_{int(samples.labels[i])}.png"
        save_latent_heatmap(latent, int(samples.labels[i]), samples.original_indices[i], path)
        heatmap_paths.append(path.name)
    (out_dir / "latent_heatmaps_manifest.txt").write_text("\n".join(heatmap_paths) + "\n", encoding="utf-8")

    print(f"Saved task (2) outputs to {out_dir}")


if __name__ == "__main__":
    main()
