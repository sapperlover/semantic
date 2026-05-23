#!/usr/bin/env python
"""Estimate latent Gaussian statistics and decode sampled latents for task (3)."""

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
    parser = argparse.ArgumentParser(description="Sample images from spatial 16D Gaussian latent distributions.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", default="outputs/ae/best_ae.pt")
    parser.add_argument("--out_dir", default="outputs/ae/task3_gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_stats_samples", type=int, default=256)
    parser.add_argument("--num_generated", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def compute_spatial_gaussian_stats(latents_nhwc: np.ndarray):
    """Compute mean, variance, and 16x16 covariance at each 8x8 spatial position."""

    means = latents_nhwc.mean(axis=0)
    vars_ = latents_nhwc.var(axis=0, ddof=1)
    covs = np.zeros((8, 8, 16, 16), dtype=np.float64)
    for h in range(8):
        for w in range(8):
            covs[h, w] = np.cov(latents_nhwc[:, h, w, :], rowvar=False, ddof=1)
    return means.astype(np.float32), vars_.astype(np.float32), covs.astype(np.float32)


def sample_latents_from_gaussians(means: np.ndarray, covs: np.ndarray, num_samples: int, eps: float, seed: int) -> np.ndarray:
    """Sample independent 16D Gaussian vectors for each spatial position."""

    generated = np.zeros((num_samples, 8, 8, 16), dtype=np.float32)
    eye = np.eye(16, dtype=np.float64)
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for sample_idx in range(num_samples):
            for h in range(8):
                for w in range(8):
                    cov = covs[h, w].astype(np.float64)
                    cov = 0.5 * (cov + cov.T) + eps * eye
                    try:
                        generated[sample_idx, h, w] = np.random.multivariate_normal(means[h, w], cov).astype(np.float32)
                    except np.linalg.LinAlgError:
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        repaired = eigvecs @ np.diag(np.maximum(eigvals, eps)) @ eigvecs.T
                        generated[sample_idx, h, w] = np.random.multivariate_normal(means[h, w], repaired).astype(np.float32)
    finally:
        np.random.set_state(old_state)
    return generated


def save_summary(path: Path, means: np.ndarray, vars_: np.ndarray, covs: np.ndarray, num_samples: int) -> None:
    """Write a clear text summary for report screenshots."""

    lines = [
        "Gaussian latent statistics for AE encoder",
        "=" * 48,
        f"Number of test samples used: {num_samples}",
        "Latent tensor from encoder: (N, 16, 8, 8)",
        "Statistics layout: means/vars=(8, 8, 16), covs=(8, 8, 16, 16)",
        "",
        f"means shape: {means.shape}",
        f"vars shape: {vars_.shape}",
        f"covs shape: {covs.shape}",
        f"overall mean value: {means.mean():.6f}",
        f"overall variance value: {vars_.mean():.6f}",
        f"mean range: [{means.min():.6f}, {means.max():.6f}]",
        f"variance range: [{vars_.min():.6f}, {vars_.max():.6f}]",
        "",
    ]

    for h in range(8):
        for w in range(8):
            lines.extend(
                [
                    f"Position (h={h}, w={w})",
                    "mean:",
                    np.array2string(means[h, w], precision=6, suppress_small=False, max_line_width=140),
                    "variance:",
                    np.array2string(vars_[h, w], precision=6, suppress_small=False, max_line_width=140),
                    "covariance:",
                    np.array2string(covs[h, w], precision=6, suppress_small=False, max_line_width=160),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_mean_var_overview(means: np.ndarray, vars_: np.ndarray, out_path: Path) -> None:
    """Save spatial and channel summaries for mean and variance."""

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), dpi=220)
    im0 = axes[0, 0].imshow(means.mean(axis=-1), cmap="coolwarm")
    axes[0, 0].set_title("Mean over channels")
    im1 = axes[0, 1].imshow(vars_.mean(axis=-1), cmap="magma")
    axes[0, 1].set_title("Variance over channels")
    axes[1, 0].bar(np.arange(16), means.mean(axis=(0, 1)))
    axes[1, 0].set_title("Mean per channel")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("Mean")
    axes[1, 1].bar(np.arange(16), vars_.mean(axis=(0, 1)))
    axes[1, 1].set_title("Variance per channel")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Variance")
    for ax in axes[0]:
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xlabel("w")
        ax.set_ylabel("h")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_generated_grid(images: torch.Tensor, out_path: Path) -> None:
    array = np.clip(images.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0)
    cols = 5
    rows = int(np.ceil(len(array) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 3.0 * rows), dpi=220)
    axes = np.asarray(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(array):
            ax.axis("off")
            continue
        ax.imshow(array[idx])
        ax.set_title(f"sample {idx + 1}", fontsize=8)
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
    samples = sample_test_items(test_set, args.num_stats_samples, seed=args.seed)
    save_selected_indices(out_dir / "stats_selected_indices.txt", samples)

    if len(samples.images) < args.num_stats_samples:
        print(f"Requested {args.num_stats_samples} samples, but test split has {len(samples.images)}. Using all available samples.")

    model = load_autoencoder(checkpoint_path, device=device)
    with torch.no_grad():
        latents = model.encode(samples.images.to(device)).detach().cpu()

    latents_nhwc = latents.numpy().transpose(0, 2, 3, 1)
    means, vars_, covs = compute_spatial_gaussian_stats(latents_nhwc)
    np.savez(out_dir / "gaussian_stats.npz", means=means, vars=vars_, covs=covs)
    save_summary(out_dir / "gaussian_stats_summary.txt", means, vars_, covs, num_samples=len(samples.images))
    save_mean_var_overview(means, vars_, out_dir / "mean_var_overview.png")

    sampled_nhwc = sample_latents_from_gaussians(means, covs, args.num_generated, args.eps, args.seed + 1000)
    sampled_nchw = torch.from_numpy(sampled_nhwc.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        generated = model.decode(sampled_nchw).clamp(0.0, 1.0).cpu()
    save_generated_grid(generated, out_dir / "generated_from_gaussian_10.png")

    print(f"Saved task (3) outputs to {out_dir}")


if __name__ == "__main__":
    main()
