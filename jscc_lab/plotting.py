"""Plotting helpers that save figures for reports and experiment logs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    ensure_dir(output_path.parent)
    return output_path


def _to_numpy_images(images) -> np.ndarray:
    """Convert tensors or arrays to NHWC float images in [0, 1]."""

    if isinstance(images, torch.Tensor):
        array = images.detach().cpu().float().numpy()
    else:
        array = np.asarray(images, dtype=np.float32)

    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected images with 3 or 4 dimensions, got {array.shape}.")
    if array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected channel dimension of size 3, got {array.shape}.")
    return np.clip(array, 0.0, 1.0)


def save_training_curves(history: Mapping[str, Sequence[float]], out_path: str | Path) -> Path:
    """Save line plots for scalar histories such as train/val loss."""

    output_path = _prepare_output(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=220)
    for name, values in history.items():
        if not values:
            continue
        ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_image_grid(
    images,
    out_path: str | Path,
    nrow: int = 8,
    titles: Sequence[str] | None = None,
) -> Path:
    """Save a grid of CIFAR images without requiring torchvision."""

    output_path = _prepare_output(out_path)
    array = _to_numpy_images(images)
    n_images = len(array)
    nrow = max(1, min(nrow, n_images))
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(ncol, nrow, figsize=(1.6 * nrow, 1.8 * ncol), dpi=220)
    axes = np.asarray(axes).reshape(ncol, nrow)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n_images:
            continue
        ax.imshow(array[idx])
        if titles is not None and idx < len(titles):
            ax.set_title(str(titles[idx]), fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_psnr_curve(snr_db: Sequence[float], psnr: Sequence[float], out_path: str | Path) -> Path:
    """Save the assignment's SNR-vs-PSNR curve."""

    output_path = _prepare_output(out_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    ax.plot(snr_db, psnr, marker="o", linewidth=1.8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Average PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_rate_distortion_curve(rates: Sequence[float], psnr: Sequence[float], out_path: str | Path) -> Path:
    """Save the Kp-derived rate-distortion curve."""

    output_path = _prepare_output(out_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    ax.plot(rates, psnr, marker="o", linewidth=1.8)
    ax.set_xlabel("Approximate Rate R")
    ax.set_ylabel("Average PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
