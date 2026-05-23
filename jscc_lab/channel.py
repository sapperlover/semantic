"""Channel utilities for Deep JSCC experiments."""

from __future__ import annotations

import math

import torch


def power_normalize(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each sample so its latent average power is approximately 1.

    Power is computed per sample as mean(z**2) over all non-batch dimensions.
    For latent shape (N, 16, 8, 8), the returned tensor keeps the same shape.
    """

    if z.ndim < 2:
        raise ValueError(f"Expected a batched latent tensor, got shape {tuple(z.shape)}.")
    dims = tuple(range(1, z.ndim))
    power = torch.mean(z**2, dim=dims, keepdim=True)
    return z / torch.sqrt(power + eps)


def snr_db_to_noise_std(snr_db: float) -> float:
    """Convert SNR in dB to AWGN standard deviation when signal power P=1.

    SNR = 10 * log10(P / noise_var). After power normalization P=1, so:
      noise_var = 10 ** (-snr_db / 10)
      noise_std = sqrt(noise_var)

    `torch.randn_like` and `torch.normal` are scaled by standard deviation,
    not variance, so callers must multiply by `noise_std`.
    """

    noise_var = 10.0 ** (-float(snr_db) / 10.0)
    return math.sqrt(noise_var)


def awgn(z: torch.Tensor, snr_db: float, training: bool = True) -> torch.Tensor:
    """Apply AWGN to a normalized latent tensor.

    When `training` is False, the input is returned unchanged. This makes the
    helper usable for ablations, while train/eval scripts pass True whenever
    they need channel noise.
    """

    if not training:
        return z
    noise_std = snr_db_to_noise_std(float(snr_db))
    return z + torch.randn_like(z) * noise_std
