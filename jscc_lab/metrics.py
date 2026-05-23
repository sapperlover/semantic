"""Reconstruction metrics for AE and Deep JSCC experiments."""

from __future__ import annotations

from typing import Callable, Dict

import torch


def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-image MSE for NCHW tensors."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}.")
    return torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.ndim)))


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """Return per-image PSNR in dB for images scaled to [0, max_value]."""

    mse = torch.clamp(batch_mse(pred, target), min=eps)
    max_tensor = torch.as_tensor(max_value, dtype=pred.dtype, device=pred.device)
    return 10.0 * torch.log10((max_tensor**2) / mse)


@torch.no_grad()
def evaluate_reconstruction(
    reconstruct: Callable[[torch.Tensor], torch.Tensor] | torch.nn.Module,
    dataloader,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Evaluate average MSE and PSNR for a model or reconstruction callable."""

    device = torch.device(device)
    if isinstance(reconstruct, torch.nn.Module):
        reconstruct.eval()
        reconstruct.to(device)

    total_mse = 0.0
    total_psnr = 0.0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch[0].to(device)
        outputs = reconstruct(images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        outputs = torch.clamp(outputs, 0.0, 1.0)

        mse = batch_mse(outputs, images)
        psnr = batch_psnr(outputs, images)
        count = images.shape[0]
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total += int(count)

    if total == 0:
        raise ValueError("No samples were evaluated.")
    return {
        "mse": total_mse / total,
        "psnr": total_psnr / total,
        "num_samples": total,
    }
