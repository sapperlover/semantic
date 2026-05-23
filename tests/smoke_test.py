#!/usr/bin/env python
"""Smoke test for the project skeleton using a generated tiny dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_dataloaders, make_splits
from jscc_lab.channel import awgn, power_normalize, snr_db_to_noise_std
from jscc_lab.metrics import batch_mse, batch_psnr, evaluate_reconstruction
from jscc_lab.models import AutoEncoder, DeepJSCC, count_parameters, model_summary
from jscc_lab.plotting import save_image_grid, save_psnr_curve, save_rate_distortion_curve, save_training_curves
from jscc_lab.utils import ensure_dir, get_device, save_json, seed_everything
from scripts.eval_rate_distortion import apply_prefix_kp_mask, flatten_latent_hwc


def main() -> None:
    seed_everything(123)
    device = get_device("auto")
    out_dir = ensure_dir(ROOT / "tests" / "_smoke_outputs")

    tiny_path = out_dir / "tiny_dataset.npz"
    raw = np.random.randint(0, 256, size=(32, 3072), dtype=np.uint8)
    labels = np.arange(32, dtype=np.int64) % 10
    np.savez(tiny_path, data=raw, labels=labels)

    images, loaded_labels = load_cifar_array_dataset(tiny_path)
    assert images.shape == (32, 3, 32, 32)
    assert images.dtype == np.float32
    assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
    assert loaded_labels.tolist() == labels.tolist()

    dataset = CIFARArrayDataset(raw, labels)
    train_set, val_set, test_set = make_splits(dataset, train=0.5, val=0.25, test=0.25, seed=123)
    assert (len(train_set), len(val_set), len(test_set)) == (16, 8, 8)

    loaders = make_dataloaders(
        dataset=dataset,
        batch_size=8,
        train=0.5,
        val=0.25,
        test=0.25,
        seed=123,
    )
    batch_images, _ = next(iter(loaders["train"]))
    model = AutoEncoder()
    with torch.no_grad():
        latent = model.encode(batch_images)
        decoded = model.decode(latent)
    assert latent.shape == (8, 16, 8, 8)
    assert decoded.shape == (8, 3, 32, 32)
    assert float(decoded.min()) >= 0.0 and float(decoded.max()) <= 1.0
    assert count_parameters(model) > 0
    assert "8 x 8 x 16" in model_summary(model)

    jscc = DeepJSCC(snr_db=7)
    with torch.no_grad():
        z_norm = jscc.encode_normalized(batch_images)
        z_noisy = awgn(z_norm, snr_db=7, training=True)
        jscc_out = jscc(batch_images, snr_db=7, add_noise=True)
    assert z_norm.shape == (8, 16, 8, 8)
    assert torch.allclose(torch.mean(z_norm**2, dim=(1, 2, 3)), torch.ones(8), atol=1e-4)
    assert z_noisy.shape == z_norm.shape
    assert jscc_out.shape == (8, 3, 32, 32)
    assert float(jscc_out.min()) >= 0.0 and float(jscc_out.max()) <= 1.0
    assert snr_db_to_noise_std(10) > 0
    assert power_normalize(torch.ones(2, 16, 8, 8)).shape == (2, 16, 8, 8)

    hwc_latent = torch.arange(8 * 8 * 16, dtype=torch.float32).reshape(8, 8, 16)
    chw_latent = hwc_latent.permute(2, 0, 1).contiguous()
    assert torch.equal(apply_prefix_kp_mask(chw_latent, 1024), chw_latent)
    assert torch.count_nonzero(apply_prefix_kp_mask(chw_latent, 0)).item() == 0
    masked_small = apply_prefix_kp_mask(chw_latent, 5)
    masked_flat, _ = flatten_latent_hwc(masked_small)
    expected_flat = torch.zeros(1, 1024)
    expected_flat[0, :5] = torch.arange(5, dtype=torch.float32)
    assert torch.equal(masked_flat, expected_flat)

    recon = torch.clamp(batch_images + 0.01 * torch.randn_like(batch_images), 0.0, 1.0)
    mse = batch_mse(recon, batch_images)
    psnr = batch_psnr(recon, batch_images)
    assert mse.shape == (8,)
    assert psnr.shape == (8,)

    identity_metrics = evaluate_reconstruction(lambda x: x, loaders["test"], device=device)
    assert identity_metrics["num_samples"] == 8
    assert identity_metrics["mse"] < 1e-8

    curve_path = save_training_curves(
        {"train_loss": [0.12, 0.08, 0.05], "val_loss": [0.14, 0.09, 0.06]},
        out_dir / "training_curves.png",
    )
    grid_path = save_image_grid(batch_images, out_dir / "image_grid.png", nrow=4)
    psnr_path = save_psnr_curve([1, 4, 7, 13, 19], [18.0, 20.0, 23.0, 27.0, 30.0], out_dir / "psnr_curve.png")
    rd_path = save_rate_distortion_curve(
        [128 / 3072, 256 / 3072, 512 / 3072, 1024 / 3072],
        [16.0, 19.0, 23.0, 27.0],
        out_dir / "rate_distortion.png",
    )
    summary_path = save_json(
        {
            "tiny_dataset": str(tiny_path),
            "device": str(device),
            "metrics": identity_metrics,
            "plots": [str(curve_path), str(grid_path), str(psnr_path), str(rd_path)],
        },
        out_dir / "summary.json",
    )

    print("Smoke test passed.")
    print(f"Tiny dataset: {tiny_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
