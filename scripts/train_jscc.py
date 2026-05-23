#!/usr/bin/env python
"""Train a power-normalized Deep JSCC model for homework task (5)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from jscc_lab.channel import snr_db_to_noise_std
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.models import DeepJSCC, count_parameters
from jscc_lab.plotting import save_training_curves
from jscc_lab.utils import ensure_dir, get_device, save_checkpoint, seed_everything


def snr_tag(snr_db: float) -> str:
    """Create filesystem-safe SNR tags such as 7 or 7p5."""

    value = float(snr_db)
    if value.is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Deep JSCC at one channel SNR.")
    parser.add_argument("--train_snr_db", type=float, required=True, help="Training channel SNR in dB.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or directory.")
    parser.add_argument("--out_dir", default=None, help="Output directory, default outputs/jscc/snr_{snr}.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def maybe_create_tiny_dataset(data_path: Path, seed: int) -> None:
    """Create the documented tiny fixture when quick debug commands request it."""

    if data_path.exists() or data_path.name != "tiny_dataset.npz":
        return
    ensure_dir(data_path.parent)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(256, 3072), dtype=np.uint8)
    labels = np.arange(256, dtype=np.int64) % 10
    np.savez(data_path, data=raw, labels=labels)
    print(f"Created tiny dataset fixture: {data_path}")


def cap_subset(dataset, max_samples: int | None):
    """Limit a split for short debug runs."""

    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def jscc_model_summary(model: DeepJSCC, train_snr_db: float, device: torch.device) -> str:
    """Return a report-friendly summary for the Deep JSCC model."""

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 32, 32), device=device)
        raw_latent = model.encode(dummy)
        normalized_latent = model.encode_normalized(dummy)
        output = model(dummy, snr_db=train_snr_db, add_noise=True)
        normalized_power = torch.mean(normalized_latent**2, dim=(1, 2, 3)).item()
    if was_training:
        model.train()

    noise_var = 10.0 ** (-float(train_snr_db) / 10.0)
    noise_std = snr_db_to_noise_std(train_snr_db)
    lines = [
        repr(model),
        "",
        f"Training SNR: {float(train_snr_db):.4g} dB",
        f"AWGN conversion after power normalization P=1: noise_var=10^(-SNR/10)={noise_var:.8f}",
        f"torch.randn_like is scaled by noise_std=sqrt(noise_var)={noise_std:.8f}, not by variance.",
        f"Input tensor shape: {tuple(dummy.shape)}",
        f"Raw PyTorch latent tensor shape (N,C,H,W): {tuple(raw_latent.shape)}",
        f"Normalized PyTorch latent tensor shape (N,C,H,W): {tuple(normalized_latent.shape)}",
        "Per-image latent code shape for the report (H x W x C): 8 x 8 x 16",
        f"Flattened latent elements per image K: {raw_latent.shape[1] * raw_latent.shape[2] * raw_latent.shape[3]}",
        f"Mean normalized latent power for dummy sample: {normalized_power:.6f}",
        f"Output tensor shape: {tuple(output.shape)}",
        f"Trainable parameters: {count_parameters(model, trainable_only=True):,}",
        f"Total parameters: {count_parameters(model, trainable_only=False):,}",
    ]
    return "\n".join(lines)


def run_epoch(
    model: DeepJSCC,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train_snr_db: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one epoch with normalized latent, AWGN, decoder, and MSE loss."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            recon = model(images, snr_db=train_snr_db, add_noise=True)
            loss = criterion(recon, images)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += int(batch_size)

    if total_samples == 0:
        raise ValueError("The dataloader did not produce any samples.")
    return total_loss / total_samples


def write_history_csv(history: Iterable[Dict[str, float]], path: Path) -> None:
    rows = list(history)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    seed_everything(args.seed)
    device = get_device(args.device)
    tag = snr_tag(args.train_snr_db)
    out_dir = ensure_dir(Path(args.out_dir or f"outputs/jscc/snr_{tag}").expanduser())

    data_path = Path(args.data_path).expanduser()
    maybe_create_tiny_dataset(data_path, args.seed)

    images, labels = load_cifar_array_dataset(data_path)
    dataset = CIFARArrayDataset(images, labels)
    train_set, val_set, _ = make_splits(
        dataset,
        train=args.train_split,
        val=args.val_split,
        test=args.test_split,
        seed=args.seed,
    )
    train_set = cap_subset(train_set, args.max_train_samples)
    val_set = cap_subset(val_set, args.max_eval_samples)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_set, args.batch_size, True, args.seed, args.num_workers, pin_memory)
    val_loader = make_loader(val_set, args.batch_size, False, args.seed, args.num_workers, pin_memory)

    model = DeepJSCC(snr_db=args.train_snr_db).to(device)
    summary_text = jscc_model_summary(model, args.train_snr_db, device)
    print(summary_text)
    (out_dir / "model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    history = []
    best_path = out_dir / f"best_jscc_snr{tag}.pt"
    last_path = out_dir / f"last_jscc_snr{tag}.pt"
    extra = {
        "train_snr_db": float(args.train_snr_db),
        "noise_std": snr_db_to_noise_std(args.train_snr_db),
        "latent_shape_hwc": [8, 8, 16],
        "data_path": str(data_path),
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, args.train_snr_db, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, args.train_snr_db, optimizer=None)
        is_best = val_loss <= best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        history.append(row)
        print(
            f"SNR {float(args.train_snr_db):.4g} dB | epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
        )

        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss},
                extra=extra,
            )

        write_history_csv(history, out_dir / "history.csv")
        save_training_curves(
            {
                "train_loss": [item["train_loss"] for item in history],
                "val_loss": [item["val_loss"] for item in history],
            },
            out_dir / "loss_curve.png",
        )

    save_checkpoint(
        last_path,
        model,
        optimizer=optimizer,
        epoch=args.epochs,
        metrics=history[-1],
        extra=extra,
    )

    print(f"Saved Deep JSCC SNR {float(args.train_snr_db):.4g} dB outputs to {out_dir}")


if __name__ == "__main__":
    main()
