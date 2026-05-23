#!/usr/bin/env python
"""Train the baseline autoencoder for homework task (1)."""

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

from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.models import AutoEncoder, model_summary
from jscc_lab.plotting import save_training_curves
from jscc_lab.utils import ensure_dir, get_device, save_checkpoint, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 AE baseline.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or directory.")
    parser.add_argument("--out_dir", default="outputs/ae", help="Directory for AE outputs.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for train samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for validation samples.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio.")
    return parser


def maybe_create_tiny_dataset(data_path: Path, seed: int) -> None:
    """Create the documented tiny fixture when the quick-test path is requested."""

    if data_path.exists() or data_path.name != "tiny_dataset.npz":
        return
    ensure_dir(data_path.parent)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(256, 3072), dtype=np.uint8)
    labels = np.arange(256, dtype=np.int64) % 10
    np.savez(data_path, data=raw, labels=labels)
    print(f"Created tiny dataset fixture: {data_path}")


def cap_subset(dataset, max_samples: int | None):
    """Limit a split for quick debug runs without changing the base dataset."""

    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one train or validation epoch and return average MSE loss."""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            recon = model(images)
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
    ensure_dir(path.parent)
    rows = list(history)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def write_latent_shape(path: Path, model: AutoEncoder, sample_batch: torch.Tensor, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        latent = model.encode(sample_batch.to(device))
        recon = model.decode(latent)

    text = "\n".join(
        [
            f"Input batch shape: {tuple(sample_batch.shape)}",
            f"PyTorch latent tensor shape (N,C,H,W): {tuple(latent.shape)}",
            "Per-image latent code shape for the report (H x W x C): 8 x 8 x 16",
            f"Flattened latent elements per image K: {latent.shape[1] * latent.shape[2] * latent.shape[3]}",
            f"Decoder output shape: {tuple(recon.shape)}",
            "Decoder output pixel range: [0, 1] by final Sigmoid",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(Path(args.out_dir).expanduser())

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

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, seed=args.seed, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, seed=args.seed, num_workers=args.num_workers)

    model = AutoEncoder().to(device)
    summary_text = model_summary(model, device=device)
    print(summary_text)
    (out_dir / "model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    first_batch, _ = next(iter(train_loader))
    write_latent_shape(out_dir / "latent_shape.txt", model, first_batch, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        best_val_loss = min(best_val_loss, val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
        )

        if val_loss <= best_val_loss:
            save_checkpoint(
                out_dir / "best_ae.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
                extra={"latent_shape_hwc": [8, 8, 16], "data_path": str(data_path)},
            )

        write_history_csv(history, out_dir / "ae_history.csv")
        save_training_curves(
            {
                "train_loss": [item["train_loss"] for item in history],
                "val_loss": [item["val_loss"] for item in history],
            },
            out_dir / "ae_train_loss.png",
        )

    save_checkpoint(
        out_dir / "last_ae.pt",
        model,
        optimizer=optimizer,
        epoch=args.epochs,
        metrics=history[-1],
        extra={"latent_shape_hwc": [8, 8, 16], "data_path": str(data_path)},
    )

    print(f"Saved AE outputs to {out_dir}")


if __name__ == "__main__":
    main()
