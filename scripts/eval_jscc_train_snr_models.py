#!/usr/bin/env python
"""Evaluate task (5) Deep JSCC models trained at multiple SNR values."""

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
from torch.utils.data import DataLoader, Subset

from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, seed_everything


def snr_tag(snr_db: float) -> str:
    value = float(snr_db)
    if value.is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def parse_snr_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained Deep JSCC SNR sweep models.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt_dir", default="outputs/jscc", help="Directory containing snr_{snr}/ subdirectories.")
    parser.add_argument("--out_dir", default="outputs/jscc")
    parser.add_argument("--snr_list", default="1,4,7,13,19", help="Comma-separated training SNR values.")
    parser.add_argument("--eval_snr_db", type=float, default=None, help="Optional fixed eval SNR for all models.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def cap_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def resolve_checkpoint(ckpt_dir: Path, train_snr_db: float) -> Path:
    tag = snr_tag(train_snr_db)
    best = ckpt_dir / f"snr_{tag}" / f"best_jscc_snr{tag}.pt"
    last = ckpt_dir / f"snr_{tag}" / f"last_jscc_snr{tag}.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"Could not find checkpoint for SNR {train_snr_db}: {best} or {last}")


def load_model(ckpt_path: Path, train_snr_db: float, device: torch.device) -> DeepJSCC:
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = DeepJSCC(snr_db=train_snr_db)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model: DeepJSCC, dataloader: DataLoader, device: torch.device, eval_snr_db: float) -> Dict[str, float]:
    """Evaluate reconstruction under AWGN at `eval_snr_db`."""

    total_mse = 0.0
    total_psnr = 0.0
    total = 0
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        recon = model(images, snr_db=eval_snr_db, add_noise=True).clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        count = images.shape[0]
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total += int(count)
    if total == 0:
        raise ValueError("The evaluation dataloader did not produce any samples.")
    return {"avg_mse": total_mse / total, "avg_psnr": total_psnr / total, "num_samples": total}


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["train_snr_db", "eval_snr_db", "avg_mse", "avg_psnr", "ckpt_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_curve(rows: List[Dict[str, object]], out_path: Path) -> None:
    train_snrs = [float(row["train_snr_db"]) for row in rows]
    psnrs = [float(row["avg_psnr"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(train_snrs, psnrs, marker="o", linewidth=1.8)
    ax.set_xlabel("Training SNR (dB)")
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_xticks(train_snrs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    train_snrs = parse_snr_list(args.snr_list)

    data_path = Path(args.data_path).expanduser()
    images, labels = load_cifar_array_dataset(data_path)
    dataset = CIFARArrayDataset(images, labels)
    _, _, test_set = make_splits(
        dataset,
        train=args.train_split,
        val=args.val_split,
        test=args.test_split,
        seed=args.seed,
    )
    test_set = cap_subset(test_set, args.max_eval_samples)
    dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: List[Dict[str, object]] = []
    for train_snr in train_snrs:
        eval_snr = float(train_snr if args.eval_snr_db is None else args.eval_snr_db)
        ckpt_path = resolve_checkpoint(ckpt_dir, train_snr)
        model = load_model(ckpt_path, train_snr, device)
        metrics = evaluate_model(model, dataloader, device, eval_snr)
        row = {
            "train_snr_db": float(train_snr),
            "eval_snr_db": float(eval_snr),
            "avg_mse": metrics["avg_mse"],
            "avg_psnr": metrics["avg_psnr"],
            "ckpt_path": str(ckpt_path),
        }
        rows.append(row)
        print(
            f"train_snr={train_snr:g} dB eval_snr={eval_snr:g} dB "
            f"avg_mse={metrics['avg_mse']:.6f} avg_psnr={metrics['avg_psnr']:.3f}"
        )

    csv_path = out_dir / "task5_snr_psnr.csv"
    curve_path = out_dir / "task5_snr_psnr_curve.png"
    write_csv(rows, csv_path)
    save_curve(rows, curve_path)
    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")


if __name__ == "__main__":
    main()
