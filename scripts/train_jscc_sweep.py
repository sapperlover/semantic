#!/usr/bin/env python
"""Train the five Deep JSCC models required by homework task (5)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jscc_lab.utils import ensure_dir


def parse_snr_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--snr_list must contain at least one SNR value.")
    return values


def snr_tag(snr_db: float) -> str:
    value = float(snr_db)
    if value.is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train all task (5) Deep JSCC SNR models.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or CIFAR directory.")
    parser.add_argument("--out_dir", default="outputs/jscc", help="Root output directory for task (5).")
    parser.add_argument("--snr_list", default="1,4,7,13,19", help="Comma-separated training SNR values.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_eval", action="store_true", help="Only train models; do not create task5 CSV/curve.")
    return parser


def add_optional(cmd: List[str], name: str, value) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def run_command(cmd: List[str], step_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"[task5] {step_name}")
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data_path).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    snr_values = parse_snr_list(args.snr_list)

    for snr in snr_values:
        tag = snr_tag(snr)
        snr_dir = out_dir / f"snr_{tag}"
        cmd = [
            sys.executable,
            "scripts/train_jscc.py",
            "--data_path",
            str(data_path),
            "--train_snr_db",
            str(snr),
            "--out_dir",
            str(snr_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        add_optional(cmd, "--max_train_samples", args.max_train_samples)
        add_optional(cmd, "--max_eval_samples", args.max_eval_samples)
        run_command(cmd, f"Train Deep JSCC model at SNR={snr:g} dB")

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "scripts/eval_jscc_train_snr_models.py",
            "--data_path",
            str(data_path),
            "--ckpt_dir",
            str(out_dir),
            "--out_dir",
            str(out_dir),
            "--snr_list",
            ",".join(str(int(snr)) if float(snr).is_integer() else str(snr) for snr in snr_values),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        add_optional(eval_cmd, "--max_eval_samples", args.max_eval_samples)
        run_command(eval_cmd, "Evaluate task (5) SNR-PSNR curve")

    print(f"\nTask (5) outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
