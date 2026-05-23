#!/usr/bin/env python
"""Run all required non-innovation AE / Deep JSCC homework outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jscc_lab.results import validate_outputs, write_manifest
from jscc_lab.utils import ensure_dir


SNR_VALUES = [1, 4, 7, 13, 19]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all baseline outputs required by the AE / Deep JSCC homework.")
    parser.add_argument("--data_path", required=True, help="Path to .npz, CIFAR batch, or CIFAR directory.")
    parser.add_argument("--out_dir", default="outputs/full_run")
    parser.add_argument("--ae_epochs", type=int, default=100)
    parser.add_argument("--jscc_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="Use tiny sample counts and 1 epoch to verify the whole pipeline.")
    return parser


def run_command(cmd: List[str], step_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"[run_all] {step_name}")
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_if_present(cmd: List[str], name: str, value) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def main() -> None:
    args = build_parser().parse_args()

    data_path = Path(args.data_path).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())
    ae_dir = out_dir / "ae"
    gaussian_dir = out_dir / "gaussian"
    perturb_dir = out_dir / "perturb"
    jscc_dir = out_dir / "jscc"
    task6_dir = out_dir / "task6"
    task7_dir = out_dir / "task7"
    for directory in [ae_dir, gaussian_dir, perturb_dir, jscc_dir, task6_dir, task7_dir]:
        ensure_dir(directory)

    ae_epochs = 1 if args.quick else args.ae_epochs
    jscc_epochs = 1 if args.quick else args.jscc_epochs
    batch_size = min(args.batch_size, 16) if args.quick else args.batch_size
    max_train_samples = 128 if args.quick else None
    max_eval_samples = 64 if args.quick else None
    task2_images = 10
    gaussian_stats_samples = 16 if args.quick else 256
    task6_images = 16 if args.quick else 500
    task7_images = 16 if args.quick else 500

    common = [
        "--data_path",
        str(data_path),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]

    commands: List[Dict[str, object]] = []

    cmd = [
        sys.executable,
        "scripts/train_ae.py",
        *common,
        "--out_dir",
        str(ae_dir),
        "--epochs",
        str(ae_epochs),
        "--batch_size",
        str(batch_size),
    ]
    add_if_present(cmd, "--max_train_samples", max_train_samples)
    add_if_present(cmd, "--max_eval_samples", max_eval_samples)
    commands.append({"step": "Task (1): train AE", "cmd": cmd})

    commands.append(
        {
            "step": "Task (2): AE reconstructions and latent heatmaps",
            "cmd": [
                sys.executable,
                "scripts/visualize_ae_latent.py",
                *common,
                "--checkpoint",
                str(ae_dir / "best_ae.pt"),
                "--out_dir",
                str(ae_dir),
                "--num_images",
                str(task2_images),
            ],
        }
    )

    commands.append(
        {
            "step": "Task (3): Gaussian latent statistics and sampling",
            "cmd": [
                sys.executable,
                "scripts/gaussian_latent_sampling.py",
                *common,
                "--checkpoint",
                str(ae_dir / "best_ae.pt"),
                "--out_dir",
                str(gaussian_dir),
                "--num_stats_samples",
                str(gaussian_stats_samples),
            ],
        }
    )

    commands.append(
        {
            "step": "Task (4): latent perturbation",
            "cmd": [
                sys.executable,
                "scripts/perturb_latent.py",
                *common,
                "--checkpoint",
                str(ae_dir / "best_ae.pt"),
                "--out_dir",
                str(perturb_dir),
                "--selected_indices",
                str(ae_dir / "selected_indices.txt"),
                "--noise_std",
                "0.1",
            ],
        }
    )

    for snr in SNR_VALUES:
        cmd = [
            sys.executable,
            "scripts/train_jscc.py",
            *common,
            "--train_snr_db",
            str(snr),
            "--out_dir",
            str(jscc_dir / f"snr_{snr}"),
            "--epochs",
            str(jscc_epochs),
            "--batch_size",
            str(batch_size),
        ]
        add_if_present(cmd, "--max_train_samples", max_train_samples)
        add_if_present(cmd, "--max_eval_samples", max_eval_samples)
        commands.append({"step": f"Task (5): train Deep JSCC SNR={snr} dB", "cmd": cmd})

    cmd = [
        sys.executable,
        "scripts/eval_jscc_train_snr_models.py",
        *common,
        "--ckpt_dir",
        str(jscc_dir),
        "--out_dir",
        str(jscc_dir),
        "--batch_size",
        str(batch_size),
    ]
    add_if_present(cmd, "--max_eval_samples", max_eval_samples)
    commands.append({"step": "Task (5): evaluate five JSCC models", "cmd": cmd})

    commands.append(
        {
            "step": "Task (6): SNR=7 model noise sweep",
            "cmd": [
                sys.executable,
                "scripts/eval_noise_sweep.py",
                *common,
                "--ckpt",
                str(jscc_dir / "snr_7" / "best_jscc_snr7.pt"),
                "--out_dir",
                str(task6_dir),
                "--num_images",
                str(task6_images),
                "--batch_size",
                str(batch_size),
            ],
        }
    )

    commands.append(
        {
            "step": "Task (7): prefix-Kp rate distortion",
            "cmd": [
                sys.executable,
                "scripts/eval_rate_distortion.py",
                *common,
                "--ckpt",
                str(jscc_dir / "snr_7" / "best_jscc_snr7.pt"),
                "--out_dir",
                str(task7_dir),
                "--num_images",
                str(task7_images),
                "--batch_size",
                str(batch_size),
            ],
        }
    )

    for item in commands:
        run_command(item["cmd"], item["step"])

    validation = validate_outputs(out_dir)
    manifest_path = write_manifest(
        out_dir,
        extra={
            "quick": bool(args.quick),
            "data_path": str(data_path),
            "seed": args.seed,
            "device": args.device,
            "ae_epochs": ae_epochs,
            "jscc_epochs": jscc_epochs,
            "batch_size": batch_size,
            "commands": commands,
        },
    )

    print("\n" + "=" * 80)
    print(f"[run_all] Wrote manifest: {manifest_path}")
    if validation["ok"]:
        print("[run_all] Output validation passed.")
    else:
        print("[run_all] Output validation failed. Missing:")
        for missing in validation["missing"]:
            print(f"  - {missing}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
