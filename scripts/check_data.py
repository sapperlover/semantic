#!/usr/bin/env python
"""Quick CLI check for the CIFAR array loader and split pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jscc_lab.cli import apply_common_overrides, build_common_parser, load_config
from jscc_lab.data import load_cifar_array_dataset, make_dataloaders
from jscc_lab.plotting import save_image_grid
from jscc_lab.utils import ensure_dir, get_device, save_json, seed_everything


def main() -> None:
    parser = build_common_parser("Check CIFAR-10 array loading and save a sample grid.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override loader batch size.")
    args = parser.parse_args()

    config = apply_common_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = get_device(config.get("device", "auto"))

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    data_path = data_cfg.get("data_path")
    if data_path is None:
        raise ValueError("--data_path is required because no data.data_path is set in the config.")
    data_path = Path(data_path).expanduser()
    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs")).expanduser())
    batch_size = args.batch_size or int(data_cfg.get("batch_size", 128))
    splits = data_cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})

    images, labels = load_cifar_array_dataset(data_path)
    loaders = make_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        train=splits.get("train", 0.8),
        val=splits.get("val", 0.1),
        test=splits.get("test", 0.1),
        seed=seed,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )

    batch_images, batch_labels = next(iter(loaders["train"]))
    grid_path = save_image_grid(batch_images[:32], out_dir / "check_data_grid.png", nrow=8)
    summary = {
        "device": str(device),
        "data_path": str(data_path),
        "num_samples": int(len(images)),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "train_batches": len(loaders["train"]),
        "val_batches": len(loaders["val"]),
        "test_batches": len(loaders["test"]),
        "sample_labels": [int(x) for x in batch_labels[:16].tolist()],
        "grid_path": str(grid_path),
    }
    summary_path = save_json(summary, out_dir / "check_data_summary.json")
    print(f"Loaded {summary['num_samples']} samples on {device}.")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
