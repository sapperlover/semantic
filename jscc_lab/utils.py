"""Small runtime helpers shared by experiment scripts."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch for reproducible experiment runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a CLI device string such as `auto`, `cpu`, or `cuda:0`."""

    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: Dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Write a JSON file with stable formatting."""

    output_path = Path(path).expanduser()
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=True)
        f.write("\n")
    return output_path


class TeeLogger:
    """A tiny file-and-console logger compatible with `print(..., file=logger)`."""

    def __init__(self, path: str | Path, mode: str = "a", stream=None):
        self.path = Path(path).expanduser()
        ensure_dir(self.path.parent)
        self.stream = sys.stdout if stream is None else stream
        self.file = self.path.open(mode, encoding="utf-8")

    def write(self, message: str) -> int:
        self.stream.write(message)
        written = self.file.write(message)
        self.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def log(self, message: str) -> None:
        self.write(f"{message}\n")

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TeeLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint with model, optimizer, and metadata."""

    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    output_path = Path(path).expanduser()
    ensure_dir(output_path.parent)
    torch.save(checkpoint, output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model and optimizer state."""

    checkpoint = torch.load(Path(path).expanduser(), map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
