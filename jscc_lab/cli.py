"""Shared command-line helpers for future experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML experiment config into a Python dictionary."""

    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_common_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Create a parser with the flags shared by all lab scripts."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--data_path", default=None, help="Path to CIFAR data; overrides config data.data_path.")
    parser.add_argument("--out_dir", default=None, help="Override output directory.")
    return parser


def apply_common_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply common CLI overrides while preserving the nested config layout."""

    updated = dict(config)
    updated.setdefault("data", {})
    updated.setdefault("output", {})

    if args.seed is not None:
        updated["seed"] = args.seed
    if args.device is not None:
        updated["device"] = args.device
    if args.data_path is not None:
        updated["data"]["data_path"] = args.data_path
    if args.out_dir is not None:
        updated["output"]["out_dir"] = args.out_dir
    return updated
