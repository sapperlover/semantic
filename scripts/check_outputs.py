#!/usr/bin/env python
"""Validate that a full non-innovation run produced the required artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jscc_lab.results import validate_outputs, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check required report outputs under an output directory.")
    parser.add_argument("--out_dir", default="outputs/full_run")
    parser.add_argument("--write_manifest", action="store_true", help="Update results_manifest.json after checking.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser()
    result = validate_outputs(out_dir)
    if args.write_manifest:
        manifest_path = write_manifest(out_dir)
        result["manifest_path"] = str(manifest_path)

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
