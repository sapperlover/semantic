#!/usr/bin/env python
"""Plot multiple task7 rate-distortion CSV files on one PSNR-vs-Kp figure."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_series(items: List[str], project_dir: Path) -> List[Tuple[str, Path]]:
    series: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Series must be LABEL=CSV_PATH, got {item!r}.")
        label, path_text = item.split("=", 1)
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"CSV file does not exist: {path}")
        series.append((label.strip(), path))
    if not series:
        raise ValueError("At least one --series LABEL=CSV_PATH is required.")
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine task7 RD CSV curves into one figure.")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab")
    parser.add_argument("--series", action="append", required=True, help="Curve as LABEL=CSV_PATH. Can be repeated.")
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--title", default="Task7 RD Comparison")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    series = parse_series(args.series, project_dir)
    out_png = Path(args.output_png).expanduser()
    if not out_png.is_absolute():
        out_png = project_dir / out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = None
    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser()
        if not out_csv.is_absolute():
            out_csv = project_dir / out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#2563eb", "#64748b", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    markers = ["o", "s", "^", "D", "v", "P"]
    combined_rows = []

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=220)
    for index, (label, path) in enumerate(series):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        kp = [int(row["Kp"]) for row in rows]
        psnr = [float(row["avg_psnr"]) for row in rows]
        mse = [float(row["avg_mse"]) for row in rows]
        ax.plot(
            kp,
            psnr,
            marker=markers[index % len(markers)],
            linewidth=1.9,
            markersize=4.5,
            color=colors[index % len(colors)],
            label=label,
        )
        for row, k, p, m in zip(rows, kp, psnr, mse):
            combined_rows.append(
                {
                    "series": label,
                    "source_csv": str(path.relative_to(project_dir)) if path.is_relative_to(project_dir) else str(path),
                    "Kp": k,
                    "R": float(row["R"]),
                    "avg_mse": m,
                    "avg_psnr": p,
                }
            )

    ax.set_xlabel("Kp kept latent elements")
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_xticks([128, 256, 384, 512, 640, 768, 896, 1024])
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.5)
    ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    if out_csv is not None:
        fieldnames = ["series", "source_csv", "Kp", "R", "avg_mse", "avg_psnr"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    print(f"Saved {out_png}")
    if out_csv is not None:
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
