"""Result manifest and output validation helpers for full non-innovation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .utils import ensure_dir


REQUIRED_OUTPUTS = [
    "ae/model_summary.txt",
    "ae/latent_shape.txt",
    "ae/recon_10.png",
    "gaussian/gaussian_stats.npz",
    "gaussian/gaussian_stats_summary.txt",
    "gaussian/generated_from_gaussian_10.png",
    "perturb/perturb_recon_10.png",
    "jscc/task5_snr_psnr.csv",
    "jscc/task5_snr_psnr_curve.png",
    "task6/task6_noise_sweep.csv",
    "task6/task6_psnr_vs_snr.png",
    "task7/task7_rate_distortion.csv",
    "task7/task7_rate_distortion_curve.png",
]


def _as_path(out_dir: str | Path) -> Path:
    return Path(out_dir).expanduser()


def key_output_paths(out_dir: str | Path) -> Dict[str, object]:
    """Return key output paths used by the non-innovation full run."""

    base = _as_path(out_dir)
    heatmap_grid = base / "ae" / "latent_heatmaps_10.png"
    heatmaps = sorted((base / "ae").glob("latent_heatmap_img_*.png"))
    manifest = {
        "ae_model_summary": str(base / "ae" / "model_summary.txt"),
        "ae_latent_shape": str(base / "ae" / "latent_shape.txt"),
        "ae_reconstruction_grid": str(base / "ae" / "recon_10.png"),
        "ae_selected_indices": str(base / "ae" / "selected_indices.txt"),
        "ae_latent_heatmaps_grid": str(heatmap_grid) if heatmap_grid.exists() else None,
        "ae_latent_heatmaps": [str(path) for path in heatmaps],
        "gaussian_stats": str(base / "gaussian" / "gaussian_stats.npz"),
        "gaussian_summary": str(base / "gaussian" / "gaussian_stats_summary.txt"),
        "gaussian_overview": str(base / "gaussian" / "mean_var_overview.png"),
        "gaussian_generated": str(base / "gaussian" / "generated_from_gaussian_10.png"),
        "perturb_reconstruction_grid": str(base / "perturb" / "perturb_recon_10.png"),
        "task5_csv": str(base / "jscc" / "task5_snr_psnr.csv"),
        "task5_curve": str(base / "jscc" / "task5_snr_psnr_curve.png"),
        "task6_csv": str(base / "task6" / "task6_noise_sweep.csv"),
        "task6_curve": str(base / "task6" / "task6_psnr_vs_snr.png"),
        "task6_summary": str(base / "task6" / "task6_summary.txt"),
        "task7_csv": str(base / "task7" / "task7_rate_distortion.csv"),
        "task7_curve": str(base / "task7" / "task7_rate_distortion_curve.png"),
        "task7_summary": str(base / "task7" / "task7_summary.txt"),
    }
    for snr in [1, 4, 7, 13, 19]:
        manifest[f"jscc_snr_{snr}_best_checkpoint"] = str(base / "jscc" / f"snr_{snr}" / f"best_jscc_snr{snr}.pt")
        manifest[f"jscc_snr_{snr}_last_checkpoint"] = str(base / "jscc" / f"snr_{snr}" / f"last_jscc_snr{snr}.pt")
        manifest[f"jscc_snr_{snr}_history"] = str(base / "jscc" / f"snr_{snr}" / "history.csv")
        manifest[f"jscc_snr_{snr}_loss_curve"] = str(base / "jscc" / f"snr_{snr}" / "loss_curve.png")
    return manifest


def validate_outputs(out_dir: str | Path) -> Dict[str, object]:
    """Check required report artifacts under `out_dir`."""

    base = _as_path(out_dir)
    missing: List[str] = []
    present: List[str] = []

    for relative in REQUIRED_OUTPUTS:
        path = base / relative
        if path.exists():
            present.append(relative)
        else:
            missing.append(relative)

    heatmap_grid = base / "ae" / "latent_heatmaps_10.png"
    heatmap_files = sorted((base / "ae").glob("latent_heatmap_img_*.png"))
    if heatmap_grid.exists():
        present.append("ae/latent_heatmaps_10.png")
    elif heatmap_files:
        present.extend(str(path.relative_to(base)) for path in heatmap_files)
    else:
        missing.append("ae/latent_heatmaps_10.png or ae/latent_heatmap_img_*.png")

    return {
        "out_dir": str(base),
        "ok": not missing,
        "present": present,
        "missing": missing,
    }


def write_manifest(out_dir: str | Path, extra: Dict[str, object] | None = None) -> Path:
    """Write `results_manifest.json` with key paths and validation status."""

    base = _as_path(out_dir)
    payload: Dict[str, object] = {
        "out_dir": str(base),
        "key_outputs": key_output_paths(base),
        "validation": validate_outputs(base),
    }
    if extra:
        payload.update(extra)

    manifest_path = base / "results_manifest.json"
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return manifest_path
