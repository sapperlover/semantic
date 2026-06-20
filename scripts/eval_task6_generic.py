#!/usr/bin/env python
"""Generic task (6) noise sweep for DeepJSCC-style checkpoints.

The original task (6) evaluates the SNR=7 DeepJSCC model at several test SNRs,
reporting MSE, PSNR, encoder time, decoder time, and an SNR-vs-PSNR curve. This
generic version keeps the same protocol while supporting the project variants
introduced in the innovation experiments:

  - plain DeepJSCC checkpoints from scripts/train_jscc.py
  - rate_conditioned_deepjscc checkpoints from train_mask_aware_random_kp.py
  - decoder_film, encoder_decoder_film, complex_decoder_film, and
    snr_rate_complex_film checkpoints from train_film_decoder_multirate.py

For rate-conditioned and FiLM models, task (6) is evaluated at the full-rate
condition by default: r=1.0, cond=[1.0, log2(1.0)=0.0], and no Kp mask.

Example:
  python scripts/eval_task6_generic.py \
    --data_path /home/lc/class/yuyi/cifar-10 \
    --checkpoint outputs/film_decoder_multirate_b8_snr7/best_film_decoder_multirate_b8_snr7.pt \
    --output_dir outputs/task6 \
    --run_name film_decoder_b8_snr7 \
    --test_snr_list 1,4,7,13,19 \
    --num_images 500 \
    --batch_size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jscc_lab.analysis import load_test_split, sample_test_items, save_selected_indices, read_selected_indices
from jscc_lab.channel import awgn, power_normalize, snr_db_to_noise_std
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import DeepJSCC
from jscc_lab.utils import ensure_dir, get_device, save_json, seed_everything
from train_film_decoder_multirate import (
    ComplexFiLMDeepJSCC,
    FiLMDecoderDeepJSCC,
    FiLMEncoderDecoderDeepJSCC,
    SNRRateFiLMDeepJSCC,
    MODEL_TYPE as FILM_MODEL_TYPE,
    MODEL_VARIANT_COMPLEX_DECODER,
    MODEL_VARIANT_DECODER,
    MODEL_VARIANT_ENCODER_DECODER,
    MODEL_VARIANT_SNR_RATE_COMPLEX,
    cond_from_kp,
)
from train_mask_aware_random_kp import RateConditionedDeepJSCC, load_state_with_rate_conditioning


RATE_CONDITIONED_MODEL_TYPE = "rate_conditioned_deepjscc"
FULL_KP = 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic task6 SNR sweep for DeepJSCC-style checkpoints.")
    parser.add_argument("--data_path", "--data_dir", dest="data_path", default="/home/lc/class/yuyi/cifar-10")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default="outputs/task6")
    parser.add_argument("--run_name", default="auto", help="Subdirectory name under output_dir; use auto to derive from checkpoint.")
    parser.add_argument("--test_snr_list", default="1,4,7,13,19")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--selected_indices", default="outputs/task6/task6_selected_indices.txt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument(
        "--model_type",
        choices=["auto", "deepjscc", RATE_CONDITIONED_MODEL_TYPE, FILM_MODEL_TYPE],
        default="auto",
        help="Normally inferred from checkpoint extra.model_type.",
    )
    parser.add_argument(
        "--model_variant",
        choices=["auto", MODEL_VARIANT_DECODER, MODEL_VARIANT_ENCODER_DECODER, MODEL_VARIANT_COMPLEX_DECODER, MODEL_VARIANT_SNR_RATE_COMPLEX],
        default="auto",
        help="Used for film_decoder_multirate checkpoints; normally inferred.",
    )
    parser.add_argument("--rate_condition", type=float, default=1.0, help="Full-rate condition r used by rate/FILM models.")
    parser.add_argument("--cond_dim", type=int, choices=[1, 2, 3], default=None, help="Override FiLM cond_dim; default reads checkpoint extra.")
    parser.add_argument("--film_hidden_dim", type=int, default=None, help="Override FiLM hidden dim; default reads checkpoint extra.")
    return parser


def resolve_path(path_text: str | Path | None, project_dir: Path) -> Path | None:
    if path_text is None:
        return None
    text = str(path_text).strip()
    if not text or text.lower() in {"none", "null", "false"}:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return project_dir / path


def parse_snr_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("--test_snr_list must contain at least one value.")
    return values


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    name = name.strip("._-")
    return name or "task6_model"


def default_run_name(checkpoint_path: Path, metadata: Dict[str, object]) -> str:
    model_type = str(metadata.get("model_type", "model"))
    variant = str(metadata.get("model_variant", "") or "")
    parent = checkpoint_path.parent.name
    stem = checkpoint_path.stem
    pieces = [piece for piece in [model_type, variant, parent, stem] if piece and piece != "auto"]
    return safe_name("_".join(pieces))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint_payload(checkpoint_path: Path) -> Tuple[Dict[str, object] | torch.Tensor, Dict[str, object], Dict[str, torch.Tensor]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        extra = dict(checkpoint.get("extra", {}) or {})
        state_dict = checkpoint["model_state"]
        return checkpoint, extra, state_dict
    if isinstance(checkpoint, dict):
        return checkpoint, {}, checkpoint
    raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}.")


def infer_model_type(extra: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        return requested
    return str(extra.get("model_type", "deepjscc"))


def infer_film_variant(extra: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        return requested
    return str(extra.get("model_variant", MODEL_VARIANT_DECODER))


def load_generic_model(
    checkpoint_path: Path,
    device: torch.device,
    requested_model_type: str,
    requested_model_variant: str,
    cond_dim_override: int | None,
    film_hidden_dim_override: int | None,
) -> Tuple[nn.Module, Dict[str, object]]:
    _, extra, state_dict = load_checkpoint_payload(checkpoint_path)
    model_type = infer_model_type(extra, requested_model_type)
    train_snr_db = float(extra.get("train_snr_db", extra.get("snr_db", 7.0)))
    metadata: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "model_type": model_type,
        "train_snr_db": train_snr_db,
        "checkpoint_extra": extra,
    }

    if model_type == "deepjscc":
        model = DeepJSCC(snr_db=train_snr_db)
        model.load_state_dict(state_dict)
        metadata["model_variant"] = "plain"
    elif model_type == RATE_CONDITIONED_MODEL_TYPE:
        model = RateConditionedDeepJSCC(snr_db=train_snr_db)
        load_info = load_state_with_rate_conditioning(model, state_dict, strict_rate_conditioned=True)
        metadata["model_variant"] = RATE_CONDITIONED_MODEL_TYPE
        metadata["load_info"] = load_info
    elif model_type == FILM_MODEL_TYPE:
        variant = infer_film_variant(extra, requested_model_variant)
        cond_dim = int(cond_dim_override if cond_dim_override is not None else extra.get("cond_dim", 2))
        film_hidden_dim = int(film_hidden_dim_override if film_hidden_dim_override is not None else extra.get("film_hidden_dim", 64))
        if variant == MODEL_VARIANT_ENCODER_DECODER:
            model = FiLMEncoderDecoderDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        elif variant == MODEL_VARIANT_COMPLEX_DECODER:
            model = ComplexFiLMDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        elif variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
            cond_dim = 3
            model = SNRRateFiLMDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        elif variant == MODEL_VARIANT_DECODER:
            model = FiLMDecoderDeepJSCC(snr_db=train_snr_db, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        else:
            raise ValueError(f"Unsupported FiLM model_variant={variant!r}.")
        model.load_state_dict(state_dict)
        metadata["model_variant"] = variant
        metadata["cond_dim"] = cond_dim
        metadata["film_hidden_dim"] = film_hidden_dim
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Use --model_type deepjscc, "
            f"{RATE_CONDITIONED_MODEL_TYPE}, or {FILM_MODEL_TYPE}."
        )

    model.to(device)
    model.eval()
    return model, metadata


def rate_tensor(rate: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((batch_size, 1), float(rate), device=device, dtype=dtype)


def film_cond(rate: float, batch_size: int, device: torch.device, dtype: torch.dtype, cond_dim: int, snr_db: float | None = None) -> torch.Tensor:
    # Reuse the same helper as training by converting r back to equivalent Kp.
    kp = int(round(float(rate) * FULL_KP))
    if kp <= 0:
        if cond_dim == 1:
            values = [float(rate)]
        elif cond_dim == 2:
            values = [float(rate), math.log2(max(float(rate), 1e-8))]
        elif cond_dim == 3:
            if snr_db is None:
                raise ValueError("cond_dim=3 requires snr_db for [r, log2(r), snr_db / 20].")
            values = [float(rate), math.log2(max(float(rate), 1e-8)), float(snr_db) / 20.0]
        else:
            raise ValueError(f"Unsupported cond_dim={cond_dim}.")
        return torch.tensor(values, dtype=dtype, device=device).view(1, cond_dim).expand(batch_size, cond_dim)
    return cond_from_kp(kp, batch_size, device, dtype, cond_dim=cond_dim, snr_db=snr_db)


@torch.no_grad()
def encode_normalized(model: nn.Module, images: torch.Tensor, metadata: Dict[str, object], rate: float, test_snr_db: float) -> torch.Tensor:
    model_type = str(metadata.get("model_type"))
    if model_type == "deepjscc":
        return power_normalize(model.encoder(images))
    if model_type == RATE_CONDITIONED_MODEL_TYPE:
        return power_normalize(model.encode(images, rate_tensor(rate, images.shape[0], images.device, images.dtype)))
    if model_type == FILM_MODEL_TYPE:
        cond_dim = int(metadata.get("cond_dim", 2))
        if str(metadata.get("model_variant")) == MODEL_VARIANT_ENCODER_DECODER:
            cond = film_cond(rate, images.shape[0], images.device, images.dtype, cond_dim, snr_db=test_snr_db)
            return power_normalize(model.encode(images, cond))
        return power_normalize(model.encode(images))
    raise ValueError(f"Unsupported model_type={model_type!r}.")


@torch.no_grad()
def decode_latent(model: nn.Module, latent: torch.Tensor, metadata: Dict[str, object], rate: float, test_snr_db: float) -> torch.Tensor:
    model_type = str(metadata.get("model_type"))
    if model_type == "deepjscc":
        return model.decoder(latent)
    if model_type == RATE_CONDITIONED_MODEL_TYPE:
        return model.decode(latent, rate_tensor(rate, latent.shape[0], latent.device, latent.dtype))
    if model_type == FILM_MODEL_TYPE:
        cond_dim = int(metadata.get("cond_dim", 2))
        cond = film_cond(rate, latent.shape[0], latent.device, latent.dtype, cond_dim, snr_db=test_snr_db)
        return model.decode(latent, cond)
    raise ValueError(f"Unsupported model_type={model_type!r}.")


@torch.no_grad()
def warmup_model(
    model: nn.Module,
    metadata: Dict[str, object],
    dataloader: DataLoader,
    device: torch.device,
    test_snr_db: float,
    rate: float,
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0:
        return
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        z = encode_normalized(model, images, metadata, rate, test_snr_db)
        z_noisy = awgn(z, test_snr_db, training=True)
        _ = decode_latent(model, z_noisy, metadata, rate, test_snr_db)
    sync_if_cuda(device)


@torch.no_grad()
def evaluate_one_snr(
    model: nn.Module,
    metadata: Dict[str, object],
    dataloader: DataLoader,
    device: torch.device,
    test_snr_db: float,
    rate: float,
) -> Dict[str, float]:
    """Evaluate metrics and model-only forward timing for one test SNR."""

    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0
    encoder_seconds = 0.0
    decoder_seconds = 0.0

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        batch_size = int(images.shape[0])

        sync_if_cuda(device)
        start = time.perf_counter()
        z = encode_normalized(model, images, metadata, rate, test_snr_db)
        sync_if_cuda(device)
        encoder_seconds += time.perf_counter() - start

        z_noisy = awgn(z, test_snr_db, training=True)

        sync_if_cuda(device)
        start = time.perf_counter()
        recon = decode_latent(model, z_noisy, metadata, rate, test_snr_db)
        sync_if_cuda(device)
        decoder_seconds += time.perf_counter() - start

        recon = recon.clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += batch_size

    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {
        "test_snr_db": float(test_snr_db),
        "noise_std": snr_db_to_noise_std(float(test_snr_db)),
        "rate_condition": float(rate),
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
        "encoder_ms_per_image": 1000.0 * encoder_seconds / total_images,
        "decoder_ms_per_image": 1000.0 * decoder_seconds / total_images,
    }


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
        "test_snr_db",
        "noise_std",
        "rate_condition",
        "avg_mse",
        "avg_psnr",
        "encoder_ms_per_image",
        "decoder_ms_per_image",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_psnr_curve(rows: List[Dict[str, float]], out_path: Path, title: str) -> None:
    snrs = [row["test_snr_db"] for row in rows]
    psnrs = [row["avg_psnr"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(snrs, psnrs, marker="o", linewidth=1.8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_xticks(snrs)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(
    out_path: Path,
    rows: List[Dict[str, float]],
    num_images: int,
    checkpoint_path: Path,
    metadata: Dict[str, object],
    selected_indices_path: Path,
    output_dir: Path,
) -> None:
    lines = [
        "Generic task (6) noise sweep summary",
        "=" * 43,
        f"Checkpoint: {checkpoint_path}",
        f"Model type: {metadata.get('model_type')}",
        f"Model variant: {metadata.get('model_variant')}",
        f"Training SNR from checkpoint: {metadata.get('train_snr_db')} dB",
        f"Rate condition used for conditioned models: {rows[0]['rate_condition'] if rows else 'n/a'}",
        f"Shared sampled test images: {num_images}",
        f"Selected indices file: {selected_indices_path}",
        f"Output directory: {output_dir}",
        "",
        "Task (6) interpretation template:",
        "当测试 SNR 低于训练 SNR 时，信道噪声更强，重建质量和 PSNR 通常下降。",
        "当测试 SNR 高于训练 SNR 时，信道噪声更弱，重建质量通常提高，但提升可能逐渐饱和。",
        "",
        "Measured rows:",
    ]
    for row in rows:
        lines.append(
            f"SNR={row['test_snr_db']:g} dB | noise_std={row['noise_std']:.6f} | "
            f"avg_mse={row['avg_mse']:.6f} | avg_psnr={row['avg_psnr']:.3f} dB | "
            f"encoder={row['encoder_ms_per_image']:.4f} ms/image | "
            f"decoder={row['decoder_ms_per_image']:.4f} ms/image"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    project_dir = Path(args.project_dir).expanduser().resolve()
    data_path = resolve_path(args.data_path, project_dir)
    checkpoint_path = resolve_path(args.checkpoint, project_dir)
    output_root = ensure_dir(resolve_path(args.output_dir, project_dir))
    selected_indices_arg = resolve_path(args.selected_indices, project_dir)
    if data_path is None or checkpoint_path is None:
        raise ValueError("--data_path and --checkpoint are required.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = get_device(args.device)
    test_snrs = parse_snr_list(args.test_snr_list)
    num_images = int(args.max_eval_samples if args.max_eval_samples is not None else args.num_images)
    if num_images <= 0:
        raise ValueError("--num_images/--max_eval_samples must be positive.")

    model, metadata = load_generic_model(
        checkpoint_path,
        device,
        requested_model_type=args.model_type,
        requested_model_variant=args.model_variant,
        cond_dim_override=args.cond_dim,
        film_hidden_dim_override=args.film_hidden_dim,
    )
    run_name = default_run_name(checkpoint_path, metadata) if args.run_name == "auto" else safe_name(args.run_name)
    out_dir = ensure_dir(output_root / run_name)

    _, test_set = load_test_split(
        data_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    selected_original_indices = None
    if selected_indices_arg is not None and selected_indices_arg.is_file():
        selected_original_indices = read_selected_indices(selected_indices_arg)
    samples = sample_test_items(test_set, num_images, seed=args.seed, selected_original_indices=selected_original_indices)
    if len(samples.images) < num_images:
        print(f"Requested {num_images} images, but test split has {len(samples.images)}. Using all available samples.")
    selected_indices_path = save_selected_indices(out_dir / "task6_selected_indices.txt", samples)

    dataloader = DataLoader(
        TensorDataset(samples.images, samples.labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: List[Dict[str, float]] = []
    for snr in test_snrs:
        # Keep AWGN draws reproducible while all SNR values share the same images.
        snr_seed = args.seed + int(round(float(snr) * 1000))
        torch.manual_seed(snr_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(snr_seed)
        warmup_model(model, metadata, dataloader, device, snr, args.rate_condition, args.warmup_batches)
        metrics = evaluate_one_snr(model, metadata, dataloader, device, snr, args.rate_condition)
        rows.append(metrics)
        print(
            f"SNR={snr:g} dB avg_mse={metrics['avg_mse']:.6f} avg_psnr={metrics['avg_psnr']:.3f} "
            f"encoder={metrics['encoder_ms_per_image']:.4f} ms/image decoder={metrics['decoder_ms_per_image']:.4f} ms/image"
        )

    csv_path = out_dir / "task6_noise_sweep.csv"
    curve_path = out_dir / "task6_psnr_vs_snr.png"
    summary_path = out_dir / "task6_summary.txt"
    config_path = out_dir / "task6_config.json"
    metadata_path = out_dir / "task6_model_metadata.json"

    write_csv(rows, csv_path)
    save_psnr_curve(rows, curve_path, title=f"Task6 PSNR vs SNR ({run_name})")
    write_summary(summary_path, rows, len(samples.images), checkpoint_path, metadata, selected_indices_path, out_dir)
    save_json(vars(args), config_path)
    metadata_json = dict(metadata)
    metadata_json["checkpoint_extra"] = json.loads(json.dumps(metadata_json.get("checkpoint_extra", {}), default=str))
    save_json(metadata_json, metadata_path)

    print(f"Saved {csv_path}")
    print(f"Saved {curve_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
