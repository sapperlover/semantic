#!/usr/bin/env python
"""Train FiLM Deep JSCC variants with multi-rate fixed energy masks.

This experiment supports decoder-only FiLM, encoder-decoder FiLM, and a
complex_decoder_film model. All variants compute several Kp losses per batch,
and the full Kp=1024 path is included in every training batch to protect the
high-rate endpoint.

Example:
  python scripts/train_film_decoder_multirate.py \
    --data_dir /home/lc/class/yuyi/cifar-10 \
    --output_dir outputs/film_decoder_multirate_b8_snr7 \
    --model_variant decoder_film \
    --snr_db 7 \
    --block_size 8 \
    --kp_list 128,256,384,512,640,768,896,1024 \
    --low_loss_weight 1.0 \
    --mid_loss_weight 1.0 \
    --high_loss_weight 1.0 \
    --full_loss_weight 1.0 \
    --from_scratch
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from eval_rate_distortion import INPUT_K, LATENT_C, LATENT_H, LATENT_K, LATENT_W, parse_kp_list
from jscc_lab.channel import awgn, power_normalize, snr_db_to_noise_std
from jscc_lab.data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from jscc_lab.metrics import batch_mse, batch_psnr
from jscc_lab.models import count_parameters
from jscc_lab.plotting import save_psnr_curve, save_rate_distortion_curve, save_training_curves
from jscc_lab.utils import TeeLogger, ensure_dir, get_device, save_checkpoint, save_json, seed_everything
from train_jscc import cap_subset, make_loader, maybe_create_tiny_dataset, snr_tag
from train_mask_aware_random_kp import (
    build_mask_bank,
    find_rank_path,
    load_and_normalize_rank,
    resolve_path,
    run_mask_sanity_checks,
    str2bool,
    write_eval_csv,
    write_mask_metadata,
)


MODEL_TYPE = "film_decoder_multirate"
MODEL_VARIANT_DECODER = "decoder_film"
MODEL_VARIANT_ENCODER_DECODER = "encoder_decoder_film"
MODEL_VARIANT_COMPLEX_DECODER = "complex_decoder_film"
MODEL_VARIANT_SNR_RATE_COMPLEX = "snr_rate_complex_film"


def group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class FiLM(nn.Module):
    """Feature-wise linear modulation: x' = x * (1 + gamma(cond)) + beta(cond)."""

    def __init__(self, cond_dim: int, num_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2:
            raise ValueError(f"Expected cond shape (N,cond_dim), got {tuple(cond.shape)}.")
        gamma_beta = self.net(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = group_norm(channels)
        self.film1 = FiLM(cond_dim, channels, hidden_dim=hidden_dim)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = group_norm(channels)
        self.film2 = FiLM(cond_dim, channels, hidden_dim=hidden_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.film1(y, cond)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.film2(y, cond)
        return self.act(y + residual)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ECABlock(nn.Module):
    """Efficient channel attention with a lightweight 1D convolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1)
        return x * self.gate(weights)


class ResSEBlock(nn.Module):
    """Unconditioned residual block with GroupNorm and SE attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = group_norm(channels)
        self.se = SEBlock(channels, reduction=reduction)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = self.se(y)
        return self.act(y + residual)


class FiLMConvBlock(nn.Module):
    """Conv -> GroupNorm -> FiLM -> activation block used by the conditioned encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        stride: int = 1,
        hidden_dim: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = group_norm(out_channels)
        self.film = FiLM(cond_dim, out_channels, hidden_dim=hidden_dim)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return self.act(x)


class UnconditionedEncoder(nn.Module):
    """Unconditioned encoder E(x) -> latent with shape (N,16,8,8)."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            group_norm(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            group_norm(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexResSEEncoder(nn.Module):
    """Stronger unconditioned encoder with residual SE blocks."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.stem64 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            group_norm(64),
            nn.SiLU(inplace=True),
        )
        self.block64 = ResSEBlock(64)
        self.down128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            group_norm(128),
            nn.SiLU(inplace=True),
        )
        self.block128_a = ResSEBlock(128)
        self.block128_b = ResSEBlock(128)
        self.out = nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem64(x)
        x = self.block64(x)
        x = self.down128(x)
        x = self.block128_a(x)
        x = self.block128_b(x)
        return self.out(x)


class FiLMEncoder(nn.Module):
    """EncoderFiLM: E(x, cond) -> latent with shape (N,16,8,8)."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.down64 = FiLMConvBlock(in_channels, 64, cond_dim, stride=2, hidden_dim=film_hidden_dim, activation="relu")
        self.block64 = FiLMResBlock(64, cond_dim, hidden_dim=film_hidden_dim)
        self.down128 = FiLMConvBlock(64, 128, cond_dim, stride=2, hidden_dim=film_hidden_dim, activation="relu")
        self.block128_a = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.conv128 = FiLMConvBlock(128, 128, cond_dim, stride=1, hidden_dim=film_hidden_dim, activation="relu")
        self.block128_b = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.out = nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.down64(x, cond)
        x = self.block64(x, cond)
        x = self.down128(x, cond)
        x = self.block128_a(x, cond)
        x = self.conv128(x, cond)
        x = self.block128_b(x, cond)
        return self.out(x)


class FiLMDecoder(nn.Module):
    """Moderately wider FiLM-conditioned decoder D(z, cond)."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 16, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            group_norm(128),
            nn.SiLU(inplace=True),
        )
        self.block128_a = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.block128_b = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            group_norm(64),
            nn.SiLU(inplace=True),
        )
        self.block64 = FiLMResBlock(64, cond_dim, hidden_dim=film_hidden_dim)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.block32 = FiLMResBlock(32, cond_dim, hidden_dim=film_hidden_dim)
        self.out = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        x = self.block128_a(x, cond)
        x = self.block128_b(x, cond)
        x = self.up1(x)
        x = self.block64(x, cond)
        x = self.up2(x)
        x = self.block32(x, cond)
        return self.out(x)


class ComplexFiLMDecoder(nn.Module):
    """FiLM decoder with residual conditioning and lightweight ECA attention."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 16, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            group_norm(128),
            nn.SiLU(inplace=True),
        )
        self.block128_a = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.eca128 = ECABlock(128)
        self.block128_b = FiLMResBlock(128, cond_dim, hidden_dim=film_hidden_dim)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            group_norm(64),
            nn.SiLU(inplace=True),
        )
        self.block64 = FiLMResBlock(64, cond_dim, hidden_dim=film_hidden_dim)
        self.eca64 = ECABlock(64)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.block32 = FiLMResBlock(32, cond_dim, hidden_dim=film_hidden_dim)
        self.out = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        x = self.block128_a(x, cond)
        x = self.eca128(x)
        x = self.block128_b(x, cond)
        x = self.up1(x)
        x = self.block64(x, cond)
        x = self.eca64(x)
        x = self.up2(x)
        x = self.block32(x, cond)
        return self.out(x)


class FiLMDecoderDeepJSCC(nn.Module):
    """Deep JSCC with unconditioned encoder and FiLM-conditioned decoder."""

    def __init__(self, snr_db: float = 7.0, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.snr_db = float(snr_db)
        self.cond_dim = int(cond_dim)
        self.model_variant = MODEL_VARIANT_DECODER
        self.encoder_conditioned = False
        self.encoder = UnconditionedEncoder(in_channels=3, latent_channels=LATENT_C)
        self.decoder = FiLMDecoder(out_channels=3, latent_channels=LATENT_C, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)


class ComplexFiLMDeepJSCC(nn.Module):
    """More expressive Deep JSCC variant with Res/SE encoder and FiLM+ECA decoder."""

    def __init__(self, snr_db: float = 7.0, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.snr_db = float(snr_db)
        self.cond_dim = int(cond_dim)
        self.model_variant = MODEL_VARIANT_COMPLEX_DECODER
        self.encoder_conditioned = False
        self.encoder = ComplexResSEEncoder(in_channels=3, latent_channels=LATENT_C)
        self.decoder = ComplexFiLMDecoder(out_channels=3, latent_channels=LATENT_C, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)


class SNRRateFiLMDeepJSCC(nn.Module):
    """SNRRateFiLM-DeepJSCC: E(x), D(y, [r, log2(r), snr_db / 20])."""

    def __init__(self, snr_db: float = 7.0, cond_dim: int = 3, film_hidden_dim: int = 64):
        super().__init__()
        if int(cond_dim) != 3:
            raise ValueError("SNRRateFiLMDeepJSCC requires cond_dim=3 for [r, log2(r), snr_db / 20].")
        self.snr_db = float(snr_db)
        self.cond_dim = 3
        self.model_variant = MODEL_VARIANT_SNR_RATE_COMPLEX
        self.encoder_conditioned = False
        self.snr_conditioned = True
        self.encoder = ComplexResSEEncoder(in_channels=3, latent_channels=LATENT_C)
        self.decoder = ComplexFiLMDecoder(out_channels=3, latent_channels=LATENT_C, cond_dim=3, film_hidden_dim=film_hidden_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)


class FiLMEncoderDecoderDeepJSCC(nn.Module):
    """Deep JSCC with FiLM-conditioned encoder and decoder."""

    def __init__(self, snr_db: float = 7.0, cond_dim: int = 2, film_hidden_dim: int = 64):
        super().__init__()
        self.snr_db = float(snr_db)
        self.cond_dim = int(cond_dim)
        self.model_variant = MODEL_VARIANT_ENCODER_DECODER
        self.encoder_conditioned = True
        self.encoder = FiLMEncoder(in_channels=3, latent_channels=LATENT_C, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        self.decoder = FiLMDecoder(out_channels=3, latent_channels=LATENT_C, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            raise ValueError("FiLMEncoderDecoderDeepJSCC.encode requires cond.")
        return self.encoder(x, cond)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_variant == MODEL_VARIANT_DECODER:
        return FiLMDecoderDeepJSCC(snr_db=args.snr_db, cond_dim=args.cond_dim, film_hidden_dim=args.film_hidden_dim)
    if args.model_variant == MODEL_VARIANT_COMPLEX_DECODER:
        return ComplexFiLMDeepJSCC(snr_db=args.snr_db, cond_dim=args.cond_dim, film_hidden_dim=args.film_hidden_dim)
    if args.model_variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
        return SNRRateFiLMDeepJSCC(snr_db=args.snr_db, cond_dim=3, film_hidden_dim=args.film_hidden_dim)
    if args.model_variant == MODEL_VARIANT_ENCODER_DECODER:
        return FiLMEncoderDecoderDeepJSCC(snr_db=args.snr_db, cond_dim=args.cond_dim, film_hidden_dim=args.film_hidden_dim)
    raise ValueError(f"Unsupported model_variant={args.model_variant!r}.")


def output_prefix(model_variant: str) -> str:
    if model_variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
        return "snr_rate_complex_film_multirate"
    if model_variant == MODEL_VARIANT_COMPLEX_DECODER:
        return "complex_film_multirate"
    if model_variant == MODEL_VARIANT_ENCODER_DECODER:
        return "film_encoder_decoder_multirate"
    return "film_decoder_multirate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FiLM multi-rate Deep JSCC variants.")
    parser.add_argument("--data_dir", "--data_path", dest="data_dir", default="/home/lc/class/yuyi/cifar-10")
    parser.add_argument("--project_dir", default="/home/lc/class/yuyi/semantic_jscc_lab")
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional matching FiLM checkpoint for weight initialization.")
    parser.add_argument("--resume", default=None, help="Resume matching FiLM checkpoint, including optimizer state.")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch; this is the default when no checkpoint is given.")
    parser.add_argument("--eval_only", action="store_true", help="Load a FiLM checkpoint and run fixed-Kp test without training.")
    parser.add_argument(
        "--model_variant",
        choices=[MODEL_VARIANT_DECODER, MODEL_VARIANT_ENCODER_DECODER, MODEL_VARIANT_COMPLEX_DECODER, MODEL_VARIANT_SNR_RATE_COMPLEX],
        default=MODEL_VARIANT_DECODER,
        help=(
            "decoder_film keeps E(x) unconditioned; encoder_decoder_film uses E(x,cond) and D(z,cond); "
            "complex_decoder_film uses a Res/SE encoder and FiLM/ECA decoder; "
            "snr_rate_complex_film adds snr_db/20 to the complex decoder FiLM condition."
        ),
    )
    parser.add_argument("--snr_db", "--train_snr_db", dest="snr_db", type=float, default=7.0)
    parser.add_argument("--kp_list", default="128,256,384,512,640,768,896,1024")
    parser.add_argument(
        "--rate_mode",
        choices=["multi", "full_only"],
        default="multi",
        help="multi samples low/mid/high and full per batch; full_only trains only Kp=1024 while keeping fixed-Kp eval.",
    )
    parser.add_argument("--block_size", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--rank_path", default=None)
    parser.add_argument("--rank_format", choices=["auto", "hwc_flat", "block_id", "channel"], default="auto")
    parser.add_argument("--kp_rounding", choices=["floor", "ceil"], default="floor")
    parser.add_argument("--noise_only_on_kept", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--low_loss_weight", type=float, default=1.0)
    parser.add_argument("--mid_loss_weight", type=float, default=1.0)
    parser.add_argument("--high_loss_weight", type=float, default=1.0)
    parser.add_argument("--full_loss_weight", type=float, default=1.0)
    parser.add_argument("--cond_dim", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--film_hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument(
        "--run_final_eval",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="After training, reload best checkpoint and write task7/task6 eval outputs. Default false; use --eval_only for testing later.",
    )
    parser.add_argument("--run_task6_eval", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--task6_snr_list", default="1,4,7,13,19")
    parser.add_argument("--task6_num_images", type=int, default=500)
    parser.add_argument("--task6_warmup_batches", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    return parser


def cond_from_kp(
    kp: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    cond_dim: int = 2,
    snr_db: float | None = None,
) -> torch.Tensor:
    r = float(kp) / float(LATENT_K)
    if cond_dim == 1:
        values = [r]
    elif cond_dim == 2:
        values = [r, math.log2(max(r, 1e-8))]
    elif cond_dim == 3:
        if snr_db is None:
            raise ValueError("cond_dim=3 requires snr_db to build [r, log2(r), snr_db / 20].")
        values = [r, math.log2(max(r, 1e-8)), float(snr_db) / 20.0]
    else:
        raise ValueError(f"Unsupported cond_dim={cond_dim}.")
    return torch.tensor(values, dtype=dtype, device=device).view(1, cond_dim).expand(batch_size, cond_dim)


def parse_snr_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("SNR list must contain at least one value.")
    return values


def kp_groups(kp_values: Sequence[int], rate_mode: str = "multi") -> Dict[str, List[int]]:
    values = sorted(int(kp) for kp in kp_values)
    if 1024 not in values:
        raise ValueError("kp_list must include Kp=1024 because the full-rate path is always required.")
    if rate_mode == "full_only":
        return {"low": [], "mid": [], "high": [], "full": [1024]}
    groups = {
        "low": [kp for kp in values if kp in {128, 256, 384}],
        "mid": [kp for kp in values if kp in {512, 640}],
        "high": [kp for kp in values if kp in {768, 896}],
        "full": [1024] if 1024 in values else [],
    }
    missing = [name for name, items in groups.items() if not items]
    if missing:
        raise ValueError(
            f"kp_list must include choices for every multi-rate group; missing {missing}. "
            "Default is 128,256,384,512,640,768,896,1024."
        )
    return groups


def select_training_rates(groups: Dict[str, List[int]], rate_mode: str) -> Dict[str, int]:
    if rate_mode == "full_only":
        return {"full": 1024}
    return {
        "low": int(random.choice(groups["low"])),
        "mid": int(random.choice(groups["mid"])),
        "high": int(random.choice(groups["high"])),
        "full": 1024,
    }


def load_initial_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | None,
    resume_path: Path | None,
    device: torch.device,
    expected_model_variant: str,
) -> Tuple[int, Dict[str, object]]:
    info: Dict[str, object] = {"mode": "from_scratch", "start_epoch": 1}
    path = resume_path if resume_path is not None else checkpoint_path
    if path is None:
        return 1, info
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{path} is not a compatible {MODEL_TYPE} checkpoint. Use --from_scratch.")
    extra = checkpoint.get("extra", {})
    if extra.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"{path} has model_type={extra.get('model_type')!r}, expected {MODEL_TYPE!r}. "
            "Use --from_scratch for this new architecture."
        )
    checkpoint_variant = str(extra.get("model_variant", MODEL_VARIANT_DECODER))
    if checkpoint_variant != expected_model_variant:
        raise ValueError(
            f"{path} has model_variant={checkpoint_variant!r}, expected {expected_model_variant!r}. "
            "Use the matching --model_variant or train this architecture with --from_scratch."
        )

    model.load_state_dict(checkpoint["model_state"])
    loaded_optimizer = False
    if resume_path is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        loaded_optimizer = True
    previous_epoch = int(checkpoint.get("epoch") or 0)
    start_epoch = previous_epoch + 1 if resume_path is not None else 1
    return start_epoch, {
        "mode": "resume" if resume_path is not None else "checkpoint_init",
        "path": str(path),
        "previous_epoch": previous_epoch,
        "start_epoch": start_epoch,
        "loaded_optimizer_state": loaded_optimizer,
        "checkpoint_extra": extra,
    }


def forward_one_rate(
    model: nn.Module,
    latent: torch.Tensor | None,
    images: torch.Tensor,
    kp: int,
    mask: torch.Tensor,
    snr_db: float,
    noise_only_on_kept: bool,
    criterion: nn.Module | None,
    cond_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    cond = cond_from_kp(kp, images.shape[0], images.device, images.dtype, cond_dim=cond_dim, snr_db=snr_db)
    if latent is None:
        latent = power_normalize(model.encode(images, cond))
    latent_tx = latent * mask
    latent_rx = awgn(latent_tx, snr_db, training=True)
    if noise_only_on_kept:
        latent_rx = latent_rx * mask
    recon = model.decode(latent_rx, cond)
    loss = criterion(recon, images) if criterion is not None else None
    return recon, loss


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    masks: Dict[int, torch.Tensor],
    groups: Dict[str, List[int]],
    weights: Dict[str, float],
    rate_mode: str,
    noise_only_on_kept: bool,
    cond_dim: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[Dict[str, float], Dict[int, int]]:
    model.train(True)
    totals = {"train_loss": 0.0, "train_low_loss": 0.0, "train_mid_loss": 0.0, "train_high_loss": 0.0, "train_full_loss": 0.0}
    group_counts = {"low": 0, "mid": 0, "high": 0, "full": 0}
    total_samples = 0
    kp_counts = {int(kp): 0 for choices in groups.values() for kp in choices}

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        selected = select_training_rates(groups, rate_mode)
        for kp in selected.values():
            kp_counts[kp] = kp_counts.get(kp, 0) + 1
        for group_name in selected:
            group_counts[group_name] += 1

        optimizer.zero_grad(set_to_none=True)
        shared_latent = None if bool(getattr(model, "encoder_conditioned", False)) else power_normalize(model.encode(images))
        losses: Dict[str, torch.Tensor] = {}
        weighted_loss = torch.zeros((), dtype=images.dtype, device=images.device)
        weight_sum = sum(float(weights[group_name]) for group_name in selected)
        if weight_sum <= 0:
            raise ValueError("At least one multi-rate loss weight must be positive.")
        for group_name, kp in selected.items():
            _, loss = forward_one_rate(
                model,
                shared_latent,
                images,
                kp,
                masks[kp],
                snr_db,
                noise_only_on_kept,
                criterion,
                cond_dim,
            )
            assert loss is not None
            losses[group_name] = loss
            weighted_loss = weighted_loss + float(weights[group_name]) * loss
        total_loss = weighted_loss / weight_sum
        total_loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        totals["train_loss"] += float(total_loss.item()) * batch_size
        for group_name, loss in losses.items():
            totals[f"train_{group_name}_loss"] += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training dataloader did not produce any samples.")
    stats: Dict[str, float] = {"train_loss": totals["train_loss"] / total_samples}
    for group_name in ["low", "mid", "high", "full"]:
        key = f"train_{group_name}_loss"
        stats[key] = totals[key] / total_samples if group_counts[group_name] > 0 else float("nan")
    return stats, kp_counts


@torch.no_grad()
def evaluate_one_kp(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    snr_db: float,
    kp: int,
    mask: torch.Tensor,
    noise_only_on_kept: bool,
    cond_dim: int,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        latent = None if bool(getattr(model, "encoder_conditioned", False)) else power_normalize(model.encode(images))
        recon, _ = forward_one_rate(model, latent, images, kp, mask, snr_db, noise_only_on_kept, None, cond_dim)
        recon = recon.clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        count = int(images.shape[0])
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += count
    if total_images == 0:
        raise ValueError("No images were evaluated.")
    return {"Kp": int(kp), "R": float(kp) / INPUT_K, "avg_mse": total_mse / total_images, "avg_psnr": total_psnr / total_images}


@torch.no_grad()
def evaluate_all_kp(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    snr_db: float,
    kp_values: Sequence[int],
    masks: Dict[int, torch.Tensor],
    noise_only_on_kept: bool,
    seed: int,
    cond_dim: int,
) -> List[Dict[str, float]]:
    rows = []
    for kp in kp_values:
        torch.manual_seed(seed + int(kp))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + int(kp))
        rows.append(evaluate_one_kp(model, dataloader, device, snr_db, int(kp), masks[int(kp)], noise_only_on_kept, cond_dim))
    return rows


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_task6_loader(
    test_set,
    num_images: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, List[int]]:
    if num_images <= 0:
        raise ValueError("--task6_num_images must be positive.")
    count = min(int(num_images), len(test_set))
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(len(test_set), generator=generator)[:count].tolist()
    subset = Subset(test_set, positions)
    loader = make_loader(subset, batch_size, False, seed, num_workers, pin_memory)
    return loader, [int(pos) for pos in positions]


@torch.no_grad()
def task6_forward_full_rate(
    model: nn.Module,
    images: torch.Tensor,
    test_snr_db: float,
    cond_dim: int,
) -> Tuple[torch.Tensor, float, float]:
    cond = cond_from_kp(LATENT_K, images.shape[0], images.device, images.dtype, cond_dim=cond_dim, snr_db=test_snr_db)

    sync_if_cuda(images.device)
    start = time.perf_counter()
    if bool(getattr(model, "encoder_conditioned", False)):
        latent = power_normalize(model.encode(images, cond))
    else:
        latent = power_normalize(model.encode(images))
    sync_if_cuda(images.device)
    encoder_seconds = time.perf_counter() - start

    latent_noisy = awgn(latent, test_snr_db, training=True)

    sync_if_cuda(images.device)
    start = time.perf_counter()
    recon = model.decode(latent_noisy, cond)
    sync_if_cuda(images.device)
    decoder_seconds = time.perf_counter() - start
    return recon, encoder_seconds, decoder_seconds


@torch.no_grad()
def warmup_task6(model: nn.Module, dataloader: DataLoader, device: torch.device, snr_db: float, cond_dim: int, warmup_batches: int) -> None:
    if warmup_batches <= 0:
        return
    model.eval()
    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        _ = task6_forward_full_rate(model, images, snr_db, cond_dim)
    sync_if_cuda(device)


@torch.no_grad()
def evaluate_task6_one_snr(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    snr_db: float,
    cond_dim: int,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0
    encoder_seconds = 0.0
    decoder_seconds = 0.0
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        recon, enc_s, dec_s = task6_forward_full_rate(model, images, snr_db, cond_dim)
        recon = recon.clamp(0.0, 1.0)
        mse = batch_mse(recon, images)
        psnr = batch_psnr(recon, images)
        batch_size = int(images.shape[0])
        total_mse += float(mse.sum().item())
        total_psnr += float(psnr.sum().item())
        total_images += batch_size
        encoder_seconds += enc_s
        decoder_seconds += dec_s
    if total_images == 0:
        raise ValueError("No task6 images were evaluated.")
    return {
        "test_snr_db": float(snr_db),
        "avg_mse": total_mse / total_images,
        "avg_psnr": total_psnr / total_images,
        "encoder_ms_per_image": 1000.0 * encoder_seconds / total_images,
        "decoder_ms_per_image": 1000.0 * decoder_seconds / total_images,
    }


def evaluate_task6_all_snr(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    snr_values: Sequence[float],
    cond_dim: int,
    seed: int,
    warmup_batches: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for snr in snr_values:
        snr_seed = seed + int(round(float(snr) * 1000))
        torch.manual_seed(snr_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(snr_seed)
        warmup_task6(model, dataloader, device, float(snr), cond_dim, warmup_batches)
        rows.append(evaluate_task6_one_snr(model, dataloader, device, float(snr), cond_dim))
    return rows


def write_task6_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_snr_db", "avg_mse", "avg_psnr", "encoder_ms_per_image", "decoder_ms_per_image"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_task6_summary(
    path: Path,
    rows: Sequence[Dict[str, float]],
    model_variant: str,
    checkpoint_path: Path,
    num_images: int,
    selected_positions_path: Path,
) -> None:
    lines = [
        "Task (6) multi-SNR evaluation summary",
        "=" * 40,
        f"Model variant: {model_variant}",
        f"Checkpoint: {checkpoint_path}",
        f"Shared task6 test images: {num_images}",
        f"Selected task6 test positions: {selected_positions_path}",
        "Rate/SNR condition for conditioned decoder: Kp=1024, r=1.0; cond=[1,0] when cond_dim=2, cond=[1,0,snr_db/20] when cond_dim=3",
        "",
        "Measured rows:",
    ]
    for row in rows:
        lines.append(
            f"SNR={row['test_snr_db']:g} dB | avg_mse={row['avg_mse']:.6f} | "
            f"avg_psnr={row['avg_psnr']:.3f} dB | encoder={row['encoder_ms_per_image']:.4f} ms/image | "
            f"decoder={row['decoder_ms_per_image']:.4f} ms/image"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_task6_selected_positions(path: Path, positions: Sequence[int]) -> None:
    lines = ["# order test_subset_position"]
    for order, pos in enumerate(positions):
        lines.append(f"{order} {int(pos)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_and_save_task6_outputs(
    model: nn.Module,
    task6_loader: DataLoader,
    device: torch.device,
    snr_values: Sequence[float],
    cond_dim: int,
    seed: int,
    warmup_batches: int,
    prefix: str,
    out_dir: Path,
    model_variant: str,
    checkpoint_path: Path,
    selected_positions: Sequence[int],
) -> List[Dict[str, float]]:
    task6_rows = evaluate_task6_all_snr(model, task6_loader, device, snr_values, cond_dim, seed, warmup_batches)
    task6_csv = out_dir / f"{prefix}_task6_noise_sweep.csv"
    task6_png = out_dir / f"{prefix}_task6_psnr_vs_snr.png"
    task6_summary = out_dir / f"{prefix}_task6_summary.txt"
    task6_positions = out_dir / f"{prefix}_task6_selected_positions.txt"

    write_task6_csv(task6_rows, task6_csv)
    save_psnr_curve([row["test_snr_db"] for row in task6_rows], [row["avg_psnr"] for row in task6_rows], task6_png)
    save_task6_selected_positions(task6_positions, selected_positions)
    write_task6_summary(task6_summary, task6_rows, model_variant, checkpoint_path, len(selected_positions), task6_positions)
    return task6_rows


def run_additional_sanity_checks(
    masks: Dict[int, torch.Tensor],
    mask_info: Dict[int, Dict[str, int]],
    block_size: int,
    kp_values: Sequence[int],
    device: torch.device,
    snr_db: float,
    noise_only_on_kept: bool,
    cond_dim: int,
    rate_mode: str,
) -> List[str]:
    lines = run_mask_sanity_checks(masks, mask_info, block_size, kp_values)
    if 128 in masks and noise_only_on_kept:
        mask = masks[128]
        dummy = torch.randn((2, LATENT_C, LATENT_H, LATENT_W), device=device)
        rx = awgn(dummy * mask, snr_db, training=True) * mask
        zeros_ok = bool(torch.all(rx.masked_select(mask.expand_as(rx) == 0) == 0))
        if not zeros_ok:
            raise AssertionError("Untransmitted positions are nonzero after AWGN and remasking.")
        lines.append("AWGN remask sanity: untransmitted positions remain zero.")
    cond = cond_from_kp(128, 2, device, torch.float32, cond_dim=cond_dim, snr_db=snr_db)
    if cond.shape != (2, cond_dim):
        raise AssertionError(f"Condition shape mismatch: expected (2,{cond_dim}), got {tuple(cond.shape)}.")
    lines.append(f"Condition sanity: Kp=128 cond[0]={cond[0].detach().cpu().tolist()}.")
    if rate_mode == "full_only":
        lines.append("Rate-mode sanity: every training batch uses only full Kp=1024.")
    else:
        lines.append("Multi-rate sanity: every training batch samples low/mid/high and includes full Kp=1024 by construction.")
    return lines


def write_history_csv(history: Iterable[Dict[str, float]], path: Path, kp_values: Sequence[int]) -> None:
    rows = list(history)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_low_loss",
        "train_mid_loss",
        "train_high_loss",
        "train_full_loss",
        "val_loss",
        "val_psnr",
        "best_val_loss",
    ] + [f"kp_count_{int(kp)}" for kp in kp_values]
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    rows: List[Dict[str, float]],
    checkpoint_info: Dict[str, object],
    final_eval_checkpoint: Path,
    rank_metadata: Dict[str, object],
    out_dir: Path,
    num_train: int,
    num_val: int,
    num_test: int,
) -> None:
    prefix = output_prefix(args.model_variant)
    if args.model_variant == MODEL_VARIANT_ENCODER_DECODER:
        model_description = "Encoder: FiLM-conditioned E(x, cond); Decoder: FiLM-conditioned D(z, cond)"
    elif args.model_variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
        model_description = "Encoder: unconditioned Res/SE E(x); Decoder: FiLM-conditioned D(z, [r, log2(r), snr_db/20]) with ECA"
    elif args.model_variant == MODEL_VARIANT_COMPLEX_DECODER:
        model_description = "Encoder: unconditioned Res/SE E(x); Decoder: FiLM-conditioned D(z, cond) with ECA"
    else:
        model_description = "Encoder: unconditioned E(x); Decoder: FiLM-conditioned D(z, cond)"
    if args.rate_mode == "full_only":
        rate_mode_description = "Every training batch uses only full Kp=1024."
    else:
        rate_mode_description = "Every training batch includes low/mid/high sampled rates plus full Kp=1024."
    lines = [
        f"{prefix} Deep JSCC summary",
        "=" * 46,
        f"Output directory: {out_dir}",
        f"Dataset path: {args.data_dir}",
        f"Checkpoint mode: {checkpoint_info.get('mode')}",
        f"Checkpoint path: {checkpoint_info.get('path', 'none')}",
        f"Final test checkpoint: {final_eval_checkpoint}",
        f"SNR: {args.snr_db:g} dB",
        f"block_size: {args.block_size}",
        f"kp_list: {parse_kp_list(args.kp_list)}",
        f"loss weights: low={args.low_loss_weight}, mid={args.mid_loss_weight}, high={args.high_loss_weight}, full={args.full_loss_weight}",
        "loss normalization: weighted sum divided by active loss weight sum",
        f"noise_only_on_kept: {bool(args.noise_only_on_kept)}",
        f"cond_dim: {args.cond_dim}; condition=[r, log2(r)] when cond_dim=2; [r, log2(r), snr_db/20] when cond_dim=3",
        f"model_variant: {args.model_variant}",
        f"rate_mode: {args.rate_mode}",
        model_description,
        rate_mode_description,
        f"Rank file: {rank_metadata.get('rank_path')}",
        f"Rank format: requested={rank_metadata.get('requested_rank_format')}, resolved={rank_metadata.get('resolved_rank_format')}",
        f"Rank input type: {rank_metadata.get('rank_input_type')}",
        f"Train/val/test samples used: {num_train}/{num_val}/{num_test}",
        "",
        "Final fixed-Kp evaluation rows:",
    ]
    for row in rows:
        lines.append(f"Kp={int(row['Kp'])} | R={row['R']:.6f} | avg_mse={row['avg_mse']:.6f} | avg_psnr={row['avg_psnr']:.3f} dB")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.model_variant == MODEL_VARIANT_SNR_RATE_COMPLEX:
        args.cond_dim = 3
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.eval_every < 1:
        raise ValueError("--eval_every must be at least 1.")
    if args.save_every < 0:
        raise ValueError("--save_every must be non-negative.")

    seed_everything(args.seed)
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    tag = snr_tag(args.snr_db)
    prefix = output_prefix(args.model_variant)
    out_dir = ensure_dir(resolve_path(args.output_dir, project_dir) or (project_dir / f"outputs/{prefix}_b{args.block_size}_snr{tag}"))
    log = TeeLogger(out_dir / ("eval.log" if args.eval_only else "train.log"), mode="a" if args.eval_only else "w")

    try:
        device = get_device(args.device)
        data_dir = Path(args.data_dir).expanduser()
        maybe_create_tiny_dataset(data_dir, args.seed)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

        kp_values = parse_kp_list(args.kp_list)
        groups = kp_groups(kp_values, args.rate_mode)
        weights = {
            "low": float(args.low_loss_weight),
            "mid": float(args.mid_loss_weight),
            "high": float(args.high_loss_weight),
            "full": float(args.full_loss_weight),
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError(f"Loss weights must be non-negative, got {weights}.")
        if sum(weights.values()) <= 0:
            raise ValueError(f"At least one loss weight must be positive, got {weights}.")
        rank_path = resolve_path(args.rank_path, project_dir) if args.rank_path else find_rank_path(project_dir, args.block_size)
        rank_indices, rank_metadata = load_and_normalize_rank(rank_path, args.block_size, args.rank_format)
        sanity_kp_values = sorted(set(int(kp) for kp in kp_values) | {128, 1024})
        sanity_masks, sanity_mask_info, _ = build_mask_bank(rank_indices, args.block_size, sanity_kp_values, args.kp_rounding, device)
        sanity_lines = run_additional_sanity_checks(
            sanity_masks,
            sanity_mask_info,
            args.block_size,
            sanity_kp_values,
            device,
            args.snr_db,
            bool(args.noise_only_on_kept),
            args.cond_dim,
            args.rate_mode,
        )
        masks = {int(kp): sanity_masks[int(kp)] for kp in kp_values}
        mask_info = {int(kp): sanity_mask_info[int(kp)] for kp in kp_values}

        shutil.copy2(rank_path, out_dir / f"source_rank_b{args.block_size}.npy")
        np.save(out_dir / f"normalized_block_rank_indices_b{args.block_size}.npy", rank_indices)
        save_json(rank_metadata, out_dir / "rank_metadata.json")
        write_mask_metadata(mask_info, out_dir / "mask_metadata.csv")

        images, labels = load_cifar_array_dataset(data_dir)
        dataset = CIFARArrayDataset(images, labels)
        train_set, val_set, test_set = make_splits(dataset, train=args.train_split, val=args.val_split, test=args.test_split, seed=args.seed)
        train_set = cap_subset(train_set, args.max_train_samples)
        val_set = cap_subset(val_set, args.max_eval_samples)
        test_set = cap_subset(test_set, args.max_eval_samples)

        pin_memory = device.type == "cuda"
        train_loader = make_loader(train_set, args.batch_size, True, args.seed, args.num_workers, pin_memory)
        val_loader = make_loader(val_set, args.batch_size, False, args.seed, args.num_workers, pin_memory)
        test_loader = make_loader(test_set, args.batch_size, False, args.seed, args.num_workers, pin_memory)
        task6_snr_values = parse_snr_list(args.task6_snr_list)
        task6_loader = None
        task6_positions: List[int] = []
        if bool(args.run_task6_eval):
            task6_loader, task6_positions = make_task6_loader(
                test_set,
                args.task6_num_images,
                args.batch_size,
                args.seed,
                args.num_workers,
                pin_memory,
            )

        best_path = out_dir / f"best_{prefix}_b{args.block_size}_snr{tag}.pt"
        last_path = out_dir / f"last_{prefix}_b{args.block_size}_snr{tag}.pt"
        model = build_model(args).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        resume_path = resolve_path(args.resume, project_dir)
        checkpoint_path = None if args.from_scratch else resolve_path(args.checkpoint, project_dir)
        if args.eval_only and checkpoint_path is None and resume_path is None:
            if best_path.is_file():
                checkpoint_path = best_path
            else:
                raise ValueError(
                    "--eval_only requires --checkpoint or --resume unless the output directory contains "
                    f"{best_path.name}."
                )
        start_epoch, checkpoint_info = load_initial_checkpoint(model, optimizer, checkpoint_path, resume_path, device, args.model_variant)
        if args.eval_only and checkpoint_info.get("mode") == "from_scratch":
            raise ValueError("--eval_only cannot run from scratch; pass --checkpoint or --resume.")
        if not args.eval_only and start_epoch > args.epochs:
            raise ValueError(f"Resume checkpoint starts at epoch {start_epoch}, but --epochs={args.epochs}.")

        summary_text = "\n".join(
            [
                repr(model),
                "",
                f"FiLM multi-rate training SNR: {float(args.snr_db):.4g} dB",
                f"model_variant: {args.model_variant}",
                f"rate_mode: {args.rate_mode}",
                f"AWGN noise std after power normalization: {snr_db_to_noise_std(args.snr_db):.8f}",
                "Latent shape CHW: 16 x 8 x 8; report HWC: 8 x 8 x 16",
                f"block_size: {args.block_size}",
                f"Rank path: {rank_path}",
                f"Rank format: requested={rank_metadata.get('requested_rank_format')} resolved={rank_metadata.get('resolved_rank_format')}",
                f"Loss weights: {weights}",
                "Multi-rate loss normalization: weighted sum divided by active loss weight sum",
                f"noise_only_on_kept: {bool(args.noise_only_on_kept)}",
                f"cond_dim: {args.cond_dim}; condition=[r, log2(r)] when cond_dim=2; [r, log2(r), snr_db/20] when cond_dim=3",
                f"Optimizer: Adam(lr={args.lr}); scheduler: none",
                f"Trainable parameters: {count_parameters(model, trainable_only=True):,}",
                f"Total parameters: {count_parameters(model, trainable_only=False):,}",
            ]
        )
        (out_dir / ("eval_model_summary.txt" if args.eval_only else "model_summary.txt")).write_text(summary_text + "\n", encoding="utf-8")
        log.log(summary_text)
        for line in sanity_lines:
            log.log(f"Sanity: {line}")

        config = vars(args).copy()
        config.update(
            {
                "resolved_output_dir": str(out_dir),
                "resolved_rank_path": str(rank_path),
                "checkpoint_info": checkpoint_info,
                "rank_metadata": rank_metadata,
                "mask_info": {str(kp): info for kp, info in mask_info.items()},
                "kp_groups": groups,
                "rate_mode": args.rate_mode,
                "loss_weights": weights,
                "loss_normalization": "weighted_sum_divided_by_active_weight_sum",
                "model_type": MODEL_TYPE,
                "model_variant": args.model_variant,
                "optimizer": "Adam",
                "scheduler": "none",
                "device": str(device),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "test_samples": len(test_set),
                "task6_samples": len(task6_positions),
                "task6_snr_list": task6_snr_values,
                "run_final_eval": bool(args.run_final_eval),
            }
        )
        save_json(config, out_dir / ("eval_config.json" if args.eval_only else "config.json"))

        if args.eval_only:
            eval_checkpoint = Path(str(checkpoint_info.get("path"))).expanduser()
            log.log(f"Eval-only fixed-Kp test uses checkpoint: {eval_checkpoint}")
            test_rows = evaluate_all_kp(model, test_loader, device, args.snr_db, kp_values, masks, bool(args.noise_only_on_kept), args.seed, args.cond_dim)
            eval_csv = out_dir / f"{prefix}_eval_block{args.block_size}.csv"
            rd_png = out_dir / f"{prefix}_rd_block{args.block_size}.png"
            task7_csv = out_dir / f"{prefix}_task7_rate_distortion.csv"
            task7_png = out_dir / f"{prefix}_task7_rate_distortion_curve.png"
            write_eval_csv(test_rows, eval_csv)
            save_rate_distortion_curve([row["R"] for row in test_rows], [row["avg_psnr"] for row in test_rows], rd_png)
            write_eval_csv(test_rows, task7_csv)
            save_rate_distortion_curve([row["R"] for row in test_rows], [row["avg_psnr"] for row in test_rows], task7_png)
            write_summary(
                out_dir / f"{prefix}_summary.txt",
                args,
                test_rows,
                checkpoint_info,
                eval_checkpoint,
                rank_metadata,
                out_dir,
                len(train_set),
                len(val_set),
                len(test_set),
            )
            log.log(f"Saved eval CSV: {eval_csv}")
            log.log(f"Saved RD curve: {rd_png}")
            log.log(f"Saved task7 CSV: {task7_csv}")
            log.log(f"Saved task7 curve: {task7_png}")
            if bool(args.run_task6_eval):
                assert task6_loader is not None
                task6_rows = run_and_save_task6_outputs(
                    model,
                    task6_loader,
                    device,
                    task6_snr_values,
                    args.cond_dim,
                    args.seed,
                    args.task6_warmup_batches,
                    prefix,
                    out_dir,
                    args.model_variant,
                    eval_checkpoint,
                    task6_positions,
                )
                log.log(f"Saved task6 multi-SNR outputs for {len(task6_rows)} SNR values.")
            return

        history: List[Dict[str, float]] = []
        best_val_loss = float("inf")
        extra = {
            "model_type": MODEL_TYPE,
            "model_variant": args.model_variant,
            "train_snr_db": float(args.snr_db),
            "noise_std": snr_db_to_noise_std(args.snr_db),
            "latent_shape_hwc": [8, 8, 16],
            "data_path": str(data_dir),
            "block_size": int(args.block_size),
            "kp_list": [int(kp) for kp in kp_values],
            "kp_groups": groups,
            "rate_mode": args.rate_mode,
            "loss_weights": weights,
            "loss_normalization": "weighted_sum_divided_by_active_weight_sum",
            "rank_path": str(rank_path),
            "rank_format": args.rank_format,
            "noise_only_on_kept": bool(args.noise_only_on_kept),
            "cond_dim": int(args.cond_dim),
            "film_hidden_dim": int(args.film_hidden_dim),
            "optimizer": "Adam",
            "scheduler": "none",
            "mask_info": {str(kp): info for kp, info in mask_info.items()},
            "task6_snr_list": task6_snr_values,
            "task6_num_images": int(args.task6_num_images),
            "run_task6_eval": bool(args.run_task6_eval),
            "run_final_eval": bool(args.run_final_eval),
        }

        for epoch in range(start_epoch, args.epochs + 1):
            train_stats, kp_counts = run_train_epoch(
                model,
                train_loader,
                criterion,
                device,
                args.snr_db,
                masks,
                groups,
                weights,
                args.rate_mode,
                bool(args.noise_only_on_kept),
                args.cond_dim,
                optimizer,
            )
            should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            val_rows = evaluate_all_kp(model, val_loader, device, args.snr_db, kp_values, masks, bool(args.noise_only_on_kept), args.seed, args.cond_dim) if should_eval else None
            val_loss = float(np.mean([row["avg_mse"] for row in val_rows])) if val_rows else float("nan")
            val_psnr = float(np.mean([row["avg_psnr"] for row in val_rows])) if val_rows else float("nan")
            is_best = bool(val_rows) and val_loss <= best_val_loss
            if is_best:
                best_val_loss = val_loss

            row: Dict[str, float] = {
                "epoch": int(epoch),
                **train_stats,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "best_val_loss": best_val_loss,
            }
            for kp in kp_values:
                row[f"kp_count_{int(kp)}"] = int(kp_counts.get(int(kp), 0))
            history.append(row)
            log.log(
                f"epoch {epoch:03d}/{args.epochs:03d} train_loss={train_stats['train_loss']:.6f} "
                f"low={train_stats['train_low_loss']:.6f} mid={train_stats['train_mid_loss']:.6f} "
                f"high={train_stats['train_high_loss']:.6f} full={train_stats['train_full_loss']:.6f} "
                f"val_loss={val_loss:.6f} val_psnr={val_psnr:.3f} best_val_loss={best_val_loss:.6f} kp_counts={kp_counts}"
            )

            if is_best:
                save_checkpoint(best_path, model, optimizer=optimizer, epoch=epoch, metrics=row, extra=extra)
            if args.save_every and epoch % args.save_every == 0:
                save_checkpoint(
                    out_dir / f"checkpoint_epoch_{epoch:03d}_{prefix}_b{args.block_size}_snr{tag}.pt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    extra=extra,
                )

            write_history_csv(history, out_dir / "history.csv", kp_values)
            save_training_curves(
                {
                    "train_loss": [item["train_loss"] for item in history],
                    "train_low_loss": [item["train_low_loss"] for item in history],
                    "train_mid_loss": [item["train_mid_loss"] for item in history],
                    "train_high_loss": [item["train_high_loss"] for item in history],
                    "train_full_loss": [item["train_full_loss"] for item in history],
                    "val_loss": [item["val_loss"] for item in history],
                },
                out_dir / "loss_curve.png",
            )

        save_checkpoint(last_path, model, optimizer=optimizer, epoch=args.epochs, metrics=history[-1], extra=extra)
        if not best_path.exists():
            save_checkpoint(best_path, model, optimizer=optimizer, epoch=args.epochs, metrics=history[-1], extra=extra)
        if not bool(args.run_final_eval):
            log.log(f"Saved best checkpoint: {best_path}")
            log.log(f"Saved last checkpoint: {last_path}")
            log.log("Skipped final task7/task6 evaluation after training because --run_final_eval is false.")
            return

        best_model = build_model(args).to(device)
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=args.lr)
        _, best_eval_info = load_initial_checkpoint(best_model, best_optimizer, best_path, None, device, args.model_variant)
        log.log(f"Final fixed-Kp test uses best checkpoint: {best_path}")
        log.log(f"Best checkpoint load info: {best_eval_info}")

        test_rows = evaluate_all_kp(best_model, test_loader, device, args.snr_db, kp_values, masks, bool(args.noise_only_on_kept), args.seed, args.cond_dim)
        eval_csv = out_dir / f"{prefix}_eval_block{args.block_size}.csv"
        rd_png = out_dir / f"{prefix}_rd_block{args.block_size}.png"
        task7_csv = out_dir / f"{prefix}_task7_rate_distortion.csv"
        task7_png = out_dir / f"{prefix}_task7_rate_distortion_curve.png"
        write_eval_csv(test_rows, eval_csv)
        save_rate_distortion_curve([row["R"] for row in test_rows], [row["avg_psnr"] for row in test_rows], rd_png)
        write_eval_csv(test_rows, task7_csv)
        save_rate_distortion_curve([row["R"] for row in test_rows], [row["avg_psnr"] for row in test_rows], task7_png)
        write_summary(out_dir / f"{prefix}_summary.txt", args, test_rows, checkpoint_info, best_path, rank_metadata, out_dir, len(train_set), len(val_set), len(test_set))

        if bool(args.run_task6_eval):
            assert task6_loader is not None
            task6_rows = run_and_save_task6_outputs(
                best_model,
                task6_loader,
                device,
                task6_snr_values,
                args.cond_dim,
                args.seed,
                args.task6_warmup_batches,
                prefix,
                out_dir,
                args.model_variant,
                best_path,
                task6_positions,
            )
            log.log(f"Saved task6 multi-SNR outputs for {len(task6_rows)} SNR values.")

        log.log(f"Saved best checkpoint: {best_path}")
        log.log(f"Saved last checkpoint: {last_path}")
        log.log(f"Saved final eval CSV: {eval_csv}")
        log.log(f"Saved RD curve: {rd_png}")
        log.log(f"Saved task7 CSV: {task7_csv}")
        log.log(f"Saved task7 curve: {task7_png}")
    finally:
        log.close()


if __name__ == "__main__":
    main()

# Example b8 channel-energy FiLM multi-rate training:
# python scripts/train_film_decoder_multirate.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --output_dir outputs/film_decoder_multirate_b8_snr7 \
#   --model_variant decoder_film \
#   --snr_db 7 \
#   --block_size 8 \
#   --kp_list 128,256,384,512,640,768,896,1024 \
#   --low_loss_weight 1.0 \
#   --mid_loss_weight 1.0 \
#   --high_loss_weight 1.0 \
#   --full_loss_weight 1.0 \
#   --from_scratch
#
# Example b8 channel-energy complex Res/SE + FiLM/ECA multi-rate training:
# python scripts/train_film_decoder_multirate.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --output_dir outputs/complex_film_multirate_b8_snr7 \
#   --model_variant complex_decoder_film \
#   --snr_db 7 \
#   --block_size 8 \
#   --kp_list 128,256,384,512,640,768,896,1024 \
#   --task6_snr_list 1,4,7,13,19 \
#   --task6_num_images 500 \
#   --low_loss_weight 1.0 \
#   --mid_loss_weight 1.0 \
#   --high_loss_weight 1.0 \
#   --full_loss_weight 1.0 \
#   --from_scratch
#
# Example b8 channel-energy SNRRateFiLM complex multi-rate training:
# python scripts/train_film_decoder_multirate.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --output_dir outputs/snr_rate_complex_film_multirate_b8_snr7 \
#   --model_variant snr_rate_complex_film \
#   --snr_db 7 \
#   --block_size 8 \
#   --kp_list 128,256,384,512,640,768,896,1024 \
#   --task6_snr_list 1,4,7,13,19 \
#   --task6_num_images 500 \
#   --low_loss_weight 1.0 \
#   --mid_loss_weight 1.0 \
#   --high_loss_weight 1.0 \
#   --full_loss_weight 1.0 \
#   --from_scratch
#
# Example b8 channel-energy encoder+decoder FiLM multi-rate training:
# python scripts/train_film_decoder_multirate.py \
#   --data_dir /home/lc/class/yuyi/cifar-10 \
#   --output_dir outputs/film_encoder_decoder_multirate_b8_snr7 \
#   --model_variant encoder_decoder_film \
#   --snr_db 7 \
#   --block_size 8 \
#   --kp_list 128,256,384,512,640,768,896,1024 \
#   --low_loss_weight 1.0 \
#   --mid_loss_weight 1.0 \
#   --high_loss_weight 1.0 \
#   --full_loss_weight 1.0 \
#   --from_scratch
