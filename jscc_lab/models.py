"""Neural network modules for the CIFAR-10 AE / Deep JSCC lab."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .channel import awgn, power_normalize


class ConvEncoder(nn.Module):
    """Convolutional encoder mapping 32x32 RGB images to 8x8x16 latent codes."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Keep 8x8 spatial size and set the required latent channel count.
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Convolutional decoder mapping 8x8x16 latent codes back to 32x32 RGB images."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoEncoder(nn.Module):
    """Baseline autoencoder used by homework task (1)."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_channels=latent_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class DeepJSCC(nn.Module):
    """Deep JSCC model with power-normalized latent code and AWGN channel."""

    def __init__(self, snr_db: float = 7.0, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        self.snr_db = float(snr_db)
        self.encoder = ConvEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_channels=latent_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and normalize each sample to average latent power P=1."""

        return power_normalize(self.encode(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, snr_db: float | None = None, add_noise: bool = True) -> torch.Tensor:
        z = self.encode_normalized(x)
        z_noisy = awgn(z, self.snr_db if snr_db is None else float(snr_db), training=add_noise)
        return self.decode(z_noisy)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally only those with gradients enabled."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def model_summary(
    model: AutoEncoder,
    input_shape: Tuple[int, int, int] = (3, 32, 32),
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> str:
    """Return a compact text summary including the required latent shape."""

    device = torch.device(device)
    was_training = model.training
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros((batch_size, *input_shape), device=device)
        latent = model.encode(dummy)
        output = model.decode(latent)

    if was_training:
        model.train()

    lines = [
        repr(model),
        "",
        f"Input tensor shape: {tuple(dummy.shape)}",
        f"PyTorch latent tensor shape (N,C,H,W): {tuple(latent.shape)}",
        "Per-image latent code shape for the report (H x W x C): 8 x 8 x 16",
        f"Flattened latent elements per image K: {latent.shape[1] * latent.shape[2] * latent.shape[3]}",
        f"Output tensor shape: {tuple(output.shape)}",
        f"Trainable parameters: {count_parameters(model, trainable_only=True):,}",
        f"Total parameters: {count_parameters(model, trainable_only=False):,}",
    ]
    return "\n".join(lines)
