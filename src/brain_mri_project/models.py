from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        layers.extend(
            [
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_channels, dropout=dropout * 0.5)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, dropout=dropout)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, dropout=dropout)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return e1, e2, e3, b


class UNetDecoder3D(nn.Module):
    def __init__(self, out_channels: int, base_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4, dropout=dropout)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2, dropout=dropout)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels, dropout=dropout * 0.5)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = UNetEncoder3D(in_channels, base_channels, dropout=dropout)
        self.decoder = UNetDecoder3D(out_channels, base_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(*self.encoder(x))


def adapt_first_conv_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    old_in = weight.shape[1]
    if old_in == in_channels:
        return weight
    if old_in == 1:
        return weight.repeat(1, in_channels, 1, 1, 1) / float(in_channels)
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    repeated = weight.repeat(1, (in_channels + old_in - 1) // old_in, 1, 1, 1)
    return repeated[:, :in_channels] * (old_in / float(in_channels))


def load_encoder_state_with_channel_adaptation(
    model: UNet3D,
    encoder_state: dict[str, torch.Tensor],
    in_channels: int,
) -> tuple[list[str], list[str]]:
    state = dict(encoder_state)
    first_key = "enc1.block.0.weight"
    if first_key in state:
        state[first_key] = adapt_first_conv_weight(state[first_key], in_channels)
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)
