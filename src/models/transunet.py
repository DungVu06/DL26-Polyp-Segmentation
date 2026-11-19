from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((skip, x), dim=1))


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        input_size: int = 224,
        name: str = "transunet",
    ) -> None:
        super().__init__()
        del name  # Kept in Hydra config for readable run names.
        bottleneck_channels = base_channels * 16
        if bottleneck_channels % transformer_heads != 0:
            raise ValueError("base_channels * 16 must be divisible by transformer_heads")

        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = DownBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = DownBlock(base_channels * 8, bottleneck_channels)

        grid_size = max(1, input_size // 16)
        self.position_embedding = nn.Parameter(torch.zeros(1, grid_size * grid_size, bottleneck_channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_channels,
            nhead=transformer_heads,
            dim_feedforward=int(bottleneck_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.transformer_norm = nn.LayerNorm(bottleneck_channels)

        self.decoder4 = UpBlock(bottleneck_channels, base_channels * 8, base_channels * 8)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _positional_embedding(self, height: int, width: int) -> torch.Tensor:
        original_tokens = self.position_embedding.shape[1]
        original_grid = int(math.sqrt(original_tokens))
        if original_grid * original_grid != original_tokens:
            raise RuntimeError("Position embedding must represent a square grid.")
        positional = self.position_embedding.transpose(1, 2).reshape(1, -1, original_grid, original_grid)
        positional = functional.interpolate(positional, size=(height, width), mode="bicubic", align_corners=False)
        return positional.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        bottleneck = self.bottleneck(e4)

        batch_size, channels, height, width = bottleneck.shape
        tokens = bottleneck.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens + self._positional_embedding(height, width))
        bottleneck = self.transformer_norm(tokens).transpose(1, 2).reshape(batch_size, channels, height, width)

        x = self.decoder4(bottleneck, e4)
        x = self.decoder3(x, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)
        return self.head(x)
