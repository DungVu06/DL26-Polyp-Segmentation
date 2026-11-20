from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class SegFormer(nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        name: str = "segformer",
        **kwargs,
    ) -> None:
        super().__init__()
        del name, kwargs
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels

        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError(
                "The 'transformers' package is required to run SegFormer. "
                "Please run: pip install transformers"
            ) from error

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=out_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = out_channels
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input images tensor with shape (B, 3, H, W).
        Returns:
            Logits tensor with shape (B, out_channels, H, W).
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, out_channels, H/4, W/4)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = functional.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits
