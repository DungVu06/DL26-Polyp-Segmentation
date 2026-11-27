from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn


def _zero_stage_projection(
    _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(output)


class SegFormer(nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        name: str = "segformer",
        decoder_stages: Sequence[int] | None = None,
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

        if decoder_stages is not None:
            self._select_decoder_stages(decoder_stages)

    def _select_decoder_stages(self, decoder_stages: Sequence[int]) -> None:
        stage_count = self.model.config.num_encoder_blocks
        available = set(range(1, stage_count + 1))
        stages = tuple(decoder_stages)
        if (
            not stages
            or any(type(stage) is not int or stage not in available for stage in stages)
            or len(set(stages)) != len(stages)
        ):
            raise ValueError(f"decoder_stages must contain unique stage numbers from 1 to {stage_count}")
        if set(stages) == available:
            return

        head = self.model.decode_head
        projections = getattr(head, "linear_projections", None)
        if projections is None:
            projections = getattr(head, "linear_c", None)
        if projections is None or len(projections) != stage_count:
            raise RuntimeError("The installed Transformers SegFormer decoder has unsupported stage projections")

        # Mask features after projection, including its bias, while keeping the fusion shape.
        for stage, projection in enumerate(projections, start=1):
            if stage not in stages:
                projection.register_forward_hook(_zero_stage_projection)

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
