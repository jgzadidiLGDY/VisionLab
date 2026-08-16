"""Compact custom CNN for VisionLab Phase 2.

The model is intentionally small and explicit. It establishes the
data-to-logits path that later training phases can build on without adding
trainer, checkpoint, dataset-loading, or registry behavior here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CustomCNNConfig:
    input_channels: int = 3
    image_size: tuple[int, int] = (32, 32)
    num_classes: int = 10
    feature_channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not self.feature_channels:
            raise ValueError("feature_channels must contain at least one block")
        if any(channels <= 0 for channels in self.feature_channels):
            raise ValueError("feature_channels values must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")


class CustomCNN(nn.Module):
    """Small CIFAR-sized CNN that returns raw logits."""

    def __init__(self, config: CustomCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CustomCNNConfig()

        blocks: list[nn.Module] = []
        in_channels = self.config.input_channels
        for out_channels in self.config.feature_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            in_channels = out_channels

        self.feature_blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.feature_channels[-1], self.config.num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        self._validate_input(inputs)
        x = inputs
        for block in self.feature_blocks:
            x = block(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)

    def intermediate_shapes(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Return concise, stable stage shapes for a dummy forward pass."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        width, height = self.config.image_size
        model_device = device if device is not None else next(self.parameters()).device
        x = torch.zeros(
            batch_size,
            self.config.input_channels,
            height,
            width,
            device=model_device,
        )

        shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.shape)}
        with torch.no_grad():
            for index, block in enumerate(self.feature_blocks, start=1):
                x = block(x)
                shapes[f"block{index}"] = tuple(x.shape)
            x = self.pool(x)
            shapes["pooled"] = tuple(x.shape)
            x = self.flatten(x)
            shapes["flattened"] = tuple(x.shape)
            x = self.classifier(x)
            shapes["logits"] = tuple(x.shape)
        return shapes

    def _validate_input(self, inputs: Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "CustomCNN expects input shape N x C x H x W; "
                f"received tensor with rank {inputs.ndim}"
            )
        _, channels, height, width = inputs.shape
        expected_width, expected_height = self.config.image_size
        if channels != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, got {channels}"
            )
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                "expected spatial size "
                f"{(expected_width, expected_height)}, got {(width, height)}"
            )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
