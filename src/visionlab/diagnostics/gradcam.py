"""Grad-CAM style spatial diagnostics for image classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: Tensor
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    predicted_index: int
    confidence: float
    target_class_index: int


def compute_gradcam(
    model: nn.Module,
    inputs: Tensor,
    *,
    target_layer: nn.Module,
    target_class_index: int,
) -> GradCAMResult:
    """Compute a deterministic Grad-CAM heatmap for a single image batch."""

    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single-image batch shaped 1 x C x H x W")
    if target_class_index < 0:
        raise ValueError("target_class_index must be non-negative")

    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        activations.append(output)

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        if not grad_output or grad_output[0] is None:
            raise ValueError("target layer gradients unavailable")
        gradients.append(grad_output[0])

    model.eval()
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits_tensor = model(inputs.requires_grad_(True))
        if logits_tensor.ndim != 2 or logits_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects model logits shaped 1 x num_classes")
        if target_class_index >= logits_tensor.shape[1]:
            raise ValueError("target_class_index is outside model output range")
        score = logits_tensor[0, target_class_index]
        score.backward()
    finally:
        handle_forward.remove()
        handle_backward.remove()

    if not activations:
        raise ValueError("target layer activations unavailable")
    if not gradients:
        raise ValueError("target layer gradients unavailable")

    activation = activations[-1].detach()
    gradient = gradients[-1].detach()
    if activation.ndim != 4 or gradient.ndim != 4:
        raise ValueError("target layer must produce spatial feature maps")
    if activation.shape != gradient.shape:
        raise ValueError("target layer activation and gradient shapes differ")
    if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
        raise ValueError("Grad-CAM activation or gradient contains non-finite values")

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam,
        size=tuple(inputs.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if not torch.isfinite(cam).all():
        raise ValueError("Grad-CAM heatmap contains non-finite values")
    maximum = cam.max()
    if float(maximum.item()) <= 0.0:
        raise ValueError("Grad-CAM heatmap is all-empty")
    heatmap = (cam / maximum).detach().cpu().contiguous()
    probabilities_tensor = torch.softmax(logits_tensor.detach().cpu(), dim=1)[0]
    confidence, predicted = probabilities_tensor.max(dim=0)
    return GradCAMResult(
        heatmap=heatmap,
        logits=tuple(float(value) for value in logits_tensor.detach().cpu()[0].tolist()),
        probabilities=tuple(float(value) for value in probabilities_tensor.tolist()),
        predicted_index=int(predicted.item()),
        confidence=float(confidence.item()),
        target_class_index=target_class_index,
    )
