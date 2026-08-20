"""Versioned augmentation profiles for VisionLab CIFAR-10 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TransformSpec:
    name: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class AugmentationProfile:
    profile_id: str
    version: str
    description: str
    train_only: bool
    transforms: tuple[TransformSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "description": self.description,
            "train_only": self.train_only,
            "transforms": [transform.to_dict() for transform in self.transforms],
        }


PHASE4_NO_AUGMENTATION_PROFILE = AugmentationProfile(
    profile_id="phase4-control-no-augmentation",
    version="1.0",
    description="Phase 4-compatible control: no train-time augmentation.",
    train_only=True,
    transforms=(),
)

PHASE5A_CANDIDATE_FLIP_CROP_PROFILE = AugmentationProfile(
    profile_id="phase5a-candidate-horizontal-flip-random-crop",
    version="1.0",
    description=(
        "Candidate CIFAR-10 train-only profile using horizontal flip and "
        "padding-based random crop for visual review before Phase 5B."
    ),
    train_only=True,
    transforms=(
        TransformSpec(
            name="random_horizontal_flip",
            parameters={"probability": 0.5},
        ),
        TransformSpec(
            name="random_crop_with_padding",
            parameters={
                "output_size": [32, 32],
                "padding": 4,
                "padding_mode": "constant",
                "fill": 0.0,
            },
        ),
    ),
)

PHASE5A_PROFILE_REGISTRY = (
    PHASE4_NO_AUGMENTATION_PROFILE,
    PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
)


def profile_registry_dict() -> dict[str, Any]:
    return {
        "registry_id": "visionlab-phase5a-augmentation-profiles",
        "version": "1.0",
        "scope": "Phase 5A augmentation profile and smoke verification",
        "profiles": [profile.to_dict() for profile in PHASE5A_PROFILE_REGISTRY],
    }


def get_augmentation_profile(profile_id: str) -> AugmentationProfile:
    for profile in PHASE5A_PROFILE_REGISTRY:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"unknown augmentation profile: {profile_id}")


def apply_augmentation_profile(
    tensor: Tensor,
    profile: AugmentationProfile,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply a profile to an unnormalized C x H x W tensor in [0, 1]."""

    augmented = tensor.detach().clone()
    for transform in profile.transforms:
        if transform.name == "random_horizontal_flip":
            probability = float(transform.parameters["probability"])
            if torch.rand((), generator=generator).item() < probability:
                augmented = torch.flip(augmented, dims=(2,))
        elif transform.name == "random_crop_with_padding":
            augmented = _random_crop_with_padding(
                augmented,
                output_size=tuple(transform.parameters["output_size"]),
                padding=int(transform.parameters["padding"]),
                fill=float(transform.parameters["fill"]),
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported transform: {transform.name}")
    return augmented


def _random_crop_with_padding(
    tensor: Tensor,
    *,
    output_size: tuple[int, int],
    padding: int,
    fill: float,
    generator: torch.Generator | None,
) -> Tensor:
    if tensor.ndim != 3:
        raise ValueError("augmentation tensor must have shape C x H x W")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    output_height, output_width = output_size
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_size values must be positive")

    padded = F.pad(tensor, (padding, padding, padding, padding), value=fill)
    _, padded_height, padded_width = padded.shape
    if output_height > padded_height or output_width > padded_width:
        raise ValueError("output_size cannot exceed padded image size")

    max_top = padded_height - output_height
    max_left = padded_width - output_width
    top = int(torch.randint(max_top + 1, (), generator=generator).item())
    left = int(torch.randint(max_left + 1, (), generator=generator).item())
    return padded[:, top : top + output_height, left : left + output_width].contiguous()
