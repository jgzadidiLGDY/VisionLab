"""Versioned degradation profiles for Phase 8A robustness preparation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


PHASE8A_DEGRADATION_REGISTRY_ID = "visionlab-phase8a-degradation-profiles"
PHASE8A_DEGRADATION_REGISTRY_VERSION = "1.0"
PHASE8A_DEFAULT_SEED = 20260825
PHASE8A_INPUT_CONTRACT = {
    "shape": "C x H x W",
    "channels": "RGB, exactly 3 channels",
    "value_range": [0.0, 1.0],
    "normalization": "unnormalized unit tensor",
}
PHASE8A_OUTPUT_CONTRACT = {
    "shape": "preserve input C x H x W",
    "channels": "preserve RGB channels",
    "value_range": [0.0, 1.0],
    "finite": True,
}
PHASE8A_SEED_POLICY = (
    "Deterministic transforms ignore seed. Stochastic transforms require a base seed; "
    "the effective seed is derived from profile_id, version, severity_id, base seed, "
    "sample_id, and source_id so repeated sample access is order-independent."
)


@dataclass(frozen=True)
class DegradationSeverity:
    severity_id: str
    level: int
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity_id": self.severity_id,
            "level": self.level,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class DegradationProfile:
    profile_id: str
    version: str
    degradation_type: str
    deterministic: bool
    seed_policy: str
    description: str
    rationale: str
    severities: tuple[DegradationSeverity, ...]
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "degradation_type": self.degradation_type,
            "deterministic": self.deterministic,
            "seed_policy": self.seed_policy,
            "description": self.description,
            "rationale": self.rationale,
            "input_contract": dict(self.input_contract),
            "output_contract": dict(self.output_contract),
            "severity_levels": [severity.to_dict() for severity in self.severities],
        }


PHASE8A_GAUSSIAN_NOISE_PROFILE = DegradationProfile(
    profile_id="phase8a-gaussian-noise",
    version="1.0",
    degradation_type="gaussian_noise",
    deterministic=False,
    seed_policy=PHASE8A_SEED_POLICY,
    description="Add zero-mean Gaussian noise to unnormalized RGB tensors before preprocessing.",
    rationale=(
        "The standard-deviation schedule is bounded for CIFAR-10-sized images: S1 is "
        "lightly visible, while S5 is materially noisy without intentionally erasing all "
        "class evidence. Parameters are fixed before any model robustness evaluation."
    ),
    severities=(
        DegradationSeverity("S1", 1, {"std": 0.03}),
        DegradationSeverity("S2", 2, {"std": 0.06}),
        DegradationSeverity("S3", 3, {"std": 0.09}),
        DegradationSeverity("S4", 4, {"std": 0.12}),
        DegradationSeverity("S5", 5, {"std": 0.15}),
    ),
    input_contract=PHASE8A_INPUT_CONTRACT,
    output_contract=PHASE8A_OUTPUT_CONTRACT,
)

PHASE8A_GAUSSIAN_BLUR_PROFILE = DegradationProfile(
    profile_id="phase8a-gaussian-blur",
    version="1.0",
    degradation_type="gaussian_blur",
    deterministic=True,
    seed_policy=PHASE8A_SEED_POLICY,
    description="Apply depthwise Gaussian blur to unnormalized RGB tensors before preprocessing.",
    rationale=(
        "The kernel/sigma schedule increases blur gradually while keeping odd kernels "
        "small enough for 32 x 32 CIFAR-10 images. Parameters are fixed before any model "
        "robustness evaluation."
    ),
    severities=(
        DegradationSeverity("S1", 1, {"kernel_size": 3, "sigma": 0.4}),
        DegradationSeverity("S2", 2, {"kernel_size": 3, "sigma": 0.7}),
        DegradationSeverity("S3", 3, {"kernel_size": 5, "sigma": 1.0}),
        DegradationSeverity("S4", 4, {"kernel_size": 5, "sigma": 1.3}),
        DegradationSeverity("S5", 5, {"kernel_size": 7, "sigma": 1.6}),
    ),
    input_contract=PHASE8A_INPUT_CONTRACT,
    output_contract=PHASE8A_OUTPUT_CONTRACT,
)

PHASE8A_BRIGHTNESS_SHIFT_PROFILE = DegradationProfile(
    profile_id="phase8a-brightness-shift",
    version="1.0",
    degradation_type="brightness_shift",
    deterministic=True,
    seed_policy=PHASE8A_SEED_POLICY,
    description="Apply a fixed darkening offset to unnormalized RGB tensors before preprocessing.",
    rationale=(
        "The additive schedule simulates progressively darker capture conditions while "
        "clamping to the valid unit range. Parameters are fixed before any model "
        "robustness evaluation."
    ),
    severities=(
        DegradationSeverity("S1", 1, {"delta": -0.08}),
        DegradationSeverity("S2", 2, {"delta": -0.16}),
        DegradationSeverity("S3", 3, {"delta": -0.24}),
        DegradationSeverity("S4", 4, {"delta": -0.32}),
        DegradationSeverity("S5", 5, {"delta": -0.40}),
    ),
    input_contract=PHASE8A_INPUT_CONTRACT,
    output_contract=PHASE8A_OUTPUT_CONTRACT,
)

PHASE8A_CONTRAST_REDUCTION_PROFILE = DegradationProfile(
    profile_id="phase8a-contrast-reduction",
    version="1.0",
    degradation_type="contrast_reduction",
    deterministic=True,
    seed_policy=PHASE8A_SEED_POLICY,
    description="Reduce contrast around each image's per-channel spatial mean before preprocessing.",
    rationale=(
        "The factor schedule progressively compresses contrast while preserving global "
        "color bias. Parameters are fixed before any model robustness evaluation."
    ),
    severities=(
        DegradationSeverity("S1", 1, {"factor": 0.90}),
        DegradationSeverity("S2", 2, {"factor": 0.80}),
        DegradationSeverity("S3", 3, {"factor": 0.70}),
        DegradationSeverity("S4", 4, {"factor": 0.60}),
        DegradationSeverity("S5", 5, {"factor": 0.50}),
    ),
    input_contract=PHASE8A_INPUT_CONTRACT,
    output_contract=PHASE8A_OUTPUT_CONTRACT,
)

PHASE8A_DEGRADATION_PROFILES = (
    PHASE8A_GAUSSIAN_NOISE_PROFILE,
    PHASE8A_GAUSSIAN_BLUR_PROFILE,
    PHASE8A_BRIGHTNESS_SHIFT_PROFILE,
    PHASE8A_CONTRAST_REDUCTION_PROFILE,
)


def degradation_registry_dict() -> dict[str, Any]:
    return {
        "registry_id": PHASE8A_DEGRADATION_REGISTRY_ID,
        "version": PHASE8A_DEGRADATION_REGISTRY_VERSION,
        "scope": "Phase 8A degradation registry, visual QA, and tiny smoke",
        "phase": "8A",
        "model_evaluation_allowed": False,
        "profiles": [profile.to_dict() for profile in PHASE8A_DEGRADATION_PROFILES],
    }


def get_degradation_profile(profile_id: str) -> DegradationProfile:
    for profile in PHASE8A_DEGRADATION_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"unknown degradation profile: {profile_id}")


def get_degradation_severity(
    profile: DegradationProfile,
    severity_id: str,
) -> DegradationSeverity:
    for severity in profile.severities:
        if severity.severity_id == severity_id:
            return severity
    raise ValueError(f"unknown severity {severity_id} for degradation profile {profile.profile_id}")


def apply_degradation(
    tensor: Tensor,
    *,
    profile_id: str,
    severity_id: str,
    seed: int | None = None,
    sample_id: str = "",
    source_id: str = "",
) -> Tensor:
    """Apply a Phase 8A degradation to an unnormalized RGB tensor in [0, 1]."""

    profile = get_degradation_profile(profile_id)
    severity = get_degradation_severity(profile, severity_id)
    _validate_unit_rgb_tensor(tensor)
    degraded = tensor.detach().clone()

    if profile.degradation_type == "gaussian_noise":
        if seed is None:
            raise ValueError("gaussian_noise degradation requires an explicit seed")
        generator = torch.Generator(device=degraded.device)
        generator.manual_seed(_effective_seed(profile, severity, seed, sample_id, source_id))
        noise = torch.randn(
            degraded.shape,
            generator=generator,
            dtype=degraded.dtype,
            device=degraded.device,
        )
        degraded = degraded + noise * float(severity.parameters["std"])
    elif profile.degradation_type == "gaussian_blur":
        degraded = _gaussian_blur(
            degraded,
            kernel_size=int(severity.parameters["kernel_size"]),
            sigma=float(severity.parameters["sigma"]),
        )
    elif profile.degradation_type == "brightness_shift":
        degraded = degraded + float(severity.parameters["delta"])
    elif profile.degradation_type == "contrast_reduction":
        factor = float(severity.parameters["factor"])
        channel_mean = degraded.mean(dim=(1, 2), keepdim=True)
        degraded = channel_mean + (degraded - channel_mean) * factor
    else:
        raise ValueError(f"unsupported degradation type: {profile.degradation_type}")

    degraded = degraded.clamp(0.0, 1.0).contiguous()
    _validate_unit_rgb_tensor(degraded)
    return degraded


class DegradedSampleDataset(Dataset):
    """Wrap prediction-aware samples and preserve metadata after raw-image degradation."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        profile_id: str,
        severity_id: str,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.profile = get_degradation_profile(profile_id)
        self.severity = get_degradation_severity(self.profile, severity_id)
        self.seed = seed
        if not self.profile.deterministic and self.seed is None:
            raise ValueError(f"{self.profile.profile_id} requires an explicit seed")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        sample_id = str(sample["sample_id"])
        source_id = str(sample.get("source_id", sample_id))
        degraded = apply_degradation(
            sample["raw_input"],
            profile_id=self.profile.profile_id,
            severity_id=self.severity.severity_id,
            seed=self.seed,
            sample_id=sample_id,
            source_id=source_id,
        )
        return {
            **sample,
            "input": degraded,
            "raw_input": degraded,
            "sample_id": sample_id,
            "source_id": source_id,
            "degradation_profile_id": self.profile.profile_id,
            "degradation_profile_version": self.profile.version,
            "degradation_severity_id": self.severity.severity_id,
            "degradation_seed": self.seed,
        }


def _validate_unit_rgb_tensor(tensor: Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError("degradation input must have shape C x H x W")
    if tuple(tensor.shape)[0] != 3:
        raise ValueError("degradation input must have exactly 3 RGB channels")
    if not torch.isfinite(tensor).all():
        raise ValueError("degradation input must contain only finite values")
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError("degradation input must be in [0, 1]")


def _effective_seed(
    profile: DegradationProfile,
    severity: DegradationSeverity,
    seed: int,
    sample_id: str,
    source_id: str,
) -> int:
    payload = "|".join(
        [
            profile.profile_id,
            profile.version,
            severity.severity_id,
            str(int(seed)),
            sample_id,
            source_id,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _gaussian_blur(tensor: Tensor, *, kernel_size: int, sigma: float) -> Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("gaussian blur kernel_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("gaussian blur sigma must be positive")
    radius = kernel_size // 2
    positions = torch.arange(
        -radius,
        radius + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    kernel_1d = torch.exp(-(positions**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    padded = F.pad(tensor.unsqueeze(0), (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=3).squeeze(0)
