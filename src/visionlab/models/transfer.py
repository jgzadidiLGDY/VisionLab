"""Transfer-learning model contracts for VisionLab Phase 6A."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


TRANSFER_MODEL_CONFIG_VERSION = "phase6a-transfer-model-v1"
RESNET18_ARCHITECTURE = "torchvision.models.resnet18"
RESNET18_WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
RESNET18_WEIGHT_CHECKPOINT = "resnet18-f37072fd.pth"
PHASE6A_PREPROCESSING_ID = "phase6a-resnet18-imagenet1k-v1-preprocessing"
PHASE6A_FREEZE_MODE = "frozen_backbone_head_only"
PHASE6C_FINETUNE_MODE = "finetune_layer4_head"
EXPECTED_PHASE6A_PARAMETER_COUNTS = {
    "total": 11_181_642,
    "trainable": 5_130,
    "frozen": 11_176_512,
}


@dataclass(frozen=True)
class ResNet18PreprocessingContract:
    profile_id: str = PHASE6A_PREPROCESSING_ID
    input_size: tuple[int, int] = (224, 224)
    resize_size: int = 256
    crop_size: int = 224
    interpolation: str = "bilinear"
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    source: str = RESNET18_WEIGHT_ENUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "input_size": list(self.input_size),
            "resize_size": self.resize_size,
            "crop_size": self.crop_size,
            "interpolation": self.interpolation,
            "normalization_mean": list(self.normalization_mean),
            "normalization_std": list(self.normalization_std),
            "source": self.source,
            "separate_from_phase4b_custom_cnn_preprocessing": True,
            "comparison_limitation": (
                "The pretrained route uses ImageNet source pretraining, 224x224 "
                "inputs, ImageNet normalization, ResNet-18 architecture scale, "
                "and a different preprocessing pipeline from the Phase 4B CustomCNN."
            ),
        }


@dataclass(frozen=True)
class TransferModelConfig:
    architecture: str = RESNET18_ARCHITECTURE
    weight_identity: str = RESNET18_WEIGHT_ENUM
    num_classes: int = 10
    input_size: tuple[int, int] = (224, 224)
    preprocessing_id: str = PHASE6A_PREPROCESSING_ID
    freeze_mode: str = PHASE6A_FREEZE_MODE
    config_version: str = TRANSFER_MODEL_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.architecture != RESNET18_ARCHITECTURE:
            raise ValueError("Phase 6A supports exactly torchvision.models.resnet18")
        if self.weight_identity != RESNET18_WEIGHT_ENUM:
            raise ValueError("Phase 6A supports exactly ResNet18_Weights.IMAGENET1K_V1")
        if self.num_classes != 10:
            raise ValueError("Phase 6A transfer model must use 10 CIFAR-10 classes")
        if self.input_size != (224, 224):
            raise ValueError("Phase 6A ResNet-18 input_size must be (224, 224)")
        if self.preprocessing_id != PHASE6A_PREPROCESSING_ID:
            raise ValueError("Phase 6A preprocessing_id does not match ResNet-18 contract")
        if self.freeze_mode not in {PHASE6A_FREEZE_MODE, PHASE6C_FINETUNE_MODE}:
            raise ValueError(
                "transfer model freeze_mode must be frozen_backbone_head_only "
                "or finetune_layer4_head"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "architecture": self.architecture,
            "weight_identity": self.weight_identity,
            "weight_url": selected_resnet18_weights().url,
            "weight_checkpoint": RESNET18_WEIGHT_CHECKPOINT,
            "num_classes": self.num_classes,
            "input_size": list(self.input_size),
            "preprocessing_id": self.preprocessing_id,
            "freeze_mode": self.freeze_mode,
            "classifier_head": {
                "module": "torch.nn.Linear",
                "in_features": 512,
                "out_features": self.num_classes,
            },
        }


@dataclass(frozen=True)
class WeightCacheProbe:
    weight_identity: str
    url: str
    expected_filename: str
    cache_dir: Path
    expected_path: Path
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_identity": self.weight_identity,
            "url": self.url,
            "expected_filename": self.expected_filename,
            "cache_dir": str(self.cache_dir),
            "expected_path": str(self.expected_path),
            "exists": self.exists,
            "download_attempted": False,
        }


class TransferResNet18(nn.Module):
    """Frozen-backbone ResNet-18 transfer model returning raw CIFAR-10 logits."""

    def __init__(
        self,
        config: TransferModelConfig | None = None,
        *,
        pretrained_weights_loaded: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or TransferModelConfig()
        self.preprocessing = ResNet18PreprocessingContract()
        self.pretrained_weights_loaded = bool(pretrained_weights_loaded)
        weights = selected_resnet18_weights() if pretrained_weights_loaded else None
        self.model = resnet18(weights=weights)
        if self.model.fc.in_features != 512:
            raise ValueError("expected ResNet-18 classifier input features to equal 512")
        self.model.fc = nn.Linear(512, self.config.num_classes)
        self._freeze_backbone()

    def forward(self, inputs: Tensor) -> Tensor:
        self._validate_input(inputs)
        return self.model(inputs)

    def train(self, mode: bool = True) -> "TransferResNet18":
        super().train(mode)
        if self.config.freeze_mode in {PHASE6A_FREEZE_MODE, PHASE6C_FINETUNE_MODE}:
            for name, module in self.model.named_children():
                if name == "fc":
                    continue
                if self.config.freeze_mode == PHASE6C_FINETUNE_MODE and name == "layer4":
                    continue
                else:
                    module.eval()
        return self

    def parameter_summary(self) -> dict[str, Any]:
        return transfer_parameter_summary(self)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "config": self.config.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "parameter_counts": self.parameter_summary()["counts"],
            "frozen_trainable_summary": self.parameter_summary()["groups"],
        }

    def _freeze_backbone(self) -> None:
        for name, parameter in self.model.named_parameters():
            if self.config.freeze_mode == PHASE6A_FREEZE_MODE:
                parameter.requires_grad = name.startswith("fc.")
            elif self.config.freeze_mode == PHASE6C_FINETUNE_MODE:
                parameter.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
            else:
                raise ValueError(f"unsupported freeze mode: {self.config.freeze_mode}")
        self.train(True)

    def _validate_input(self, inputs: Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "TransferResNet18 expects input shape N x C x H x W; "
                f"received tensor with rank {inputs.ndim}"
            )
        _, channels, height, width = inputs.shape
        expected_width, expected_height = self.config.input_size
        if channels != 3:
            raise ValueError(f"expected 3 input channels, got {channels}")
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                "expected spatial size "
                f"{(expected_width, expected_height)}, got {(width, height)}"
            )


def selected_resnet18_weights() -> ResNet18_Weights:
    return ResNet18_Weights.IMAGENET1K_V1


def probe_resnet18_weight_cache() -> WeightCacheProbe:
    weights = selected_resnet18_weights()
    filename = weights.url.rsplit("/", 1)[-1]
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    cache_dir = torch_home / "hub" / "checkpoints"
    expected_path = cache_dir / filename
    return WeightCacheProbe(
        weight_identity=RESNET18_WEIGHT_ENUM,
        url=weights.url,
        expected_filename=filename,
        cache_dir=cache_dir,
        expected_path=expected_path,
        exists=expected_path.exists(),
    )


def build_phase6a_transfer_model(
    *,
    load_pretrained: bool,
    config: TransferModelConfig | None = None,
) -> TransferResNet18:
    resolved_config = config or TransferModelConfig()
    if load_pretrained:
        probe = probe_resnet18_weight_cache()
        if not probe.exists:
            raise FileNotFoundError(
                "ResNet18_Weights.IMAGENET1K_V1 is not cached; download approval is required"
            )
    model = TransferResNet18(
        config=resolved_config,
        pretrained_weights_loaded=load_pretrained,
    )
    if resolved_config.freeze_mode == PHASE6A_FREEZE_MODE:
        verify_phase6a_parameter_counts(model)
    elif transfer_parameter_summary(model)["counts"]["total"] != EXPECTED_PHASE6A_PARAMETER_COUNTS["total"]:
        raise ValueError("Phase 6C fine-tuning model total parameter count changed unexpectedly")
    return model


def transfer_parameter_summary(model: nn.Module) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named_parameters)
    trainable = sum(
        parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
    )
    frozen = total - trainable
    return {
        "counts": {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        },
        "groups": {
            "trainable": [
                name for name, parameter in named_parameters if parameter.requires_grad
            ],
            "frozen_prefixes": sorted(
                {
                    name.split(".", 2)[1] if name.startswith("model.") else name.split(".", 1)[0]
                    for name, parameter in named_parameters
                    if not parameter.requires_grad
                }
            ),
        },
    }


def verify_phase6a_parameter_counts(model: nn.Module) -> dict[str, int]:
    counts = transfer_parameter_summary(model)["counts"]
    if counts != EXPECTED_PHASE6A_PARAMETER_COUNTS:
        raise ValueError(
            "Phase 6A ResNet-18 parameter counts differ from the approved contract: "
            f"expected {EXPECTED_PHASE6A_PARAMETER_COUNTS}, got {counts}"
        )
    return dict(counts)
