"""Preprocessing helpers for Phase 6 transfer-learning paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from visionlab.data.cifar10 import DataLoaderPolicy, SplitDatasetBundle, _seed_worker, to_unit_tensor
from visionlab.data.manifests import PreprocessingSpec
from visionlab.models.transfer import (
    PHASE6A_PREPROCESSING_ID,
    ResNet18PreprocessingContract,
    selected_resnet18_weights,
)


TRANSFER_PREPROCESSING_SOURCE_SPEC = PreprocessingSpec(
    image_size=(32, 32),
    color_mode="RGB",
    value_range=(0.0, 1.0),
    normalization_mean=(0.0, 0.0, 0.0),
    normalization_std=(1.0, 1.0, 1.0),
    deterministic=True,
)


@dataclass(frozen=True)
class TransferPreprocessingResult:
    tensor: Tensor
    contract: ResNet18PreprocessingContract


class TransferTrainingView(Dataset):
    """Expose a prediction-aware dataset with ResNet-18 preprocessing for training."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.dataset[index]
        return preprocess_resnet18_imagenet_tensor(sample["raw_input"]), int(sample["label"])


class TransferPredictionView(Dataset):
    """Expose sample metadata with ResNet-18 preprocessing for evaluation artifacts."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return {
            "input": preprocess_resnet18_imagenet_tensor(sample["raw_input"]),
            "label": int(sample["label"]),
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "source_id": sample["source_id"],
        }


@dataclass(frozen=True)
class TransferDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    prediction_val: DataLoader
    prediction_test: DataLoader
    policy: DataLoaderPolicy


def build_transfer_dataloaders(
    datasets: SplitDatasetBundle,
    policy: DataLoaderPolicy,
) -> TransferDataLoaders:
    train_generator = torch.Generator()
    train_generator.manual_seed(policy.seed)

    return TransferDataLoaders(
        train=DataLoader(
            TransferTrainingView(datasets.train),
            batch_size=policy.batch_size,
            shuffle=policy.train_shuffle,
            generator=train_generator,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=policy.drop_last,
        ),
        val=DataLoader(
            TransferTrainingView(datasets.val),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        test=DataLoader(
            TransferTrainingView(datasets.test),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        prediction_val=DataLoader(
            TransferPredictionView(datasets.val),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        prediction_test=DataLoader(
            TransferPredictionView(datasets.test),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        policy=policy,
    )


def preprocess_resnet18_imagenet_tensor(image: Any) -> Tensor:
    """Apply the exact Torchvision transform attached to the selected ResNet-18 weights."""

    unit_tensor = to_unit_tensor(image, TRANSFER_PREPROCESSING_SOURCE_SPEC)
    transform = selected_resnet18_weights().transforms()
    transformed = transform(unit_tensor)
    _validate_transfer_tensor(transformed)
    return transformed.contiguous()


def phase6b_preprocessing_contract_dict() -> dict[str, Any]:
    contract = ResNet18PreprocessingContract()
    data = contract.to_dict()
    data["actual_transform_source"] = "ResNet18_Weights.IMAGENET1K_V1.transforms()"
    data["phase6b_verified_application"] = True
    return data


def _validate_transfer_tensor(tensor: Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError("transfer preprocessing must return C x H x W")
    if tuple(tensor.shape) != (3, 224, 224):
        raise ValueError(f"expected transfer tensor shape (3, 224, 224), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("transfer preprocessing produced non-finite values")
