"""CIFAR-10 split and DataLoader plumbing for Phase 4 baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from visionlab.data.augmentation import (
    AugmentationProfile,
    apply_augmentation_profile,
)
from visionlab.data.manifests import (
    ClassMapping,
    DatasetIdentity,
    PreprocessingSpec,
)
from visionlab.data.splits import stratified_validation_indices


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_SPLIT_SEED = 20260814
CIFAR10_VALIDATION_PER_CLASS = 500
CIFAR10_IDENTITY = DatasetIdentity(
    dataset_id="cifar10",
    version="phase1b-registered",
    source="University of Toronto CIFAR-10 via torchvision.datasets.CIFAR10",
    license_or_usage="Original source requests citation; UCI metadata lists CC BY 4.0.",
    description="Provisional core development dataset for VisionLab baseline work.",
)
CIFAR10_PREPROCESSING = PreprocessingSpec(
    image_size=(32, 32),
    color_mode="RGB",
    value_range=(0.0, 1.0),
    normalization_mean=(0.5, 0.5, 0.5),
    normalization_std=(0.5, 0.5, 0.5),
    deterministic=True,
)
MATERIAL_CIFAR10_SPLIT_COUNTS = {
    "train": 45_000,
    "val": 5_000,
    "test": 10_000,
}


@dataclass(frozen=True)
class DataLoaderPolicy:
    batch_size: int
    seed: int
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    train_shuffle: bool = True
    eval_shuffle: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not self.train_shuffle:
            raise ValueError("Phase 4 policy expects shuffled training batches")
        if self.eval_shuffle:
            raise ValueError("Phase 4 policy expects non-shuffled eval batches")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "train_shuffle": self.train_shuffle,
            "eval_shuffle": self.eval_shuffle,
        }


@dataclass(frozen=True)
class SplitDatasetBundle:
    train: "VisionLabSplitDataset"
    val: "VisionLabSplitDataset"
    test: "VisionLabSplitDataset"
    validation_indices: frozenset[int]
    identity: DatasetIdentity = CIFAR10_IDENTITY
    classes: ClassMapping = ClassMapping(CIFAR10_CLASSES)
    preprocessing: PreprocessingSpec = CIFAR10_PREPROCESSING

    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "identity": {
                "dataset_id": self.identity.dataset_id,
                "version": self.identity.version,
                "source": self.identity.source,
                "license_or_usage": self.identity.license_or_usage,
                "description": self.identity.description,
            },
            "classes": list(self.classes.names),
            "preprocessing": {
                "image_size": list(self.preprocessing.image_size),
                "color_mode": self.preprocessing.color_mode,
                "value_range": list(self.preprocessing.value_range),
                "normalization_mean": list(self.preprocessing.normalization_mean),
                "normalization_std": list(self.preprocessing.normalization_std),
                "deterministic": self.preprocessing.deterministic,
            },
            "split_counts": self.split_counts(),
            "split_policy": {
                "validation_seed": CIFAR10_SPLIT_SEED,
                "validation_indices_count": len(self.validation_indices),
                "test_partition": "upstream test only",
            },
        }


@dataclass(frozen=True)
class Phase4DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    prediction_val: DataLoader
    prediction_test: DataLoader
    policy: DataLoaderPolicy


class VisionLabSplitDataset(Dataset):
    """Dataset wrapper that preserves sample identity for prediction artifacts."""

    def __init__(
        self,
        upstream: Dataset,
        *,
        split: str,
        upstream_partition: str,
        indices: list[int],
        class_names: tuple[str, ...] = CIFAR10_CLASSES,
        preprocessing: PreprocessingSpec = CIFAR10_PREPROCESSING,
        augmentation_profile: AugmentationProfile | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if augmentation_profile is not None and split != "train":
            raise ValueError("augmentation profiles may only be attached to the train split")
        self.upstream = upstream
        self.split = split
        self.upstream_partition = upstream_partition
        self.indices = list(indices)
        self.class_names = tuple(class_names)
        self.preprocessing = preprocessing
        self.augmentation_profile = augmentation_profile

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index])
        image, label = self.upstream[source_index]
        label_index = int(label)
        raw_tensor = to_unit_tensor(image, self.preprocessing)
        tensor = raw_tensor
        if self.augmentation_profile is not None:
            tensor = apply_augmentation_profile(tensor, self.augmentation_profile)
        tensor = normalize_tensor(tensor, self.preprocessing)
        sample_id = f"cifar10-{self.upstream_partition}-{source_index:05d}"
        return {
            "input": tensor,
            "raw_input": raw_tensor,
            "label": label_index,
            "sample_id": sample_id,
            "split": self.split,
            "source_id": sample_id,
        }


class TrainingView(Dataset):
    """Expose a prediction-aware dataset as the pair expected by the trainer."""

    def __init__(self, dataset: VisionLabSplitDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.dataset[index]
        return sample["input"], int(sample["label"])


def build_cifar10_split_datasets(
    root: str | Path = "data",
    *,
    download: bool = False,
    validation_per_class: int = CIFAR10_VALIDATION_PER_CLASS,
    split_seed: int = CIFAR10_SPLIT_SEED,
    upstream_train: Dataset | None = None,
    upstream_test: Dataset | None = None,
    train_augmentation_profile: AugmentationProfile | None = None,
) -> SplitDatasetBundle:
    """Build registered train/val/test datasets from CIFAR-10 upstream partitions."""

    if upstream_train is None or upstream_test is None:
        upstream_train, upstream_test = _load_torchvision_cifar10(root, download)

    class_names = _extract_classes(upstream_train)
    if class_names != CIFAR10_CLASSES:
        raise ValueError("upstream train CIFAR-10 class order does not match registration")
    test_class_names = _extract_classes(upstream_test)
    if test_class_names != CIFAR10_CLASSES:
        raise ValueError("upstream test CIFAR-10 class order does not match registration")

    train_labels = _extract_targets(upstream_train)
    validation_indices = stratified_validation_indices(
        train_labels,
        validation_per_class=validation_per_class,
        seed=split_seed,
    )
    train_indices = [
        index for index in range(len(train_labels)) if index not in validation_indices
    ]
    test_indices = list(range(len(_extract_targets(upstream_test))))

    return SplitDatasetBundle(
        train=VisionLabSplitDataset(
            upstream_train,
            split="train",
            upstream_partition="train",
            indices=train_indices,
            class_names=class_names,
            augmentation_profile=train_augmentation_profile,
        ),
        val=VisionLabSplitDataset(
            upstream_train,
            split="val",
            upstream_partition="train",
            indices=sorted(validation_indices),
            class_names=class_names,
        ),
        test=VisionLabSplitDataset(
            upstream_test,
            split="test",
            upstream_partition="test",
            indices=test_indices,
            class_names=class_names,
        ),
        validation_indices=frozenset(validation_indices),
    )


def verify_material_cifar10_contract(
    datasets: SplitDatasetBundle,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Verify the registered CIFAR-10 material-training split contract."""

    expected = expected_counts or MATERIAL_CIFAR10_SPLIT_COUNTS
    if tuple(datasets.classes.names) != CIFAR10_CLASSES:
        raise ValueError("registered CIFAR-10 class order does not match Phase 1B")
    counts = datasets.split_counts()
    if counts != expected:
        raise ValueError(f"registered CIFAR-10 split counts {counts} do not match {expected}")
    if len(datasets.validation_indices) != expected["val"]:
        raise ValueError("validation index count does not match registered validation split")
    if not set(datasets.train.indices).isdisjoint(datasets.val.indices):
        raise ValueError("train and validation source indices overlap")
    if datasets.test.upstream_partition != "test":
        raise ValueError("test split must use the upstream test partition")
    if datasets.train.upstream_partition != "train" or datasets.val.upstream_partition != "train":
        raise ValueError("train and validation splits must use the upstream train partition")
    return {
        "classes": list(datasets.classes.names),
        "split_counts": counts,
        "validation_indices_count": len(datasets.validation_indices),
        "test_partition": datasets.test.upstream_partition,
        "status": "passed",
    }


def build_phase4_dataloaders(
    datasets: SplitDatasetBundle,
    policy: DataLoaderPolicy,
) -> Phase4DataLoaders:
    train_generator = torch.Generator()
    train_generator.manual_seed(policy.seed)

    return Phase4DataLoaders(
        train=DataLoader(
            TrainingView(datasets.train),
            batch_size=policy.batch_size,
            shuffle=policy.train_shuffle,
            generator=train_generator,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=policy.drop_last,
        ),
        val=DataLoader(
            TrainingView(datasets.val),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        test=DataLoader(
            TrainingView(datasets.test),
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        prediction_val=DataLoader(
            datasets.val,
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        prediction_test=DataLoader(
            datasets.test,
            batch_size=policy.batch_size,
            shuffle=policy.eval_shuffle,
            num_workers=policy.num_workers,
            worker_init_fn=_seed_worker if policy.num_workers else None,
            pin_memory=policy.pin_memory,
            drop_last=False,
        ),
        policy=policy,
    )


def to_normalized_tensor(image: Any, preprocessing: PreprocessingSpec) -> Tensor:
    return normalize_tensor(to_unit_tensor(image, preprocessing), preprocessing)


def to_unit_tensor(image: Any, preprocessing: PreprocessingSpec) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach().clone().float()
        if tensor.ndim == 3 and tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
    else:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to convert PIL/numpy images") from exc
        array = np.asarray(image, dtype="float32")
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    if tensor.ndim != 3:
        raise ValueError("image tensor must have shape C x H x W or H x W x C")
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0

    expected_width, expected_height = preprocessing.image_size
    channels, height, width = tensor.shape
    expected_channels = 3 if preprocessing.color_mode == "RGB" else 1
    if channels != expected_channels:
        raise ValueError(f"expected {expected_channels} channels, got {channels}")
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"expected image size {(expected_width, expected_height)}, got {(width, height)}"
        )
    return tensor.contiguous()


def normalize_tensor(tensor: Tensor, preprocessing: PreprocessingSpec) -> Tensor:
    if tensor.ndim != 3:
        raise ValueError("image tensor must have shape C x H x W")
    mean = torch.tensor(preprocessing.normalization_mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(preprocessing.normalization_std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean) / std


def _extract_classes(dataset: Dataset) -> tuple[str, ...]:
    classes = getattr(dataset, "classes", None)
    if classes is None:
        raise ValueError("upstream dataset must expose classes")
    return tuple(str(name) for name in classes)


def _extract_targets(dataset: Dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("upstream dataset must expose targets")
    return [int(label) for label in targets]


def _load_torchvision_cifar10(root: str | Path, download: bool) -> tuple[Dataset, Dataset]:
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("torchvision is required for CIFAR-10 loading") from exc
    root_path = Path(root)
    return (
        CIFAR10(root=str(root_path), train=True, download=download),
        CIFAR10(root=str(root_path), train=False, download=download),
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
