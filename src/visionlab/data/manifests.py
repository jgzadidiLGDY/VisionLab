"""Minimal dataset and manifest contracts.

The contract records the dataset identity, split membership, labels, and
preprocessing metadata needed before VisionLab begins model work. It avoids
dataset-specific assumptions so a later CIFAR-10 manifest can use the same
shape as the committed tiny fixture manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    version: str
    source: str
    license_or_usage: str
    description: str = ""


@dataclass(frozen=True)
class ClassMapping:
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("class mapping must contain at least one class")
        cleaned = tuple(name.strip() for name in self.names)
        if any(not name for name in cleaned):
            raise ValueError("class names must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("class names must be unique")
        object.__setattr__(self, "names", cleaned)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}

    def contains(self, label: str) -> bool:
        return label in self.class_to_idx


@dataclass(frozen=True)
class PreprocessingSpec:
    image_size: tuple[int, int]
    color_mode: str
    value_range: tuple[float, float]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    deterministic: bool = True

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height")
        if self.color_mode not in {"RGB", "L"}:
            raise ValueError("color_mode must be RGB or L")
        lower, upper = self.value_range
        if lower >= upper:
            raise ValueError("value_range lower bound must be below upper bound")
        expected_channels = 3 if self.color_mode == "RGB" else 1
        if len(self.normalization_mean) != expected_channels:
            raise ValueError("normalization_mean length must match color channels")
        if len(self.normalization_std) != expected_channels:
            raise ValueError("normalization_std length must match color channels")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("normalization_std values must be positive")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    label: str
    relative_path: str
    source_id: str = ""
    group_id: str = ""
    checksum: str = ""


@dataclass(frozen=True)
class DatasetManifest:
    identity: DatasetIdentity
    classes: ClassMapping
    split_names: tuple[str, ...]
    preprocessing: PreprocessingSpec
    samples: tuple[SampleRecord, ...]

    def __post_init__(self) -> None:
        if not self.split_names:
            raise ValueError("manifest must define at least one split")
        cleaned = tuple(split.strip() for split in self.split_names)
        if any(not split for split in cleaned):
            raise ValueError("split names must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("split names must be unique")
        object.__setattr__(self, "split_names", cleaned)

    def samples_for_split(self, split: str) -> tuple[SampleRecord, ...]:
        return tuple(sample for sample in self.samples if sample.split == split)

    def class_counts_by_split(self) -> dict[str, dict[str, int]]:
        counts = {
            split: {class_name: 0 for class_name in self.classes.names}
            for split in self.split_names
        }
        for sample in self.samples:
            if sample.split in counts and sample.label in counts[sample.split]:
                counts[sample.split][sample.label] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": {
                "dataset_id": self.identity.dataset_id,
                "version": self.identity.version,
                "source": self.identity.source,
                "license_or_usage": self.identity.license_or_usage,
                "description": self.identity.description,
            },
            "classes": list(self.classes.names),
            "split_names": list(self.split_names),
            "preprocessing": {
                "image_size": list(self.preprocessing.image_size),
                "color_mode": self.preprocessing.color_mode,
                "value_range": list(self.preprocessing.value_range),
                "normalization_mean": list(self.preprocessing.normalization_mean),
                "normalization_std": list(self.preprocessing.normalization_std),
                "deterministic": self.preprocessing.deterministic,
            },
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "label": sample.label,
                    "relative_path": sample.relative_path,
                    "source_id": sample.source_id,
                    "group_id": sample.group_id,
                    "checksum": sample.checksum,
                }
                for sample in self.samples
            ],
        }
