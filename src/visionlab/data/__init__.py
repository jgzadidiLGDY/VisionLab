"""Dataset contract and validation helpers for VisionLab."""

from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10_IDENTITY,
    CIFAR10_PREPROCESSING,
    CIFAR10_SPLIT_SEED,
    CIFAR10_VALIDATION_PER_CLASS,
    DataLoaderPolicy,
    MATERIAL_CIFAR10_SPLIT_COUNTS,
    Phase4DataLoaders,
    SplitDatasetBundle,
    TrainingView,
    VisionLabSplitDataset,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
    verify_material_cifar10_contract,
)
from visionlab.data.manifests import (
    ClassMapping,
    DatasetIdentity,
    DatasetManifest,
    PreprocessingSpec,
    SampleRecord,
)
from visionlab.data.validation import ValidationIssue, ValidationReport, validate_manifest
from visionlab.data.splits import stratified_validation_indices

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_IDENTITY",
    "CIFAR10_PREPROCESSING",
    "CIFAR10_SPLIT_SEED",
    "CIFAR10_VALIDATION_PER_CLASS",
    "ClassMapping",
    "DataLoaderPolicy",
    "DatasetIdentity",
    "DatasetManifest",
    "MATERIAL_CIFAR10_SPLIT_COUNTS",
    "Phase4DataLoaders",
    "PreprocessingSpec",
    "SampleRecord",
    "SplitDatasetBundle",
    "TrainingView",
    "ValidationIssue",
    "ValidationReport",
    "VisionLabSplitDataset",
    "build_cifar10_split_datasets",
    "build_phase4_dataloaders",
    "stratified_validation_indices",
    "validate_manifest",
    "verify_material_cifar10_contract",
]
