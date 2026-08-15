"""Dataset contract and validation helpers for VisionLab."""

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
    "ClassMapping",
    "DatasetIdentity",
    "DatasetManifest",
    "PreprocessingSpec",
    "SampleRecord",
    "ValidationIssue",
    "ValidationReport",
    "stratified_validation_indices",
    "validate_manifest",
]
