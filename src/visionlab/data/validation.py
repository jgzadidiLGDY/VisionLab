"""Validation for VisionLab dataset manifests and tiny fixture images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from visionlab.data.manifests import DatasetManifest, SampleRecord


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    sample_id: str = ""


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    class_counts_by_split: dict[str, dict[str, int]]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    channels: int
    max_value: int

    @property
    def color_mode(self) -> str:
        return "RGB" if self.channels == 3 else "L"


def validate_manifest(manifest: DatasetManifest, dataset_root: Path) -> ValidationReport:
    root = dataset_root.resolve()
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    _validate_unique_sample_ids(manifest.samples, errors)
    _validate_non_empty_splits(manifest, warnings)

    for sample in manifest.samples:
        _validate_sample_record(manifest, sample, root, errors)

    return ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        class_counts_by_split=manifest.class_counts_by_split(),
    )


def read_ascii_netpbm_info(path: Path) -> ImageInfo:
    tokens = _netpbm_tokens(path)
    if len(tokens) < 4:
        raise ValueError("Netpbm file is incomplete")

    magic = tokens[0]
    if magic not in {"P2", "P3"}:
        raise ValueError("only ASCII P2/P3 Netpbm fixtures are supported")

    width = _positive_int(tokens[1], "width")
    height = _positive_int(tokens[2], "height")
    max_value = _positive_int(tokens[3], "max value")
    channels = 3 if magic == "P3" else 1
    expected_values = width * height * channels
    values = tokens[4:]

    if len(values) != expected_values:
        raise ValueError(
            f"expected {expected_values} pixel values, found {len(values)}"
        )
    for value in values:
        pixel = int(value)
        if pixel < 0 or pixel > max_value:
            raise ValueError("pixel value is outside the declared range")

    return ImageInfo(width=width, height=height, channels=channels, max_value=max_value)


def _validate_unique_sample_ids(
    samples: tuple[SampleRecord, ...], errors: list[ValidationIssue]
) -> None:
    seen: set[str] = set()
    for sample in samples:
        if not sample.sample_id:
            errors.append(ValidationIssue("empty_sample_id", "sample_id is required"))
        elif sample.sample_id in seen:
            errors.append(
                ValidationIssue(
                    "duplicate_sample_id",
                    f"sample_id {sample.sample_id!r} appears more than once",
                    sample.sample_id,
                )
            )
        else:
            seen.add(sample.sample_id)


def _validate_non_empty_splits(
    manifest: DatasetManifest, warnings: list[ValidationIssue]
) -> None:
    for split in manifest.split_names:
        if not manifest.samples_for_split(split):
            warnings.append(
                ValidationIssue(
                    "empty_split",
                    f"split {split!r} has no samples",
                )
            )


def _validate_sample_record(
    manifest: DatasetManifest,
    sample: SampleRecord,
    root: Path,
    errors: list[ValidationIssue],
) -> None:
    if sample.split not in manifest.split_names:
        errors.append(
            ValidationIssue(
                "unknown_split",
                f"sample split {sample.split!r} is not declared",
                sample.sample_id,
            )
        )
    if not manifest.classes.contains(sample.label):
        errors.append(
            ValidationIssue(
                "unknown_label",
                f"sample label {sample.label!r} is not declared",
                sample.sample_id,
            )
        )

    sample_path = _safe_sample_path(root, sample)
    if sample_path is None:
        errors.append(
            ValidationIssue(
                "unsafe_path",
                f"relative_path {sample.relative_path!r} must stay under dataset root",
                sample.sample_id,
            )
        )
        return

    if not sample_path.exists():
        errors.append(
            ValidationIssue(
                "missing_file",
                f"sample file does not exist: {sample.relative_path}",
                sample.sample_id,
            )
        )
        return

    try:
        info = read_ascii_netpbm_info(sample_path)
    except (OSError, ValueError) as exc:
        errors.append(
            ValidationIssue(
                "invalid_image",
                f"sample image is invalid: {exc}",
                sample.sample_id,
            )
        )
        return

    expected_width, expected_height = manifest.preprocessing.image_size
    if (info.width, info.height) != (expected_width, expected_height):
        errors.append(
            ValidationIssue(
                "image_size_mismatch",
                (
                    f"sample image size {(info.width, info.height)} does not match "
                    f"expected {(expected_width, expected_height)}"
                ),
                sample.sample_id,
            )
        )
    if info.color_mode != manifest.preprocessing.color_mode:
        errors.append(
            ValidationIssue(
                "color_mode_mismatch",
                (
                    f"sample color mode {info.color_mode!r} does not match "
                    f"expected {manifest.preprocessing.color_mode!r}"
                ),
                sample.sample_id,
            )
        )


def _safe_sample_path(root: Path, sample: SampleRecord) -> Path | None:
    path = PurePosixPath(sample.relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    candidate = (root / Path(*path.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _netpbm_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        content = line.split("#", 1)[0]
        tokens.extend(content.split())
    return tokens


def _positive_int(raw: str, field_name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value
