"""CIFAR-10.1 v6 cross-source dataset contract for Phase 8C-1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from visionlab.data.cifar10 import CIFAR10_CLASSES
from visionlab.data.manifests import (
    ClassMapping,
    DatasetIdentity,
    DatasetManifest,
    PreprocessingSpec,
    SampleRecord,
)


CIFAR10_1_DATASET_ID = "cifar10-1"
CIFAR10_1_VERSION = "v6"
CIFAR10_1_EXPECTED_SAMPLE_COUNT = 2_000
CIFAR10_1_IMAGE_SIZE = (32, 32)
CIFAR10_1_IMAGE_SHAPE_HWC = (32, 32, 3)
CIFAR10_1_SPLIT_NAME = "cross_source_test"
CIFAR10_1_USAGE = "cross-source evaluation-only; never train, tune, or select checkpoints"
CIFAR10_1_SOURCE = "CIFAR-10.1 v6 official release"
CIFAR10_1_SOURCE_URLS = (
    "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
    "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
)
CIFAR10_1_LICENSE_OR_USAGE = (
    "Use according to the upstream CIFAR-10.1 release terms; preserve source identity."
)
CIFAR10_1_IDENTITY = DatasetIdentity(
    dataset_id=CIFAR10_1_DATASET_ID,
    version=CIFAR10_1_VERSION,
    source=CIFAR10_1_SOURCE,
    license_or_usage=CIFAR10_1_LICENSE_OR_USAGE,
    description="Phase 8C-1 cross-source evaluation-only CIFAR-10.1 v6 dataset contract.",
)
CIFAR10_1_PREPROCESSING = PreprocessingSpec(
    image_size=CIFAR10_1_IMAGE_SIZE,
    color_mode="RGB",
    value_range=(0.0, 1.0),
    normalization_mean=(0.0, 0.0, 0.0),
    normalization_std=(1.0, 1.0, 1.0),
    deterministic=True,
)
CIFAR10_1_EXPECTED_FILENAMES = (
    "cifar10.1_v6_data.npy",
    "cifar10.1_v6_labels.npy",
)
CIFAR10_1_SEARCH_DIR_NAMES = (
    "cifar10.1",
    "cifar10_1",
    "cifar10-1",
    "cifar10_1_v6",
    "cifar10-1-v6",
)


@dataclass(frozen=True)
class Cifar101Availability:
    available: bool
    root: Path | None
    data_path: Path | None
    labels_path: Path | None
    checked_roots: tuple[Path, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "root": str(self.root) if self.root is not None else "",
            "data_path": str(self.data_path) if self.data_path is not None else "",
            "labels_path": str(self.labels_path) if self.labels_path is not None else "",
            "checked_roots": [str(path) for path in self.checked_roots],
            "reason": self.reason,
        }


class Cifar101Dataset(Dataset):
    """Prediction-aware CIFAR-10.1 v6 dataset preserving sample metadata."""

    def __init__(
        self,
        images: Any,
        labels: Any,
        *,
        split: str = CIFAR10_1_SPLIT_NAME,
        source_id_prefix: str = "cifar10-1-v6",
        require_expected_count: bool = True,
    ) -> None:
        validate_cifar10_1_split(split)
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.split = split
        self.source_id_prefix = source_id_prefix
        if self.images.ndim != 4:
            raise ValueError("CIFAR-10.1 images must have shape N x 32 x 32 x 3")
        if tuple(self.images.shape[1:]) != CIFAR10_1_IMAGE_SHAPE_HWC:
            raise ValueError("CIFAR-10.1 images must have shape N x 32 x 32 x 3")
        if self.labels.ndim != 1:
            raise ValueError("CIFAR-10.1 labels must have shape N")
        if len(self.images) != len(self.labels):
            raise ValueError("CIFAR-10.1 image and label counts must match")
        if require_expected_count and len(self.images) != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
            raise ValueError("CIFAR-10.1 v6 must contain exactly 2,000 samples")
        if not torch.isfinite(self.images.float()).all():
            raise ValueError("CIFAR-10.1 images must contain only finite values")
        if int(self.labels.min().item()) < 0 or int(self.labels.max().item()) >= len(CIFAR10_CLASSES):
            raise ValueError("CIFAR-10.1 labels must be within CIFAR-10 class bounds")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        raw = _to_unit_chw(self.images[index])
        label = int(self.labels[index].item())
        sample_id = f"cifar10-1-v6-{index:05d}"
        return {
            "input": raw.detach().clone(),
            "raw_input": raw,
            "label": label,
            "sample_id": sample_id,
            "split": self.split,
            "source_id": f"{self.source_id_prefix}-{index:05d}",
            "dataset_id": CIFAR10_1_DATASET_ID,
            "dataset_version": CIFAR10_1_VERSION,
            "usage": CIFAR10_1_USAGE,
        }


def cifar10_1_contract_dict() -> dict[str, Any]:
    return {
        "identity": {
            "dataset_id": CIFAR10_1_IDENTITY.dataset_id,
            "version": CIFAR10_1_IDENTITY.version,
            "source": CIFAR10_1_IDENTITY.source,
            "source_urls": list(CIFAR10_1_SOURCE_URLS),
            "license_or_usage": CIFAR10_1_IDENTITY.license_or_usage,
            "description": CIFAR10_1_IDENTITY.description,
        },
        "expected_sample_count": CIFAR10_1_EXPECTED_SAMPLE_COUNT,
        "image_shape": list(CIFAR10_1_IMAGE_SHAPE_HWC),
        "classes": list(CIFAR10_CLASSES),
        "split_name": CIFAR10_1_SPLIT_NAME,
        "usage": CIFAR10_1_USAGE,
        "preprocessing": {
            "raw_representation": "unnormalized RGB unit tensor C x H x W in [0, 1]",
            "model_specific_preprocessing": "applied later by the evaluation runner",
            "image_size": list(CIFAR10_1_PREPROCESSING.image_size),
            "color_mode": CIFAR10_1_PREPROCESSING.color_mode,
            "value_range": list(CIFAR10_1_PREPROCESSING.value_range),
            "deterministic": CIFAR10_1_PREPROCESSING.deterministic,
        },
        "phase_boundary": {
            "phase": "8C-1",
            "material_model_evaluation_allowed": False,
            "official_cifar10_test_evaluation_allowed": False,
            "training_or_tuning_allowed": False,
            "model_selection_allowed": False,
            "ood_detection_claim_allowed": False,
        },
    }


def find_local_cifar10_1_v6(root: str | Path = "data") -> Cifar101Availability:
    root_path = Path(root)
    checked: list[Path] = []
    candidate_roots = [root_path]
    candidate_roots.extend(root_path / name for name in CIFAR10_1_SEARCH_DIR_NAMES)
    for candidate in candidate_roots:
        checked.append(candidate)
        data_path = candidate / CIFAR10_1_EXPECTED_FILENAMES[0]
        labels_path = candidate / CIFAR10_1_EXPECTED_FILENAMES[1]
        if data_path.exists() and labels_path.exists():
            return Cifar101Availability(
                available=True,
                root=candidate,
                data_path=data_path,
                labels_path=labels_path,
                checked_roots=tuple(checked),
                reason="found expected CIFAR-10.1 v6 .npy files",
            )
    return Cifar101Availability(
        available=False,
        root=None,
        data_path=None,
        labels_path=None,
        checked_roots=tuple(checked),
        reason="CIFAR-10.1 v6 files are not present locally; explicit acquisition approval required",
    )


def load_cifar10_1_v6(root: str | Path = "data") -> Cifar101Dataset:
    availability = find_local_cifar10_1_v6(root)
    if not availability.available or availability.data_path is None or availability.labels_path is None:
        raise FileNotFoundError(availability.reason)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to load CIFAR-10.1 v6 .npy files") from exc
    images = np.load(availability.data_path)
    labels = np.load(availability.labels_path)
    return Cifar101Dataset(images, labels, require_expected_count=True)


def validate_cifar10_1_split(split: str) -> None:
    if split != CIFAR10_1_SPLIT_NAME:
        raise ValueError("CIFAR-10.1 v6 is evaluation-only and exposes only cross_source_test")


def validate_cifar10_1_usage(usage: str) -> None:
    if usage != "evaluation":
        raise ValueError("CIFAR-10.1 v6 may only be used for evaluation, not train/tune/select")


def build_cifar10_1_manifest(dataset: Cifar101Dataset) -> DatasetManifest:
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        label_index = int(sample["label"])
        records.append(
            SampleRecord(
                sample_id=str(sample["sample_id"]),
                split=str(sample["split"]),
                label=CIFAR10_CLASSES[label_index],
                relative_path=f"{CIFAR10_1_VERSION}/{index:05d}",
                source_id=str(sample["source_id"]),
                group_id=f"cifar10-1-v6-{index:05d}",
                checksum="",
            )
        )
    return DatasetManifest(
        identity=CIFAR10_1_IDENTITY,
        classes=ClassMapping(CIFAR10_CLASSES),
        split_names=(CIFAR10_1_SPLIT_NAME,),
        preprocessing=CIFAR10_1_PREPROCESSING,
        samples=tuple(records),
    )


def cifar10_1_sample_label_digest(dataset: Cifar101Dataset) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        sample = dataset[index]
        digest.update(str(sample["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(sample["label"])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cifar10_1_file_digest(availability: Cifar101Availability) -> dict[str, str]:
    if not availability.available or availability.data_path is None or availability.labels_path is None:
        return {}
    return {
        "data_sha256": _sha256_file(availability.data_path),
        "labels_sha256": _sha256_file(availability.labels_path),
    }


def phase8c1_tiny_fixture_dataset(sample_count: int = 4) -> Cifar101Dataset:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    images = torch.zeros(sample_count, 32, 32, 3, dtype=torch.uint8)
    labels = torch.arange(sample_count, dtype=torch.long) % len(CIFAR10_CLASSES)
    for index in range(sample_count):
        channel = index % 3
        images[index, :, :, channel] = 32 + index * 16
    return Cifar101Dataset(images, labels, require_expected_count=False)


def verify_cifar10_1_dataset_contract(
    dataset: Cifar101Dataset,
    *,
    require_expected_count: bool = True,
) -> dict[str, Any]:
    if require_expected_count and len(dataset) != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
        raise ValueError("CIFAR-10.1 v6 must contain exactly 2,000 samples")
    if dataset.split != CIFAR10_1_SPLIT_NAME:
        raise ValueError("CIFAR-10.1 split identity must remain cross_source_test")
    sample_ids = []
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        raw = sample["raw_input"]
        if tuple(raw.shape) != (3, 32, 32):
            raise ValueError("CIFAR-10.1 raw tensors must have shape 3 x 32 x 32")
        if not torch.isfinite(raw).all():
            raise ValueError("CIFAR-10.1 raw tensors must contain only finite values")
        if float(raw.min().item()) < 0.0 or float(raw.max().item()) > 1.0:
            raise ValueError("CIFAR-10.1 raw tensors must be in [0, 1]")
        if sample["dataset_id"] != CIFAR10_1_DATASET_ID or sample["dataset_version"] != CIFAR10_1_VERSION:
            raise ValueError("CIFAR-10.1 sample metadata must preserve dataset identity")
        sample_ids.append(str(sample["sample_id"]))
        labels.append(int(sample["label"]))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("CIFAR-10.1 sample IDs must be stable and unique")
    class_counts = {CIFAR10_CLASSES[index]: labels.count(index) for index in range(len(CIFAR10_CLASSES))}
    return {
        "status": "passed",
        "dataset_id": CIFAR10_1_DATASET_ID,
        "version": CIFAR10_1_VERSION,
        "split": dataset.split,
        "sample_count": len(dataset),
        "image_shape": list(CIFAR10_1_IMAGE_SHAPE_HWC),
        "class_count": len(CIFAR10_CLASSES),
        "class_names": list(CIFAR10_CLASSES),
        "class_counts": class_counts,
        "sample_label_digest": cifar10_1_sample_label_digest(dataset),
        "usage": CIFAR10_1_USAGE,
    }


def select_phase8c1_visual_samples(dataset: Cifar101Dataset) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        label = int(sample["label"])
        if label not in seen:
            selected.append(
                {
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "source_id": sample["source_id"],
                    "split": sample["split"],
                    "label_index": label,
                    "label": CIFAR10_CLASSES[label],
                }
            )
            seen.add(label)
        if len(seen) == len(CIFAR10_CLASSES):
            break
    if len(selected) != len(CIFAR10_CLASSES):
        raise ValueError("visual QA requires at least one CIFAR-10.1 sample per class")
    return selected


def write_phase8c1_visual_grid(
    dataset: Cifar101Dataset,
    visual_manifest: list[dict[str, Any]],
    path: Path,
) -> Path:
    from PIL import Image, ImageDraw

    cell = 64
    label_height = 18
    columns = 5
    rows = 2
    image = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(image)
    for position, record in enumerate(visual_manifest):
        row = position // columns
        col = position % columns
        x = col * cell
        y = row * (cell + label_height)
        sample = dataset[int(record["index"])]
        tile = _tensor_to_image(sample["raw_input"]).resize((cell, cell), Image.Resampling.NEAREST)
        image.paste(tile, (x, y + label_height))
        draw.text((x + 2, y + 2), str(record["label"]), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_phase8c1_preflight_artifacts(
    output_dir: Path | str,
    *,
    data_root: Path | str = "data",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    artifact_dir = output_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    availability = find_local_cifar10_1_v6(data_root)
    contract = cifar10_1_contract_dict()
    availability_dict = availability.to_dict()
    availability_dict["file_digests"] = cifar10_1_file_digest(availability)
    fixture = phase8c1_tiny_fixture_dataset()
    fixture_report = verify_cifar10_1_dataset_contract(fixture, require_expected_count=False)
    fixture_manifest = build_cifar10_1_manifest(fixture).to_dict()
    real_report: dict[str, Any] | None = None
    real_manifest: dict[str, Any] | None = None
    visual_manifest: list[dict[str, Any]] = []
    visual_grid_path: Path | None = None
    if availability.available:
        real_dataset = load_cifar10_1_v6(data_root)
        real_report = verify_cifar10_1_dataset_contract(real_dataset, require_expected_count=True)
        real_report["file_digests"] = availability_dict["file_digests"]
        real_report["source_urls"] = list(CIFAR10_1_SOURCE_URLS)
        real_manifest = build_cifar10_1_manifest(real_dataset).to_dict()
        visual_manifest = select_phase8c1_visual_samples(real_dataset)
        visual_grid_path = artifact_dir / "phase8c1_cifar10_1_v6_visual_grid.png"
        write_phase8c1_visual_grid(real_dataset, visual_manifest, visual_grid_path)

    contract_path = artifact_dir / "phase8c1_cifar10_1_contract.json"
    availability_path = artifact_dir / "phase8c1_local_availability.json"
    fixture_report_path = artifact_dir / "phase8c1_tiny_fixture_smoke.json"
    fixture_manifest_path = artifact_dir / "phase8c1_tiny_fixture_manifest.json"
    real_report_path = artifact_dir / "phase8c1_cifar10_1_v6_registration.json"
    real_manifest_path = artifact_dir / "phase8c1_cifar10_1_v6_manifest.json"
    visual_manifest_path = artifact_dir / "phase8c1_visual_sample_manifest.json"
    note_path = output_path / "phase8c1_inspection_note.md"
    result_path = output_path / "phase8c1_result.json"

    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    availability_path.write_text(json.dumps(availability_dict, indent=2), encoding="utf-8")
    fixture_report_path.write_text(json.dumps(fixture_report, indent=2), encoding="utf-8")
    fixture_manifest_path.write_text(json.dumps(fixture_manifest, indent=2), encoding="utf-8")
    if real_report is not None and real_manifest is not None:
        real_report_path.write_text(json.dumps(real_report, indent=2), encoding="utf-8")
        real_manifest_path.write_text(json.dumps(real_manifest, indent=2), encoding="utf-8")
        visual_manifest_path.write_text(json.dumps(visual_manifest, indent=2), encoding="utf-8")

    paths: dict[str, Path] = {
        "contract": contract_path,
        "local_availability": availability_path,
        "tiny_fixture_smoke": fixture_report_path,
        "tiny_fixture_manifest": fixture_manifest_path,
    }
    if visual_grid_path is not None:
        paths.update(
            {
                "real_registration": real_report_path,
                "real_manifest": real_manifest_path,
                "visual_sample_manifest": visual_manifest_path,
                "visual_grid": visual_grid_path,
            }
        )
    _write_phase8c1_note(
        note_path,
        availability=availability,
        fixture_report=fixture_report,
        real_report=real_report,
        visual_manifest=visual_manifest,
        visual_grid_path=visual_grid_path,
        paths=paths,
    )
    result: dict[str, Any] = {
        "status": "real_registration_and_visual_qa_complete" if availability.available else "blocked_external_data_required",
        "phase": "8C-1",
        "real_cifar10_1_registration_complete": availability.available,
        "real_visual_qa_complete": availability.available,
        "model_evaluation_performed": False,
        "contract": str(contract_path),
        "local_availability": str(availability_path),
        "tiny_fixture_smoke": str(fixture_report_path),
        "tiny_fixture_manifest": str(fixture_manifest_path),
        "inspection_note": str(note_path),
    }
    if availability.available:
        result.update(
            {
                "real_registration": str(real_report_path),
                "real_manifest": str(real_manifest_path),
                "visual_sample_manifest": str(visual_manifest_path),
                "visual_grid": str(visual_grid_path),
            }
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result"] = str(result_path)
    return result


def _to_unit_chw(image: Tensor) -> Tensor:
    tensor = image.detach().clone()
    if tensor.ndim != 3 or tuple(tensor.shape) != CIFAR10_1_IMAGE_SHAPE_HWC:
        raise ValueError("CIFAR-10.1 sample image must have shape 32 x 32 x 3")
    tensor = tensor.float()
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    if float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
        raise ValueError("CIFAR-10.1 image values must be in [0, 1]")
    return tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_phase8c1_note(
    path: Path,
    *,
    availability: Cifar101Availability,
    fixture_report: dict[str, Any],
    real_report: dict[str, Any] | None,
    visual_manifest: list[dict[str, Any]],
    visual_grid_path: Path | None,
    paths: dict[str, Path],
) -> Path:
    lines = [
        "# Phase 8C-1 CIFAR-10.1 v6 Registration Inspection Note",
        "",
        "Status: implementation/preflight artifact; Phase 8C-1 is not closed.",
        "",
        "## Scope",
        "",
        "Phase 8C-1 registers the CIFAR-10.1 v6 cross-source/evaluation-only contract and verifies tiny fixture mechanics. It does not evaluate any model checkpoint.",
        "",
        "## Local Data Availability",
        "",
        f"- Available locally: `{availability.available}`",
        f"- Reason: {availability.reason}",
    ]
    if not availability.available:
        lines.extend(
            [
                "- Required files were not found locally.",
                "- Explicit builder approval is required before downloading or acquiring CIFAR-10.1 v6.",
                "- Real CIFAR-10.1 v6 registration and visual QA are blocked until acquisition is approved and completed.",
            ]
        )
    else:
        file_digests = real_report.get("file_digests", {}) if real_report else {}
        lines.extend(
            [
                f"- Source URLs: `{CIFAR10_1_SOURCE_URLS[0]}`, `{CIFAR10_1_SOURCE_URLS[1]}`",
                f"- Sample count: `{real_report['sample_count'] if real_report else ''}`",
                f"- Image shape: `{CIFAR10_1_IMAGE_SHAPE_HWC}`",
                f"- Sample-label digest: `{real_report['sample_label_digest'] if real_report else ''}`",
                f"- File digests: `{json.dumps(file_digests, sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Tiny Fixture Smoke",
            "",
            f"- Status: `{fixture_report['status']}`",
            f"- Fixture sample count: `{fixture_report['sample_count']}`",
            f"- Fixture sample-label digest: `{fixture_report['sample_label_digest']}`",
            "",
            "## Visual QA",
            "",
            f"- Real visual grid: `{visual_grid_path}`" if visual_grid_path is not None else "- Real CIFAR-10.1 v6 visual QA was not generated because the dataset is not locally available.",
            "- Visual QA is qualitative only and does not prove label correctness, OOD difficulty, robustness, or model behavior.",
            "",
            "## Fixed Visual Samples",
            "",
        ]
    )
    if visual_manifest:
        for sample in visual_manifest:
            lines.append(
                f"- `{sample['sample_id']}`: label `{sample['label']}`, source `{sample['source_id']}`"
            )
    else:
        lines.append("- None; real dataset is unavailable.")
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- Pending builder manual visual review.",
            "- Record only visible image/label plausibility observations from the generated grid.",
            "",
            "## Approval Recommendation",
            "",
            "- Pending builder review.",
            "- Registration evidence supports proceeding to a Phase 8C-1 phase check once manual visual QA is reviewed.",
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for name, artifact_path in paths.items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.extend(
        [
            "",
            "## Explicit Non-Claims",
            "",
            "- No Phase 4B, Phase 6B-2, or Phase 6C-2 checkpoint evaluation occurred.",
            "- No official CIFAR-10 test split evaluation occurred.",
            "- No material OOD/cross-source evaluation occurred.",
            "- No training, tuning, model selection, checkpoint modification, Phase 8C-2, or Phase 9 work occurred.",
            "- No OOD-detection claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tensor_to_image(tensor: Tensor):
    from PIL import Image

    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")
