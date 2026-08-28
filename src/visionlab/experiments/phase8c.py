"""Phase 8C-2A cross-source evaluation preflight for CIFAR-10.1 v6."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from visionlab.data.cifar10 import CIFAR10_CLASSES, CIFAR10_PREPROCESSING, normalize_tensor
from visionlab.data.cifar10_1 import (
    CIFAR10_1_DATASET_ID,
    CIFAR10_1_EXPECTED_SAMPLE_COUNT,
    CIFAR10_1_IMAGE_SHAPE_HWC,
    CIFAR10_1_SPLIT_NAME,
    CIFAR10_1_USAGE,
    CIFAR10_1_VERSION,
    cifar10_1_contract_dict,
    cifar10_1_file_digest,
    find_local_cifar10_1_v6,
    load_cifar10_1_v6,
    validate_cifar10_1_usage,
    verify_cifar10_1_dataset_contract,
)
from visionlab.data.transfer_preprocessing import preprocess_resnet18_imagenet_tensor
from visionlab.evaluation.metrics import classification_metrics_from_predictions
from visionlab.evaluation.calibration import calibration_summary
from visionlab.evaluation.classification import evaluate_classification
from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import PHASE6C_RUN_ID
from visionlab.experiments.phase7 import (
    PHASE7_NUM_CALIBRATION_BINS,
    Phase7RunReference,
    _load_phase4b_model,
    _load_phase6b2_model,
    _load_phase6c_model,
    phase7_references,
    sha256_file,
)


PHASE8C2A_RUN_ID = "phase8c2a-cifar10-1-v6-cross-source-preflight"
PHASE8C2A_OUTPUT_DIR = Path("outputs") / PHASE8C2A_RUN_ID
PHASE8C2B_RUN_ID = 'phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation'
PHASE8C2B_OUTPUT_DIR = Path('outputs') / PHASE8C2B_RUN_ID
PHASE8C1_OUTPUT_DIR = Path("outputs") / "phase8c1-cifar10-1-registration-visual-qa-tiny-smoke"
PHASE8C1_SAMPLE_LABEL_DIGEST = "2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75"
PHASE8C2_EXPECTED_MODEL_ROWS = 3
PHASE8C2_EXPECTED_DELTA_ROWS = 3
PHASE7_COMPARISON_TABLE = Path("outputs") / "phase7-evaluation-harness-and-calibration" / "artifacts" / "phase7_comparison_table.csv"
PHASE8C2_TINY_SMOKE_SAMPLE_COUNT = 6
PHASE8C2B_BATCH_SIZE = 8
PHASE8C2B_RUNTIME_GUARD_SECONDS = 45 * 60


@dataclass(frozen=True)
class Phase8C2AResult:
    run_dir: Path
    run_id: str
    status: str
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "status": self.status,
            "artifact_paths": dict(self.artifact_paths),
        }


class Cifar101Subset(Dataset):
    def __init__(self, dataset: Dataset, *, sample_count: int) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sample_count > len(dataset):
            raise ValueError("sample_count cannot exceed dataset length")
        self.dataset = dataset
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        return self.dataset[index]


class Phase8C2CustomPredictionView(Dataset):
    preprocessing_id = "phase4-cifar10-normalization"

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        raw = _raw_cifar10_1_tensor(sample)
        return _sample_with_model_input(
            sample,
            raw_cross_source_input=raw,
            model_input=normalize_tensor(raw, CIFAR10_PREPROCESSING),
            preprocessing_id=self.preprocessing_id,
        )


class Phase8C2TransferPredictionView(Dataset):
    preprocessing_id = "phase6a-resnet18-imagenet1k-v1-preprocessing"

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        raw = _raw_cifar10_1_tensor(sample)
        return _sample_with_model_input(
            sample,
            raw_cross_source_input=raw,
            model_input=preprocess_resnet18_imagenet_tensor(raw),
            preprocessing_id=self.preprocessing_id,
        )


def run_phase8c2a_preflight(
    run_dir: Path | str = PHASE8C2A_OUTPUT_DIR,
    *,
    data_root: str | Path = "data",
) -> Phase8C2AResult:
    """Write Phase 8C-2A preflight artifacts without material model evaluation."""

    run_path = Path(run_dir)
    verify_phase8c2_output_isolation(run_path)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar10_1_v6(data_root)
    dataset_report = verify_phase8c2_dataset_identity(dataset)
    references = phase7_references()
    checkpoint_manifest = build_phase8c2_checkpoint_manifest(references)
    historical_reference = load_phase7_historical_test_reference()
    verify_historical_reference_rows(historical_reference, references)
    material_contract = build_phase8c2_material_contract(
        dataset_report=dataset_report,
        checkpoint_manifest=checkpoint_manifest,
        historical_reference=historical_reference,
    )
    verify_phase8c2_material_contract(material_contract)

    tiny_subset = Cifar101Subset(dataset, sample_count=PHASE8C2_TINY_SMOKE_SAMPLE_COUNT)
    preprocessing_report = verify_phase8c2_preprocessing_order(tiny_subset)
    tiny_smoke = build_phase8c2_tiny_smoke_report(tiny_subset)
    artifact_schema = phase8c2_expected_artifact_schema()

    contract_path = artifact_dir / "phase8c2a_material_contract.json"
    dataset_path = artifact_dir / "phase8c2a_cifar10_1_dataset_identity.json"
    checkpoint_path = artifact_dir / "phase8c2a_checkpoint_manifest.json"
    reference_path = artifact_dir / "phase8c2a_historical_phase7_test_reference.json"
    preprocessing_path = artifact_dir / "phase8c2a_preprocessing_verification.json"
    tiny_smoke_path = artifact_dir / "phase8c2a_tiny_smoke.json"
    schema_path = artifact_dir / "phase8c2a_expected_artifact_schema.json"
    validation_path = artifact_dir / "phase8c2a_artifact_validation.json"
    report_path = run_path / "phase8c2a_preflight_report.md"
    result_path = run_path / "phase8c2a_result.json"

    _write_json(contract_path, material_contract)
    _write_json(dataset_path, dataset_report)
    _write_json(checkpoint_path, checkpoint_manifest)
    _write_json(reference_path, historical_reference)
    _write_json(preprocessing_path, preprocessing_report)
    _write_json(tiny_smoke_path, tiny_smoke)
    _write_json(schema_path, artifact_schema)
    artifact_validation = verify_phase8c2a_artifacts(
        [
            contract_path,
            dataset_path,
            checkpoint_path,
            reference_path,
            preprocessing_path,
            tiny_smoke_path,
            schema_path,
        ]
    )
    _write_json(validation_path, artifact_validation)
    write_phase8c2a_report(
        report_path,
        dataset_report=dataset_report,
        checkpoint_manifest=checkpoint_manifest,
        historical_reference=historical_reference,
        preprocessing_report=preprocessing_report,
        tiny_smoke=tiny_smoke,
    )

    artifact_paths = {
        "material_contract": str(contract_path),
        "dataset_identity": str(dataset_path),
        "checkpoint_manifest": str(checkpoint_path),
        "historical_phase7_test_reference": str(reference_path),
        "preprocessing_verification": str(preprocessing_path),
        "tiny_smoke": str(tiny_smoke_path),
        "expected_artifact_schema": str(schema_path),
        "artifact_validation": str(validation_path),
        "preflight_report": str(report_path),
    }
    result = Phase8C2AResult(
        run_dir=run_path,
        run_id=PHASE8C2A_RUN_ID,
        status="completed_preflight_only",
        artifact_paths=artifact_paths,
    )
    artifact_paths["result"] = str(result_path)
    _write_json(result_path, result.to_dict())
    return result


def verify_phase8c2_dataset_identity(dataset: Dataset) -> dict[str, Any]:
    report = verify_cifar10_1_dataset_contract(dataset, require_expected_count=True)
    if report["dataset_id"] != CIFAR10_1_DATASET_ID or report["version"] != CIFAR10_1_VERSION:
        raise ValueError("Phase 8C-2 requires CIFAR-10.1 v6")
    if report["split"] != CIFAR10_1_SPLIT_NAME:
        raise ValueError("Phase 8C-2 requires cross_source_test split")
    if report["sample_count"] != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
        raise ValueError("Phase 8C-2 requires exactly 2,000 CIFAR-10.1 samples")
    if report["sample_label_digest"] != PHASE8C1_SAMPLE_LABEL_DIGEST:
        raise ValueError("CIFAR-10.1 sample-label digest does not match accepted Phase 8C-1")
    if tuple(report["class_names"]) != CIFAR10_CLASSES:
        raise ValueError("CIFAR-10.1 class map must exactly match CIFAR-10")
    validate_cifar10_1_usage("evaluation")
    report["accepted_phase8c1_sample_label_digest"] = PHASE8C1_SAMPLE_LABEL_DIGEST
    report["evaluation_only_enforced"] = True
    return report


def build_phase8c2_checkpoint_manifest(references: tuple[Any, ...]) -> dict[str, Any]:
    if len(references) != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2 requires exactly three fixed checkpoint references")
    checkpoints = []
    for reference in references:
        if not reference.checkpoint_path.exists():
            raise FileNotFoundError(f"missing fixed checkpoint for {reference.run_id}: {reference.checkpoint_path}")
        checkpoints.append(
            {
                "run_id": reference.run_id,
                "display_name": reference.display_name,
                "model_family": reference.model_family,
                "checkpoint_path": str(reference.checkpoint_path),
                "checkpoint_tag": reference.checkpoint_tag,
                "checkpoint_sha256": sha256_file(reference.checkpoint_path),
                "fixed_reference": True,
            }
        )
    return {
        "status": "passed",
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "checkpoint_mutation_allowed": False,
    }


def load_phase7_historical_test_reference(path: Path = PHASE7_COMPARISON_TABLE) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing accepted Phase 7 comparison table: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                rows.append(row)
    if len(rows) != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2A expects exactly three historical Phase 7 test rows")
    return {
        "status": "passed",
        "source": str(path),
        "reference_split": "test",
        "reference_semantics": "previously accepted Phase 7 official CIFAR-10 test summaries; not rerun in Phase 8C-2A",
        "official_test_rerun_performed": False,
        "rows": [_coerce_metric_row(row) for row in rows],
    }


def verify_historical_reference_rows(reference: dict[str, Any], references: tuple[Any, ...]) -> dict[str, Any]:
    expected_run_ids = [item.run_id for item in references]
    actual_run_ids = [row["run_id"] for row in reference["rows"]]
    if actual_run_ids != expected_run_ids:
        raise ValueError("historical Phase 7 test reference rows do not match fixed checkpoint order")
    if reference.get("official_test_rerun_performed") is not False:
        raise ValueError("Phase 8C-2A must not rerun official CIFAR-10 test evaluation")
    return {"status": "passed"}


def build_phase8c2_material_contract(
    *,
    dataset_report: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    historical_reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "8C-2A",
        "run_id": PHASE8C2A_RUN_ID,
        "future_material_run_id": PHASE8C2B_RUN_ID,
        "scope": "cross-source evaluation runner contracts, preflight, tiny smoke, artifact validation, and tests only",
        "material_cross_source_evaluation_performed": False,
        "future_material_dataset": {
            "dataset_id": dataset_report["dataset_id"],
            "version": dataset_report["version"],
            "split": dataset_report["split"],
            "sample_count": dataset_report["sample_count"],
            "sample_label_digest": dataset_report["sample_label_digest"],
            "usage": CIFAR10_1_USAGE,
        },
        "expected_future_model_metric_rows": PHASE8C2_EXPECTED_MODEL_ROWS,
        "expected_future_cross_source_delta_rows": PHASE8C2_EXPECTED_DELTA_ROWS,
        "checkpoint_manifest": checkpoint_manifest,
        "historical_indistribution_reference": historical_reference,
        "metric_semantics_source": "Phase 7 evaluation, metrics, and calibration helpers; unchanged",
        "calibration_bins": PHASE7_NUM_CALIBRATION_BINS,
        "official_cifar10_test_rerun_performed": False,
        "training_or_tuning_performed": False,
        "model_selection_performed": False,
        "checkpoint_mutation_performed": False,
        "phase8c2b_started": False,
        "phase9_started": False,
        "unsupported_ood_detection_claim": False,
    }


def verify_phase8c2_material_contract(contract: dict[str, Any]) -> dict[str, Any]:
    future = contract["future_material_dataset"]
    if future["dataset_id"] != CIFAR10_1_DATASET_ID or future["version"] != CIFAR10_1_VERSION:
        raise ValueError("Phase 8C-2 material contract must use CIFAR-10.1 v6")
    if future["split"] != CIFAR10_1_SPLIT_NAME:
        raise ValueError("Phase 8C-2 material contract must use cross_source_test")
    if future["sample_count"] != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
        raise ValueError("Phase 8C-2 material contract must expect 2,000 samples")
    if future["sample_label_digest"] != PHASE8C1_SAMPLE_LABEL_DIGEST:
        raise ValueError("Phase 8C-2 material contract digest must match Phase 8C-1")
    if contract["expected_future_model_metric_rows"] != 3:
        raise ValueError("Phase 8C-2 material contract must expect three model metric rows")
    if contract["expected_future_cross_source_delta_rows"] != 3:
        raise ValueError("Phase 8C-2 material contract must expect three cross-source delta rows")
    forbidden_flags = [
        "material_cross_source_evaluation_performed",
        "official_cifar10_test_rerun_performed",
        "training_or_tuning_performed",
        "model_selection_performed",
        "checkpoint_mutation_performed",
        "phase8c2b_started",
        "phase9_started",
        "unsupported_ood_detection_claim",
    ]
    for flag in forbidden_flags:
        if contract.get(flag) is not False:
            raise ValueError(f"Phase 8C-2A contract must keep {flag}=False")
    return {"status": "passed"}


def verify_phase8c2_preprocessing_order(dataset: Dataset) -> dict[str, Any]:
    rows = []
    custom_view = Phase8C2CustomPredictionView(dataset)
    transfer_view = Phase8C2TransferPredictionView(dataset)
    for index in range(len(dataset)):
        custom = custom_view[index]
        transfer = transfer_view[index]
        if custom["sample_id"] != transfer["sample_id"]:
            raise ValueError("preprocessing views produced mismatched sample IDs")
        if int(custom["label"]) != int(transfer["label"]):
            raise ValueError("preprocessing views produced mismatched labels")
        if not torch.equal(custom["raw_cross_source_input"], transfer["raw_cross_source_input"]):
            raise ValueError("raw CIFAR-10.1 unit tensor must be identical before model-specific preprocessing")
        rows.append(
            {
                "sample_id": custom["sample_id"],
                "label": int(custom["label"]),
                "raw_shape": list(custom["raw_cross_source_input"].shape),
                "custom_input_shape": list(custom["input"].shape),
                "transfer_input_shape": list(transfer["input"].shape),
                "custom_preprocessing_id": custom["preprocessing_id"],
                "transfer_preprocessing_id": transfer["preprocessing_id"],
                "raw_inputs_identical_before_preprocessing": True,
            }
        )
    return {
        "status": "passed",
        "sample_count": len(dataset),
        "model_specific_preprocessing_after_raw_unit_tensor": True,
        "checks": rows,
    }


def build_phase8c2_tiny_smoke_report(dataset: Dataset) -> dict[str, Any]:
    predictions = [index % len(CIFAR10_CLASSES) for index in range(len(dataset))]
    labels = [int(dataset[index]["label"]) for index in range(len(dataset))]
    probabilities = []
    for prediction in predictions:
        row = [0.05 for _ in CIFAR10_CLASSES]
        row[prediction] = 0.55
        normalizer = sum(row)
        probabilities.append([value / normalizer for value in row])
    confidences = [max(row) for row in probabilities]
    metrics = classification_metrics_from_predictions(
        labels,
        predictions,
        probabilities,
        class_names=CIFAR10_CLASSES,
    )
    calibration = calibration_summary(
        confidences,
        [true == predicted for true, predicted in zip(labels, predictions)],
        num_bins=PHASE7_NUM_CALIBRATION_BINS,
    )
    return {
        "status": "passed",
        "sample_count": len(dataset),
        "metrics_are_material_results": False,
        "metrics_are_conclusive": False,
        "model_checkpoint_evaluation_performed": False,
        "purpose": "verify Phase 7 metric/calibration helper compatibility on tiny synthetic predictions only",
        "metric_semantics_source": "Phase 7 helpers reused unchanged",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["averages"]["f1"]["macro"],
        "ece_10_bin": calibration.expected_calibration_error,
    }


def phase8c2_expected_artifact_schema() -> dict[str, Any]:
    return {
        "status": "planned",
        "phase8c2b_run_id": PHASE8C2B_RUN_ID,
        "required_future_material_artifacts": [
            "phase8c2b_material_run_contract.json",
            "artifacts/phase8c2_cross_source_metrics.csv",
            "artifacts/phase8c2_cross_source_deltas.csv",
            "artifacts/phase8c2_sample_alignment.json",
            "artifacts/phase8c2_artifact_validation.json",
            "phase8c2_cross_source_report.md",
            "phase8c2b_result.json",
        ],
        "expected_future_model_metric_rows": PHASE8C2_EXPECTED_MODEL_ROWS,
        "expected_future_cross_source_delta_rows": PHASE8C2_EXPECTED_DELTA_ROWS,
        "metric_columns": [
            "run_id",
            "dataset_id",
            "dataset_version",
            "split",
            "total_examples",
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ece",
            "average_confidence",
            "incorrect_average_confidence",
        ],
        "delta_columns": [
            "run_id",
            "reference_split",
            "cross_source_split",
            "accuracy_delta_from_phase7_test",
            "balanced_accuracy_delta_from_phase7_test",
            "macro_f1_delta_from_phase7_test",
            "ece_delta_from_phase7_test",
            "average_confidence_delta_from_phase7_test",
            "incorrect_average_confidence_delta_from_phase7_test",
        ],
    }


def verify_phase8c2_output_isolation(run_dir: Path | str) -> dict[str, Any]:
    run_path = Path(run_dir)
    forbidden = [
        Path("outputs") / "phase7-evaluation-harness-and-calibration",
        Path("outputs") / "phase8a-degradation-registry-visual-qa-tiny-smoke",
        Path("outputs") / "phase8b1-robustness-plumbing-validation-smoke",
        Path("outputs") / "phase8b2a-validation-robustness-runner-preflight",
        Path("outputs") / "phase8b2b-fixed-checkpoint-validation-robustness-sweep",
        PHASE8C1_OUTPUT_DIR,
    ]
    for forbidden_path in forbidden:
        if run_path == forbidden_path or forbidden_path in run_path.parents:
            raise ValueError(f"Phase 8C-2A output directory must not overlap {forbidden_path}")
    return {"status": "passed", "run_dir": str(run_path)}


def verify_phase8c2a_artifacts(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size <= 0:
            raise ValueError(f"artifact is empty: {path}")
    return {"status": "passed", "file_count": len(paths), "files": [str(path) for path in paths]}


def write_phase8c2a_report(
    path: Path,
    *,
    dataset_report: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    historical_reference: dict[str, Any],
    preprocessing_report: dict[str, Any],
    tiny_smoke: dict[str, Any],
) -> Path:
    lines = [
        "# Phase 8C-2A Cross-Source Evaluation Preflight Report",
        "",
        "This report is preflight and tiny-smoke evidence only. It is not material cross-source/OOD evaluation evidence.",
        "",
        "## Scope",
        "",
        f"- Phase: `8C-2A`.",
        f"- Future material run: `{PHASE8C2B_RUN_ID}`.",
        f"- Dataset: `{dataset_report['dataset_id']}` version `{dataset_report['version']}` split `{dataset_report['split']}`.",
        f"- Future material sample count: `{dataset_report['sample_count']}`.",
        f"- Fixed checkpoints: `{checkpoint_manifest['checkpoint_count']}`.",
        f"- Expected future model metric rows: `{PHASE8C2_EXPECTED_MODEL_ROWS}`.",
        f"- Expected future cross-source delta rows: `{PHASE8C2_EXPECTED_DELTA_ROWS}`.",
        "- Official CIFAR-10 test metrics are historical Phase 7 reference summaries only and were not rerun.",
        "",
        "## Dataset Identity",
        "",
        f"- Sample-label digest: `{dataset_report['sample_label_digest']}`.",
        f"- File SHA-256 digests: `{json.dumps(dataset_report.get('file_digests', {}), sort_keys=True)}`.",
        f"- Class distribution: `{json.dumps(dataset_report['class_counts'], sort_keys=True)}`.",
        "",
        "## Preprocessing Verification",
        "",
        f"- Status: `{preprocessing_report['status']}`.",
        "- CustomCNN and ResNet-18 views share identical raw CIFAR-10.1 unit tensors before model-specific preprocessing.",
        "",
        "## Historical Reference Boundary",
        "",
        f"- Source: `{historical_reference['source']}`.",
        f"- Semantics: {historical_reference['reference_semantics']}.",
        "",
        "## Tiny Smoke",
        "",
        f"- Status: `{tiny_smoke['status']}`.",
        f"- Sample count: `{tiny_smoke['sample_count']}`.",
        "- Tiny smoke metrics are non-material and non-conclusive.",
        "- No checkpoint inference was performed by the tiny smoke.",
        "",
        "## Explicit Non-Claims",
        "",
        "- No material 2,000-sample CIFAR-10.1 evaluation occurred.",
        "- No official CIFAR-10 test split rerun occurred.",
        "- No training, tuning, model selection, or checkpoint modification occurred.",
        "- No Phase 8C-2B or Phase 9 work occurred.",
        "- No OOD-detection, robustness, deployment-reliability, or general model-superiority claim is made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _raw_cifar10_1_tensor(sample: dict[str, Any]) -> Tensor:
    raw = sample["raw_input"].detach().clone().contiguous()
    if tuple(raw.shape) != (3, 32, 32):
        raise ValueError("raw CIFAR-10.1 tensor must have shape 3 x 32 x 32")
    if not torch.isfinite(raw).all():
        raise ValueError("raw CIFAR-10.1 tensor must be finite")
    if float(raw.min().item()) < 0.0 or float(raw.max().item()) > 1.0:
        raise ValueError("raw CIFAR-10.1 tensor must be in [0, 1]")
    return raw


def _sample_with_model_input(
    sample: dict[str, Any],
    *,
    raw_cross_source_input: Tensor,
    model_input: Tensor,
    preprocessing_id: str,
) -> dict[str, Any]:
    return {
        **sample,
        "input": model_input,
        "raw_cross_source_input": raw_cross_source_input,
        "sample_id": str(sample["sample_id"]),
        "source_id": str(sample.get("source_id", sample["sample_id"])),
        "label": int(sample["label"]),
        "split": CIFAR10_1_SPLIT_NAME,
        "preprocessing_id": preprocessing_id,
    }


def _coerce_metric_row(row: dict[str, str]) -> dict[str, Any]:
    numeric = {
        "total_examples",
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_macro",
        "pr_auc_macro",
        "ece",
        "average_confidence",
        "incorrect_average_confidence",
    }
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key in numeric:
            output[key] = int(value) if key == "total_examples" else float(value)
        else:
            output[key] = value
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path






@dataclass(frozen=True)
class Phase8C2BResult:
    run_dir: Path
    run_id: str
    status: str
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "status": self.status,
            "artifact_paths": dict(self.artifact_paths),
        }


def build_phase8c2b_material_contract(
    *,
    dataset_report: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    historical_reference: dict[str, Any],
    runtime_projection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "8C-2B",
        "run_id": PHASE8C2B_RUN_ID,
        "scope": "single material CIFAR-10.1 v6 cross-source evaluation for three fixed accepted checkpoints",
        "dataset": {
            "dataset_id": dataset_report["dataset_id"],
            "version": dataset_report["version"],
            "split": dataset_report["split"],
            "sample_count": dataset_report["sample_count"],
            "sample_label_digest": dataset_report["sample_label_digest"],
            "usage": CIFAR10_1_USAGE,
        },
        "checkpoint_manifest": checkpoint_manifest,
        "historical_indistribution_reference": historical_reference,
        "runtime_projection": runtime_projection,
        "expected_cross_source_metric_rows": PHASE8C2_EXPECTED_MODEL_ROWS,
        "expected_historical_reference_delta_rows": PHASE8C2_EXPECTED_DELTA_ROWS,
        "metric_semantics_source": "Phase 7 evaluation, metrics, and calibration helpers; unchanged",
        "calibration_bins": PHASE7_NUM_CALIBRATION_BINS,
        "raw_input_invariant": "Each model receives the same raw CIFAR-10.1 unit RGB tensor before model-specific preprocessing.",
        "official_cifar10_test_rerun_performed": False,
        "training_or_tuning_performed": False,
        "model_selection_performed": False,
        "checkpoint_mutation_performed": False,
        "additional_ood_dataset_evaluation_performed": False,
        "phase9_started": False,
        "unsupported_ood_detection_claim": False,
    }


def verify_phase8c2b_material_contract(contract: dict[str, Any]) -> dict[str, Any]:
    dataset = contract["dataset"]
    if dataset["dataset_id"] != CIFAR10_1_DATASET_ID or dataset["version"] != CIFAR10_1_VERSION:
        raise ValueError("Phase 8C-2B must use registered CIFAR-10.1 v6")
    if dataset["split"] != CIFAR10_1_SPLIT_NAME:
        raise ValueError("Phase 8C-2B must use cross_source_test only")
    if dataset["sample_count"] != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
        raise ValueError("Phase 8C-2B must evaluate exactly 2,000 CIFAR-10.1 samples")
    if dataset["sample_label_digest"] != PHASE8C1_SAMPLE_LABEL_DIGEST:
        raise ValueError("Phase 8C-2B sample-label digest must match accepted Phase 8C-1")
    if contract.get("expected_cross_source_metric_rows") != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2B must expect three cross-source metric rows")
    if contract.get("expected_historical_reference_delta_rows") != PHASE8C2_EXPECTED_DELTA_ROWS:
        raise ValueError("Phase 8C-2B must expect three historical-reference delta rows")
    if contract["checkpoint_manifest"].get("checkpoint_count") != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2B requires exactly three fixed checkpoints")
    forbidden_flags = [
        "official_cifar10_test_rerun_performed",
        "training_or_tuning_performed",
        "model_selection_performed",
        "checkpoint_mutation_performed",
        "additional_ood_dataset_evaluation_performed",
        "phase9_started",
        "unsupported_ood_detection_claim",
    ]
    for flag in forbidden_flags:
        if contract.get(flag) is not False:
            raise ValueError(f"Phase 8C-2B contract must keep {flag}=False")
    projection = contract.get("runtime_projection", {})
    if projection.get("guard_status") != "passed":
        raise TimeoutError("Phase 8C-2B runtime guard did not pass")
    return {"status": "passed"}


def build_phase8c2b_runtime_projection() -> dict[str, Any]:
    estimated_minutes = 1.49
    guard_minutes = PHASE8C2B_RUNTIME_GUARD_SECONDS / 60.0
    guard_status = "passed" if estimated_minutes * 60.0 <= PHASE8C2B_RUNTIME_GUARD_SECONDS else "failed"
    return {
        "status": "estimate_only",
        "source": "Scaled from accepted Phase 8B-1 approximate 77.92-minute estimate for 315,000 model-condition examples to 6,000 Phase 8C-2B model examples.",
        "estimated_model_examples": PHASE8C2_EXPECTED_MODEL_ROWS * CIFAR10_1_EXPECTED_SAMPLE_COUNT,
        "estimated_minutes": estimated_minutes,
        "runtime_guard_minutes": guard_minutes,
        "guard_status": guard_status,
        "configuration_change_allowed_without_approval": False,
    }


def verify_phase8c2b_runtime_guard(projection: dict[str, Any]) -> dict[str, Any]:
    if projection.get("guard_status") != "passed":
        raise TimeoutError("Projected Phase 8C-2B runtime exceeds the approved 45-minute guard")
    return {"status": "passed", "estimated_minutes": projection["estimated_minutes"]}


def run_phase8c2b_material_evaluation(
    run_dir: Path | str = PHASE8C2B_OUTPUT_DIR,
    *,
    data_root: str | Path = "data",
) -> Phase8C2BResult:
    """Run the approved CIFAR-10.1 v6 cross-source material evaluation."""

    run_path = Path(run_dir)
    verify_phase8c2_output_isolation(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar10_1_v6(data_root)
    dataset_report = verify_phase8c2_dataset_identity(dataset)
    availability = find_local_cifar10_1_v6(data_root)
    dataset_report["file_digests"] = cifar10_1_file_digest(availability)
    references = phase7_references()
    checkpoint_manifest = build_phase8c2_checkpoint_manifest(references)
    historical_reference = load_phase7_historical_test_reference()
    verify_historical_reference_rows(historical_reference, references)
    runtime_projection = build_phase8c2b_runtime_projection()
    verify_phase8c2b_runtime_guard(runtime_projection)
    preprocessing_report = verify_phase8c2_preprocessing_order(Cifar101Subset(dataset, sample_count=PHASE8C2_TINY_SMOKE_SAMPLE_COUNT))

    contract = build_phase8c2b_material_contract(
        dataset_report=dataset_report,
        checkpoint_manifest=checkpoint_manifest,
        historical_reference=historical_reference,
        runtime_projection=runtime_projection,
    )
    verify_phase8c2b_material_contract(contract)

    contract_path = run_path / "phase8c2b_material_run_contract.json"
    dataset_path = artifact_dir / "phase8c2b_cifar10_1_dataset_identity.json"
    checkpoint_path = artifact_dir / "phase8c2b_checkpoint_manifest.json"
    reference_path = artifact_dir / "phase8c2b_historical_phase7_test_reference.json"
    runtime_path = artifact_dir / "phase8c2b_runtime_projection.json"
    preprocessing_path = artifact_dir / "phase8c2b_preprocessing_verification.json"
    _write_json(contract_path, contract)
    _write_json(dataset_path, dataset_report)
    _write_json(checkpoint_path, checkpoint_manifest)
    _write_json(reference_path, historical_reference)
    _write_json(runtime_path, runtime_projection)
    _write_json(preprocessing_path, preprocessing_report)

    metrics_rows: list[dict[str, Any]] = []
    split_results: dict[str, dict[str, Any]] = {}
    started_at = time.time()

    for reference in references:
        before_sha = sha256_file(reference.checkpoint_path)
        model = _load_phase8c2_model(reference)
        view = _phase8c2_prediction_view_for_reference(reference, dataset)
        loader = DataLoader(
            view,
            batch_size=PHASE8C2B_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        model_started_at = time.time()
        evaluation = evaluate_classification(
            model,
            loader,
            class_names=CIFAR10_CLASSES,
            split=CIFAR10_1_SPLIT_NAME,
        )
        elapsed_seconds = time.time() - model_started_at
        after_sha = sha256_file(reference.checkpoint_path)
        if before_sha != after_sha:
            raise ValueError(f"checkpoint mutated during Phase 8C-2B evaluation: {reference.run_id}")
        summary = _phase8c2_summary_from_evaluation(evaluation)
        if summary["total_examples"] != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"Phase 8C-2B expected 2,000 examples for {reference.run_id}")
        metrics_rows.append(
            {
                "run_id": reference.run_id,
                "display_name": reference.display_name,
                "dataset_id": CIFAR10_1_DATASET_ID,
                "dataset_version": CIFAR10_1_VERSION,
                "split": CIFAR10_1_SPLIT_NAME,
                **summary,
                "elapsed_seconds": elapsed_seconds,
                "metrics_are_cross_source_results": True,
                "metrics_are_ood_detection_results": False,
            }
        )
        split_results[reference.run_id] = {
            "sample_ids": [record.sample_id for record in evaluation.predictions],
            "true_labels": [record.true_label for record in evaluation.predictions],
            "source_ids": [record.source_id for record in evaluation.predictions],
        }

    delta_rows = phase8c2b_historical_reference_delta_rows(metrics_rows, historical_reference)
    alignment_report = verify_phase8c2b_sample_alignment(split_results, dataset_report=dataset_report)

    metrics_path = artifact_dir / "phase8c2_cross_source_metrics.csv"
    deltas_path = artifact_dir / "phase8c2_cross_source_deltas.csv"
    alignment_path = artifact_dir / "phase8c2_sample_alignment.json"
    _write_phase8c2b_metrics_csv(metrics_rows, metrics_path)
    _write_phase8c2b_delta_csv(delta_rows, deltas_path)
    _write_json(alignment_path, alignment_report)

    artifact_validation_path = artifact_dir / "phase8c2_artifact_validation.json"
    artifact_validation = verify_phase8c2b_artifacts(
        [
            contract_path,
            dataset_path,
            checkpoint_path,
            reference_path,
            runtime_path,
            preprocessing_path,
            metrics_path,
            deltas_path,
            alignment_path,
        ],
        metrics_rows=metrics_rows,
        delta_rows=delta_rows,
        alignment_report=alignment_report,
    )
    _write_json(artifact_validation_path, artifact_validation)

    report_path = run_path / "phase8c2_cross_source_report.md"
    write_phase8c2b_report(
        report_path,
        metrics_rows=metrics_rows,
        delta_rows=delta_rows,
        dataset_report=dataset_report,
        runtime_projection=runtime_projection,
        elapsed_seconds=time.time() - started_at,
    )

    artifact_paths = {
        "material_run_contract": str(contract_path),
        "dataset_identity": str(dataset_path),
        "checkpoint_manifest": str(checkpoint_path),
        "historical_phase7_test_reference": str(reference_path),
        "runtime_projection": str(runtime_path),
        "preprocessing_verification": str(preprocessing_path),
        "cross_source_metrics": str(metrics_path),
        "cross_source_deltas": str(deltas_path),
        "sample_alignment": str(alignment_path),
        "artifact_validation": str(artifact_validation_path),
        "cross_source_report": str(report_path),
    }
    result = Phase8C2BResult(
        run_dir=run_path,
        run_id=PHASE8C2B_RUN_ID,
        status="completed_at_phase_check_boundary",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase8c2b_result.json"
    artifact_paths["result"] = str(result_path)
    _write_json(result_path, result.to_dict())
    return result


def phase8c2b_historical_reference_delta_rows(
    metrics_rows: list[dict[str, Any]],
    historical_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_by_run_id = {row["run_id"]: row for row in historical_reference["rows"]}
    rows = []
    for row in metrics_rows:
        reference = reference_by_run_id[row["run_id"]]
        rows.append(
            {
                "run_id": row["run_id"],
                "reference_split": historical_reference["reference_split"],
                "cross_source_split": row["split"],
                "reference_total_examples": reference["total_examples"],
                "cross_source_total_examples": row["total_examples"],
                "accuracy_delta_from_phase7_test": row["accuracy"] - reference["accuracy"],
                "balanced_accuracy_delta_from_phase7_test": row["balanced_accuracy"] - reference["balanced_accuracy"],
                "macro_f1_delta_from_phase7_test": row["macro_f1"] - reference["macro_f1"],
                "ece_delta_from_phase7_test": row["ece"] - reference["ece"],
                "average_confidence_delta_from_phase7_test": row["average_confidence"] - reference["average_confidence"],
                "incorrect_average_confidence_delta_from_phase7_test": row["incorrect_average_confidence"] - reference["incorrect_average_confidence"],
                "reference_semantics": historical_reference["reference_semantics"],
                "reference_was_rerun_in_phase8c2b": False,
            }
        )
    if len(rows) != PHASE8C2_EXPECTED_DELTA_ROWS:
        raise ValueError("Phase 8C-2B must produce exactly three historical-reference delta rows")
    return rows


def verify_phase8c2b_sample_alignment(
    split_results: dict[str, dict[str, Any]],
    *,
    dataset_report: dict[str, Any],
) -> dict[str, Any]:
    if len(split_results) != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2B sample alignment requires all three model results")
    baseline_key = next(iter(split_results))
    baseline = split_results[baseline_key]
    for run_id, result in split_results.items():
        if len(result["sample_ids"]) != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"Phase 8C-2B expected 2,000 sample IDs for {run_id}")
        if result["sample_ids"] != baseline["sample_ids"]:
            raise ValueError("Phase 8C-2B sample IDs are not aligned across checkpoints")
        if result["true_labels"] != baseline["true_labels"]:
            raise ValueError("Phase 8C-2B labels are not aligned across checkpoints")
        if result["source_ids"] != baseline["source_ids"]:
            raise ValueError("Phase 8C-2B source IDs are not aligned across checkpoints")
    return {
        "status": "passed",
        "checkpoint_count": len(split_results),
        "sample_count_per_checkpoint": CIFAR10_1_EXPECTED_SAMPLE_COUNT,
        "sample_label_digest": dataset_report["sample_label_digest"],
        "source_dataset_id": dataset_report["dataset_id"],
        "source_dataset_version": dataset_report["version"],
        "split": dataset_report["split"],
        "sample_ids_and_labels_aligned_across_checkpoints": True,
        "source_ids_aligned_across_checkpoints": True,
    }


def verify_phase8c2b_artifacts(
    paths: list[Path],
    *,
    metrics_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    alignment_report: dict[str, Any],
) -> dict[str, Any]:
    verify_phase8c2a_artifacts(paths)
    if len(metrics_rows) != PHASE8C2_EXPECTED_MODEL_ROWS:
        raise ValueError("Phase 8C-2B must produce exactly three cross-source metric rows")
    if len(delta_rows) != PHASE8C2_EXPECTED_DELTA_ROWS:
        raise ValueError("Phase 8C-2B must produce exactly three historical-reference delta rows")
    for row in metrics_rows:
        if row["total_examples"] != CIFAR10_1_EXPECTED_SAMPLE_COUNT:
            raise ValueError("each Phase 8C-2B metric row must contain exactly 2,000 examples")
        if row.get("metrics_are_ood_detection_results") is not False:
            raise ValueError("Phase 8C-2B metrics must not be labeled as OOD-detection results")
    if alignment_report.get("status") != "passed":
        raise ValueError("Phase 8C-2B sample alignment did not pass")
    return {
        "status": "passed",
        "file_count": len(paths),
        "cross_source_metric_rows": len(metrics_rows),
        "historical_reference_delta_rows": len(delta_rows),
        "sample_count_per_model": CIFAR10_1_EXPECTED_SAMPLE_COUNT,
        "official_cifar10_test_rerun_performed": False,
        "phase9_started": False,
        "files": [str(path) for path in paths],
    }


def write_phase8c2b_report(
    path: Path,
    *,
    metrics_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    dataset_report: dict[str, Any],
    runtime_projection: dict[str, Any],
    elapsed_seconds: float,
) -> Path:
    lines = [
        "# Phase 8C-2B CIFAR-10.1 v6 Cross-Source Evaluation Report",
        "",
        "This report records CIFAR-10.1 v6 cross-source/distribution-shift evidence only. It is not a general OOD-detection, deployment-reliability, real-world robustness, or general model-superiority claim.",
        "",
        "## Scope",
        "",
        f"- Run ID: `{PHASE8C2B_RUN_ID}`.",
        f"- Dataset: `{dataset_report['dataset_id']}` version `{dataset_report['version']}` split `{dataset_report['split']}`.",
        f"- Samples per checkpoint: `{dataset_report['sample_count']}`.",
        f"- Sample-label digest: `{dataset_report['sample_label_digest']}`.",
        f"- File SHA-256 digests: `{json.dumps(dataset_report.get('file_digests', {}), sort_keys=True)}`.",
        f"- Runtime projection: `{runtime_projection['estimated_minutes']}` minutes; guard `{runtime_projection['runtime_guard_minutes']}` minutes; status `{runtime_projection['guard_status']}`.",
        f"- Wall-clock material evaluation elapsed seconds: `{elapsed_seconds}`.",
        "- Official CIFAR-10 test metrics are previously accepted Phase 7 historical references only; the official test split was not rerun.",
        "- No training, tuning, model selection, checkpoint modification, additional OOD dataset evaluation, or Phase 9 work occurred.",
        "",
        "## Cross-Source Metrics",
        "",
        "| Run | N | Loss | Accuracy | Balanced Accuracy | Macro F1 | ECE | Avg Confidence | Incorrect Avg Confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            "| {run_id} | {total_examples} | {loss:.6f} | {accuracy:.6f} | {balanced_accuracy:.6f} | {macro_f1:.6f} | {ece:.6f} | {average_confidence:.6f} | {incorrect_average_confidence:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Historical-Reference Deltas",
            "",
            "Deltas are computed as CIFAR-10.1 v6 cross-source metric minus the corresponding previously accepted Phase 7 official CIFAR-10 test summary. They are not paired-prediction deltas and they do not rerun the official test split.",
            "",
            "| Run | Accuracy Delta | Balanced Accuracy Delta | Macro F1 Delta | ECE Delta | Avg Confidence Delta | Incorrect Avg Confidence Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {run_id} | {accuracy_delta_from_phase7_test:.6f} | {balanced_accuracy_delta_from_phase7_test:.6f} | {macro_f1_delta_from_phase7_test:.6f} | {ece_delta_from_phase7_test:.6f} | {average_confidence_delta_from_phase7_test:.6f} | {incorrect_average_confidence_delta_from_phase7_test:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- These results are limited to CIFAR-10.1 v6 as registered in Phase 8C-1.",
            "- They do not establish label correctness, general OOD detection, deployment reliability, real-world robustness, or general model superiority.",
            "- They do not evaluate the official CIFAR-10 test split in Phase 8C-2B.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_phase8c2_model(reference: Phase7RunReference):
    if reference.run_id == PHASE4B_RUN_ID:
        return _load_phase4b_model(reference)
    if reference.run_id == PHASE6B2_RUN_ID:
        return _load_phase6b2_model(reference)
    if reference.run_id == PHASE6C_RUN_ID:
        return _load_phase6c_model(reference)
    raise ValueError(f"unknown Phase 8C-2B reference: {reference.run_id}")


def _phase8c2_prediction_view_for_reference(reference: Phase7RunReference, dataset: Dataset) -> Dataset:
    if reference.run_id == PHASE4B_RUN_ID:
        return Phase8C2CustomPredictionView(dataset)
    if reference.run_id in {PHASE6B2_RUN_ID, PHASE6C_RUN_ID}:
        return Phase8C2TransferPredictionView(dataset)
    raise ValueError(f"unknown Phase 8C-2B reference: {reference.run_id}")


def _phase8c2_summary_from_evaluation(evaluation) -> dict[str, Any]:
    true_indices = []
    predicted_indices = []
    probabilities = []
    confidences = []
    correct = []
    for record in evaluation.predictions:
        if record.true_index is None or record.predicted_index is None:
            raise ValueError("Phase 8C-2B predictions must include true/predicted indices")
        true_indices.append(record.true_index)
        predicted_indices.append(record.predicted_index)
        probabilities.append(list(record.probabilities))
        confidences.append(record.confidence)
        correct.append(record.correct)
    metrics = classification_metrics_from_predictions(
        true_indices,
        predicted_indices,
        probabilities,
        class_names=CIFAR10_CLASSES,
    )
    calibration = calibration_summary(
        confidences,
        correct,
        num_bins=PHASE7_NUM_CALIBRATION_BINS,
    )
    return {
        "total_examples": evaluation.summary.total_examples,
        "loss": evaluation.summary.loss,
        "accuracy": evaluation.summary.accuracy,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["averages"]["f1"]["macro"],
        "ece": calibration.expected_calibration_error,
        "average_confidence": calibration.average_confidence,
        "incorrect_average_confidence": calibration.incorrect_average_confidence,
    }


def _write_phase8c2b_metrics_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "run_id",
        "display_name",
        "dataset_id",
        "dataset_version",
        "split",
        "total_examples",
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ece",
        "average_confidence",
        "incorrect_average_confidence",
        "elapsed_seconds",
        "metrics_are_cross_source_results",
        "metrics_are_ood_detection_results",
    ]
    _write_csv(rows, path, fieldnames)
    return path


def _write_phase8c2b_delta_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "run_id",
        "reference_split",
        "cross_source_split",
        "reference_total_examples",
        "cross_source_total_examples",
        "accuracy_delta_from_phase7_test",
        "balanced_accuracy_delta_from_phase7_test",
        "macro_f1_delta_from_phase7_test",
        "ece_delta_from_phase7_test",
        "average_confidence_delta_from_phase7_test",
        "incorrect_average_confidence_delta_from_phase7_test",
        "reference_semantics",
        "reference_was_rerun_in_phase8c2b",
    ]
    _write_csv(rows, path, fieldnames)
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
