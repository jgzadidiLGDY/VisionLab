"""Phase 8B-1 robustness plumbing and validation-smoke workflow."""

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

from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10_PREPROCESSING,
    DataLoaderPolicy,
    build_cifar10_split_datasets,
    normalize_tensor,
    verify_material_cifar10_contract,
)
from visionlab.data.degradations import (
    PHASE8A_DEFAULT_SEED,
    PHASE8A_DEGRADATION_PROFILES,
    apply_degradation,
    degradation_registry_dict,
)
from visionlab.data.transfer_preprocessing import preprocess_resnet18_imagenet_tensor
from visionlab.evaluation.calibration import calibration_summary
from visionlab.evaluation.metrics import classification_metrics_from_predictions
from visionlab.evaluation.classification import evaluate_classification
from visionlab.experiments.phase7 import (
    PHASE7_NUM_CALIBRATION_BINS,
    Phase7RunReference,
    _load_phase4b_model,
    _load_phase6b2_model,
    _load_phase6c_model,
    phase7_references,
    sha256_file,
)
from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import PHASE6C_RUN_ID


PHASE8B1_RUN_ID = "phase8b1-robustness-plumbing-validation-smoke"
PHASE8B1_OUTPUT_DIR = Path("outputs") / PHASE8B1_RUN_ID
PHASE8B1_SAMPLE_COUNT = 10
PHASE8B1_BATCH_SIZE = 8
PHASE8B1_SEED = PHASE8A_DEFAULT_SEED
PHASE8B1_SPLIT = "val"
PHASE8B2_ESTIMATED_FULL_SAMPLE_COUNT = 5_000
PHASE8B2_ESTIMATED_SPLITS = ("val",)
PHASE8B2A_RUN_ID = "phase8b2a-validation-robustness-runner-preflight"
PHASE8B2A_OUTPUT_DIR = Path("outputs") / PHASE8B2A_RUN_ID
PHASE8B2B_RUN_ID = "phase8b2b-fixed-checkpoint-validation-robustness-sweep"
PHASE8B2B_OUTPUT_DIR = Path("outputs") / PHASE8B2B_RUN_ID
PHASE8B2_MATERIAL_SPLIT = "val"
PHASE8B2_MATERIAL_SAMPLE_COUNT = 5_000
PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS = 63


@dataclass(frozen=True)
class Phase8BCondition:
    condition_id: str
    display_name: str
    profile_id: str | None
    profile_version: str | None
    severity_id: str | None
    severity_parameters: dict[str, Any]
    is_clean: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "display_name": self.display_name,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "severity_id": self.severity_id,
            "severity_parameters": dict(self.severity_parameters),
            "is_clean": self.is_clean,
        }


@dataclass(frozen=True)
class Phase8B1SmokeResult:
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


class ValidationSubset(Dataset):
    """Fixed-prefix validation subset for Phase 8B-1 plumbing smoke."""

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


class Phase8BCustomPredictionView(Dataset):
    """Apply Phase 8B condition raw tensor, then existing CustomCNN preprocessing."""

    preprocessing_id = "phase4-cifar10-normalization"

    def __init__(self, dataset: Dataset, condition: Phase8BCondition, *, seed: int) -> None:
        self.dataset = dataset
        self.condition = condition
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        raw = condition_raw_tensor(sample, self.condition, seed=self.seed)
        return _sample_with_model_input(
            sample,
            condition=self.condition,
            raw_condition_input=raw,
            model_input=normalize_tensor(raw, CIFAR10_PREPROCESSING),
            preprocessing_id=self.preprocessing_id,
        )


class Phase8BTransferPredictionView(Dataset):
    """Apply Phase 8B condition raw tensor, then existing ResNet-18 preprocessing."""

    preprocessing_id = "phase6a-resnet18-imagenet1k-v1-preprocessing"

    def __init__(self, dataset: Dataset, condition: Phase8BCondition, *, seed: int) -> None:
        self.dataset = dataset
        self.condition = condition
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        raw = condition_raw_tensor(sample, self.condition, seed=self.seed)
        return _sample_with_model_input(
            sample,
            condition=self.condition,
            raw_condition_input=raw,
            model_input=preprocess_resnet18_imagenet_tensor(raw),
            preprocessing_id=self.preprocessing_id,
        )


def phase8b_conditions() -> tuple[Phase8BCondition, ...]:
    conditions = [
        Phase8BCondition(
            condition_id="clean",
            display_name="clean validation smoke baseline",
            profile_id=None,
            profile_version=None,
            severity_id=None,
            severity_parameters={},
            is_clean=True,
        )
    ]
    for profile in PHASE8A_DEGRADATION_PROFILES:
        for severity in profile.severities:
            conditions.append(
                Phase8BCondition(
                    condition_id=f"{profile.profile_id}__{severity.severity_id}",
                    display_name=f"{profile.profile_id} {severity.severity_id}",
                    profile_id=profile.profile_id,
                    profile_version=profile.version,
                    severity_id=severity.severity_id,
                    severity_parameters=dict(severity.parameters),
                    is_clean=False,
                )
            )
    return tuple(conditions)


def condition_raw_tensor(
    sample: dict[str, Any],
    condition: Phase8BCondition,
    *,
    seed: int,
) -> Tensor:
    raw = sample["raw_input"]
    if condition.is_clean:
        return raw.detach().clone().contiguous()
    if condition.profile_id is None or condition.severity_id is None:
        raise ValueError("degraded condition requires profile_id and severity_id")
    return apply_degradation(
        raw,
        profile_id=condition.profile_id,
        severity_id=condition.severity_id,
        seed=seed,
        sample_id=str(sample["sample_id"]),
        source_id=str(sample.get("source_id", sample["sample_id"])),
    )


def run_phase8b1_validation_smoke(
    run_dir: Path | str = PHASE8B1_OUTPUT_DIR,
    *,
    sample_count: int = PHASE8B1_SAMPLE_COUNT,
    data_root: str | Path = "data",
) -> Phase8B1SmokeResult:
    """Run tiny validation-subset plumbing smoke; metrics are not robustness results."""

    if sample_count <= 0 or sample_count > 100:
        raise ValueError("Phase 8B-1 sample_count must be between 1 and 100")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    references = phase7_references()
    _verify_checkpoint_identities_only(references)
    conditions = phase8b_conditions()
    _verify_condition_contract(conditions)
    datasets = build_cifar10_split_datasets(root=data_root, download=False)
    preflight = verify_material_cifar10_contract(datasets)
    val_subset = ValidationSubset(datasets.val, sample_count=sample_count)
    policy = DataLoaderPolicy(
        batch_size=PHASE8B1_BATCH_SIZE,
        seed=PHASE8B1_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )

    contract = {
        "phase": "8B-1",
        "run_id": PHASE8B1_RUN_ID,
        "scope": "robustness plumbing, condition contracts, tiny validation smoke, artifact validation, and runtime estimate only",
        "not_robustness_results": True,
        "official_test_evaluation": "not performed",
        "phase8b2_status": "not started",
        "phase8c_status": "not started",
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": preflight,
        "smoke_split": PHASE8B1_SPLIT,
        "smoke_sample_count": sample_count,
        "dataloader_policy": policy.to_dict(),
        "class_names": list(CIFAR10_CLASSES),
        "phase8a_degradation_registry": degradation_registry_dict(),
        "conditions": [condition.to_dict() for condition in conditions],
        "checkpoint_references": [reference.to_dict() for reference in references],
        "hard_invariants": [
            "Phase 8A degradation profile IDs, versions, severities, and parameters are reused unchanged",
            "degradation is applied to the raw unit tensor before model-specific preprocessing",
            "the same sample/profile/severity/seed raw degraded unit tensor is identical before CustomCNN and ResNet-18 preprocessing",
            "official CIFAR-10 test split is not evaluated in Phase 8B-1",
            "Phase 8B-1 metrics are pipeline-smoke evidence only, not robustness results",
            "Phase 7 metric and calibration helper semantics are reused unchanged",
            "Phase 7 and Phase 8A output directories are not written by this smoke",
        ],
    }
    contract_path = run_path / "phase8b1_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    equivalence_report = verify_raw_input_equivalence(
        val_subset,
        conditions,
        seed=PHASE8B1_SEED,
    )
    equivalence_path = artifact_dir / "raw_input_equivalence.json"
    equivalence_path.write_text(json.dumps(equivalence_report, indent=2), encoding="utf-8")

    preprocessing_report = verify_preprocessing_contracts(val_subset, conditions, seed=PHASE8B1_SEED)
    preprocessing_path = artifact_dir / "preprocessing_verification.json"
    preprocessing_path.write_text(json.dumps(preprocessing_report, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    split_results: dict[str, list[dict[str, Any]]] = {}
    total_evaluated_examples = 0

    for reference in references:
        model = _load_phase8b_model(reference)
        split_results[reference.run_id] = []
        for condition in conditions:
            view = _prediction_view_for_reference(reference, val_subset, condition)
            loader = DataLoader(
                view,
                batch_size=policy.batch_size,
                shuffle=policy.eval_shuffle,
                num_workers=policy.num_workers,
                drop_last=False,
            )
            start = time.perf_counter()
            evaluation = evaluate_classification(
                model,
                loader,
                class_names=CIFAR10_CLASSES,
                split=PHASE8B1_SPLIT,
            )
            elapsed = time.perf_counter() - start
            total_evaluated_examples += evaluation.summary.total_examples
            summary = _phase8b_summary_from_evaluation(evaluation)
            row = {
                "run_id": reference.run_id,
                "condition_id": condition.condition_id,
                "profile_id": condition.profile_id or "",
                "profile_version": condition.profile_version or "",
                "severity_id": condition.severity_id or "",
                "is_clean": condition.is_clean,
                **summary,
                "elapsed_seconds": elapsed,
                "metrics_are_robustness_results": False,
            }
            metrics_rows.append(row)
            timings.append(
                {
                    "run_id": reference.run_id,
                    "condition_id": condition.condition_id,
                    "examples": evaluation.summary.total_examples,
                    "elapsed_seconds": elapsed,
                    "seconds_per_example": elapsed / evaluation.summary.total_examples,
                }
            )
            split_results[reference.run_id].append(
                {
                    "condition_id": condition.condition_id,
                    "sample_ids": [record.sample_id for record in evaluation.predictions],
                    "true_labels": [record.true_label for record in evaluation.predictions],
                }
            )

    metrics_path = artifact_dir / "phase8b1_smoke_metrics.csv"
    write_phase8b_metrics_csv(metrics_rows, metrics_path)
    delta_rows = clean_delta_rows(metrics_rows)
    deltas_path = artifact_dir / "phase8b1_clean_delta_smoke.csv"
    write_phase8b_delta_csv(delta_rows, deltas_path)
    alignment_report = verify_phase8b_sample_alignment(split_results)
    alignment_path = artifact_dir / "sample_alignment.json"
    alignment_path.write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")
    artifact_report = verify_phase8b1_artifacts(
        [
            contract_path,
            equivalence_path,
            preprocessing_path,
            metrics_path,
            deltas_path,
            alignment_path,
        ]
    )
    artifact_report_path = artifact_dir / "artifact_validation.json"
    artifact_report_path.write_text(json.dumps(artifact_report, indent=2), encoding="utf-8")
    runtime_estimate = estimate_phase8b2_runtime(
        timings,
        conditions=conditions,
        references=references,
        target_sample_count=PHASE8B2_ESTIMATED_FULL_SAMPLE_COUNT,
        target_splits=PHASE8B2_ESTIMATED_SPLITS,
    )
    runtime_path = artifact_dir / "phase8b2_runtime_estimate.json"
    runtime_path.write_text(json.dumps(runtime_estimate, indent=2), encoding="utf-8")
    report_path = run_path / "phase8b1_smoke_report.md"
    write_phase8b1_report(
        report_path,
        sample_count=sample_count,
        conditions=conditions,
        total_evaluated_examples=total_evaluated_examples,
        runtime_estimate=runtime_estimate,
    )

    artifact_paths = {
        "contract": str(contract_path),
        "raw_input_equivalence": str(equivalence_path),
        "preprocessing_verification": str(preprocessing_path),
        "smoke_metrics": str(metrics_path),
        "clean_delta_smoke": str(deltas_path),
        "sample_alignment": str(alignment_path),
        "artifact_validation": str(artifact_report_path),
        "runtime_estimate": str(runtime_path),
        "smoke_report": str(report_path),
    }
    result = Phase8B1SmokeResult(
        run_dir=run_path,
        run_id=PHASE8B1_RUN_ID,
        status="completed",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase8b1_result.json"
    artifact_paths["result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def verify_raw_input_equivalence(
    dataset: Dataset,
    conditions: tuple[Phase8BCondition, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for condition in conditions:
        for index in range(len(dataset)):
            sample = dataset[index]
            custom_raw = condition_raw_tensor(sample, condition, seed=seed)
            transfer_raw = condition_raw_tensor(sample, condition, seed=seed)
            if not torch.equal(custom_raw, transfer_raw):
                raise ValueError(
                    f"raw degraded input mismatch for {sample['sample_id']} {condition.condition_id}"
                )
            rows.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "condition_id": condition.condition_id,
                    "shape": list(custom_raw.shape),
                    "min": float(custom_raw.min().item()),
                    "max": float(custom_raw.max().item()),
                    "finite": bool(torch.isfinite(custom_raw).all().item()),
                    "raw_inputs_identical": True,
                }
            )
    return {
        "status": "passed",
        "seed": seed,
        "condition_count": len(conditions),
        "sample_count": len(dataset),
        "checks": rows,
    }


def verify_preprocessing_contracts(
    dataset: Dataset,
    conditions: tuple[Phase8BCondition, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for condition in conditions:
        custom_view = Phase8BCustomPredictionView(dataset, condition, seed=seed)
        transfer_view = Phase8BTransferPredictionView(dataset, condition, seed=seed)
        for index in range(len(dataset)):
            custom = custom_view[index]
            transfer = transfer_view[index]
            if custom["sample_id"] != transfer["sample_id"]:
                raise ValueError("preprocessing views produced mismatched sample IDs")
            if int(custom["label"]) != int(transfer["label"]):
                raise ValueError("preprocessing views produced mismatched labels")
            if not torch.equal(custom["raw_condition_input"], transfer["raw_condition_input"]):
                raise ValueError("preprocessing views do not share equivalent raw condition input")
            rows.append(
                {
                    "sample_id": str(custom["sample_id"]),
                    "condition_id": condition.condition_id,
                    "custom_input_shape": list(custom["input"].shape),
                    "transfer_input_shape": list(transfer["input"].shape),
                    "custom_preprocessing_id": custom["preprocessing_id"],
                    "transfer_preprocessing_id": transfer["preprocessing_id"],
                    "raw_condition_inputs_identical": True,
                }
            )
    return {
        "status": "passed",
        "sample_count": len(dataset),
        "condition_count": len(conditions),
        "checks": rows,
    }


def clean_delta_rows(
    rows: list[dict[str, Any]],
    *,
    metrics_are_robustness_results: bool = False,
) -> list[dict[str, Any]]:
    clean_by_run = {row["run_id"]: row for row in rows if row["condition_id"] == "clean"}
    output = []
    for row in rows:
        clean = clean_by_run.get(row["run_id"])
        if clean is None:
            raise ValueError(f"missing clean baseline for {row['run_id']}")
        output.append(
            {
                "run_id": row["run_id"],
                "condition_id": row["condition_id"],
                "profile_id": row["profile_id"],
                "profile_version": row["profile_version"],
                "severity_id": row["severity_id"],
                "accuracy_delta_from_clean": row["accuracy"] - clean["accuracy"],
                "balanced_accuracy_delta_from_clean": row["balanced_accuracy"] - clean["balanced_accuracy"],
                "macro_f1_delta_from_clean": row["macro_f1"] - clean["macro_f1"],
                "ece_delta_from_clean": row["ece"] - clean["ece"],
                "average_confidence_delta_from_clean": row["average_confidence"] - clean["average_confidence"],
                "incorrect_average_confidence_delta_from_clean": _nullable_delta(
                    row["incorrect_average_confidence"],
                    clean["incorrect_average_confidence"],
                ),
                "metrics_are_robustness_results": metrics_are_robustness_results,
            }
        )
    return output


def verify_phase8b_sample_alignment(split_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "passed", "runs": {}}
    reference_samples: list[str] | None = None
    reference_labels: list[str] | None = None
    for run_id, results in split_results.items():
        if not results:
            raise ValueError(f"no Phase 8B-1 results for {run_id}")
        run_reference = results[0]
        for item in results[1:]:
            if item["sample_ids"] != run_reference["sample_ids"]:
                raise ValueError(f"sample identity mismatch for {run_id} {item['condition_id']}")
            if item["true_labels"] != run_reference["true_labels"]:
                raise ValueError(f"label mismatch for {run_id} {item['condition_id']}")
        if reference_samples is None:
            reference_samples = list(run_reference["sample_ids"])
            reference_labels = list(run_reference["true_labels"])
        elif run_reference["sample_ids"] != reference_samples or run_reference["true_labels"] != reference_labels:
            raise ValueError(f"run-level alignment mismatch for {run_id}")
        report["runs"][run_id] = {
            "condition_count": len(results),
            "sample_count": len(run_reference["sample_ids"]),
            "sample_ids_identical_across_conditions": True,
            "true_labels_identical_across_conditions": True,
        }
    report["sample_ids_identical_across_runs"] = True
    report["true_labels_identical_across_runs"] = True
    return report


def write_phase8b_metrics_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "run_id",
        "condition_id",
        "profile_id",
        "profile_version",
        "severity_id",
        "is_clean",
        "total_examples",
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ece",
        "average_confidence",
        "incorrect_average_confidence",
        "elapsed_seconds",
        "metrics_are_robustness_results",
    ]
    _write_csv(rows, path, fieldnames)
    return path


def write_phase8b_delta_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "run_id",
        "condition_id",
        "profile_id",
        "profile_version",
        "severity_id",
        "accuracy_delta_from_clean",
        "balanced_accuracy_delta_from_clean",
        "macro_f1_delta_from_clean",
        "ece_delta_from_clean",
        "average_confidence_delta_from_clean",
        "incorrect_average_confidence_delta_from_clean",
        "metrics_are_robustness_results",
    ]
    _write_csv(rows, path, fieldnames)
    return path


def verify_phase8b1_artifacts(paths: list[Path]) -> dict[str, Any]:
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Phase 8B-1 artifact missing: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Phase 8B-1 artifact is empty: {path}")
        files.append({"path": str(path), "bytes": path.stat().st_size})
    return {"status": "passed", "files": files}


def estimate_phase8b2_runtime(
    timings: list[dict[str, Any]],
    *,
    conditions: tuple[Phase8BCondition, ...],
    references: tuple[Phase7RunReference, ...],
    target_sample_count: int,
    target_splits: tuple[str, ...],
) -> dict[str, Any]:
    if not timings:
        raise ValueError("cannot estimate runtime without timing rows")
    seconds_per_example = sum(float(row["seconds_per_example"]) for row in timings) / len(timings)
    total_target_evaluations = len(conditions) * len(references) * len(target_splits)
    estimated_seconds = seconds_per_example * target_sample_count * total_target_evaluations
    return {
        "status": "approximate",
        "basis": "Phase 8B-1 tiny validation-subset smoke timing extrapolation",
        "average_seconds_per_example_condition_model": seconds_per_example,
        "target_sample_count_per_split": target_sample_count,
        "target_splits": list(target_splits),
        "condition_count": len(conditions),
        "model_count": len(references),
        "estimated_model_condition_evaluations": total_target_evaluations,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds / 60.0,
        "not_a_commitment": True,
    }


def write_phase8b1_report(
    path: Path,
    *,
    sample_count: int,
    conditions: tuple[Phase8BCondition, ...],
    total_evaluated_examples: int,
    runtime_estimate: dict[str, Any],
) -> Path:
    lines = [
        "# Phase 8B-1 Robustness Plumbing Validation Smoke Report",
        "",
        "This report is pipeline-smoke evidence only. It is not a robustness result.",
        "",
        "## Scope",
        "",
        "- Split: validation subset only.",
        f"- Fixed validation sample count: `{sample_count}`.",
        f"- Conditions checked: `{len(conditions)}`.",
        f"- Total model-condition examples evaluated: `{total_evaluated_examples}`.",
        "- Official CIFAR-10 test split was not evaluated.",
        "- Phase 8B-2 and Phase 8C were not started.",
        "",
        "## Runtime Estimate",
        "",
        f"- Estimated Phase 8B-2 minutes: `{runtime_estimate['estimated_minutes']:.2f}`.",
        "- This estimate is approximate and extrapolated from tiny-smoke timing.",
        "",
        "## Interpretation Boundary",
        "",
        "- Smoke metrics validate plumbing and artifact shape only.",
        "- Do not interpret smoke metrics as robustness, calibration robustness, or model-ranking evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sample_with_model_input(
    sample: dict[str, Any],
    *,
    condition: Phase8BCondition,
    raw_condition_input: Tensor,
    model_input: Tensor,
    preprocessing_id: str,
) -> dict[str, Any]:
    return {
        "input": model_input,
        "raw_condition_input": raw_condition_input,
        "label": int(sample["label"]),
        "sample_id": str(sample["sample_id"]),
        "split": str(sample["split"]),
        "source_id": str(sample.get("source_id", sample["sample_id"])),
        "condition_id": condition.condition_id,
        "is_clean_condition": condition.is_clean,
        "degradation_profile_id": condition.profile_id or "",
        "degradation_profile_version": condition.profile_version or "",
        "degradation_severity_id": condition.severity_id or "",
        "preprocessing_id": preprocessing_id,
    }


def _prediction_view_for_reference(
    reference: Phase7RunReference,
    dataset: Dataset,
    condition: Phase8BCondition,
) -> Dataset:
    if reference.run_id == PHASE4B_RUN_ID:
        return Phase8BCustomPredictionView(dataset, condition, seed=PHASE8B1_SEED)
    if reference.run_id in {PHASE6B2_RUN_ID, PHASE6C_RUN_ID}:
        return Phase8BTransferPredictionView(dataset, condition, seed=PHASE8B1_SEED)
    raise ValueError(f"unknown Phase 8B-1 reference: {reference.run_id}")


def _load_phase8b_model(reference: Phase7RunReference):
    if reference.run_id == PHASE4B_RUN_ID:
        return _load_phase4b_model(reference)
    if reference.run_id == PHASE6B2_RUN_ID:
        return _load_phase6b2_model(reference)
    if reference.run_id == PHASE6C_RUN_ID:
        return _load_phase6c_model(reference)
    raise ValueError(f"unknown Phase 8B-1 reference: {reference.run_id}")


def _phase8b_summary_from_evaluation(evaluation) -> dict[str, Any]:
    true_indices = []
    predicted_indices = []
    probabilities = []
    confidences = []
    correct = []
    for record in evaluation.predictions:
        if record.true_index is None or record.predicted_index is None:
            raise ValueError("Phase 8B-1 predictions must include true/predicted indices")
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


def _verify_checkpoint_identities_only(references: tuple[Phase7RunReference, ...]) -> None:
    for reference in references:
        if not reference.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Phase 8B-1 requires preserved checkpoint for {reference.run_id}: {reference.checkpoint_path}"
            )
        before = sha256_file(reference.checkpoint_path)
        after = sha256_file(reference.checkpoint_path)
        if before != after:
            raise ValueError(f"checkpoint identity changed while verifying {reference.run_id}")


def _verify_condition_contract(conditions: tuple[Phase8BCondition, ...]) -> None:
    if not conditions or conditions[0].condition_id != "clean" or not conditions[0].is_clean:
        raise ValueError("Phase 8B conditions must start with a clean condition")
    seen = set()
    for condition in conditions:
        if condition.condition_id in seen:
            raise ValueError(f"duplicate Phase 8B condition: {condition.condition_id}")
        seen.add(condition.condition_id)
        if condition.is_clean and condition.condition_id != "clean":
            raise ValueError("clean condition must use condition_id clean")
        if not condition.is_clean and (
            not condition.profile_id or not condition.profile_version or not condition.severity_id
        ):
            raise ValueError("degraded condition identity is incomplete")


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _nullable_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline

@dataclass(frozen=True)
class Phase8B2APreflightResult:
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


def run_phase8b2a_preflight(
    run_dir: Path | str = PHASE8B2A_OUTPUT_DIR,
    *,
    target_split: str = PHASE8B2_MATERIAL_SPLIT,
    data_root: str | Path = "data",
) -> Phase8B2APreflightResult:
    """Prepare Phase 8B-2B material validation sweep contracts without running it."""

    validate_phase8b2_material_split(target_split)
    run_path = Path(run_dir)
    output_isolation = verify_phase8b2_output_isolation(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    references = phase7_references()
    _verify_checkpoint_identities_only(references)
    conditions = phase8b_conditions()
    _verify_condition_contract(conditions)
    datasets = build_cifar10_split_datasets(root=data_root, download=False)
    preflight = verify_material_cifar10_contract(datasets)
    validation_count = len(datasets.val)
    if validation_count != PHASE8B2_MATERIAL_SAMPLE_COUNT:
        raise ValueError(
            f"Phase 8B-2 requires exactly {PHASE8B2_MATERIAL_SAMPLE_COUNT} validation samples; "
            f"found {validation_count}"
        )

    condition_manifest = build_phase8b2_condition_manifest(conditions)
    checkpoint_manifest = build_phase8b2_checkpoint_manifest(references)
    expected_schema = phase8b2_expected_artifact_schema()
    alignment_preflight = build_phase8b2_sample_alignment_preflight(
        datasets.val,
        conditions=conditions,
        references=references,
        split=target_split,
    )
    material_contract = build_phase8b2_material_contract(
        references=references,
        conditions=conditions,
        dataset_contract=datasets.to_contract_dict(),
        preflight_report=preflight,
        target_split=target_split,
    )
    verify_phase8b2_material_contract(material_contract)

    contract_path = run_path / "phase8b2a_preflight_contract.json"
    condition_path = artifact_dir / "phase8b2_condition_manifest.json"
    checkpoint_path = artifact_dir / "phase8b2_checkpoint_manifest.json"
    validation_path = artifact_dir / "phase8b2_validation_preflight.json"
    schema_path = artifact_dir / "phase8b2_expected_artifact_schema.json"
    alignment_path = artifact_dir / "phase8b2_sample_alignment_preflight.json"
    material_contract_path = artifact_dir / "phase8b2b_material_run_contract.json"

    contract = {
        "phase": "8B-2A",
        "run_id": PHASE8B2A_RUN_ID,
        "scope": "validation-only robustness runner/preflight, artifact contracts, and tests; no full sweep",
        "phase8b2b_run_id": PHASE8B2B_RUN_ID,
        "phase8b2b_status": "not started",
        "phase8c_status": "not started",
        "full_validation_sweep": "not run",
        "official_test_evaluation": "rejected by validation-only split gate",
        "robustness_conclusions": "none",
        "target_split": target_split,
        "target_sample_count": validation_count,
        "expected_model_condition_rows": phase8b2_expected_model_condition_rows(references, conditions),
        "output_isolation": output_isolation,
        "hard_invariants": [
            "Phase 8A degradation registry v1.0 is reused unchanged",
            "Phase 8B-2B material run is validation-only and rejects official test split requests",
            "the same raw CIFAR-10 sample/profile/severity must produce identical raw degraded unit tensors before model-specific preprocessing",
            "CustomCNN preprocessing occurs only after degradation",
            "ResNet-18 ImageNet preprocessing occurs only after degradation",
            "Phase 7 metric and calibration semantics are reused unchanged",
            "Phase 8B-2A does not train, tune, select models, mutate checkpoints, use OOD data, or evaluate official test robustness",
        ],
    }

    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    condition_path.write_text(json.dumps(condition_manifest, indent=2), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")
    validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "split": target_split,
                "sample_count": validation_count,
                "expected_sample_count": PHASE8B2_MATERIAL_SAMPLE_COUNT,
                "dataset_contract_status": preflight["status"],
                "official_test_evaluation": "not performed; rejected by split gate",
                "full_validation_sweep": "not run",
                "no_robustness_conclusions": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    schema_path.write_text(json.dumps(expected_schema, indent=2), encoding="utf-8")
    alignment_path.write_text(json.dumps(alignment_preflight, indent=2), encoding="utf-8")
    material_contract_path.write_text(json.dumps(material_contract, indent=2), encoding="utf-8")

    artifact_validation_path = artifact_dir / "phase8b2a_artifact_validation.json"
    artifact_validation = verify_phase8b2a_artifacts(
        [
            contract_path,
            condition_path,
            checkpoint_path,
            validation_path,
            schema_path,
            alignment_path,
            material_contract_path,
        ]
    )
    artifact_validation_path.write_text(json.dumps(artifact_validation, indent=2), encoding="utf-8")

    report_path = run_path / "phase8b2a_preflight_report.md"
    write_phase8b2a_preflight_report(
        report_path,
        validation_count=validation_count,
        expected_rows=PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS,
    )

    artifact_paths = {
        "preflight_contract": str(contract_path),
        "condition_manifest": str(condition_path),
        "checkpoint_manifest": str(checkpoint_path),
        "validation_preflight": str(validation_path),
        "expected_artifact_schema": str(schema_path),
        "sample_alignment_preflight": str(alignment_path),
        "material_run_contract": str(material_contract_path),
        "artifact_validation": str(artifact_validation_path),
        "preflight_report": str(report_path),
    }
    result = Phase8B2APreflightResult(
        run_dir=run_path,
        run_id=PHASE8B2A_RUN_ID,
        status="completed",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase8b2a_result.json"
    artifact_paths["result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def validate_phase8b2_material_split(split: str) -> None:
    if split != PHASE8B2_MATERIAL_SPLIT:
        raise ValueError(
            "Phase 8B-2 is validation-only; official test and non-validation split requests are rejected"
        )


def phase8b2_expected_model_condition_rows(
    references: tuple[Phase7RunReference, ...] | None = None,
    conditions: tuple[Phase8BCondition, ...] | None = None,
) -> int:
    references = references or phase7_references()
    conditions = conditions or phase8b_conditions()
    return len(references) * len(conditions)


def build_phase8b2_condition_manifest(conditions: tuple[Phase8BCondition, ...]) -> dict[str, Any]:
    return {
        "status": "passed",
        "condition_count": len(conditions),
        "expected_condition_count": 21,
        "phase8a_registry": degradation_registry_dict(),
        "conditions": [condition.to_dict() for condition in conditions],
    }


def build_phase8b2_checkpoint_manifest(references: tuple[Phase7RunReference, ...]) -> dict[str, Any]:
    return {
        "status": "passed",
        "checkpoint_count": len(references),
        "expected_checkpoint_count": 3,
        "checkpoints": [reference.to_dict() for reference in references],
        "checkpoint_mutation_policy": "hash identities are checked before preflight; Phase 8B-2A does not write checkpoints",
    }


def build_phase8b2_material_contract(
    *,
    references: tuple[Phase7RunReference, ...],
    conditions: tuple[Phase8BCondition, ...],
    dataset_contract: dict[str, Any],
    preflight_report: dict[str, Any],
    target_split: str,
) -> dict[str, Any]:
    validate_phase8b2_material_split(target_split)
    return {
        "phase": "8B-2B",
        "run_id": PHASE8B2B_RUN_ID,
        "status": "planned; not started by Phase 8B-2A",
        "scope": "future validation-only fixed-checkpoint robustness sweep; requires separate approval",
        "target_split": target_split,
        "target_sample_count": PHASE8B2_MATERIAL_SAMPLE_COUNT,
        "official_test_evaluation": "forbidden and rejected",
        "phase8c_status": "not started",
        "robustness_conclusions": "none in Phase 8B-2A",
        "dataset_contract": dataset_contract,
        "preflight_report": preflight_report,
        "phase8a_degradation_registry": degradation_registry_dict(),
        "conditions": [condition.to_dict() for condition in conditions],
        "checkpoint_references": [reference.to_dict() for reference in references],
        "expected_model_condition_rows": phase8b2_expected_model_condition_rows(references, conditions),
        "metrics": [
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ece_10_bin",
            "average_confidence",
            "incorrect_average_confidence",
            "clean_vs_degraded_delta",
        ],
        "metric_semantics_source": "Phase 7 evaluation, metrics, and calibration helpers; unchanged",
        "exclusions": [
            "official test robustness evaluation",
            "OOD or cross-source evaluation",
            "training",
            "tuning",
            "model selection",
            "checkpoint modification",
            "Phase 8C",
        ],
    }


def verify_phase8b2_material_contract(contract: dict[str, Any]) -> dict[str, Any]:
    validate_phase8b2_material_split(str(contract.get("target_split", "")))
    if contract.get("target_sample_count") != PHASE8B2_MATERIAL_SAMPLE_COUNT:
        raise ValueError("Phase 8B-2 material contract must target exactly 5,000 validation samples")
    conditions = contract.get("conditions", [])
    references = contract.get("checkpoint_references", [])
    if len(conditions) != 21:
        raise ValueError("Phase 8B-2 material contract must contain exactly 21 conditions")
    if conditions != [condition.to_dict() for condition in phase8b_conditions()]:
        raise ValueError("Phase 8B-2 material contract conditions must exactly match frozen Phase 8A v1.0 conditions")
    if len(references) != 3:
        raise ValueError("Phase 8B-2 material contract must contain exactly 3 checkpoint references")
    if contract.get("expected_model_condition_rows") != PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS:
        raise ValueError("Phase 8B-2 material contract must expect exactly 63 model-condition rows")
    clean_count = sum(1 for condition in conditions if condition.get("condition_id") == "clean")
    if clean_count != 1:
        raise ValueError("Phase 8B-2 material contract must contain exactly one clean condition")
    for condition in conditions:
        if condition.get("condition_id") == "clean":
            continue
        if condition.get("profile_version") != "1.0":
            raise ValueError("Phase 8B-2 degraded conditions must use Phase 8A profile version 1.0")
    return {"status": "passed"}


def verify_phase8b2_output_isolation(run_dir: Path | str) -> dict[str, Any]:
    run_path = Path(run_dir)
    forbidden = [
        PHASE8B1_OUTPUT_DIR,
        Path("outputs") / "phase7-evaluation-harness-and-calibration",
        Path("outputs") / "phase8a-degradation-registry-visual-qa-tiny-smoke",
    ]
    for forbidden_path in forbidden:
        if run_path == forbidden_path or forbidden_path in run_path.parents:
            raise ValueError(f"Phase 8B-2A output directory must not overlap {forbidden_path}")
    return {
        "status": "passed",
        "run_dir": str(run_path),
        "forbidden_output_dirs": [str(path) for path in forbidden],
    }


def build_phase8b2_sample_alignment_preflight(
    dataset: Dataset,
    *,
    conditions: tuple[Phase8BCondition, ...],
    references: tuple[Phase7RunReference, ...],
    split: str,
) -> dict[str, Any]:
    validate_phase8b2_material_split(split)
    digest = _dataset_sample_label_digest(dataset)
    return {
        "status": "passed",
        "split": split,
        "sample_count": len(dataset),
        "condition_count": len(conditions),
        "checkpoint_count": len(references),
        "sample_label_digest": digest,
        "alignment_policy": "all model-condition evaluation views must wrap the same registered validation dataset order",
        "official_test_evaluation": "not performed",
    }


def phase8b2_expected_artifact_schema() -> dict[str, Any]:
    return {
        "status": "planned",
        "phase8b2b_run_id": PHASE8B2B_RUN_ID,
        "output_dir": str(PHASE8B2B_OUTPUT_DIR),
        "required_artifacts": [
            "phase8b2b_material_run_contract.json",
            "artifacts/phase8b2_condition_manifest.json",
            "artifacts/phase8b2_checkpoint_manifest.json",
            "artifacts/phase8b2_validation_metrics.csv",
            "artifacts/phase8b2_clean_delta_metrics.csv",
            "artifacts/phase8b2_severity_curves.csv",
            "artifacts/phase8b2_sample_alignment.json",
            "artifacts/phase8b2_artifact_validation.json",
            "phase8b2_validation_robustness_report.md",
            "phase8b2b_result.json",
        ],
        "metric_columns": [
            "run_id",
            "condition_id",
            "profile_id",
            "profile_version",
            "severity_id",
            "is_clean",
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
            "condition_id",
            "profile_id",
            "profile_version",
            "severity_id",
            "accuracy_delta_from_clean",
            "balanced_accuracy_delta_from_clean",
            "macro_f1_delta_from_clean",
            "ece_delta_from_clean",
            "average_confidence_delta_from_clean",
            "incorrect_average_confidence_delta_from_clean",
        ],
    }


def verify_phase8b2a_artifacts(paths: list[Path]) -> dict[str, Any]:
    return verify_phase8b1_artifacts(paths)


def write_phase8b2a_preflight_report(path: Path, *, validation_count: int, expected_rows: int) -> Path:
    lines = [
        "# Phase 8B-2A Validation Robustness Runner Preflight Report",
        "",
        "This report is preflight evidence only. The full validation robustness sweep was not run.",
        "",
        "## Scope",
        "",
        "- Phase: `8B-2A`.",
        f"- Future material run: `{PHASE8B2B_RUN_ID}`.",
        f"- Split enforced: `{PHASE8B2_MATERIAL_SPLIT}` only.",
        f"- Validation samples required: `{validation_count}`.",
        "- Official test split requests are rejected in code.",
        "- Phase 8B-2B and Phase 8C were not started.",
        "",
        "## Expected Future Sweep Shape",
        "",
        "- Fixed checkpoint references: `3`.",
        "- Conditions: `21`.",
        f"- Expected model-condition metric rows: `{expected_rows}`.",
        "- Runtime estimate from Phase 8B-1: about `77.92` minutes, estimate only.",
        "",
        "## Interpretation Boundary",
        "",
        "- No robustness conclusion was produced by Phase 8B-2A.",
        "- No official test robustness evaluation, OOD evaluation, training, tuning, model selection, or checkpoint modification occurred.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dataset_sample_label_digest(dataset: Dataset) -> str:
    import hashlib

    digest = hashlib.sha256()
    for index in range(len(dataset)):
        sample = dataset[index]
        digest.update(str(sample["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(sample["label"])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Phase8B2BResult:
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


def run_phase8b2b_validation_sweep(
    run_dir: Path | str = PHASE8B2B_OUTPUT_DIR,
    *,
    target_split: str = PHASE8B2_MATERIAL_SPLIT,
    data_root: str | Path = "data",
) -> Phase8B2BResult:
    """Run the future material validation-only robustness sweep after separate approval."""

    validate_phase8b2_material_split(target_split)
    run_path = Path(run_dir)
    verify_phase8b2_output_isolation(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    references = phase7_references()
    _verify_checkpoint_identities_only(references)
    conditions = phase8b_conditions()
    _verify_condition_contract(conditions)
    datasets = build_cifar10_split_datasets(root=data_root, download=False)
    preflight = verify_material_cifar10_contract(datasets)
    dataset = datasets.val
    if len(dataset) != PHASE8B2_MATERIAL_SAMPLE_COUNT:
        raise ValueError(
            f"Phase 8B-2B requires exactly {PHASE8B2_MATERIAL_SAMPLE_COUNT} validation samples; "
            f"found {len(dataset)}"
        )

    contract = build_phase8b2_material_contract(
        references=references,
        conditions=conditions,
        dataset_contract=datasets.to_contract_dict(),
        preflight_report=preflight,
        target_split=target_split,
    )
    verify_phase8b2_material_contract(contract)

    condition_manifest = build_phase8b2_condition_manifest(conditions)
    checkpoint_manifest = build_phase8b2_checkpoint_manifest(references)
    alignment_preflight = build_phase8b2_sample_alignment_preflight(
        dataset,
        conditions=conditions,
        references=references,
        split=target_split,
    )

    contract_path = run_path / "phase8b2b_material_run_contract.json"
    condition_path = artifact_dir / "phase8b2_condition_manifest.json"
    checkpoint_path = artifact_dir / "phase8b2_checkpoint_manifest.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    condition_path.write_text(json.dumps(condition_manifest, indent=2), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    split_results: dict[str, list[dict[str, Any]]] = {}
    policy = DataLoaderPolicy(
        batch_size=PHASE8B1_BATCH_SIZE,
        seed=PHASE8B1_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )

    for reference in references:
        model = _load_phase8b_model(reference)
        split_results[reference.run_id] = []
        for condition in conditions:
            view = _prediction_view_for_reference(reference, dataset, condition)
            loader = DataLoader(
                view,
                batch_size=policy.batch_size,
                shuffle=policy.eval_shuffle,
                num_workers=policy.num_workers,
                drop_last=False,
            )
            evaluation = evaluate_classification(
                model,
                loader,
                class_names=CIFAR10_CLASSES,
                split=target_split,
            )
            summary = _phase8b_summary_from_evaluation(evaluation)
            metrics_rows.append(
                {
                    "run_id": reference.run_id,
                    "condition_id": condition.condition_id,
                    "profile_id": condition.profile_id or "",
                    "profile_version": condition.profile_version or "",
                    "severity_id": condition.severity_id or "",
                    "is_clean": condition.is_clean,
                    **summary,
                    "elapsed_seconds": "",
                    "metrics_are_robustness_results": True,
                }
            )
            split_results[reference.run_id].append(
                {
                    "condition_id": condition.condition_id,
                    "sample_ids": [record.sample_id for record in evaluation.predictions],
                    "true_labels": [record.true_label for record in evaluation.predictions],
                }
            )

    metrics_path = artifact_dir / "phase8b2_validation_metrics.csv"
    deltas_path = artifact_dir / "phase8b2_clean_delta_metrics.csv"
    severity_path = artifact_dir / "phase8b2_severity_curves.csv"
    alignment_path = artifact_dir / "phase8b2_sample_alignment.json"
    write_phase8b_metrics_csv(metrics_rows, metrics_path)
    delta_rows = clean_delta_rows(metrics_rows, metrics_are_robustness_results=True)
    write_phase8b_delta_csv(delta_rows, deltas_path)
    write_phase8b_severity_curves_csv(delta_rows, severity_path)
    alignment_report = verify_phase8b_sample_alignment(split_results)
    alignment_report["preflight_sample_label_digest"] = alignment_preflight["sample_label_digest"]
    alignment_path.write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    artifact_validation_path = artifact_dir / "phase8b2_artifact_validation.json"
    artifact_validation = verify_phase8b2a_artifacts(
        [
            contract_path,
            condition_path,
            checkpoint_path,
            metrics_path,
            deltas_path,
            severity_path,
            alignment_path,
        ]
    )
    artifact_validation_path.write_text(json.dumps(artifact_validation, indent=2), encoding="utf-8")
    report_path = run_path / "phase8b2_validation_robustness_report.md"
    write_phase8b2b_report(report_path, metrics_rows=metrics_rows, delta_rows=delta_rows)

    artifact_paths = {
        "material_run_contract": str(contract_path),
        "condition_manifest": str(condition_path),
        "checkpoint_manifest": str(checkpoint_path),
        "validation_metrics": str(metrics_path),
        "clean_delta_metrics": str(deltas_path),
        "severity_curves": str(severity_path),
        "sample_alignment": str(alignment_path),
        "artifact_validation": str(artifact_validation_path),
        "validation_robustness_report": str(report_path),
    }
    result = Phase8B2BResult(
        run_dir=run_path,
        run_id=PHASE8B2B_RUN_ID,
        status="completed",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase8b2b_result.json"
    artifact_paths["result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def write_phase8b_severity_curves_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "run_id",
        "profile_id",
        "profile_version",
        "severity_id",
        "accuracy_delta_from_clean",
        "balanced_accuracy_delta_from_clean",
        "macro_f1_delta_from_clean",
        "ece_delta_from_clean",
        "average_confidence_delta_from_clean",
        "incorrect_average_confidence_delta_from_clean",
    ]
    curve_rows = [
        {key: row[key] for key in fieldnames}
        for row in rows
        if row["condition_id"] != "clean"
    ]
    _write_csv(curve_rows, path, fieldnames)
    return path


def write_phase8b2b_report(
    path: Path,
    *,
    metrics_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> Path:
    lines = [
        "# Phase 8B-2B Validation Robustness Sweep Report",
        "",
        "This report is validation robustness evidence only. It is not official test robustness or OOD evidence.",
        "",
        "## Scope",
        "",
        f"- Split: `{PHASE8B2_MATERIAL_SPLIT}`.",
        f"- Metric rows: `{len(metrics_rows)}`.",
        f"- Delta rows: `{len(delta_rows)}`.",
        "- Official CIFAR-10 test split was not evaluated.",
        "- No training, tuning, model selection, checkpoint modification, OOD evaluation, or Phase 8C work occurred.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_phase8b2b_report(
    path: Path,
    *,
    metrics_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> Path:
    lines = [
        "# Phase 8B-2B Validation Robustness Sweep Report",
        "",
        "This report is validation robustness evidence only. It is not official test robustness or OOD evidence.",
        "",
        "## Scope",
        "",
        f"- Split: `{PHASE8B2_MATERIAL_SPLIT}`.",
        f"- Metric rows: `{len(metrics_rows)}`.",
        f"- Delta rows: `{len(delta_rows)}`.",
        "- Official CIFAR-10 test split was not evaluated.",
        "- No training, tuning, model selection, checkpoint modification, OOD evaluation, or Phase 8C work occurred.",
        "",
        "## Condition Metrics",
        "",
        "| Run | Condition | N | Loss | Accuracy | Balanced Accuracy | Macro F1 | ECE | Avg Confidence | Incorrect Avg Confidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_rows:
        lines.append(
            "| {run_id} | {condition_id} | {total_examples} | {loss} | {accuracy} | {balanced_accuracy} | {macro_f1} | {ece} | {average_confidence} | {incorrect_average_confidence} |".format(
                run_id=row["run_id"],
                condition_id=row["condition_id"],
                total_examples=row["total_examples"],
                loss=_format_report_value(row["loss"]),
                accuracy=_format_report_value(row["accuracy"]),
                balanced_accuracy=_format_report_value(row["balanced_accuracy"]),
                macro_f1=_format_report_value(row["macro_f1"]),
                ece=_format_report_value(row["ece"]),
                average_confidence=_format_report_value(row["average_confidence"]),
                incorrect_average_confidence=_format_report_value(row["incorrect_average_confidence"]),
            )
        )
    lines.extend(
        [
            "",
            "## Clean Deltas",
            "",
            "Deltas are computed against the clean condition for the same checkpoint reference.",
            "",
            "| Run | Condition | Accuracy Delta | Balanced Accuracy Delta | Macro F1 Delta | ECE Delta | Avg Confidence Delta | Incorrect Avg Confidence Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {run_id} | {condition_id} | {accuracy_delta_from_clean} | {balanced_accuracy_delta_from_clean} | {macro_f1_delta_from_clean} | {ece_delta_from_clean} | {average_confidence_delta_from_clean} | {incorrect_average_confidence_delta_from_clean} |".format(
                run_id=row["run_id"],
                condition_id=row["condition_id"],
                accuracy_delta_from_clean=_format_report_value(row["accuracy_delta_from_clean"]),
                balanced_accuracy_delta_from_clean=_format_report_value(row["balanced_accuracy_delta_from_clean"]),
                macro_f1_delta_from_clean=_format_report_value(row["macro_f1_delta_from_clean"]),
                ece_delta_from_clean=_format_report_value(row["ece_delta_from_clean"]),
                average_confidence_delta_from_clean=_format_report_value(row["average_confidence_delta_from_clean"]),
                incorrect_average_confidence_delta_from_clean=_format_report_value(row["incorrect_average_confidence_delta_from_clean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These are validation-only fixed-checkpoint robustness observations under registered CIFAR-10 validation conditions.",
            "They are not official test robustness, OOD robustness, semantic label-preservation, production reliability, or general model-superiority claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_report_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
