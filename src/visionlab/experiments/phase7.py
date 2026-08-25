"""Phase 7 evaluation harness and calibration workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionlab.data import (
    CIFAR10_CLASSES,
    DataLoaderPolicy,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
    build_transfer_dataloaders,
    verify_material_cifar10_contract,
)
from visionlab.evaluation import (
    ClassificationEvaluation,
    PredictionRecord,
    calibration_summary,
    classification_metrics_from_predictions,
    confidence_histogram,
    evaluate_classification,
    write_confidence_histogram_csv,
    write_confidence_histogram_svg,
    write_confusion_matrix_svg,
    write_evaluation_artifacts,
    write_reliability_diagram_svg,
)
from visionlab.experiments.phase4b import PHASE4B_RUN_ID, PHASE4B_SEED
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID, PHASE6B2_SEED
from visionlab.experiments.phase6c import PHASE6C_RUN_ID, build_phase6c_finetune_model_from_phase6b2
from visionlab.models import CustomCNN, CustomCNNConfig, build_phase6a_transfer_model
from visionlab.training.checkpoints import load_checkpoint


PHASE7_RUN_ID = "phase7-evaluation-harness-and-calibration"
PHASE7_NUM_CALIBRATION_BINS = 10
PHASE7_OUTPUT_DIR = Path("outputs") / PHASE7_RUN_ID


@dataclass(frozen=True)
class Phase7RunReference:
    run_id: str
    display_name: str
    model_family: str
    checkpoint_path: Path
    checkpoint_tag: str = "best"
    validation_semantics: str = "diagnostic/comparison evidence only"
    test_semantics: str = "retrospective fixed-checkpoint evaluation; not model selection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "display_name": self.display_name,
            "model_family": self.model_family,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_tag": self.checkpoint_tag,
            "checkpoint_sha256": sha256_file(self.checkpoint_path) if self.checkpoint_path.exists() else None,
            "validation_semantics": self.validation_semantics,
            "test_semantics": self.test_semantics,
        }


@dataclass(frozen=True)
class Phase7SplitResult:
    run_id: str
    split: str
    artifacts: dict[str, str]
    summary: dict[str, Any]
    sample_ids: tuple[str, ...]
    true_labels: tuple[str, ...]


@dataclass(frozen=True)
class Phase7Result:
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


def phase7_references() -> tuple[Phase7RunReference, ...]:
    return (
        Phase7RunReference(
            run_id=PHASE4B_RUN_ID,
            display_name="Phase 4B CustomCNN baseline",
            model_family="custom_cnn_from_scratch",
            checkpoint_path=Path("outputs") / PHASE4B_RUN_ID / "checkpoints" / "best.pt",
        ),
        Phase7RunReference(
            run_id=PHASE6B2_RUN_ID,
            display_name="Phase 6B-2 ResNet-18 frozen feature",
            model_family="resnet18_imagenet_frozen_feature",
            checkpoint_path=Path("outputs") / PHASE6B2_RUN_ID / "checkpoints" / "best.pt",
        ),
        Phase7RunReference(
            run_id=PHASE6C_RUN_ID,
            display_name="Phase 6C-2 ResNet-18 layer4 + fc fine-tuning",
            model_family="resnet18_imagenet_layer4_fc_finetune",
            checkpoint_path=Path("outputs") / PHASE6C_RUN_ID / "checkpoints" / "best.pt",
        ),
    )


def run_phase7_evaluation(run_dir: Path | str = PHASE7_OUTPUT_DIR) -> Phase7Result:
    """Evaluate fixed accepted checkpoints without training or model selection."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    references = phase7_references()
    _verify_reference_checkpoints(references)
    datasets = build_cifar10_split_datasets(root="data", download=False)
    preflight = verify_material_cifar10_contract(datasets)

    contract = {
        "phase": "7",
        "run_id": PHASE7_RUN_ID,
        "scope": "fixed-checkpoint evaluation harness and calibration; no training or tuning",
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": preflight,
        "class_names": list(CIFAR10_CLASSES),
        "references": [reference.to_dict() for reference in references],
        "hard_invariants": [
            "all compared runs must use identical registered sample IDs within each split",
            "all compared runs must use the Phase 1B CIFAR-10 class mapping",
            "only preserved best checkpoints identified by run ID and checkpoint tag are loaded",
            "Phase 7 must not train, tune, regenerate, or overwrite checkpoints",
            "validation and official test artifacts remain separate",
        ],
        "evaluation_semantics": {
            "validation": "diagnostic/comparison evidence",
            "official_test": "retrospective fixed-checkpoint evaluation of accepted historical checkpoints",
            "model_selection": "no Phase 7 model-selection decision is made from test performance",
        },
        "calibration": {
            "num_bins": PHASE7_NUM_CALIBRATION_BINS,
            "confidence": "maximum predicted class probability",
        },
    }
    contract_path = run_path / "phase7_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    phase4_policy = DataLoaderPolicy(
        batch_size=128,
        seed=PHASE4B_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    transfer_policy = DataLoaderPolicy(
        batch_size=64,
        seed=PHASE6B2_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    phase4_loaders = build_phase4_dataloaders(datasets, phase4_policy)
    transfer_loaders = build_transfer_dataloaders(datasets, transfer_policy)

    split_results: dict[str, list[Phase7SplitResult]] = {"val": [], "test": []}
    artifact_paths: dict[str, str] = {"phase7_contract": str(contract_path)}

    for reference in references:
        print(f"Phase 7 evaluating {reference.run_id}", flush=True)
        if reference.run_id == PHASE4B_RUN_ID:
            model = _load_phase4b_model(reference)
            val_loader = phase4_loaders.prediction_val
            test_loader = phase4_loaders.prediction_test
        elif reference.run_id == PHASE6B2_RUN_ID:
            model = _load_phase6b2_model(reference)
            val_loader = transfer_loaders.prediction_val
            test_loader = transfer_loaders.prediction_test
        elif reference.run_id == PHASE6C_RUN_ID:
            model = _load_phase6c_model(reference)
            val_loader = transfer_loaders.prediction_val
            test_loader = transfer_loaders.prediction_test
        else:
            raise ValueError(f"unknown Phase 7 reference: {reference.run_id}")

        for split, loader in (("val", val_loader), ("test", test_loader)):
            print(f"Phase 7 evaluating {reference.run_id} {split}", flush=True)
            evaluation = evaluate_classification(
                model,
                loader,
                class_names=CIFAR10_CLASSES,
                split=split,
            )
            split_dir = artifact_dir / reference.run_id / split
            result = write_phase7_split_artifacts(
                evaluation,
                split_dir,
                run_id=reference.run_id,
                split=split,
            )
            split_results[split].append(result)
            for name, value in result.artifacts.items():
                artifact_paths[f"{reference.run_id}_{split}_{name}"] = value

    alignment_path = artifact_dir / "sample_alignment.json"
    alignment_report = verify_phase7_sample_alignment(split_results)
    alignment_path.write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")
    artifact_paths["sample_alignment"] = str(alignment_path)

    table_path = artifact_dir / "phase7_comparison_table.csv"
    write_phase7_comparison_table(split_results, table_path)
    artifact_paths["comparison_table"] = str(table_path)

    report_path = run_path / "phase7_model_comparison_report.md"
    write_phase7_report(split_results, alignment_report, report_path)
    artifact_paths["comparison_report"] = str(report_path)

    result = Phase7Result(
        run_dir=run_path,
        run_id=PHASE7_RUN_ID,
        status="completed",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase7_result.json"
    artifact_paths["phase7_result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def write_phase7_split_artifacts(
    evaluation: ClassificationEvaluation,
    output_dir: Path,
    *,
    run_id: str,
    split: str,
) -> Phase7SplitResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_evaluation_artifacts(evaluation, output_dir, prefix=split)
    true_indices, predicted_indices, probabilities, confidences, correct = _prediction_arrays(
        evaluation.predictions
    )
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
    histogram = confidence_histogram(
        confidences,
        correct,
        num_bins=PHASE7_NUM_CALIBRATION_BINS,
    )
    metrics_path = output_dir / f"{split}_metrics.json"
    calibration_path = output_dir / f"{split}_calibration.json"
    histogram_csv_path = output_dir / f"{split}_confidence_histogram.csv"
    reliability_svg_path = output_dir / f"{split}_reliability_diagram.svg"
    histogram_svg_path = output_dir / f"{split}_confidence_histogram.svg"
    confusion_svg_path = output_dir / f"{split}_confusion_matrix.svg"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    write_confidence_histogram_csv(histogram, histogram_csv_path)
    write_reliability_diagram_svg(
        calibration,
        reliability_svg_path,
        title=f"{run_id} {split} reliability",
    )
    write_confidence_histogram_svg(
        histogram,
        histogram_svg_path,
        title=f"{run_id} {split} confidence distribution",
    )
    write_confusion_matrix_svg(
        evaluation.summary.confusion_matrix,
        CIFAR10_CLASSES,
        confusion_svg_path,
        title=f"{run_id} {split} confusion matrix",
    )
    artifacts.update(
        {
            "metrics": str(metrics_path),
            "calibration": str(calibration_path),
            "confidence_histogram_csv": str(histogram_csv_path),
            "reliability_diagram_svg": str(reliability_svg_path),
            "confidence_histogram_svg": str(histogram_svg_path),
            "confusion_matrix_svg": str(confusion_svg_path),
        }
    )
    summary = {
        "loss": evaluation.summary.loss,
        "accuracy": evaluation.summary.accuracy,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["averages"]["f1"]["macro"],
        "weighted_f1": metrics["averages"]["f1"]["weighted"],
        "ece": calibration.expected_calibration_error,
        "average_confidence": calibration.average_confidence,
        "incorrect_average_confidence": calibration.incorrect_average_confidence,
        "roc_auc_macro": metrics["roc_auc_ovr"]["macro"],
        "pr_auc_macro": metrics["pr_auc_ovr_average_precision"]["macro"],
        "total_examples": evaluation.summary.total_examples,
    }
    return Phase7SplitResult(
        run_id=run_id,
        split=split,
        artifacts=artifacts,
        summary=summary,
        sample_ids=tuple(record.sample_id for record in evaluation.predictions),
        true_labels=tuple(record.true_label for record in evaluation.predictions),
    )


def verify_phase7_sample_alignment(
    split_results: dict[str, list[Phase7SplitResult]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "passed", "splits": {}}
    for split, results in split_results.items():
        if not results:
            raise ValueError(f"no Phase 7 results available for split {split}")
        reference = results[0]
        for candidate in results[1:]:
            if candidate.sample_ids != reference.sample_ids:
                raise ValueError(
                    f"Phase 7 sample identity mismatch for split {split}: "
                    f"{candidate.run_id} does not align with {reference.run_id}"
                )
            if candidate.true_labels != reference.true_labels:
                raise ValueError(
                    f"Phase 7 label alignment mismatch for split {split}: "
                    f"{candidate.run_id} does not align with {reference.run_id}"
                )
        report["splits"][split] = {
            "reference_run_id": reference.run_id,
            "run_ids": [item.run_id for item in results],
            "sample_count": len(reference.sample_ids),
            "first_sample_id": reference.sample_ids[0] if reference.sample_ids else None,
            "last_sample_id": reference.sample_ids[-1] if reference.sample_ids else None,
            "class_mapping": list(CIFAR10_CLASSES),
            "sample_ids_identical": True,
            "true_labels_identical": True,
        }
    return report


def write_phase7_comparison_table(
    split_results: dict[str, list[Phase7SplitResult]],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "run_id",
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
            ],
        )
        writer.writeheader()
        for split in ("val", "test"):
            for result in split_results[split]:
                writer.writerow({"split": split, "run_id": result.run_id, **result.summary})
    return path


def write_phase7_report(
    split_results: dict[str, list[Phase7SplitResult]],
    alignment_report: dict[str, Any],
    path: Path,
) -> Path:
    lines = [
        "# Phase 7 Fixed-Checkpoint Evaluation and Calibration Report",
        "",
        "This report evaluates accepted historical checkpoints without retraining, tuning, or selecting a new model from test performance.",
        "",
        "## Evaluation Semantics",
        "",
        "- Validation results are diagnostic/comparison evidence.",
        "- Official test results are retrospective fixed-checkpoint evaluations of already accepted runs.",
        "- Phase 7 does not make a new model-selection decision based on test performance.",
        "- Confidence is the maximum predicted class probability; it is not guaranteed correctness.",
        "",
        "## Sample Alignment",
        "",
        f"- Alignment status: `{alignment_report['status']}`",
    ]
    for split, item in alignment_report["splits"].items():
        lines.append(
            f"- `{split}`: `{item['sample_count']}` samples aligned across `{', '.join(item['run_ids'])}`"
        )
    lines.extend(["", "## Comparison Summary", ""])
    for split in ("val", "test"):
        lines.extend([f"### {split.upper()}", ""])
        for result in split_results[split]:
            summary = result.summary
            lines.extend(
                [
                    f"- `{result.run_id}`: accuracy `{summary['accuracy']:.6f}`, balanced accuracy `{summary['balanced_accuracy']:.6f}`, macro F1 `{summary['macro_f1']:.6f}`, ECE `{summary['ece']:.6f}`, average confidence `{summary['average_confidence']:.6f}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundaries",
            "",
            "- Phase 4B versus ResNet comparisons remain asymmetric because of ImageNet pretraining, model scale, input resolution, preprocessing, and training regime differences.",
            "- ECE and reliability diagrams are calibration diagnostics, not proof of operational reliability.",
            "- ROC-AUC and PR-AUC are one-vs-rest multiclass summaries and should be interpreted alongside class support and confusion patterns.",
            "- Robustness, OOD behavior, failure galleries, diagnostics, inference, and applied-domain behavior remain out of Phase 7 scope.",
            "",
            "## Artifact Trail",
            "",
            f"- Comparison table: `{path.parent / 'artifacts' / 'phase7_comparison_table.csv'}`",
            f"- Alignment report: `{path.parent / 'artifacts' / 'sample_alignment.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_phase4b_model(reference: Phase7RunReference):
    model_config = CustomCNNConfig(
        input_channels=3,
        image_size=(32, 32),
        num_classes=10,
        feature_channels=(32, 64, 128),
        dropout=0.0,
    )
    model = CustomCNN(model_config)
    checkpoint = load_checkpoint(reference.checkpoint_path, model=model, expected_run_id=reference.run_id)
    _verify_checkpoint_payload(checkpoint, reference)
    return model


def _load_phase6b2_model(reference: Phase7RunReference):
    model = build_phase6a_transfer_model(load_pretrained=True)
    checkpoint = load_checkpoint(reference.checkpoint_path, model=model, expected_run_id=reference.run_id)
    _verify_checkpoint_payload(checkpoint, reference)
    return model


def _load_phase6c_model(reference: Phase7RunReference):
    model, _initialization = build_phase6c_finetune_model_from_phase6b2()
    checkpoint = load_checkpoint(reference.checkpoint_path, model=model, expected_run_id=reference.run_id)
    _verify_checkpoint_payload(checkpoint, reference)
    return model


def _verify_reference_checkpoints(references: tuple[Phase7RunReference, ...]) -> None:
    for reference in references:
        if not reference.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Phase 7 requires preserved checkpoint for {reference.run_id}: {reference.checkpoint_path}"
            )


def _verify_checkpoint_payload(payload: dict[str, Any], reference: Phase7RunReference) -> None:
    if payload.get("run_id") != reference.run_id:
        raise ValueError(f"checkpoint run_id mismatch for {reference.run_id}")
    if payload.get("tag") != reference.checkpoint_tag:
        raise ValueError(f"checkpoint tag mismatch for {reference.run_id}")


def _prediction_arrays(
    predictions: tuple[PredictionRecord, ...],
) -> tuple[list[int], list[int], list[list[float]], list[float], list[bool]]:
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    probabilities: list[list[float]] = []
    confidences: list[float] = []
    correct: list[bool] = []
    for record in predictions:
        if record.true_index is None or record.predicted_index is None:
            raise ValueError("Phase 7 predictions must include true and predicted indices")
        if len(record.probabilities) != len(CIFAR10_CLASSES):
            raise ValueError("Phase 7 predictions must include one probability per class")
        true_indices.append(record.true_index)
        predicted_indices.append(record.predicted_index)
        probabilities.append(list(record.probabilities))
        confidences.append(record.confidence)
        correct.append(record.correct)
    return true_indices, predicted_indices, probabilities, confidences, correct


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
