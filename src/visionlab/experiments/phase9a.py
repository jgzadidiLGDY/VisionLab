"""Phase 9A failure tables and error galleries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from visionlab.data import CIFAR10_CLASSES, build_cifar10_split_datasets
from visionlab.evaluation import (
    FailurePrediction,
    load_prediction_csv,
    materialize_gallery_images,
    per_class_failure_summary,
    select_confusion_pair_examples,
    select_high_confidence_errors,
    select_model_disagreements,
    select_per_class_failure_examples,
    write_csv_rows,
    write_gallery_html,
    write_gallery_manifest,
)
from visionlab.experiments.phase7 import PHASE7_OUTPUT_DIR, phase7_references
from visionlab.experiments.phase8b import PHASE8B2B_OUTPUT_DIR
from visionlab.experiments.phase8c import PHASE8C2B_OUTPUT_DIR


PHASE9A_RUN_ID = "phase9a-failure-analysis-galleries"
PHASE9A_OUTPUT_DIR = Path("outputs") / PHASE9A_RUN_ID
PHASE9A_REQUIRED_CONTEXT_ID = "phase7_clean_cifar10_val"
PHASE9A_HIGH_CONFIDENCE_TOP_N = 24
PHASE9A_PER_CLASS_TOP_N = 3
PHASE9A_CONFUSION_TOP_PAIRS = 10
PHASE9A_CONFUSION_EXAMPLES_PER_PAIR = 3
PHASE9A_MODEL_DISAGREEMENT_TOP_N = 24
PHASE9A_CHECKPOINT_IDENTITY_FIELDS = (
    "checkpoint_tag",
    "checkpoint_path",
    "checkpoint_sha256",
)
PHASE9A_BASE_SELECTION_FIELDS = (
    "dataset_id",
    "dataset_version",
    "split",
    "condition_id",
    "sample_id",
    "source_id",
    "true_label",
    "predicted_label",
    "confidence",
    "selection_rule",
)


@dataclass(frozen=True)
class Phase9AResult:
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


def phase9a_selection_contract() -> dict[str, Any]:
    """Return the explicit machine-readable Phase 9A selection contract."""

    return {
        "phase": "9A",
        "run_id": PHASE9A_RUN_ID,
        "scope": "failure tables and error galleries only; no diagnostics, training, tuning, new evaluation, model selection, or closeout",
        "required_context": {
            "context_id": PHASE9A_REQUIRED_CONTEXT_ID,
            "dataset_id": "cifar10",
            "dataset_version": "phase1b-registered",
            "split": "val",
            "condition_id": "clean",
            "source_artifact_type": "existing Phase 7 validation prediction CSVs",
            "population_semantics": "clean CIFAR-10 validation fixed-checkpoint evidence from Phase 7",
        },
        "optional_context_policy": {
            "phase8b_degraded_validation": "use only if preserved prediction-level artifacts already exist; otherwise record unavailable",
            "phase8c_cifar10_1_v6_cross_source": "use only if preserved prediction-level artifacts already exist; otherwise record unavailable",
            "no_regeneration": True,
        },
        "confidence_definition": "maximum predicted class probability from preserved prediction artifact",
        "class_names": list(CIFAR10_CLASSES),
        "selection_rules": {
            "high_confidence_errors": {
                "population": "incorrect predictions only, per run/context",
                "top_n": PHASE9A_HIGH_CONFIDENCE_TOP_N,
                "ranking": ["confidence descending", "sample_id ascending"],
            },
            "per_class_failures": {
                "population": "all predictions, per run/context",
                "summary": "preserve every CIFAR-10 class with support, correct, false-negative, false-positive, and accuracy fields",
                "example_top_n_per_category": PHASE9A_PER_CLASS_TOP_N,
                "categories": ["false_negative", "false_positive"],
                "ranking": [
                    "class order from registered CIFAR-10 mapping",
                    "category order false_negative then false_positive",
                    "confidence descending",
                    "sample_id ascending",
                ],
            },
            "confusion_pair_examples": {
                "population": "incorrect predictions only, per run/context",
                "top_pairs": PHASE9A_CONFUSION_TOP_PAIRS,
                "examples_per_pair": PHASE9A_CONFUSION_EXAMPLES_PER_PAIR,
                "pair_ranking": [
                    "pair count descending",
                    "true_label ascending",
                    "predicted_label ascending",
                ],
                "example_ranking": ["confidence descending", "sample_id ascending"],
            },
            "model_disagreement_examples": {
                "population": "aligned samples with identical sample IDs, labels, context, split, and condition across compared runs",
                "definition": "disagreement exists when not all compared runs have the same predicted_label",
                "top_n": PHASE9A_MODEL_DISAGREEMENT_TOP_N,
                "ranking": [
                    "distinct_prediction_count descending",
                    "incorrect_model_count descending",
                    "confidence_spread descending",
                    "sample_id ascending",
                ],
            },
        },
        "required_manifest_fields": list(PHASE9A_CHECKPOINT_IDENTITY_FIELDS + PHASE9A_BASE_SELECTION_FIELDS),
        "artifact_schema_requirements": phase9a_artifact_schema_requirements(),
        "interpretation_boundaries": [
            "selected examples are representative only of the declared selection rule and population",
            "visual inspection does not establish causal explanations",
            "confidence does not imply correctness",
            "diagnostic interpretation is deferred to Phase 9B",
        ],
    }


def phase9a_artifact_schema_requirements() -> dict[str, list[str]]:
    """Required top-level fields for generated Phase 9A artifacts."""

    selection_fields = list(PHASE9A_CHECKPOINT_IDENTITY_FIELDS + PHASE9A_BASE_SELECTION_FIELDS)
    return {
        "high_confidence_errors": ["rank", *selection_fields, "run_id", "context_id"],
        "per_class_failure_summary": [
            "run_id",
            "context_id",
            *PHASE9A_CHECKPOINT_IDENTITY_FIELDS,
            "class_index",
            "class_name",
            "support",
            "correct",
            "false_negative_count",
            "false_positive_count",
            "accuracy",
        ],
        "per_class_failure_examples": [
            *selection_fields,
            "run_id",
            "context_id",
            "class_name",
            "failure_category",
            "rank",
        ],
        "confusion_pair_examples": [
            *selection_fields,
            "run_id",
            "context_id",
            "pair_rank",
            "example_rank",
            "pair_count",
        ],
        "model_disagreement_examples": [
            "selection_rule",
            "rank",
            *PHASE9A_CHECKPOINT_IDENTITY_FIELDS,
            "compared_checkpoint_identities",
            "sample_id",
            "context_id",
            "dataset_id",
            "dataset_version",
            "split",
            "condition_id",
            "source_id",
            "true_label",
            "disagreement",
            "distinct_prediction_count",
            "incorrect_model_count",
            "confidence_spread",
            "run_predictions",
        ],
        "failure_selection_manifest": [
            "phase",
            "run_id",
            "required_context",
            "source_prediction_artifacts",
            "checkpoint_identities",
            "artifact_schema_requirements",
            "generated_artifacts",
            "optional_contexts",
        ],
        "high_confidence_error_gallery_manifest": [
            "image_path",
            "rank",
            *selection_fields,
            "run_id",
            "context_id",
        ],
    }


def run_phase9a_failure_analysis(run_dir: Path | str = PHASE9A_OUTPUT_DIR) -> Phase9AResult:
    """Generate Phase 9A failure tables and galleries from preserved predictions only."""

    run_path = Path(run_dir)
    artifact_dir = run_path / "artifacts"
    gallery_dir = artifact_dir / "galleries"
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    contract = phase9a_selection_contract()
    artifact_paths: dict[str, str] = {}
    contract_path = run_path / "phase9a_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    artifact_paths["phase9a_contract"] = str(contract_path)

    optional_contexts = inspect_optional_prediction_contexts()
    context_predictions = load_phase7_clean_validation_predictions()
    source_prediction_artifacts = phase7_clean_validation_prediction_artifacts()
    checkpoint_fields_by_run = _checkpoint_fields_by_run()
    datasets = build_cifar10_split_datasets(root="data", download=False)

    all_high_confidence: list[dict[str, Any]] = []
    all_per_class_summary: list[dict[str, Any]] = []
    all_per_class_examples: list[dict[str, Any]] = []
    all_confusion_examples: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []

    for run_id, predictions in context_predictions.items():
        checkpoint_fields = checkpoint_fields_by_run[run_id]
        high_confidence = [
            _selection_row(
                item,
                "high_confidence_errors",
                rank=rank,
                checkpoint_fields=checkpoint_fields,
            )
            for rank, item in enumerate(
                select_high_confidence_errors(predictions, top_n=PHASE9A_HIGH_CONFIDENCE_TOP_N),
                start=1,
            )
        ]
        per_class_summary_rows = [
            {"run_id": run_id, "context_id": PHASE9A_REQUIRED_CONTEXT_ID, **checkpoint_fields, **row}
            for row in per_class_failure_summary(predictions, class_names=CIFAR10_CLASSES)
        ]
        per_class_examples = [
            {"selection_rule": "per_class_failures", **checkpoint_fields, **row}
            for row in select_per_class_failure_examples(
                predictions,
                class_names=CIFAR10_CLASSES,
                top_n_per_category=PHASE9A_PER_CLASS_TOP_N,
            )
        ]
        confusion_examples = [
            {"selection_rule": "confusion_pair_examples", **checkpoint_fields, **row}
            for row in select_confusion_pair_examples(
                predictions,
                top_pairs=PHASE9A_CONFUSION_TOP_PAIRS,
                examples_per_pair=PHASE9A_CONFUSION_EXAMPLES_PER_PAIR,
            )
        ]
        all_high_confidence.extend(high_confidence)
        all_per_class_summary.extend(per_class_summary_rows)
        all_per_class_examples.extend(per_class_examples)
        all_confusion_examples.extend(confusion_examples)
        gallery_rows.extend(high_confidence)

    disagreement_checkpoint_fields = _combined_checkpoint_identity_fields(checkpoint_fields_by_run)
    disagreement_rows = [
        {
            "selection_rule": "model_disagreement_examples",
            "rank": rank,
            **disagreement_checkpoint_fields,
            "compared_checkpoint_identities": json.dumps(checkpoint_fields_by_run, sort_keys=True),
            **row,
        }
        for rank, row in enumerate(
            select_model_disagreements(context_predictions, top_n=PHASE9A_MODEL_DISAGREEMENT_TOP_N),
            start=1,
        )
    ]

    high_conf_path = artifact_dir / "high_confidence_errors.csv"
    per_class_path = artifact_dir / "per_class_failure_summary.csv"
    per_class_examples_path = artifact_dir / "per_class_failure_examples.csv"
    confusion_path = artifact_dir / "confusion_pair_examples.csv"
    disagreement_path = artifact_dir / "model_disagreement_examples.csv"
    optional_path = artifact_dir / "optional_context_availability.json"
    write_csv_rows(all_high_confidence, high_conf_path)
    write_csv_rows(all_per_class_summary, per_class_path)
    write_csv_rows(all_per_class_examples, per_class_examples_path)
    write_csv_rows(all_confusion_examples, confusion_path)
    write_csv_rows(disagreement_rows, disagreement_path)
    optional_path.write_text(json.dumps(optional_contexts, indent=2), encoding="utf-8")
    artifact_paths.update(
        {
            "high_confidence_errors": str(high_conf_path),
            "per_class_failure_summary": str(per_class_path),
            "per_class_failure_examples": str(per_class_examples_path),
            "confusion_pair_examples": str(confusion_path),
            "model_disagreement_examples": str(disagreement_path),
            "optional_context_availability": str(optional_path),
        }
    )

    image_rows = materialize_gallery_images(
        gallery_rows,
        gallery_dir / "high_confidence_error_images",
        resolve_image=lambda sample_id: _resolve_cifar10_image(sample_id, datasets.val),
    )
    gallery_manifest_path = gallery_dir / "high_confidence_error_gallery_manifest.csv"
    gallery_html_path = gallery_dir / "high_confidence_error_gallery.html"
    write_gallery_manifest(image_rows, gallery_manifest_path)
    write_gallery_html(image_rows, gallery_html_path, title="Phase 9A High-Confidence Error Gallery")
    artifact_paths["high_confidence_error_gallery_manifest"] = str(gallery_manifest_path)
    artifact_paths["high_confidence_error_gallery_html"] = str(gallery_html_path)

    manifest_path = artifact_dir / "failure_selection_manifest.json"
    write_failure_selection_manifest(
        manifest_path,
        contract=contract,
        source_prediction_artifacts=source_prediction_artifacts,
        checkpoint_identities=checkpoint_fields_by_run,
        generated_artifacts=artifact_paths,
        optional_contexts=optional_contexts,
    )
    artifact_paths["failure_selection_manifest"] = str(manifest_path)

    schema_validation_path = artifact_dir / "phase9a_artifact_schema_validation.json"
    schema_validation = validate_phase9a_generated_artifact_schemas(artifact_paths, contract)
    schema_validation_path.write_text(json.dumps(schema_validation, indent=2), encoding="utf-8")
    artifact_paths["artifact_schema_validation"] = str(schema_validation_path)

    report_path = run_path / "phase9a_failure_analysis_report.md"
    write_phase9a_report(
        report_path,
        contract=contract,
        run_ids=tuple(context_predictions),
        optional_contexts=optional_contexts,
        counts={
            "high_confidence_errors": len(all_high_confidence),
            "per_class_summary_rows": len(all_per_class_summary),
            "per_class_failure_examples": len(all_per_class_examples),
            "confusion_pair_examples": len(all_confusion_examples),
            "model_disagreement_examples": len(disagreement_rows),
            "gallery_images": len({row["sample_id"] for row in image_rows}),
        },
        schema_validation=schema_validation,
    )
    artifact_paths["failure_analysis_report"] = str(report_path)

    result = Phase9AResult(
        run_dir=run_path,
        run_id=PHASE9A_RUN_ID,
        status="completed_for_phase_check_review",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase9a_result.json"
    artifact_paths["phase9a_result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def phase7_clean_validation_prediction_artifacts() -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for reference in phase7_references():
        path = PHASE7_OUTPUT_DIR / "artifacts" / reference.run_id / "val" / "val_predictions.csv"
        artifacts[reference.run_id] = str(path)
    return artifacts


def load_phase7_clean_validation_predictions() -> dict[str, tuple[FailurePrediction, ...]]:
    predictions: dict[str, tuple[FailurePrediction, ...]] = {}
    for reference in phase7_references():
        path = Path(phase7_clean_validation_prediction_artifacts()[reference.run_id])
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 9A requires preserved Phase 7 validation predictions: {path}"
            )
        predictions[reference.run_id] = load_prediction_csv(
            path,
            run_id=reference.run_id,
            context_id=PHASE9A_REQUIRED_CONTEXT_ID,
            dataset_id="cifar10",
            dataset_version="phase1b-registered",
            split="val",
            condition_id="clean",
        )
    return predictions


def inspect_optional_prediction_contexts() -> dict[str, Any]:
    checks = {
        "phase8b_degraded_validation": {
            "directory": PHASE8B2B_OUTPUT_DIR,
            "required_artifact_pattern": "*predictions*.csv",
        },
        "phase8c_cifar10_1_v6_cross_source": {
            "directory": PHASE8C2B_OUTPUT_DIR,
            "required_artifact_pattern": "*predictions*.csv",
        },
    }
    result: dict[str, Any] = {}
    for context_id, check in checks.items():
        directory = Path(check["directory"])
        matches = sorted(directory.rglob(str(check["required_artifact_pattern"]))) if directory.exists() else []
        result[context_id] = {
            "status": "available" if matches else "unavailable",
            "reason": "preserved prediction-level artifact found"
            if matches
            else "no preserved prediction-level artifact found; Phase 9A does not regenerate evaluations",
            "matched_prediction_artifacts": [str(path) for path in matches],
        }
    return result


def write_failure_selection_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
    source_prediction_artifacts: dict[str, str],
    checkpoint_identities: dict[str, dict[str, Any]],
    generated_artifacts: dict[str, str],
    optional_contexts: dict[str, Any],
) -> Path:
    manifest = {
        "phase": contract["phase"],
        "run_id": contract["run_id"],
        "required_context": contract["required_context"],
        "source_prediction_artifacts": source_prediction_artifacts,
        "checkpoint_identities": checkpoint_identities,
        "artifact_schema_requirements": contract["artifact_schema_requirements"],
        "generated_artifacts": dict(generated_artifacts),
        "optional_contexts": optional_contexts,
        "selection_rules": contract["selection_rules"],
        "interpretation_boundaries": contract["interpretation_boundaries"],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def validate_phase9a_generated_artifact_schemas(
    artifact_paths: dict[str, str],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or phase9a_selection_contract()
    schemas = contract["artifact_schema_requirements"]
    artifacts_to_validate = {
        "high_confidence_errors": artifact_paths["high_confidence_errors"],
        "per_class_failure_summary": artifact_paths["per_class_failure_summary"],
        "per_class_failure_examples": artifact_paths["per_class_failure_examples"],
        "confusion_pair_examples": artifact_paths["confusion_pair_examples"],
        "model_disagreement_examples": artifact_paths["model_disagreement_examples"],
        "failure_selection_manifest": artifact_paths["failure_selection_manifest"],
        "high_confidence_error_gallery_manifest": artifact_paths[
            "high_confidence_error_gallery_manifest"
        ],
    }
    results: dict[str, Any] = {"status": "passed", "artifacts": {}}
    for artifact_name, artifact_path in artifacts_to_validate.items():
        path = Path(artifact_path)
        required = set(schemas[artifact_name])
        present = set(_json_fields(path) if path.suffix == ".json" else _csv_fields(path))
        missing = sorted(required.difference(present))
        results["artifacts"][artifact_name] = {
            "path": str(path),
            "required_fields": sorted(required),
            "present_fields": sorted(present),
            "missing_fields": missing,
            "status": "failed" if missing else "passed",
        }
        if missing:
            results["status"] = "failed"
    if results["status"] != "passed":
        raise ValueError("Phase 9A generated artifact schema validation failed")
    return results


def write_phase9a_report(
    path: Path,
    *,
    contract: dict[str, Any],
    run_ids: Sequence[str],
    optional_contexts: dict[str, Any],
    counts: dict[str, int],
    schema_validation: dict[str, Any],
) -> Path:
    lines = [
        "# Phase 9A Failure Tables and Error Galleries Report",
        "",
        "Status: implementation artifacts generated for phase-check review; Phase 9A is not automatically closed or accepted.",
        "",
        "## Scope",
        "",
        "Phase 9A uses existing fixed prediction artifacts only. It does not train, tune, mutate checkpoints, run new evaluation, select models, implement diagnostics, implement inference, select an applied domain, or close Phase 9A.",
        "",
        "## Required Population",
        "",
        f"- Context: `{contract['required_context']['context_id']}`",
        f"- Dataset: `{contract['required_context']['dataset_id']}` version `{contract['required_context']['dataset_version']}`",
        f"- Split/condition: `{contract['required_context']['split']}` / `{contract['required_context']['condition_id']}`",
        f"- Compared runs: `{', '.join(run_ids)}`",
        "",
        "## Selection Rules",
        "",
        f"- High-confidence errors: incorrect predictions only; top `{PHASE9A_HIGH_CONFIDENCE_TOP_N}` per run; confidence descending, then sample ID ascending.",
        f"- Per-class failures: all `{len(CIFAR10_CLASSES)}` CIFAR-10 classes preserved with support, correct, false-negative, false-positive, and accuracy fields; examples use top `{PHASE9A_PER_CLASS_TOP_N}` by confidence descending then sample ID ascending.",
        f"- Confusion-pair examples: top `{PHASE9A_CONFUSION_TOP_PAIRS}` pairs by count descending, true label ascending, predicted label ascending; up to `{PHASE9A_CONFUSION_EXAMPLES_PER_PAIR}` examples per pair by confidence descending then sample ID ascending.",
        f"- Model disagreements: top `{PHASE9A_MODEL_DISAGREEMENT_TOP_N}` aligned samples where compared runs do not all predict the same label; rank by distinct prediction count, incorrect model count, confidence spread, then sample ID.",
        "",
        "## Artifact Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifact Schema Validation",
            "",
            f"- Status: `{schema_validation['status']}`",
            "",
            "## Optional Contexts",
            "",
        ]
    )
    for context_id, item in optional_contexts.items():
        lines.append(f"- `{context_id}`: `{item['status']}` - {item['reason']}")
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- Galleries are evidence selected by declared rules, not curated anecdotes.",
            "- Selected examples are representative only of the declared selection rule and population.",
            "- Visual inspection does not establish causal explanations.",
            "- Confidence does not imply correctness.",
            "- Diagnostic interpretation is deferred to Phase 9B.",
            "",
            "## Human Review Boundary",
            "",
            "The generated galleries should be reviewed by the builder. Observations should be recorded separately from hypotheses before any later diagnostic or intervention work.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _selection_row(
    item: FailurePrediction,
    selection_rule: str,
    *,
    rank: int,
    checkpoint_fields: dict[str, Any],
) -> dict[str, Any]:
    return {"selection_rule": selection_rule, "rank": rank, **checkpoint_fields, **item.to_dict()}


def _checkpoint_fields_by_run() -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for reference in phase7_references():
        payload = reference.to_dict()
        fields[reference.run_id] = {
            "checkpoint_tag": payload["checkpoint_tag"],
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
        }
    return fields


def _combined_checkpoint_identity_fields(
    checkpoint_fields_by_run: dict[str, dict[str, Any]],
) -> dict[str, str]:
    combined: dict[str, str] = {}
    for field in PHASE9A_CHECKPOINT_IDENTITY_FIELDS:
        combined[field] = "|".join(
            f"{run_id}={checkpoint_fields_by_run[run_id][field]}"
            for run_id in sorted(checkpoint_fields_by_run)
        )
    for run_id in sorted(checkpoint_fields_by_run):
        safe_run_id = _safe_field_name(run_id)
        for field in PHASE9A_CHECKPOINT_IDENTITY_FIELDS:
            combined[f"{safe_run_id}_{field}"] = str(checkpoint_fields_by_run[run_id][field])
    return combined


def _json_fields(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return list(payload)


def _csv_fields(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _safe_field_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _resolve_cifar10_image(sample_id: str, dataset: Any) -> Image.Image:
    prefix = "cifar10-train-"
    if not sample_id.startswith(prefix):
        raise ValueError(
            f"Phase 9A clean validation gallery expected CIFAR-10 train sample ID, got {sample_id}"
        )
    source_index = int(sample_id.removeprefix(prefix))
    if source_index not in dataset.indices:
        raise ValueError(f"sample {sample_id} is not part of the registered validation split")
    image, _label = dataset.upstream[source_index]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image
