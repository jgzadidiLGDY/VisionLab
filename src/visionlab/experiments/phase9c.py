"""Phase 9C review synthesis and failure-hypothesis scaffolding."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from visionlab.evaluation import write_csv_rows
from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import PHASE6C_RUN_ID
from visionlab.experiments.phase9a import PHASE9A_OUTPUT_DIR
from visionlab.experiments.phase9b import PHASE9B_OUTPUT_DIR


PHASE9C_RUN_ID = "phase9c-review-and-closeout"
PHASE9C_OUTPUT_DIR = Path("outputs") / PHASE9C_RUN_ID
PHASE9C_FIXED_RUN_ORDER = (PHASE4B_RUN_ID, PHASE6B2_RUN_ID, PHASE6C_RUN_ID)
HEATMAP_LOW_ENTROPY_THRESHOLD = 0.55
HEATMAP_HIGH_ENTROPY_THRESHOLD = 0.80
HEATMAP_COMPACT_TOP_AREA_THRESHOLD = 0.12
HEATMAP_DIFFUSE_TOP_AREA_THRESHOLD = 0.35


@dataclass(frozen=True)
class Phase9CResult:
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


def phase9c_review_contract() -> dict[str, Any]:
    """Return the explicit machine-readable Phase 9C review/synthesis contract."""

    return {
        "phase": "9C",
        "run_id": PHASE9C_RUN_ID,
        "scope": "review and synthesis only; no new evaluation, prediction regeneration, Grad-CAM regeneration, training, tuning, checkpoint mutation, model selection, inference, Phase 10, saliency, embeddings, UMAP, or t-SNE",
        "source_artifacts": {
            "phase9a_output_dir": str(PHASE9A_OUTPUT_DIR),
            "phase9b_output_dir": str(PHASE9B_OUTPUT_DIR),
            "phase9a_required_artifacts": [
                "high_confidence_errors.csv",
                "per_class_failure_summary.csv",
                "per_class_failure_examples.csv",
                "confusion_pair_examples.csv",
                "model_disagreement_examples.csv",
                "failure_selection_manifest.json",
            ],
            "phase9b_required_artifacts": [
                "diagnostic_selection_manifest.json",
                "gradcam_manifest.csv",
                "gradcam_schema_validation.json",
            ],
        },
        "fixed_run_order": list(PHASE9C_FIXED_RUN_ORDER),
        "review_property_classes": {
            "machine_derived": {
                "definition": "computed only from existing Phase 9A/9B artifact fields or saved heatmap tensors using deterministic criteria",
                "allowed_tags": {
                    "prediction_error": "selected row has true_label different from predicted_label",
                    "prediction_correct_control": "selected row belongs to Phase 9B correct_control population and true_label equals predicted_label",
                    "model_disagreement_member": "selected row belongs to Phase 9B model_disagreement population",
                    "high_confidence_error_member": "selected row belongs to Phase 9B high_confidence_error population",
                    "heatmap_low_spatial_entropy": f"normalized heatmap entropy <= {HEATMAP_LOW_ENTROPY_THRESHOLD}",
                    "heatmap_high_spatial_entropy": f"normalized heatmap entropy >= {HEATMAP_HIGH_ENTROPY_THRESHOLD}",
                    "heatmap_compact_top_region": f"fraction of pixels at or above 80 percent of sample max <= {HEATMAP_COMPACT_TOP_AREA_THRESHOLD}",
                    "heatmap_diffuse_top_region": f"fraction of pixels at or above 80 percent of sample max >= {HEATMAP_DIFFUSE_TOP_AREA_THRESHOLD}",
                },
            },
            "builder_observation": {
                "definition": "human visual observation supplied by builder review; absent observations are recorded as pending_builder_review",
                "default": "pending_builder_review",
            },
            "hypothesis": {
                "definition": "cautious non-causal hypothesis derived from explicit evidence; no hypothesis is generated without builder observation",
                "default": "pending_builder_review",
            },
            "unsupported_causal_claim": {
                "definition": "claims that Grad-CAM proves reasoning, attention, causality, or the cause of a failure are prohibited",
                "default": "none_recorded",
            },
        },
        "separation_contract": [
            "selected example",
            "prediction/failure",
            "confidence",
            "diagnostic output",
            "visual observation",
            "hypothesis",
            "causal claim",
        ],
        "artifact_schema_requirements": phase9c_artifact_schema_requirements(),
        "hard_stop_conditions": [
            "missing Phase 9A required artifact",
            "missing Phase 9B required artifact",
            "Phase 9B schema validation did not pass",
            "diagnostic row cannot be aligned to the declared Phase 9A population where applicable",
            "missing or malformed persisted heatmap tensor",
            "unsupported causal language appears in generated Phase 9C reports",
        ],
        "interpretation_boundaries": [
            "machine-derived review properties are not semantic visual observations",
            "builder visual observations are pending unless explicitly supplied",
            "hypotheses are cautious and non-causal",
            "Grad-CAM does not prove model reasoning, attention, causality, or why a failure occurred",
            "Phase 9C does not prescribe model changes or interventions",
        ],
    }


def phase9c_artifact_schema_requirements() -> dict[str, list[str]]:
    return {
        "phase9c_contract": [
            "phase",
            "run_id",
            "scope",
            "source_artifacts",
            "fixed_run_order",
            "review_property_classes",
            "separation_contract",
            "artifact_schema_requirements",
            "hard_stop_conditions",
            "interpretation_boundaries",
        ],
        "phase9c_result": ["run_dir", "run_id", "status", "artifact_paths"],
        "review_tag_manifest": [
            "review_id",
            "diagnostic_id",
            "selection_category",
            "selection_rank",
            "run_id",
            "sample_id",
            "true_label",
            "predicted_label",
            "confidence",
            "machine_derived_properties",
            "machine_derived_tag_basis",
            "builder_visual_observation",
            "human_review_status",
            "cautious_hypothesis",
            "unsupported_causal_claim",
            "causal_claim_status",
            "diagnostic_output_reference",
        ],
        "label_data_quality_inventory": [
            "issue_id",
            "source",
            "sample_id",
            "run_id",
            "issue_type",
            "evidence_status",
            "builder_visual_observation",
            "cautious_hypothesis",
            "resolution_status",
        ],
        "failure_hypothesis_report": [
            "phase",
            "run_id",
            "status",
            "source_artifacts",
            "review_record_count",
            "hypothesis_status",
            "limitations",
        ],
        "phase9c_artifact_schema_validation": ["status", "artifacts"],
    }


def run_phase9c_review_closeout(run_dir: Path | str = PHASE9C_OUTPUT_DIR) -> Phase9CResult:
    """Generate Phase 9C review/synthesis artifacts from existing Phase 9A/9B artifacts."""

    run_path = Path(run_dir)
    artifact_dir = run_path / "artifacts"
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    contract = phase9c_review_contract()
    artifact_paths: dict[str, str] = {}
    contract_path = run_path / "phase9c_contract.json"
    _write_json(contract_path, contract)
    artifact_paths["phase9c_contract"] = str(contract_path)

    inputs = load_phase9c_inputs()
    review_rows = build_review_tag_rows(inputs["gradcam_rows"], inputs["phase9a_keys"])
    issue_rows = build_label_data_quality_inventory(review_rows)

    review_manifest_path = artifact_dir / "review_tag_manifest.csv"
    write_csv_rows(review_rows, review_manifest_path)
    artifact_paths["review_tag_manifest"] = str(review_manifest_path)

    issue_inventory_path = artifact_dir / "label_data_quality_inventory.csv"
    write_csv_rows(issue_rows, issue_inventory_path)
    artifact_paths["label_data_quality_inventory"] = str(issue_inventory_path)

    hypothesis_report_path = artifact_dir / "failure_hypothesis_report.json"
    hypothesis_report = {
        "phase": "9C",
        "run_id": PHASE9C_RUN_ID,
        "status": "pending_builder_review",
        "source_artifacts": contract["source_artifacts"],
        "review_record_count": len(review_rows),
        "hypothesis_status": "no semantic hypotheses generated without builder visual observations",
        "limitations": contract["interpretation_boundaries"],
    }
    _write_json(hypothesis_report_path, hypothesis_report)
    artifact_paths["failure_hypothesis_report"] = str(hypothesis_report_path)

    report_path = run_path / "phase9c_review_synthesis_report.md"
    write_phase9c_report(
        report_path,
        contract=contract,
        review_rows=review_rows,
        issue_rows=issue_rows,
    )
    artifact_paths["phase9c_report"] = str(report_path)

    schema_validation_path = artifact_dir / "phase9c_artifact_schema_validation.json"
    artifact_paths["artifact_schema_validation"] = str(schema_validation_path)
    _write_json(schema_validation_path, {"status": "passed", "artifacts": {}})

    result = Phase9CResult(
        run_dir=run_path,
        run_id=PHASE9C_RUN_ID,
        status="completed_for_phase_check_review",
        artifact_paths=artifact_paths,
    )
    result_path = run_path / "phase9c_result.json"
    artifact_paths["phase9c_result"] = str(result_path)
    _write_json(result_path, result.to_dict())

    schema_validation = validate_phase9c_generated_artifact_schemas(artifact_paths, contract)
    _write_json(schema_validation_path, schema_validation)
    return result


def load_phase9c_inputs(
    *,
    phase9a_dir: Path | str = PHASE9A_OUTPUT_DIR,
    phase9b_dir: Path | str = PHASE9B_OUTPUT_DIR,
) -> dict[str, Any]:
    phase9a_path = Path(phase9a_dir)
    phase9b_path = Path(phase9b_dir)
    required_phase9a = {
        "high_confidence_errors": phase9a_path / "artifacts" / "high_confidence_errors.csv",
        "model_disagreement_examples": phase9a_path / "artifacts" / "model_disagreement_examples.csv",
        "failure_selection_manifest": phase9a_path / "artifacts" / "failure_selection_manifest.json",
        "per_class_failure_summary": phase9a_path / "artifacts" / "per_class_failure_summary.csv",
        "per_class_failure_examples": phase9a_path / "artifacts" / "per_class_failure_examples.csv",
        "confusion_pair_examples": phase9a_path / "artifacts" / "confusion_pair_examples.csv",
    }
    required_phase9b = {
        "diagnostic_selection_manifest": phase9b_path / "artifacts" / "diagnostic_selection_manifest.json",
        "gradcam_manifest": phase9b_path / "artifacts" / "gradcam_manifest.csv",
        "gradcam_schema_validation": phase9b_path / "artifacts" / "gradcam_schema_validation.json",
    }
    for name, path in {**required_phase9a, **required_phase9b}.items():
        if not path.exists():
            raise FileNotFoundError(f"Phase 9C requires existing artifact {name}: {path}")

    schema_validation = _read_json_object(required_phase9b["gradcam_schema_validation"])
    if schema_validation.get("status") != "passed":
        raise ValueError("Phase 9C requires passed Phase 9B generated-artifact validation")

    phase9a_keys = _phase9a_selected_keys(required_phase9a)
    gradcam_rows = _read_csv_rows(required_phase9b["gradcam_manifest"])
    _validate_gradcam_alignment(gradcam_rows, phase9a_keys)
    return {
        "phase9a_keys": phase9a_keys,
        "gradcam_rows": gradcam_rows,
        "required_phase9a": required_phase9a,
        "required_phase9b": required_phase9b,
    }


def build_review_tag_rows(
    gradcam_rows: Sequence[dict[str, str]],
    phase9a_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_sorted_gradcam_rows(gradcam_rows), start=1):
        heatmap = _load_heatmap(Path(row["heatmap_path"]))
        heatmap_stats = heatmap_review_properties(heatmap)
        properties = _machine_tags_for_row(row, heatmap_stats)
        basis = {
            "selection_category": row["selection_category"],
            "true_label": row["true_label"],
            "predicted_label": row["source_predicted_label"],
            "confidence": float(row["diagnostic_confidence"]),
            "heatmap": heatmap_stats,
        }
        key = (row["selection_category"], row["run_id"], row["sample_id"])
        source_alignment = "aligned_to_phase9a_selection" if key in phase9a_keys else "correct_control_from_phase7_predictions"
        rows.append(
            {
                "review_id": f"phase9c-{index:04d}",
                "diagnostic_id": row["diagnostic_id"],
                "selection_category": row["selection_category"],
                "selection_rank": int(row["selection_rank"]),
                "run_id": row["run_id"],
                "sample_id": row["sample_id"],
                "true_label": row["true_label"],
                "predicted_label": row["source_predicted_label"],
                "confidence": float(row["diagnostic_confidence"]),
                "machine_derived_properties": json.dumps(properties, sort_keys=True),
                "machine_derived_tag_basis": json.dumps(basis, sort_keys=True),
                "builder_visual_observation": "pending_builder_review",
                "human_review_status": "pending_builder_review",
                "cautious_hypothesis": "pending_builder_review",
                "unsupported_causal_claim": "none_recorded",
                "causal_claim_status": "prohibited_not_made",
                "diagnostic_output_reference": json.dumps(
                    {
                        "heatmap_path": row["heatmap_path"],
                        "overlay_path": row["overlay_path"],
                        "target_layer": row["target_layer"],
                        "source_alignment": source_alignment,
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def heatmap_review_properties(heatmap: Tensor) -> dict[str, Any]:
    heatmap = _require_heatmap_tensor(heatmap)
    flattened = heatmap.flatten().to(dtype=torch.float64)
    total = float(flattened.sum().item())
    count = flattened.numel()
    if total <= 0.0:
        raise ValueError("Phase 9C requires nonempty normalized heatmaps")
    probabilities = flattened / total
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12))).item()
    normalized_entropy = float(entropy / torch.log(torch.tensor(float(count))).item()) if count > 1 else 0.0
    top_region_fraction = float((flattened >= 0.8).to(torch.float32).mean().item())
    max_value = float(flattened.max().item())
    mean_value = float(flattened.mean().item())
    return {
        "height": int(heatmap.shape[0]),
        "width": int(heatmap.shape[1]),
        "minimum": float(flattened.min().item()),
        "maximum": max_value,
        "mean": mean_value,
        "normalized_entropy": normalized_entropy,
        "top_80pct_region_fraction": top_region_fraction,
    }


def build_label_data_quality_inventory(review_rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    sample_ids = sorted({str(row["sample_id"]) for row in review_rows})
    return [
        {
            "issue_id": f"phase9c-data-quality-{index:04d}",
            "source": "phase9c_review_scaffold",
            "sample_id": sample_id,
            "run_id": "multiple_or_review_pending",
            "issue_type": "pending_builder_review",
            "evidence_status": "not_machine_established",
            "builder_visual_observation": "pending_builder_review",
            "cautious_hypothesis": "pending_builder_review",
            "resolution_status": "unresolved",
        }
        for index, sample_id in enumerate(sample_ids, start=1)
    ]


def validate_phase9c_generated_artifact_schemas(
    artifact_paths: dict[str, str],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or phase9c_review_contract()
    schemas = contract["artifact_schema_requirements"]
    required_artifacts = {
        "phase9c_contract": artifact_paths.get("phase9c_contract"),
        "phase9c_result": artifact_paths.get("phase9c_result"),
        "review_tag_manifest": artifact_paths.get("review_tag_manifest"),
        "label_data_quality_inventory": artifact_paths.get("label_data_quality_inventory"),
        "failure_hypothesis_report": artifact_paths.get("failure_hypothesis_report"),
        "phase9c_artifact_schema_validation": artifact_paths.get("artifact_schema_validation"),
        "phase9c_report": artifact_paths.get("phase9c_report"),
    }
    results: dict[str, Any] = {"status": "passed", "artifacts": {}}
    for artifact_name, artifact_path in required_artifacts.items():
        if not artifact_path:
            results["artifacts"][artifact_name] = {"path": None, "status": "failed"}
            results["status"] = "failed"
            continue
        path = Path(artifact_path)
        artifact_result = _validate_artifact(artifact_name, path, schemas)
        results["artifacts"][artifact_name] = artifact_result
        if artifact_result["status"] != "passed":
            results["status"] = "failed"
    if results["status"] != "passed":
        raise ValueError("Phase 9C generated artifact schema validation failed")
    return results


def write_phase9c_report(
    path: Path,
    *,
    contract: dict[str, Any],
    review_rows: Sequence[dict[str, Any]],
    issue_rows: Sequence[dict[str, Any]],
) -> Path:
    counts_by_category: dict[str, int] = {}
    for row in review_rows:
        counts_by_category[str(row["selection_category"])] = counts_by_category.get(str(row["selection_category"]), 0) + 1
    lines = [
        "# Phase 9C Review Synthesis and Failure-Hypothesis Scaffold",
        "",
        "Status: implementation artifacts generated for phase-check review; Phase 9C and overall Phase 9 are not automatically closed or accepted.",
        "",
        "## Scope",
        "",
        "Phase 9C is a review/synthesis phase only. It does not run new evaluation, regenerate predictions or Grad-CAM diagnostics, train, tune, mutate checkpoints, select models, implement inference, begin Phase 10, or add saliency, embeddings, UMAP, or t-SNE.",
        "",
        "## Evidence Separation",
        "",
        "- Selected example: preserved Phase 9A/9B selection row and sample identity.",
        "- Prediction/failure: preserved labels, predicted label, selection category, and confidence context.",
        "- Diagnostic output: preserved Phase 9B heatmap and overlay references plus deterministic heatmap statistics.",
        "- Machine-derived review property: generated only from deterministic artifact fields and heatmap statistics declared in the contract.",
        "- Builder/human visual observation: `pending_builder_review` unless supplied by the builder.",
        "- Cautious hypothesis: `pending_builder_review` unless supported by explicit review evidence.",
        "- Unsupported causal claim: prohibited; no causal claim is made.",
        "",
        "## Review Counts",
        "",
        f"- Review records: `{len(review_rows)}`",
        f"- Data-quality inventory rows: `{len(issue_rows)}`",
    ]
    for category, count in sorted(counts_by_category.items()):
        lines.append(f"- `{category}`: `{count}`")
    lines.extend(
        [
            "",
            "## Machine-Derived Properties",
            "",
            "Machine-derived tags are limited to deterministic properties in `phase9c_contract.json`, including prediction correctness/category membership and heatmap entropy/top-region area. Semantic tags such as `possible_label_noise`, `class_similarity`, and `background_context_possible` are not machine-generated in Phase 9C.",
            "",
            "## Hypothesis Status",
            "",
            "No semantic failure hypothesis is asserted without builder visual observations. The generated hypothesis fields remain `pending_builder_review`.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in contract["interpretation_boundaries"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Grad-CAM does not prove model reasoning, attention, causality, or the cause of a failure.",
        ]
    )
    _assert_no_unsupported_causal_language(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _phase9a_selected_keys(required_phase9a: dict[str, Path]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in _read_csv_rows(required_phase9a["high_confidence_errors"]):
        keys.add(("high_confidence_error", row["run_id"], row["sample_id"]))
    disagreement_rows = _read_csv_rows(required_phase9a["model_disagreement_examples"])
    for row in disagreement_rows:
        for run_id in PHASE9C_FIXED_RUN_ORDER:
            keys.add(("model_disagreement", run_id, row["sample_id"]))
    return keys


def _validate_gradcam_alignment(
    gradcam_rows: Sequence[dict[str, str]],
    phase9a_keys: set[tuple[str, str, str]],
) -> None:
    for row in gradcam_rows:
        category = row["selection_category"]
        key = (category, row["run_id"], row["sample_id"])
        if category in {"high_confidence_error", "model_disagreement"} and key not in phase9a_keys:
            raise ValueError(f"Phase 9C cannot align diagnostic row to Phase 9A selection: {key}")
        if category == "correct_control" and row["true_label"] != row["source_predicted_label"]:
            raise ValueError(f"Phase 9C correct control is not correct: {row['diagnostic_id']}")


def _machine_tags_for_row(row: dict[str, str], heatmap_stats: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if row["true_label"] == row["source_predicted_label"] and row["selection_category"] == "correct_control":
        tags.append("prediction_correct_control")
    elif row["true_label"] != row["source_predicted_label"]:
        tags.append("prediction_error")
    if row["selection_category"] == "high_confidence_error":
        tags.append("high_confidence_error_member")
    if row["selection_category"] == "model_disagreement":
        tags.append("model_disagreement_member")
    entropy = float(heatmap_stats["normalized_entropy"])
    top_area = float(heatmap_stats["top_80pct_region_fraction"])
    if entropy <= HEATMAP_LOW_ENTROPY_THRESHOLD:
        tags.append("heatmap_low_spatial_entropy")
    if entropy >= HEATMAP_HIGH_ENTROPY_THRESHOLD:
        tags.append("heatmap_high_spatial_entropy")
    if top_area <= HEATMAP_COMPACT_TOP_AREA_THRESHOLD:
        tags.append("heatmap_compact_top_region")
    if top_area >= HEATMAP_DIFFUSE_TOP_AREA_THRESHOLD:
        tags.append("heatmap_diffuse_top_region")
    return tags


def _validate_artifact(
    artifact_name: str,
    path: Path,
    schemas: dict[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "passed"}
    if not path.exists():
        result.update({"status": "failed", "missing": True})
        return result
    if artifact_name in schemas:
        required = set(schemas[artifact_name])
        present = set(_json_fields(path) if path.suffix == ".json" else _csv_fields(path))
        missing = sorted(required.difference(present))
        result.update({"required_fields": sorted(required), "present_fields": sorted(present), "missing_fields": missing})
        if missing:
            result["status"] = "failed"
    if artifact_name == "phase9c_contract":
        payload = _read_json_object(path)
        if payload.get("phase") != "9C" or payload.get("run_id") != PHASE9C_RUN_ID:
            result["status"] = "failed"
            result["identity_error"] = True
        machine_tags = payload.get("review_property_classes", {}).get("machine_derived", {}).get("allowed_tags")
        if not isinstance(machine_tags, dict) or "possible_label_noise" in machine_tags:
            result["status"] = "failed"
            result["machine_tag_schema_error"] = True
    elif artifact_name == "phase9c_result":
        payload = _read_json_object(path)
        if payload.get("run_id") != PHASE9C_RUN_ID or not isinstance(payload.get("artifact_paths"), dict):
            result["status"] = "failed"
            result["identity_error"] = True
    elif artifact_name == "review_tag_manifest":
        bad_rows = []
        for row in _read_csv_rows(path):
            if row["builder_visual_observation"] != "pending_builder_review":
                bad_rows.append({"review_id": row["review_id"], "error": "builder observation was machine-filled"})
            if row["cautious_hypothesis"] != "pending_builder_review":
                bad_rows.append({"review_id": row["review_id"], "error": "hypothesis was machine-filled"})
            if row["unsupported_causal_claim"] != "none_recorded":
                bad_rows.append({"review_id": row["review_id"], "error": "unsupported causal claim recorded"})
        if bad_rows:
            result["status"] = "failed"
            result["bad_rows"] = bad_rows
    elif artifact_name == "phase9c_artifact_schema_validation":
        payload = _read_json_object(path)
        if payload.get("status") != "passed":
            result["status"] = "failed"
            result["schema_validation_status_error"] = payload.get("status")
    elif artifact_name == "phase9c_report":
        text = path.read_text(encoding="utf-8")
        required_phrases = [
            "Phase 9C is a review/synthesis phase only",
            "pending_builder_review",
            "Unsupported causal claim: prohibited",
            "Grad-CAM does not prove model reasoning",
        ]
        missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
        if missing_phrases:
            result["status"] = "failed"
            result["missing_required_phrases"] = missing_phrases
        try:
            _assert_no_unsupported_causal_language(text.splitlines())
        except ValueError as exc:
            result["status"] = "failed"
            result["causal_language_error"] = str(exc)
    return result


def _sorted_gradcam_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    order = {run_id: index for index, run_id in enumerate(PHASE9C_FIXED_RUN_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            _category_order(row["selection_category"]),
            int(row["selection_rank"]),
            order.get(row["run_id"], 999),
            row["sample_id"],
            row["diagnostic_id"],
        ),
    )


def _category_order(category: str) -> int:
    return {"high_confidence_error": 0, "model_disagreement": 1, "correct_control": 2}.get(category, 99)


def _load_heatmap(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 9B heatmap: {path}")
    return _require_heatmap_tensor(torch.load(path, map_location="cpu"))


def _require_heatmap_tensor(heatmap: Tensor) -> Tensor:
    if not isinstance(heatmap, Tensor):
        raise ValueError("heatmap artifact must be a torch Tensor")
    if heatmap.ndim != 2:
        raise ValueError("heatmap artifact must be a 2D tensor")
    if not torch.isfinite(heatmap).all():
        raise ValueError("heatmap artifact contains non-finite values")
    if float(heatmap.min().item()) < 0.0 or float(heatmap.max().item()) > 1.000001:
        raise ValueError("heatmap artifact is outside normalized [0, 1] range")
    if float(heatmap.max().item()) <= 0.0:
        raise ValueError("heatmap artifact is empty")
    return heatmap.detach().cpu()


def _assert_no_unsupported_causal_language(lines: Sequence[str]) -> None:
    text = "\n".join(lines).lower()
    banned_phrases = [
        "proves what the model looked at",
        "proves model reasoning",
        "establishes causality",
        "explains why an error occurred",
        "cause of the failure is",
    ]
    for phrase in banned_phrases:
        if phrase in text:
            raise ValueError(f"unsupported causal language found: {phrase}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_fields(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _json_fields(path: Path) -> list[str]:
    return list(_read_json_object(path))


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
