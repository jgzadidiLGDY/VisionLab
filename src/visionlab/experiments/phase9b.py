"""Phase 9B spatial diagnostics and interpretability artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import Tensor, nn

from visionlab.data import (
    CIFAR10_CLASSES,
    build_cifar10_split_datasets,
    preprocess_resnet18_imagenet_tensor,
)
from visionlab.data.cifar10 import CIFAR10_PREPROCESSING, normalize_tensor
from visionlab.diagnostics import compute_gradcam
from visionlab.evaluation import FailurePrediction, load_prediction_csv, write_csv_rows, write_gallery_html
from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import PHASE6C_RUN_ID
from visionlab.experiments.phase7 import (
    PHASE7_OUTPUT_DIR,
    Phase7RunReference,
    _load_phase4b_model,
    _load_phase6b2_model,
    _load_phase6c_model,
    phase7_references,
    sha256_file,
)
from visionlab.experiments.phase9a import (
    PHASE9A_OUTPUT_DIR,
    PHASE9A_REQUIRED_CONTEXT_ID,
    load_phase7_clean_validation_predictions,
)


PHASE9B_RUN_ID = "phase9b-spatial-diagnostics"
PHASE9B_OUTPUT_DIR = Path("outputs") / PHASE9B_RUN_ID
PHASE9B_HIGH_CONFIDENCE_TOP_N_PER_RUN = 8
PHASE9B_DISAGREEMENT_TOP_N = 8
PHASE9B_CORRECT_CONTROL_TOP_N_PER_RUN = 8
PHASE9B_FIXED_RUN_ORDER = (PHASE4B_RUN_ID, PHASE6B2_RUN_ID, PHASE6C_RUN_ID)


@dataclass(frozen=True)
class Phase9BResult:
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


def phase9b_diagnostic_contract() -> dict[str, Any]:
    """Return the explicit machine-readable Phase 9B diagnostic contract."""

    return {
        "phase": "9B",
        "run_id": PHASE9B_RUN_ID,
        "scope": "Grad-CAM-style spatial diagnostics only; no new evaluation, training, tuning, model selection, checkpoint mutation, inference, embeddings, UMAP, saliency, Phase 9C, or closeout",
        "source_population": {
            "phase9a_output_dir": str(PHASE9A_OUTPUT_DIR),
            "context_id": PHASE9A_REQUIRED_CONTEXT_ID,
            "dataset_id": "cifar10",
            "dataset_version": "phase1b-registered",
            "split": "val",
            "condition_id": "clean",
            "population_semantics": "machine-selected Phase 9A examples plus deterministic correct controls from existing Phase 7 clean validation predictions",
        },
        "fixed_run_order": list(PHASE9B_FIXED_RUN_ORDER),
        "selection_rules": {
            "high_confidence_error": {
                "source_artifact": "Phase 9A high_confidence_errors.csv",
                "population": "machine-selected Phase 9A high-confidence errors",
                "top_n_per_run": PHASE9B_HIGH_CONFIDENCE_TOP_N_PER_RUN,
                "ranking": ["existing Phase 9A rank ascending"],
            },
            "model_disagreement": {
                "source_artifact": "Phase 9A model_disagreement_examples.csv",
                "population": "machine-selected Phase 9A model-disagreement samples",
                "top_n_samples": PHASE9B_DISAGREEMENT_TOP_N,
                "row_expansion": "one diagnostic row per selected sample per compared fixed model",
                "ranking": [
                    "existing Phase 9A disagreement rank ascending",
                    "fixed checkpoint order declared in this contract",
                ],
            },
            "correct_control": {
                "source_artifact": "existing Phase 7 clean CIFAR-10 validation prediction CSVs",
                "population": "correct predictions only, per fixed run",
                "top_n_per_run": PHASE9B_CORRECT_CONTROL_TOP_N_PER_RUN,
                "ranking": ["confidence descending", "sample_id ascending"],
                "interpretation": "controls only; not representative explanations of model behavior",
            },
        },
        "target_layers": {
            PHASE4B_RUN_ID: "CustomCNN.feature_blocks[-1]",
            PHASE6B2_RUN_ID: "TransferResNet18.model.layer4",
            PHASE6C_RUN_ID: "TransferResNet18.model.layer4",
        },
        "diagnostic_method": {
            "name": "Grad-CAM-style weighted activation map",
            "target_class": "preserved predicted class for the selected model/sample",
            "normalization": "per-sample min-zero/ReLU then divide by positive max",
            "prediction_context": "diagnostic pass may record logits/probabilities/predicted label for traceability only; these are not new evaluation metrics",
        },
        "hard_stop_conditions": [
            "missing Phase 9A artifact",
            "sample ID or label mismatch",
            "checkpoint SHA mismatch",
            "target layer cannot be resolved",
            "gradients unavailable",
            "non-finite or all-empty heatmap",
            "inability to map a selected sample to its source image",
        ],
        "artifact_schema_requirements": phase9b_artifact_schema_requirements(),
        "interpretation_boundaries": [
            "Grad-CAM output is diagnostic evidence, not proof of what the model looked at",
            "Grad-CAM does not establish causal explanations",
            "confidence does not imply correctness",
            "visual observations must be separated from interpretation and hypothesis",
            "Phase 9B does not prescribe model changes or interventions",
        ],
    }


def phase9b_artifact_schema_requirements() -> dict[str, list[str]]:
    return {
        "phase9b_contract": [
            "phase",
            "run_id",
            "scope",
            "source_population",
            "fixed_run_order",
            "selection_rules",
            "target_layers",
            "diagnostic_method",
            "hard_stop_conditions",
            "artifact_schema_requirements",
            "interpretation_boundaries",
        ],
        "phase9b_result": [
            "run_dir",
            "run_id",
            "status",
            "artifact_paths",
        ],
        "diagnostic_selection_manifest": [
            "phase",
            "run_id",
            "source_population",
            "fixed_checkpoints",
            "selection_rules",
            "target_layers",
            "selected_diagnostics",
        ],
        "gradcam_manifest": [
            "diagnostic_id",
            "selection_category",
            "selection_rank",
            "run_id",
            "checkpoint_tag",
            "checkpoint_path",
            "checkpoint_sha256",
            "dataset_id",
            "dataset_version",
            "split",
            "condition_id",
            "context_id",
            "sample_id",
            "source_id",
            "true_label",
            "source_predicted_label",
            "target_class_index",
            "target_class_label",
            "diagnostic_predicted_label",
            "diagnostic_confidence",
            "target_layer",
            "preprocessing_id",
            "model_input_height",
            "model_input_width",
            "heatmap_path",
            "overlay_path",
            "visual_observation",
            "interpretation_hypothesis",
            "limitations",
            "prediction_context_semantics",
        ],
        "gradcam_gallery_manifest": [
            "image_path",
            "diagnostic_id",
            "selection_category",
            "selection_rank",
            "run_id",
            "sample_id",
            "true_label",
            "predicted_label",
            "confidence",
            "target_layer",
        ],
        "gradcam_schema_validation": [
            "status",
            "artifacts",
            "heatmap_artifacts",
            "overlay_artifacts",
        ],
    }


def run_phase9b_spatial_diagnostics(run_dir: Path | str = PHASE9B_OUTPUT_DIR) -> Phase9BResult:
    run_path = Path(run_dir)
    artifact_dir = run_path / "artifacts"
    heatmap_dir = artifact_dir / "heatmaps"
    overlay_dir = artifact_dir / "overlays"
    gallery_dir = artifact_dir / "galleries"
    for directory in (run_path, artifact_dir, heatmap_dir, overlay_dir, gallery_dir):
        directory.mkdir(parents=True, exist_ok=True)

    contract = phase9b_diagnostic_contract()
    artifact_paths: dict[str, str] = {}
    contract_path = run_path / "phase9b_contract.json"
    _write_json(contract_path, contract)
    artifact_paths["phase9b_contract"] = str(contract_path)

    references = {reference.run_id: reference for reference in phase7_references()}
    _verify_fixed_checkpoint_identities(references)
    phase9a_checkpoint_identities = _read_phase9a_checkpoint_identities()
    for run_id, reference in references.items():
        expected_sha = phase9a_checkpoint_identities[run_id]["checkpoint_sha256"]
        actual_sha = sha256_file(reference.checkpoint_path)
        if actual_sha != expected_sha:
            raise ValueError(f"checkpoint SHA mismatch for {run_id}")

    predictions = load_phase7_clean_validation_predictions()
    selected = select_phase9b_diagnostic_rows(predictions_by_run=predictions)
    datasets = build_cifar10_split_datasets(root="data", download=False)
    samples = {datasets.val[index]["sample_id"]: datasets.val[index] for index in range(len(datasets.val))}
    models = _load_models(references)
    before_sha = {run_id: sha256_file(references[run_id].checkpoint_path) for run_id in PHASE9B_FIXED_RUN_ORDER}

    manifest_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    for row in selected:
        sample_id = row["sample_id"]
        if sample_id not in samples:
            raise ValueError(f"unable to map selected sample to source image: {sample_id}")
        sample = samples[sample_id]
        if int(sample["label"]) != int(row["true_index"]):
            raise ValueError(f"sample ID or label mismatch for {sample_id}")

        run_id = row["run_id"]
        model = models[run_id]
        target_layer = resolve_phase9b_target_layer(model, run_id)
        input_tensor, preprocessing_id = _diagnostic_input(sample, run_id)
        target_class_index = int(row["predicted_index"])
        result = compute_gradcam(
            model,
            input_tensor.unsqueeze(0),
            target_layer=target_layer,
            target_class_index=target_class_index,
        )
        target_class_label = CIFAR10_CLASSES[target_class_index]
        diagnostic_predicted_label = CIFAR10_CLASSES[result.predicted_index]
        heatmap_path = heatmap_dir / f"{row['diagnostic_id']}.pt"
        overlay_path = overlay_dir / f"{row['diagnostic_id']}.png"
        torch.save(result.heatmap, heatmap_path)
        _write_overlay(_source_image(sample), result.heatmap, overlay_path)

        reference = references[run_id]
        manifest_row = {
            "diagnostic_id": row["diagnostic_id"],
            "selection_category": row["selection_category"],
            "selection_rank": row["selection_rank"],
            "run_id": run_id,
            "checkpoint_tag": reference.checkpoint_tag,
            "checkpoint_path": str(reference.checkpoint_path),
            "checkpoint_sha256": before_sha[run_id],
            "dataset_id": row["dataset_id"],
            "dataset_version": row["dataset_version"],
            "split": row["split"],
            "condition_id": row["condition_id"],
            "context_id": row["context_id"],
            "sample_id": sample_id,
            "source_id": row["source_id"],
            "true_label": row["true_label"],
            "source_predicted_label": row["predicted_label"],
            "target_class_index": target_class_index,
            "target_class_label": target_class_label,
            "diagnostic_predicted_label": diagnostic_predicted_label,
            "diagnostic_confidence": result.confidence,
            "target_layer": contract["target_layers"][run_id],
            "preprocessing_id": preprocessing_id,
            "model_input_height": int(input_tensor.shape[-2]),
            "model_input_width": int(input_tensor.shape[-1]),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "visual_observation": "pending_builder_review",
            "interpretation_hypothesis": "deferred; Grad-CAM does not establish causality",
            "limitations": "Grad-CAM is a coarse spatial diagnostic and may be sensitive to target layer, preprocessing, and model architecture.",
            "prediction_context_semantics": contract["diagnostic_method"]["prediction_context"],
            "logits": json.dumps(result.logits),
            "probabilities": json.dumps(result.probabilities),
        }
        manifest_rows.append(manifest_row)
        gallery_rows.append(
            {
                "image_path": _relative_html_path(overlay_path, gallery_dir),
                "diagnostic_id": row["diagnostic_id"],
                "selection_category": row["selection_category"],
                "selection_rank": row["selection_rank"],
                "run_id": run_id,
                "sample_id": sample_id,
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "confidence": result.confidence,
                "target_layer": contract["target_layers"][run_id],
            }
        )

    after_sha = {run_id: sha256_file(references[run_id].checkpoint_path) for run_id in PHASE9B_FIXED_RUN_ORDER}
    if before_sha != after_sha:
        raise ValueError("checkpoint immutability check failed")

    gradcam_manifest_path = artifact_dir / "gradcam_manifest.csv"
    write_csv_rows(manifest_rows, gradcam_manifest_path)
    artifact_paths["gradcam_manifest"] = str(gradcam_manifest_path)

    gallery_manifest_path = gallery_dir / "gradcam_gallery_manifest.csv"
    gallery_html_path = gallery_dir / "gradcam_gallery.html"
    write_csv_rows(gallery_rows, gallery_manifest_path)
    write_gallery_html(gallery_rows, gallery_html_path, title="Phase 9B Grad-CAM Spatial Diagnostics")
    artifact_paths["gradcam_gallery_manifest"] = str(gallery_manifest_path)
    artifact_paths["gradcam_gallery_html"] = str(gallery_html_path)

    selection_manifest_path = artifact_dir / "diagnostic_selection_manifest.json"
    write_diagnostic_selection_manifest(
        selection_manifest_path,
        contract=contract,
        references=references,
        selected=selected,
        generated_artifacts=artifact_paths,
    )
    artifact_paths["diagnostic_selection_manifest"] = str(selection_manifest_path)

    schema_validation_path = artifact_dir / "gradcam_schema_validation.json"
    legacy_schema_validation_path = artifact_dir / "phase9b_artifact_schema_validation.json"
    report_path = run_path / "phase9b_spatial_diagnostics_report.md"
    result_path = run_path / "phase9b_result.json"
    artifact_paths["gradcam_schema_validation"] = str(schema_validation_path)
    artifact_paths["artifact_schema_validation"] = str(legacy_schema_validation_path)
    artifact_paths["spatial_diagnostics_report"] = str(report_path)
    artifact_paths["phase9b_result"] = str(result_path)

    counts = {
        "diagnostic_rows": len(manifest_rows),
        "high_confidence_error_rows": sum(1 for row in manifest_rows if row["selection_category"] == "high_confidence_error"),
        "model_disagreement_rows": sum(1 for row in manifest_rows if row["selection_category"] == "model_disagreement"),
        "correct_control_rows": sum(1 for row in manifest_rows if row["selection_category"] == "correct_control"),
        "overlay_images": len(gallery_rows),
    }
    provisional_validation = {"status": "pending_complete_validation"}
    _write_json(schema_validation_path, {"status": "passed", "artifacts": {}, "heatmap_artifacts": {}, "overlay_artifacts": {}})
    _write_json(legacy_schema_validation_path, {"status": "passed", "artifacts": {}, "heatmap_artifacts": {}, "overlay_artifacts": {}})
    write_phase9b_report(
        report_path,
        contract=contract,
        counts=counts,
        schema_validation=provisional_validation,
    )

    result = Phase9BResult(
        run_dir=run_path,
        run_id=PHASE9B_RUN_ID,
        status="completed_for_phase_check_review",
        artifact_paths=artifact_paths,
    )
    _write_json(result_path, result.to_dict())

    schema_validation = validate_phase9b_generated_artifact_schemas(artifact_paths, contract)
    _write_json(schema_validation_path, schema_validation)
    _write_json(legacy_schema_validation_path, schema_validation)
    write_phase9b_report(
        report_path,
        contract=contract,
        counts=counts,
        schema_validation=schema_validation,
    )
    _write_json(result_path, result.to_dict())
    return result


def select_phase9b_diagnostic_rows(
    *,
    predictions_by_run: dict[str, Sequence[FailurePrediction]] | None = None,
    phase9a_dir: Path | str = PHASE9A_OUTPUT_DIR,
) -> tuple[dict[str, Any], ...]:
    predictions_by_run = predictions_by_run or load_phase7_clean_validation_predictions()
    phase9a_path = Path(phase9a_dir)
    high_confidence_path = phase9a_path / "artifacts" / "high_confidence_errors.csv"
    disagreement_path = phase9a_path / "artifacts" / "model_disagreement_examples.csv"
    if not high_confidence_path.exists():
        raise FileNotFoundError(f"missing Phase 9A artifact: {high_confidence_path}")
    if not disagreement_path.exists():
        raise FileNotFoundError(f"missing Phase 9A artifact: {disagreement_path}")

    rows: list[dict[str, Any]] = []
    high_confidence_rows = _read_csv_rows(high_confidence_path)
    for run_id in PHASE9B_FIXED_RUN_ORDER:
        run_rows = sorted(
            [row for row in high_confidence_rows if row["run_id"] == run_id],
            key=lambda row: int(row["rank"]),
        )[:PHASE9B_HIGH_CONFIDENCE_TOP_N_PER_RUN]
        for rank, row in enumerate(run_rows, start=1):
            rows.append(_diagnostic_selection_row("high_confidence_error", rank, row, run_id))

    disagreement_rows = sorted(_read_csv_rows(disagreement_path), key=lambda row: int(row["rank"]))[
        :PHASE9B_DISAGREEMENT_TOP_N
    ]
    prediction_lookup = _prediction_lookup(predictions_by_run)
    for disagreement in disagreement_rows:
        sample_id = disagreement["sample_id"]
        for run_id in PHASE9B_FIXED_RUN_ORDER:
            prediction = prediction_lookup[run_id][sample_id]
            rows.append(
                _diagnostic_selection_row(
                    "model_disagreement",
                    int(disagreement["rank"]),
                    prediction.to_dict(),
                    run_id,
                    extra={"phase9a_disagreement_rank": disagreement["rank"]},
                )
            )

    for run_id in PHASE9B_FIXED_RUN_ORDER:
        correct = sorted(
            [item for item in predictions_by_run[run_id] if item.correct],
            key=lambda item: (-item.confidence, item.sample_id),
        )[:PHASE9B_CORRECT_CONTROL_TOP_N_PER_RUN]
        for rank, prediction in enumerate(correct, start=1):
            rows.append(_diagnostic_selection_row("correct_control", rank, prediction.to_dict(), run_id))

    return tuple(_with_diagnostic_ids(rows))


def resolve_phase9b_target_layer(model: nn.Module, run_id: str) -> nn.Module:
    if run_id == PHASE4B_RUN_ID:
        feature_blocks = getattr(model, "feature_blocks", None)
        if feature_blocks is None or len(feature_blocks) == 0:
            raise ValueError("target layer cannot be resolved for CustomCNN")
        return feature_blocks[-1]
    if run_id in {PHASE6B2_RUN_ID, PHASE6C_RUN_ID}:
        wrapped = getattr(model, "model", None)
        layer4 = getattr(wrapped, "layer4", None)
        if layer4 is None:
            raise ValueError(f"target layer cannot be resolved for {run_id}")
        return layer4
    raise ValueError(f"unknown Phase 9B run_id: {run_id}")


def write_diagnostic_selection_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
    references: dict[str, Phase7RunReference],
    selected: Sequence[dict[str, Any]],
    generated_artifacts: dict[str, str],
) -> Path:
    manifest = {
        "phase": contract["phase"],
        "run_id": contract["run_id"],
        "source_population": contract["source_population"],
        "fixed_checkpoints": {
            run_id: references[run_id].to_dict() for run_id in PHASE9B_FIXED_RUN_ORDER
        },
        "selection_rules": contract["selection_rules"],
        "target_layers": contract["target_layers"],
        "selected_diagnostics": list(selected),
        "generated_artifacts": dict(generated_artifacts),
        "interpretation_boundaries": contract["interpretation_boundaries"],
    }
    _write_json(path, manifest)
    return path


def validate_phase9b_generated_artifact_schemas(
    artifact_paths: dict[str, str],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete generated Phase 9B artifact set."""

    contract = contract or phase9b_diagnostic_contract()
    schemas = contract["artifact_schema_requirements"]
    required_artifacts = {
        "phase9b_contract": artifact_paths.get("phase9b_contract"),
        "phase9b_result": artifact_paths.get("phase9b_result"),
        "diagnostic_selection_manifest": artifact_paths.get("diagnostic_selection_manifest"),
        "gradcam_manifest": artifact_paths.get("gradcam_manifest"),
        "gradcam_schema_validation": artifact_paths.get("gradcam_schema_validation")
        or artifact_paths.get("artifact_schema_validation"),
        "spatial_diagnostics_report": artifact_paths.get("spatial_diagnostics_report"),
        "gradcam_gallery_manifest": artifact_paths.get("gradcam_gallery_manifest"),
        "gradcam_gallery_html": artifact_paths.get("gradcam_gallery_html"),
    }
    results: dict[str, Any] = {"status": "passed", "artifacts": {}}
    for artifact_name, artifact_path in required_artifacts.items():
        if not artifact_path:
            results["artifacts"][artifact_name] = {
                "path": None,
                "status": "failed",
                "missing_artifact_path": True,
            }
            results["status"] = "failed"
            continue
        path = Path(artifact_path)
        artifact_result = _validate_phase9b_artifact_file(artifact_name, path, schemas)
        results["artifacts"][artifact_name] = artifact_result
        if artifact_result["status"] != "passed":
            results["status"] = "failed"

    gradcam_manifest_path = Path(str(required_artifacts["gradcam_manifest"] or ""))
    if gradcam_manifest_path.exists():
        heatmaps = _validate_heatmap_artifacts(gradcam_manifest_path)
        overlays = _validate_overlay_artifacts(gradcam_manifest_path)
    else:
        heatmaps = {"status": "failed", "count": 0, "failures": ["missing gradcam manifest"]}
        overlays = {"status": "failed", "count": 0, "failures": ["missing gradcam manifest"]}
    results["heatmap_artifacts"] = heatmaps
    results["overlay_artifacts"] = overlays
    if heatmaps["status"] != "passed" or overlays["status"] != "passed":
        results["status"] = "failed"

    if results["status"] != "passed":
        raise ValueError("Phase 9B generated artifact schema validation failed")
    return results


def write_phase9b_report(
    path: Path,
    *,
    contract: dict[str, Any],
    counts: dict[str, int],
    schema_validation: dict[str, Any],
) -> Path:
    lines = [
        "# Phase 9B Spatial Diagnostics and Interpretability Artifacts Report",
        "",
        "Status: implementation artifacts generated for phase-check review; Phase 9B is not automatically closed or accepted.",
        "",
        "## Scope",
        "",
        "Phase 9B generates Grad-CAM-style spatial diagnostics only. It does not train, tune, mutate checkpoints, regenerate Phase 7/8 predictions, run new evaluation, select models, implement inference, create embeddings/UMAP, implement saliency, perform Phase 9C closeout, or select an applied domain.",
        "",
        "## Diagnostic Populations",
        "",
        f"- High-confidence errors: top `{PHASE9B_HIGH_CONFIDENCE_TOP_N_PER_RUN}` per run from Phase 9A high-confidence error rank.",
        f"- Model disagreements: top `{PHASE9B_DISAGREEMENT_TOP_N}` Phase 9A disagreement samples, expanded to one diagnostic row per fixed model in declared order.",
        f"- Correct controls: top `{PHASE9B_CORRECT_CONTROL_TOP_N_PER_RUN}` correct predictions per run from existing Phase 7 clean validation predictions by confidence descending and sample ID ascending.",
        "",
        "## Diagnostic Method",
        "",
        f"- Method: {contract['diagnostic_method']['name']}.",
        "- Target class: preserved predicted class for the selected model/sample.",
        "- Prediction logits and probabilities recorded by the diagnostic pass are traceability context only, not new evaluation metrics.",
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
            "## Review Table Semantics",
            "",
            "- Selected example: identified by deterministic selection category, rank, run, and sample ID.",
            "- Model prediction context: preserved predicted label from Phase 7/9A plus diagnostic-pass traceability values.",
            "- Grad-CAM output: raw heatmap and overlay image generated from the declared target layer.",
            "- Direct visual observation: left as `pending_builder_review` for human inspection.",
            "- Interpretation/hypothesis: deferred and explicitly non-causal.",
            "- Limitations: included per row and in this report.",
            "",
            "## Limitations",
            "",
            "- Grad-CAM does not prove what the model looked at.",
            "- Grad-CAM does not prove reasoning or establish why an error occurred.",
            "- Heatmaps can be coarse, architecture-dependent, layer-dependent, and preprocessing-dependent.",
            "- Correct-control examples are controls only and are not representative explanations of model behavior.",
            "- Diagnostic interpretation remains bounded to the selected Phase 9B population.",
            "",
            "## Visual QA Boundary",
            "",
            "The generated gallery should be inspected for overlay alignment, nonblank heatmaps, saturation patterns, and unambiguous model/context labels. Observations from that review should remain separate from hypotheses.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _verify_fixed_checkpoint_identities(references: dict[str, Phase7RunReference]) -> None:
    for run_id in PHASE9B_FIXED_RUN_ORDER:
        reference = references[run_id]
        if not reference.checkpoint_path.exists():
            raise FileNotFoundError(f"missing fixed checkpoint for {run_id}: {reference.checkpoint_path}")
        if sha256_file(reference.checkpoint_path) != reference.to_dict()["checkpoint_sha256"]:
            raise ValueError(f"checkpoint SHA mismatch for {run_id}")


def _read_phase9a_checkpoint_identities() -> dict[str, dict[str, Any]]:
    manifest_path = PHASE9A_OUTPUT_DIR / "artifacts" / "failure_selection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Phase 9A artifact: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    identities = manifest.get("checkpoint_identities")
    if not isinstance(identities, dict):
        raise ValueError("Phase 9A manifest is missing checkpoint identities")
    return identities


def _load_models(references: dict[str, Phase7RunReference]) -> dict[str, nn.Module]:
    return {
        PHASE4B_RUN_ID: _load_phase4b_model(references[PHASE4B_RUN_ID]),
        PHASE6B2_RUN_ID: _load_phase6b2_model(references[PHASE6B2_RUN_ID]),
        PHASE6C_RUN_ID: _load_phase6c_model(references[PHASE6C_RUN_ID]),
    }


def _diagnostic_input(sample: dict[str, Any], run_id: str) -> tuple[Tensor, str]:
    if run_id == PHASE4B_RUN_ID:
        return normalize_tensor(sample["raw_input"], CIFAR10_PREPROCESSING), "phase4-cifar10-normalization"
    if run_id in {PHASE6B2_RUN_ID, PHASE6C_RUN_ID}:
        return preprocess_resnet18_imagenet_tensor(sample["raw_input"]), "phase6a-resnet18-imagenet1k-v1-preprocessing"
    raise ValueError(f"unknown run_id: {run_id}")


def _source_image(sample: dict[str, Any]) -> Image.Image:
    tensor = sample["raw_input"].detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _write_overlay(source: Image.Image, heatmap: Tensor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_image = Image.fromarray(
        (heatmap.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8"),
        mode="L",
    )
    base = source.convert("RGB").resize(heatmap_image.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", heatmap_image.size, (255, 0, 0))
    overlay = Image.blend(base, red, alpha=0.35)
    composed = Image.composite(overlay, base, heatmap_image)
    composed.save(path)
    return path


def _relative_html_path(path: Path, base_dir: Path) -> str:
    if path.is_absolute():
        return path.as_posix()
    try:
        return Path("..").joinpath(path.relative_to(base_dir.parent)).as_posix()
    except ValueError:
        return path.as_posix()
def _diagnostic_selection_row(
    category: str,
    rank: int,
    row: dict[str, Any],
    run_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = {
        "selection_category": category,
        "selection_rank": rank,
        "run_id": run_id,
        "dataset_id": row["dataset_id"],
        "dataset_version": row["dataset_version"],
        "split": row["split"],
        "condition_id": row["condition_id"],
        "context_id": row["context_id"],
        "sample_id": row["sample_id"],
        "source_id": row["source_id"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": float(row["confidence"]),
        "true_index": int(row["true_index"]),
        "predicted_index": int(row["predicted_index"]),
    }
    if extra:
        selected.update(extra)
    return selected


def _with_diagnostic_ids(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    updated = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["diagnostic_id"] = f"phase9b-{index:04d}-{_safe_name(item['selection_category'])}-{_safe_name(item['run_id'])}-{_safe_name(item['sample_id'])}"
        updated.append(item)
    return tuple(updated)


def _prediction_lookup(
    predictions_by_run: dict[str, Sequence[FailurePrediction]],
) -> dict[str, dict[str, FailurePrediction]]:
    return {
        run_id: {prediction.sample_id: prediction for prediction in predictions}
        for run_id, predictions in predictions_by_run.items()
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _validate_phase9b_artifact_file(
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
        missing_fields = sorted(required.difference(present))
        result.update(
            {
                "required_fields": sorted(required),
                "present_fields": sorted(present),
                "missing_fields": missing_fields,
            }
        )
        if missing_fields:
            result["status"] = "failed"
    if artifact_name == "phase9b_contract":
        payload = _read_json_object(path)
        if payload.get("phase") != "9B" or payload.get("run_id") != PHASE9B_RUN_ID:
            result["status"] = "failed"
            result["identity_error"] = True
    elif artifact_name == "phase9b_result":
        payload = _read_json_object(path)
        if payload.get("run_id") != PHASE9B_RUN_ID:
            result["status"] = "failed"
            result["identity_error"] = True
        artifact_paths = payload.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            result["status"] = "failed"
            result["artifact_paths_error"] = "missing or not an object"
    elif artifact_name == "gradcam_schema_validation":
        payload = _read_json_object(path)
        if payload.get("status") != "passed":
            result["status"] = "failed"
            result["schema_validation_status_error"] = payload.get("status")
    elif artifact_name == "spatial_diagnostics_report":
        text = path.read_text(encoding="utf-8")
        required_phrases = [
            "Phase 9B Spatial Diagnostics",
            "not automatically closed or accepted",
            "Grad-CAM does not prove",
            "not new evaluation metrics",
            "pending_builder_review",
        ]
        missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
        result["missing_required_phrases"] = missing_phrases
        if missing_phrases:
            result["status"] = "failed"
    elif artifact_name == "gradcam_gallery_manifest":
        missing_images = []
        for row in _read_csv_rows(path):
            image_path = Path(row["image_path"])
            resolved = image_path if image_path.is_absolute() else path.parent / image_path
            if not resolved.exists():
                missing_images.append(row["image_path"])
        result["missing_gallery_images"] = missing_images
        if missing_images:
            result["status"] = "failed"
    elif artifact_name == "gradcam_gallery_html":
        text = path.read_text(encoding="utf-8")
        if "Phase 9B Grad-CAM Spatial Diagnostics" not in text or "<img" not in text:
            result["status"] = "failed"
            result["html_error"] = "missing title or image tags"
        if "outputs\\phase9b-spatial-diagnostics" in text or "outputs/phase9b-spatial-diagnostics" in text:
            result["status"] = "failed"
            result["html_path_error"] = "gallery HTML contains repo-root-relative image paths"
    return result


def _validate_heatmap_artifacts(manifest_path: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    rows = _read_csv_rows(manifest_path)
    for row in rows:
        path = Path(row["heatmap_path"])
        expected_shape = (int(row["model_input_height"]), int(row["model_input_width"]))
        if not path.exists():
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": "missing heatmap", "path": str(path)})
            continue
        try:
            heatmap = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive artifact validation path
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": f"unreadable heatmap: {exc}", "path": str(path)})
            continue
        if not isinstance(heatmap, Tensor):
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": "heatmap is not a tensor", "path": str(path)})
            continue
        if tuple(heatmap.shape) != expected_shape:
            failures.append(
                {
                    "diagnostic_id": row["diagnostic_id"],
                    "error": "unexpected heatmap shape",
                    "path": str(path),
                    "expected_shape": list(expected_shape),
                    "actual_shape": list(heatmap.shape),
                }
            )
            continue
        if not torch.isfinite(heatmap).all():
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": "non-finite heatmap", "path": str(path)})
            continue
        minimum = float(heatmap.min().item())
        maximum = float(heatmap.max().item())
        if minimum < 0.0 or maximum > 1.000001 or maximum <= 0.0:
            failures.append(
                {
                    "diagnostic_id": row["diagnostic_id"],
                    "error": "heatmap outside normalized nonempty range",
                    "path": str(path),
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
    return {"status": "failed" if failures else "passed", "count": len(rows), "failures": failures}


def _validate_overlay_artifacts(manifest_path: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    rows = _read_csv_rows(manifest_path)
    for row in rows:
        path = Path(row["overlay_path"])
        expected_size = (int(row["model_input_width"]), int(row["model_input_height"]))
        if not path.exists():
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": "missing overlay", "path": str(path)})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                actual_size = image.size
                mode = image.mode
        except Exception as exc:  # pragma: no cover - defensive artifact validation path
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": f"unreadable overlay: {exc}", "path": str(path)})
            continue
        if actual_size != expected_size:
            failures.append(
                {
                    "diagnostic_id": row["diagnostic_id"],
                    "error": "unexpected overlay size",
                    "path": str(path),
                    "expected_size": list(expected_size),
                    "actual_size": list(actual_size),
                }
            )
        if mode not in {"RGB", "RGBA"}:
            failures.append({"diagnostic_id": row["diagnostic_id"], "error": f"unexpected overlay mode {mode}", "path": str(path)})
    return {"status": "failed" if failures else "passed", "count": len(rows), "failures": failures}


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
