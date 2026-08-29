import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from visionlab.diagnostics import compute_gradcam
from visionlab.evaluation.failures import FailurePrediction, write_csv_rows
from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import PHASE6C_RUN_ID
from visionlab.experiments.phase9b import (
    PHASE9B_FIXED_RUN_ORDER,
    _write_overlay,
    phase9b_diagnostic_contract,
    resolve_phase9b_target_layer,
    select_phase9b_diagnostic_rows,
    validate_phase9b_generated_artifact_schemas,
)
from visionlab.models import CustomCNN, CustomCNNConfig, build_phase6a_transfer_model


CLASSES = ("airplane", "automobile", "bird")


def pred(
    sample_id,
    true_label,
    predicted_label,
    confidence,
    *,
    run_id,
):
    return FailurePrediction(
        run_id=run_id,
        context_id="phase7_clean_cifar10_val",
        dataset_id="cifar10",
        dataset_version="phase1b-registered",
        split="val",
        condition_id="clean",
        sample_id=sample_id,
        source_id=sample_id,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        correct=true_label == predicted_label,
        true_index=CLASSES.index(true_label),
        predicted_index=CLASSES.index(predicted_label),
    )


def write_csv_with_fields(path, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: "x" for field in fields})


class TinyCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(0.25)
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.5],
                        [0.5, 1.0],
                        [1.0, 1.0],
                    ]
                )
            )

    def forward(self, inputs):
        x = torch.relu(self.features(inputs))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class Phase9BSpatialDiagnosticsTest(unittest.TestCase):
    def test_contract_defines_bounded_diagnostic_scope(self):
        contract = phase9b_diagnostic_contract()

        self.assertEqual(contract["phase"], "9B")
        self.assertIn("no new evaluation", contract["scope"])
        self.assertEqual(contract["fixed_run_order"], list(PHASE9B_FIXED_RUN_ORDER))
        self.assertEqual(
            contract["selection_rules"]["model_disagreement"]["row_expansion"],
            "one diagnostic row per selected sample per compared fixed model",
        )
        self.assertIn("Grad-CAM does not establish causal explanations", contract["interpretation_boundaries"])

    def test_selection_is_deterministic_and_expands_disagreements_by_fixed_model_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            high_rows = []
            predictions = {}
            for run_id in PHASE9B_FIXED_RUN_ORDER:
                run_predictions = (
                    pred("s1", "airplane", "automobile", 0.99, run_id=run_id),
                    pred("s2", "bird", "bird", 0.98, run_id=run_id),
                    pred("s3", "airplane", "airplane", 0.97, run_id=run_id),
                )
                predictions[run_id] = run_predictions
                high_rows.append({"rank": 1, **run_predictions[0].to_dict()})
            write_csv_rows(high_rows, artifact_dir / "high_confidence_errors.csv")
            write_csv_rows(
                [
                    {
                        "rank": 1,
                        "sample_id": "s1",
                        "run_predictions": json.dumps([]),
                    }
                ],
                artifact_dir / "model_disagreement_examples.csv",
            )

            rows = select_phase9b_diagnostic_rows(
                predictions_by_run=predictions,
                phase9a_dir=root,
            )

        disagreement = [row for row in rows if row["selection_category"] == "model_disagreement"]
        self.assertEqual([row["run_id"] for row in disagreement], list(PHASE9B_FIXED_RUN_ORDER))
        correct_controls = [row for row in rows if row["selection_category"] == "correct_control"]
        self.assertEqual(len(correct_controls), 6)
        self.assertEqual(correct_controls[0]["sample_id"], "s2")

    def test_selection_hard_stops_on_missing_phase9a_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                select_phase9b_diagnostic_rows(predictions_by_run={}, phase9a_dir=Path(tmp))

    def test_target_layer_resolution_for_custom_and_resnet_models(self):
        custom = CustomCNN(CustomCNNConfig(feature_channels=(4, 8)))
        self.assertIs(resolve_phase9b_target_layer(custom, PHASE4B_RUN_ID), custom.feature_blocks[-1])

        transfer = build_phase6a_transfer_model(load_pretrained=False)
        self.assertIs(resolve_phase9b_target_layer(transfer, PHASE6B2_RUN_ID), transfer.model.layer4)
        self.assertIs(resolve_phase9b_target_layer(transfer, PHASE6C_RUN_ID), transfer.model.layer4)

    def test_target_layer_resolution_fails_for_unknown_model(self):
        with self.assertRaises(ValueError):
            resolve_phase9b_target_layer(nn.Linear(2, 2), PHASE4B_RUN_ID)

    def test_gradcam_heatmap_is_finite_normalized_and_input_sized(self):
        model = TinyCamModel()
        inputs = torch.ones(1, 3, 8, 8)

        result = compute_gradcam(
            model,
            inputs,
            target_layer=model.features,
            target_class_index=2,
        )

        self.assertEqual(tuple(result.heatmap.shape), (8, 8))
        self.assertTrue(torch.isfinite(result.heatmap).all())
        self.assertGreaterEqual(float(result.heatmap.min()), 0.0)
        self.assertLessEqual(float(result.heatmap.max()), 1.0)
        self.assertEqual(result.target_class_index, 2)

    def test_gradcam_hard_stops_on_all_empty_heatmap(self):
        model = TinyCamModel()
        with torch.no_grad():
            model.classifier.weight.fill_(-1.0)

        with self.assertRaises(ValueError):
            compute_gradcam(
                model,
                torch.ones(1, 3, 8, 8),
                target_layer=model.features,
                target_class_index=0,
            )

    def test_generated_artifact_schema_validation_checks_complete_artifact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))

            result = validate_phase9b_generated_artifact_schemas(
                artifact_paths,
                phase9b_diagnostic_contract(),
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            sorted(result["artifacts"]),
            [
                "diagnostic_selection_manifest",
                "gradcam_gallery_html",
                "gradcam_gallery_manifest",
                "gradcam_manifest",
                "gradcam_schema_validation",
                "phase9b_contract",
                "phase9b_result",
                "spatial_diagnostics_report",
            ],
        )
        self.assertEqual(result["heatmap_artifacts"]["status"], "passed")
        self.assertEqual(result["overlay_artifacts"]["status"], "passed")

    def test_generated_artifact_schema_validation_fails_missing_required_manifest_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            gradcam_manifest = Path(artifact_paths["gradcam_manifest"])
            with gradcam_manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            fields = [field for field in rows[0] if field != "checkpoint_sha256"]
            with gradcam_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({field: rows[0][field] for field in fields})

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_malformed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            Path(artifact_paths["phase9b_contract"]).write_text(json.dumps({"phase": "9A"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_malformed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            Path(artifact_paths["phase9b_result"]).write_text(
                json.dumps({"run_dir": "x", "run_id": "wrong", "status": "x", "artifact_paths": {}}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_bad_heatmap_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            with Path(artifact_paths["gradcam_manifest"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            torch.save(torch.full((4, 4), float("nan")), rows[0]["heatmap_path"])

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_bad_overlay_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            with Path(artifact_paths["gradcam_manifest"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            Path(rows[0]["overlay_path"]).write_text("not a png", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_report_without_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            Path(artifact_paths["spatial_diagnostics_report"]).write_text("thin report", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def test_generated_artifact_schema_validation_fails_gallery_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            Path(artifact_paths["gradcam_gallery_html"]).write_text("<html></html>", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )
    def test_generated_artifact_schema_validation_fails_broken_gallery_manifest_image_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_complete_validation_fixture(Path(tmp))
            gallery_manifest = Path(artifact_paths["gradcam_gallery_manifest"])
            fields = phase9b_diagnostic_contract()["artifact_schema_requirements"]["gradcam_gallery_manifest"]
            with gallery_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                row = {field: "x" for field in fields}
                row["image_path"] = "missing.png"
                writer.writerow(row)

            with self.assertRaises(ValueError):
                validate_phase9b_generated_artifact_schemas(
                    artifact_paths,
                    phase9b_diagnostic_contract(),
                )

    def _write_complete_validation_fixture(self, root: Path) -> dict[str, str]:
        contract = phase9b_diagnostic_contract()
        schemas = contract["artifact_schema_requirements"]
        heatmap = root / "heatmap.pt"
        overlay = root / "overlay.png"
        torch.save(torch.ones(4, 4), heatmap)
        _write_overlay(Image.new("RGB", (4, 4)), torch.ones(4, 4), overlay)

        contract_path = root / "phase9b_contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        result_path = root / "phase9b_result.json"
        selection_manifest = root / "diagnostic_selection_manifest.json"
        selection_manifest.write_text(
            json.dumps({field: "x" for field in schemas["diagnostic_selection_manifest"]}),
            encoding="utf-8",
        )
        gradcam_manifest = root / "gradcam_manifest.csv"
        gradcam_fields = list(schemas["gradcam_manifest"])
        with gradcam_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=gradcam_fields)
            writer.writeheader()
            row = {field: "x" for field in gradcam_fields}
            row.update(
                {
                    "diagnostic_id": "d1",
                    "heatmap_path": str(heatmap),
                    "overlay_path": str(overlay),
                    "model_input_height": "4",
                    "model_input_width": "4",
                }
            )
            writer.writerow(row)
        gallery_manifest = root / "gradcam_gallery_manifest.csv"
        with gallery_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=schemas["gradcam_gallery_manifest"])
            writer.writeheader()
            row = {field: "x" for field in schemas["gradcam_gallery_manifest"]}
            row["image_path"] = overlay.name
            writer.writerow(row)
        schema_validation = root / "gradcam_schema_validation.json"
        schema_validation.write_text(
            json.dumps({"status": "passed", "artifacts": {}, "heatmap_artifacts": {}, "overlay_artifacts": {}}),
            encoding="utf-8",
        )
        report = root / "phase9b_spatial_diagnostics_report.md"
        report.write_text(
            "Phase 9B Spatial Diagnostics\nnot automatically closed or accepted\n"
            "Grad-CAM does not prove\nnot new evaluation metrics\npending_builder_review\n",
            encoding="utf-8",
        )
        gallery_html = root / "gradcam_gallery.html"
        gallery_html.write_text("<html><title>Phase 9B Grad-CAM Spatial Diagnostics</title><img src='overlay.png'></html>", encoding="utf-8")
        artifact_paths = {
            "phase9b_contract": str(contract_path),
            "phase9b_result": str(result_path),
            "diagnostic_selection_manifest": str(selection_manifest),
            "gradcam_manifest": str(gradcam_manifest),
            "gradcam_schema_validation": str(schema_validation),
            "spatial_diagnostics_report": str(report),
            "gradcam_gallery_manifest": str(gallery_manifest),
            "gradcam_gallery_html": str(gallery_html),
        }
        result_path.write_text(
            json.dumps(
                {
                    "run_dir": str(root),
                    "run_id": "phase9b-spatial-diagnostics",
                    "status": "completed_for_phase_check_review",
                    "artifact_paths": artifact_paths,
                }
            ),
            encoding="utf-8",
        )
        return artifact_paths

if __name__ == "__main__":
    unittest.main()
