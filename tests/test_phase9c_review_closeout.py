import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from visionlab.experiments.phase4b import PHASE4B_RUN_ID
from visionlab.experiments.phase9c import (
    PHASE9C_RUN_ID,
    build_label_data_quality_inventory,
    build_review_tag_rows,
    heatmap_review_properties,
    load_phase9c_inputs,
    phase9c_review_contract,
    validate_phase9c_generated_artifact_schemas,
)


class Phase9CReviewCloseoutTest(unittest.TestCase):
    def test_contract_classifies_review_properties_and_scope(self):
        contract = phase9c_review_contract()

        self.assertEqual(contract["phase"], "9C")
        self.assertIn("review and synthesis only", contract["scope"])
        self.assertIn("no new evaluation", contract["scope"])
        classes = contract["review_property_classes"]
        self.assertIn("machine_derived", classes)
        self.assertIn("builder_observation", classes)
        self.assertIn("hypothesis", classes)
        self.assertIn("unsupported_causal_claim", classes)
        self.assertNotIn("possible_label_noise", classes["machine_derived"]["allowed_tags"])
        self.assertIn("Grad-CAM does not prove", " ".join(contract["interpretation_boundaries"]))

    def test_heatmap_properties_are_deterministic_and_normalized(self):
        heatmap = torch.zeros(4, 4)
        heatmap[0, 0] = 1.0

        props = heatmap_review_properties(heatmap)

        self.assertEqual(props["height"], 4)
        self.assertEqual(props["width"], 4)
        self.assertEqual(props["maximum"], 1.0)
        self.assertEqual(props["top_80pct_region_fraction"], 1 / 16)
        self.assertGreaterEqual(props["normalized_entropy"], 0.0)
        self.assertLessEqual(props["normalized_entropy"], 1.0)

    def test_heatmap_properties_reject_bad_heatmaps(self):
        with self.assertRaises(ValueError):
            heatmap_review_properties(torch.zeros(4, 4))
        with self.assertRaises(ValueError):
            heatmap_review_properties(torch.full((4, 4), float("nan")))
        with self.assertRaises(ValueError):
            heatmap_review_properties(torch.ones(1, 1, 1))

    def test_review_rows_keep_semantic_fields_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heatmap = root / "heatmap.pt"
            torch.save(torch.ones(4, 4), heatmap)
            rows = [
                {
                    "diagnostic_id": "d1",
                    "selection_category": "high_confidence_error",
                    "selection_rank": "1",
                    "run_id": PHASE4B_RUN_ID,
                    "sample_id": "s1",
                    "true_label": "cat",
                    "source_predicted_label": "dog",
                    "diagnostic_confidence": "0.9",
                    "heatmap_path": str(heatmap),
                    "overlay_path": str(root / "overlay.png"),
                    "target_layer": "layer",
                }
            ]

            review = build_review_tag_rows(rows, {("high_confidence_error", PHASE4B_RUN_ID, "s1")})

        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["builder_visual_observation"], "pending_builder_review")
        self.assertEqual(review[0]["cautious_hypothesis"], "pending_builder_review")
        self.assertEqual(review[0]["unsupported_causal_claim"], "none_recorded")
        tags = json.loads(review[0]["machine_derived_properties"])
        self.assertIn("prediction_error", tags)
        self.assertIn("high_confidence_error_member", tags)

    def test_label_data_quality_inventory_is_pending_not_machine_established(self):
        rows = [
            {"sample_id": "s2"},
            {"sample_id": "s1"},
            {"sample_id": "s1"},
        ]

        inventory = build_label_data_quality_inventory(rows)

        self.assertEqual([row["sample_id"] for row in inventory], ["s1", "s2"])
        self.assertTrue(all(row["issue_type"] == "pending_builder_review" for row in inventory))
        self.assertTrue(all(row["evidence_status"] == "not_machine_established" for row in inventory))

    def test_load_inputs_hard_stops_on_missing_phase9a_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_phase9c_inputs(phase9a_dir=Path(tmp), phase9b_dir=Path(tmp))

    def test_load_inputs_rejects_failed_phase9b_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase9a, phase9b = self._write_input_fixture(root)
            validation = phase9b / "artifacts" / "gradcam_schema_validation.json"
            validation.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_phase9c_inputs(phase9a_dir=phase9a, phase9b_dir=phase9b)

    def test_load_inputs_rejects_unaligned_phase9b_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase9a, phase9b = self._write_input_fixture(root)
            manifest = phase9b / "artifacts" / "gradcam_manifest.csv"
            rows = self._read_csv(manifest)
            rows[0]["sample_id"] = "missing"
            self._write_csv(manifest, rows)

            with self.assertRaises(ValueError):
                load_phase9c_inputs(phase9a_dir=phase9a, phase9b_dir=phase9b)

    def test_generated_artifact_schema_validation_checks_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_phase9c_validation_fixture(Path(tmp))

            result = validate_phase9c_generated_artifact_schemas(artifact_paths, phase9c_review_contract())

        self.assertEqual(result["status"], "passed")
        self.assertIn("review_tag_manifest", result["artifacts"])
        self.assertIn("label_data_quality_inventory", result["artifacts"])

    def test_generated_artifact_schema_validation_rejects_machine_filled_human_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_phase9c_validation_fixture(Path(tmp))
            review_path = Path(artifact_paths["review_tag_manifest"])
            rows = self._read_csv(review_path)
            rows[0]["builder_visual_observation"] = "looks like background"
            self._write_csv(review_path, rows)

            with self.assertRaises(ValueError):
                validate_phase9c_generated_artifact_schemas(artifact_paths, phase9c_review_contract())

    def test_generated_artifact_schema_validation_rejects_causal_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_paths = self._write_phase9c_validation_fixture(Path(tmp))
            review_path = Path(artifact_paths["review_tag_manifest"])
            rows = self._read_csv(review_path)
            rows[0]["unsupported_causal_claim"] = "Grad-CAM proves model reasoning"
            self._write_csv(review_path, rows)

            with self.assertRaises(ValueError):
                validate_phase9c_generated_artifact_schemas(artifact_paths, phase9c_review_contract())

    def _write_input_fixture(self, root: Path):
        phase9a = root / "phase9a"
        phase9b = root / "phase9b"
        (phase9a / "artifacts").mkdir(parents=True)
        (phase9b / "artifacts").mkdir(parents=True)
        high_conf = [
            {
                "rank": "1",
                "run_id": PHASE4B_RUN_ID,
                "sample_id": "s1",
            }
        ]
        self._write_csv(phase9a / "artifacts" / "high_confidence_errors.csv", high_conf)
        self._write_csv(phase9a / "artifacts" / "model_disagreement_examples.csv", [{"rank": "1", "sample_id": "s2"}])
        for name in ["per_class_failure_summary", "per_class_failure_examples", "confusion_pair_examples"]:
            self._write_csv(phase9a / "artifacts" / f"{name}.csv", [{"x": "y"}])
        (phase9a / "artifacts" / "failure_selection_manifest.json").write_text(json.dumps({"phase": "9A"}), encoding="utf-8")

        torch.save(torch.ones(4, 4), phase9b / "artifacts" / "heatmap.pt")
        gradcam_row = {
            "diagnostic_id": "d1",
            "selection_category": "high_confidence_error",
            "selection_rank": "1",
            "run_id": PHASE4B_RUN_ID,
            "sample_id": "s1",
            "true_label": "cat",
            "source_predicted_label": "dog",
            "diagnostic_confidence": "0.9",
            "heatmap_path": str(phase9b / "artifacts" / "heatmap.pt"),
            "overlay_path": str(phase9b / "artifacts" / "overlay.png"),
            "target_layer": "layer",
        }
        self._write_csv(phase9b / "artifacts" / "gradcam_manifest.csv", [gradcam_row])
        (phase9b / "artifacts" / "diagnostic_selection_manifest.json").write_text(json.dumps({"phase": "9B"}), encoding="utf-8")
        (phase9b / "artifacts" / "gradcam_schema_validation.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
        return phase9a, phase9b

    def _write_phase9c_validation_fixture(self, root: Path) -> dict[str, str]:
        contract = phase9c_review_contract()
        schemas = contract["artifact_schema_requirements"]
        paths = {
            "phase9c_contract": root / "phase9c_contract.json",
            "phase9c_result": root / "phase9c_result.json",
            "review_tag_manifest": root / "review_tag_manifest.csv",
            "label_data_quality_inventory": root / "label_data_quality_inventory.csv",
            "failure_hypothesis_report": root / "failure_hypothesis_report.json",
            "artifact_schema_validation": root / "phase9c_artifact_schema_validation.json",
            "phase9c_report": root / "phase9c_review_synthesis_report.md",
        }
        paths["phase9c_contract"].write_text(json.dumps(contract), encoding="utf-8")
        paths["phase9c_result"].write_text(
            json.dumps({"run_dir": str(root), "run_id": PHASE9C_RUN_ID, "status": "x", "artifact_paths": {}}),
            encoding="utf-8",
        )
        review_row = {field: "x" for field in schemas["review_tag_manifest"]}
        review_row.update(
            {
                "review_id": "r1",
                "builder_visual_observation": "pending_builder_review",
                "cautious_hypothesis": "pending_builder_review",
                "unsupported_causal_claim": "none_recorded",
            }
        )
        self._write_csv(paths["review_tag_manifest"], [review_row])
        issue_row = {field: "x" for field in schemas["label_data_quality_inventory"]}
        self._write_csv(paths["label_data_quality_inventory"], [issue_row])
        paths["failure_hypothesis_report"].write_text(
            json.dumps({field: "x" for field in schemas["failure_hypothesis_report"]}),
            encoding="utf-8",
        )
        paths["artifact_schema_validation"].write_text(json.dumps({"status": "passed", "artifacts": {}}), encoding="utf-8")
        paths["phase9c_report"].write_text(
            "Phase 9C is a review/synthesis phase only\n"
            "pending_builder_review\n"
            "Unsupported causal claim: prohibited\n"
            "Grad-CAM does not prove model reasoning\n",
            encoding="utf-8",
        )
        return {key: str(value) for key, value in paths.items()}

    def _read_csv(self, path: Path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_csv(self, path: Path, rows):
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
