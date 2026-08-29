import csv
import json
import tempfile
import unittest
from pathlib import Path

from visionlab.evaluation.failures import (
    FailurePrediction,
    align_predictions_by_sample,
    confusion_pair_summary,
    per_class_failure_summary,
    select_confusion_pair_examples,
    select_high_confidence_errors,
    select_model_disagreements,
    select_per_class_failure_examples,
)
from visionlab.evaluation.galleries import write_gallery_manifest, write_placeholder_gallery_images
from visionlab.experiments.phase9a import (
    _combined_checkpoint_identity_fields,
    _selection_row,
    phase9a_selection_contract,
    validate_phase9a_generated_artifact_schemas,
)


CLASSES = ("airplane", "automobile", "bird")


def pred(
    sample_id,
    true_label,
    predicted_label,
    confidence,
    *,
    run_id="run-a",
    context_id="ctx",
):
    return FailurePrediction(
        run_id=run_id,
        context_id=context_id,
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


class Phase9AFailureSelectionTest(unittest.TestCase):
    def test_contract_makes_selection_rules_explicit(self):
        contract = phase9a_selection_contract()

        self.assertEqual(contract["phase"], "9A")
        self.assertEqual(contract["required_context"]["split"], "val")
        self.assertTrue(contract["optional_context_policy"]["no_regeneration"])
        self.assertEqual(
            contract["selection_rules"]["high_confidence_errors"]["ranking"],
            ["confidence descending", "sample_id ascending"],
        )
        self.assertIn("confidence", contract["required_manifest_fields"])
        self.assertIn("model_disagreement_examples", contract["artifact_schema_requirements"])

    def test_high_confidence_errors_rank_by_confidence_then_sample_id(self):
        predictions = (
            pred("s3", "airplane", "bird", 0.91),
            pred("s1", "airplane", "automobile", 0.95),
            pred("s0", "airplane", "airplane", 0.99),
            pred("s2", "airplane", "bird", 0.95),
        )

        selected = select_high_confidence_errors(predictions, top_n=3)

        self.assertEqual([item.sample_id for item in selected], ["s1", "s2", "s3"])
        self.assertTrue(all(not item.correct for item in selected))

    def test_per_class_failure_summary_preserves_all_classes_and_counts(self):
        predictions = (
            pred("s1", "airplane", "automobile", 0.8),
            pred("s2", "airplane", "airplane", 0.7),
            pred("s3", "bird", "airplane", 0.9),
        )

        rows = per_class_failure_summary(predictions, class_names=CLASSES)

        self.assertEqual([row["class_name"] for row in rows], list(CLASSES))
        airplane = rows[0]
        self.assertEqual(airplane["support"], 2)
        self.assertEqual(airplane["false_negative_count"], 1)
        self.assertEqual(airplane["false_positive_count"], 1)
        automobile = rows[1]
        self.assertEqual(automobile["support"], 0)
        self.assertEqual(automobile["false_positive_count"], 1)

    def test_per_class_failure_examples_are_deterministic_by_category(self):
        predictions = (
            pred("s2", "airplane", "bird", 0.9),
            pred("s1", "airplane", "automobile", 0.9),
            pred("s3", "bird", "airplane", 0.8),
        )

        rows = select_per_class_failure_examples(
            predictions,
            class_names=CLASSES,
            top_n_per_category=2,
        )

        airplane_rows = [row for row in rows if row["class_name"] == "airplane"]
        self.assertEqual(airplane_rows[0]["failure_category"], "false_negative")
        self.assertEqual(airplane_rows[0]["sample_id"], "s1")
        self.assertEqual(airplane_rows[1]["sample_id"], "s2")
        self.assertEqual(airplane_rows[2]["failure_category"], "false_positive")
        self.assertEqual(airplane_rows[2]["sample_id"], "s3")

    def test_confusion_pair_summary_and_examples_use_declared_ranking(self):
        predictions = (
            pred("s4", "bird", "airplane", 0.99),
            pred("s1", "airplane", "bird", 0.8),
            pred("s2", "airplane", "bird", 0.9),
            pred("s3", "automobile", "bird", 0.95),
        )

        summary = confusion_pair_summary(predictions)
        examples = select_confusion_pair_examples(predictions, top_pairs=2, examples_per_pair=2)

        self.assertEqual(summary[0]["true_label"], "airplane")
        self.assertEqual(summary[0]["predicted_label"], "bird")
        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual([row["sample_id"] for row in examples[:2]], ["s2", "s1"])
        self.assertEqual(examples[2]["true_label"], "automobile")

    def test_model_disagreement_requires_hard_sample_and_label_alignment(self):
        run_a = (pred("s1", "airplane", "airplane", 0.8, run_id="a"),)
        run_b = (pred("s2", "airplane", "bird", 0.7, run_id="b"),)

        with self.assertRaises(ValueError):
            align_predictions_by_sample({"a": run_a, "b": run_b})

        run_c = (pred("s1", "bird", "bird", 0.7, run_id="c"),)
        with self.assertRaises(ValueError):
            align_predictions_by_sample({"a": run_a, "c": run_c})

    def test_model_disagreement_ranking_is_deterministic(self):
        run_a = (
            pred("s1", "airplane", "airplane", 0.90, run_id="a"),
            pred("s2", "airplane", "airplane", 0.60, run_id="a"),
            pred("s3", "bird", "bird", 0.55, run_id="a"),
        )
        run_b = (
            pred("s1", "airplane", "automobile", 0.20, run_id="b"),
            pred("s2", "airplane", "bird", 0.95, run_id="b"),
            pred("s3", "bird", "bird", 0.50, run_id="b"),
        )
        run_c = (
            pred("s1", "airplane", "bird", 0.40, run_id="c"),
            pred("s2", "airplane", "bird", 0.65, run_id="c"),
            pred("s3", "bird", "airplane", 0.99, run_id="c"),
        )

        rows = select_model_disagreements({"a": run_a, "b": run_b, "c": run_c}, top_n=3)

        self.assertEqual(rows[0]["sample_id"], "s1")
        self.assertEqual(rows[0]["distinct_prediction_count"], 3)
        self.assertEqual(rows[1]["sample_id"], "s2")
        self.assertEqual(rows[2]["sample_id"], "s3")

    def test_disagreement_rows_have_explicit_checkpoint_identity_fields(self):
        checkpoint_fields = _combined_checkpoint_identity_fields(
            {
                "run-a": {
                    "checkpoint_tag": "best",
                    "checkpoint_path": "outputs/run-a/best.pt",
                    "checkpoint_sha256": "aaa",
                },
                "run-b": {
                    "checkpoint_tag": "best",
                    "checkpoint_path": "outputs/run-b/best.pt",
                    "checkpoint_sha256": "bbb",
                },
            }
        )

        self.assertEqual(checkpoint_fields["checkpoint_tag"], "run-a=best|run-b=best")
        self.assertEqual(
            checkpoint_fields["checkpoint_sha256"],
            "run-a=aaa|run-b=bbb",
        )
        self.assertEqual(checkpoint_fields["run_a_checkpoint_path"], "outputs/run-a/best.pt")
        self.assertEqual(checkpoint_fields["run_b_checkpoint_path"], "outputs/run-b/best.pt")

    def test_selected_example_rows_include_required_manifest_fields(self):
        contract = phase9a_selection_contract()
        row = _selection_row(
            pred("s1", "airplane", "bird", 0.9),
            "high_confidence_errors",
            rank=1,
            checkpoint_fields={
                "checkpoint_tag": "best",
                "checkpoint_path": "outputs/run/checkpoints/best.pt",
                "checkpoint_sha256": "abc123",
            },
        )

        for field in contract["required_manifest_fields"]:
            self.assertIn(field, row)

    def test_generated_artifact_schema_validation_checks_every_phase9a_output(self):
        contract = phase9a_selection_contract()
        schemas = contract["artifact_schema_requirements"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_paths = {}
            csv_artifacts = [
                "high_confidence_errors",
                "per_class_failure_summary",
                "per_class_failure_examples",
                "confusion_pair_examples",
                "model_disagreement_examples",
                "high_confidence_error_gallery_manifest",
            ]
            for artifact_name in csv_artifacts:
                path = tmp_path / f"{artifact_name}.csv"
                write_csv_with_fields(path, schemas[artifact_name])
                artifact_paths[artifact_name] = str(path)

            manifest_path = tmp_path / "failure_selection_manifest.json"
            manifest = {field: "x" for field in schemas["failure_selection_manifest"]}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            artifact_paths["failure_selection_manifest"] = str(manifest_path)

            result = validate_phase9a_generated_artifact_schemas(artifact_paths, contract)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            sorted(result["artifacts"]),
            sorted(
                [
                    "high_confidence_errors",
                    "per_class_failure_summary",
                    "per_class_failure_examples",
                    "confusion_pair_examples",
                    "model_disagreement_examples",
                    "failure_selection_manifest",
                    "high_confidence_error_gallery_manifest",
                ]
            ),
        )

    def test_generated_artifact_schema_validation_fails_for_missing_disagreement_identity(self):
        contract = phase9a_selection_contract()
        schemas = contract["artifact_schema_requirements"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_paths = {}
            for artifact_name, fields in schemas.items():
                if artifact_name == "failure_selection_manifest":
                    path = tmp_path / f"{artifact_name}.json"
                    path.write_text(json.dumps({field: "x" for field in fields}), encoding="utf-8")
                else:
                    path = tmp_path / f"{artifact_name}.csv"
                    output_fields = list(fields)
                    if artifact_name == "model_disagreement_examples":
                        output_fields.remove("checkpoint_sha256")
                    write_csv_with_fields(path, output_fields)
                artifact_paths[artifact_name] = str(path)

            with self.assertRaises(ValueError):
                validate_phase9a_generated_artifact_schemas(artifact_paths, contract)

    def test_gallery_manifest_is_generated_from_machine_selected_rows(self):
        rows = [
            {
                "sample_id": "s2",
                "true_label": "airplane",
                "predicted_label": "bird",
                "confidence": 0.9,
                "run_id": "a",
            },
            {
                "sample_id": "s1",
                "true_label": "bird",
                "predicted_label": "airplane",
                "confidence": 0.8,
                "run_id": "a",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_rows = write_placeholder_gallery_images(rows, tmp_path / "images")
            manifest_path = write_gallery_manifest(image_rows, tmp_path / "gallery_manifest.csv")
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))

        self.assertEqual([row["sample_id"] for row in manifest_rows], ["s2", "s1"])
        self.assertTrue(manifest_rows[0]["image_path"].endswith("s2.png"))

    def test_disagreement_run_predictions_field_is_json(self):
        run_a = (pred("s1", "airplane", "airplane", 0.9, run_id="a"),)
        run_b = (pred("s1", "airplane", "bird", 0.8, run_id="b"),)

        rows = select_model_disagreements({"a": run_a, "b": run_b}, top_n=1)
        payload = json.loads(rows[0]["run_predictions"])

        self.assertEqual([item["run_id"] for item in payload], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
