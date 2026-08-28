import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from visionlab.experiments.phase7 import Phase7RunReference
from visionlab.experiments.phase8b import (
    PHASE8B1_OUTPUT_DIR,
    PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS,
    PHASE8B2_MATERIAL_SAMPLE_COUNT,
    PHASE8B2_MATERIAL_SPLIT,
    build_phase8b2_checkpoint_manifest,
    build_phase8b2_condition_manifest,
    build_phase8b2_material_contract,
    build_phase8b2_sample_alignment_preflight,
    phase8b2_expected_artifact_schema,
    phase8b2_expected_model_condition_rows,
    phase8b_conditions,
    run_phase8b2b_validation_sweep,
    validate_phase8b2_material_split,
    verify_phase8b2_material_contract,
    verify_phase8b2_output_isolation,
    verify_phase8b2a_artifacts,
    write_phase8b2b_report,
)


class TinyValidationDataset(Dataset):
    def __init__(self, count=3):
        self.samples = []
        for index in range(count):
            self.samples.append(
                {
                    "input": torch.zeros(3, 32, 32),
                    "raw_input": torch.zeros(3, 32, 32),
                    "label": index % 10,
                    "sample_id": f"tiny-val-{index:05d}",
                    "split": "val",
                    "source_id": f"tiny-source-{index:05d}",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def fake_references():
    return (
        Phase7RunReference(
            run_id="phase4b-cifar10-custom-cnn-baseline-001",
            display_name="Custom",
            model_family="custom",
            checkpoint_path=Path("missing-custom.pt"),
        ),
        Phase7RunReference(
            run_id="phase6b2-cifar10-resnet18-frozen-feature-001",
            display_name="Frozen",
            model_family="transfer",
            checkpoint_path=Path("missing-frozen.pt"),
        ),
        Phase7RunReference(
            run_id="phase6c-cifar10-resnet18-layer4-finetune-001",
            display_name="Finetune",
            model_family="transfer",
            checkpoint_path=Path("missing-finetune.pt"),
        ),
    )


class Phase8B2APreflightTest(unittest.TestCase):
    def test_validation_only_gate_accepts_val_and_rejects_official_test(self):
        validate_phase8b2_material_split(PHASE8B2_MATERIAL_SPLIT)

        with self.assertRaisesRegex(ValueError, "validation-only"):
            validate_phase8b2_material_split("test")

    def test_expected_row_count_is_three_models_by_twenty_one_conditions(self):
        rows = phase8b2_expected_model_condition_rows(fake_references(), phase8b_conditions())

        self.assertEqual(rows, PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS)
        self.assertEqual(rows, 63)

    def test_condition_manifest_preserves_frozen_phase8a_identity(self):
        manifest = build_phase8b2_condition_manifest(phase8b_conditions())

        self.assertEqual(manifest["condition_count"], 21)
        self.assertEqual(manifest["phase8a_registry"]["version"], "1.0")
        noise_s5 = next(
            condition for condition in manifest["conditions"]
            if condition["condition_id"] == "phase8a-gaussian-noise__S5"
        )
        self.assertEqual(noise_s5["severity_parameters"], {"std": 0.15})

    def test_checkpoint_manifest_preserves_three_fixed_reference_ids(self):
        manifest = build_phase8b2_checkpoint_manifest(fake_references())

        self.assertEqual(manifest["checkpoint_count"], 3)
        self.assertEqual(
            [item["run_id"] for item in manifest["checkpoints"]],
            [
                "phase4b-cifar10-custom-cnn-baseline-001",
                "phase6b2-cifar10-resnet18-frozen-feature-001",
                "phase6c-cifar10-resnet18-layer4-finetune-001",
            ],
        )

    def test_material_contract_is_validation_only_and_rejects_mutations(self):
        contract = build_phase8b2_material_contract(
            references=fake_references(),
            conditions=phase8b_conditions(),
            dataset_contract={"split_counts": {"val": PHASE8B2_MATERIAL_SAMPLE_COUNT}},
            preflight_report={"status": "passed"},
            target_split="val",
        )

        self.assertEqual(verify_phase8b2_material_contract(contract), {"status": "passed"})
        self.assertEqual(contract["target_sample_count"], 5000)
        self.assertEqual(contract["expected_model_condition_rows"], 63)
        self.assertEqual(contract["official_test_evaluation"], "forbidden and rejected")

        bad_split = dict(contract)
        bad_split["target_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            verify_phase8b2_material_contract(bad_split)

        bad_count = dict(contract)
        bad_count["target_sample_count"] = 10
        with self.assertRaisesRegex(ValueError, "5,000"):
            verify_phase8b2_material_contract(bad_count)

        bad_conditions = dict(contract)
        mutated_conditions = [dict(condition) for condition in contract["conditions"]]
        mutated_conditions[1] = dict(mutated_conditions[1])
        mutated_conditions[1]["severity_parameters"] = {"std": 0.99}
        bad_conditions["conditions"] = mutated_conditions
        with self.assertRaisesRegex(ValueError, "exactly match"):
            verify_phase8b2_material_contract(bad_conditions)

    def test_output_isolation_rejects_phase8b1_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "phase8b2a"
            report = verify_phase8b2_output_isolation(good)

            self.assertEqual(report["status"], "passed")

        with self.assertRaisesRegex(ValueError, "must not overlap"):
            verify_phase8b2_output_isolation(PHASE8B1_OUTPUT_DIR)

    def test_sample_alignment_preflight_records_validation_identity_digest(self):
        dataset = TinyValidationDataset(count=2)

        report = build_phase8b2_sample_alignment_preflight(
            dataset,
            conditions=phase8b_conditions(),
            references=fake_references(),
            split="val",
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["condition_count"], 21)
        self.assertEqual(report["checkpoint_count"], 3)
        self.assertEqual(len(report["sample_label_digest"]), 64)

    def test_future_material_runner_rejects_test_split_before_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "validation-only"):
                run_phase8b2b_validation_sweep(Path(tmp), target_split="test")

    def test_expected_artifact_schema_includes_metrics_and_delta_outputs(self):
        schema = phase8b2_expected_artifact_schema()

        self.assertIn("artifacts/phase8b2_validation_metrics.csv", schema["required_artifacts"])
        self.assertIn("artifacts/phase8b2_clean_delta_metrics.csv", schema["required_artifacts"])
        self.assertIn("accuracy", schema["metric_columns"])
        self.assertIn("accuracy_delta_from_clean", schema["delta_columns"])

    def test_phase8b2b_report_includes_condition_metrics_and_clean_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            metrics_rows = [
                {
                    "run_id": "run-a",
                    "condition_id": "clean",
                    "total_examples": 5000,
                    "loss": 1.0,
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.7,
                    "macro_f1": 0.6,
                    "ece": 0.1,
                    "average_confidence": 0.9,
                    "incorrect_average_confidence": 0.5,
                }
            ]
            delta_rows = [
                {
                    "run_id": "run-a",
                    "condition_id": "clean",
                    "accuracy_delta_from_clean": 0.0,
                    "balanced_accuracy_delta_from_clean": 0.0,
                    "macro_f1_delta_from_clean": 0.0,
                    "ece_delta_from_clean": 0.0,
                    "average_confidence_delta_from_clean": 0.0,
                    "incorrect_average_confidence_delta_from_clean": 0.0,
                }
            ]

            write_phase8b2b_report(path, metrics_rows=metrics_rows, delta_rows=delta_rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn("## Condition Metrics", text)
        self.assertIn("## Clean Deltas", text)
        self.assertIn("not official test robustness", text)
    def test_artifact_validation_rejects_empty_or_missing_preflight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.json"
            empty = root / "empty.json"
            good.write_text("{}", encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            self.assertEqual(verify_phase8b2a_artifacts([good])["status"], "passed")
            with self.assertRaises(ValueError):
                verify_phase8b2a_artifacts([empty])
            with self.assertRaises(FileNotFoundError):
                verify_phase8b2a_artifacts([root / "missing.json"])


if __name__ == "__main__":
    unittest.main()
