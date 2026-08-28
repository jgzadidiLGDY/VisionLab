import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from visionlab.data.cifar10 import CIFAR10_CLASSES
from visionlab.data.cifar10_1 import (
    CIFAR10_1_EXPECTED_SAMPLE_COUNT,
    CIFAR10_1_SPLIT_NAME,
    CIFAR10_1_VERSION,
    load_cifar10_1_v6,
)
from visionlab.experiments.phase7 import Phase7RunReference, phase7_references
from visionlab.experiments.phase8c import (
    PHASE8C1_OUTPUT_DIR,
    PHASE8C1_SAMPLE_LABEL_DIGEST,
    PHASE8C2_EXPECTED_DELTA_ROWS,
    PHASE8C2_EXPECTED_MODEL_ROWS,
    PHASE8C2A_OUTPUT_DIR,
    PHASE8C2B_RUN_ID,
    Phase8C2CustomPredictionView,
    Phase8C2TransferPredictionView,
    build_phase8c2_checkpoint_manifest,
    build_phase8c2_material_contract,
    build_phase8c2_tiny_smoke_report,
    load_phase7_historical_test_reference,
    phase8c2_expected_artifact_schema,
    run_phase8c2a_preflight,
    verify_historical_reference_rows,
    verify_phase8c2_dataset_identity,
    verify_phase8c2_material_contract,
    verify_phase8c2_output_isolation,
    verify_phase8c2_preprocessing_order,
    verify_phase8c2a_artifacts,
)


class TinyCifar101Dataset(Dataset):
    def __init__(self, count=3):
        self.samples = []
        for index in range(count):
            raw = torch.zeros(3, 32, 32)
            raw[index % 3] = 0.25 + 0.1 * index
            self.samples.append(
                {
                    "input": raw.clone(),
                    "raw_input": raw,
                    "label": index % 10,
                    "sample_id": f"cifar10-1-v6-{index:05d}",
                    "source_id": f"cifar10-1-v6-{index:05d}",
                    "split": CIFAR10_1_SPLIT_NAME,
                    "dataset_id": "cifar10-1",
                    "dataset_version": "v6",
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


class Phase8C2APreflightTest(unittest.TestCase):
    def test_registered_cifar10_1_identity_and_digest_are_exact(self):
        dataset = load_cifar10_1_v6("data")
        report = verify_phase8c2_dataset_identity(dataset)

        self.assertEqual(report["dataset_id"], "cifar10-1")
        self.assertEqual(report["version"], CIFAR10_1_VERSION)
        self.assertEqual(report["split"], CIFAR10_1_SPLIT_NAME)
        self.assertEqual(report["sample_count"], CIFAR10_1_EXPECTED_SAMPLE_COUNT)
        self.assertEqual(report["sample_label_digest"], PHASE8C1_SAMPLE_LABEL_DIGEST)
        self.assertEqual(tuple(report["class_names"]), CIFAR10_CLASSES)
        self.assertTrue(report["evaluation_only_enforced"])

    def test_historical_phase7_reference_uses_three_test_rows_without_rerun(self):
        reference = load_phase7_historical_test_reference()
        result = verify_historical_reference_rows(reference, phase7_references())

        self.assertEqual(result, {"status": "passed"})
        self.assertFalse(reference["official_test_rerun_performed"])
        self.assertEqual(len(reference["rows"]), 3)
        self.assertEqual([row["split"] for row in reference["rows"]], ["test", "test", "test"])

    def test_checkpoint_manifest_preserves_three_fixed_identities(self):
        manifest = build_phase8c2_checkpoint_manifest(phase7_references())

        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["checkpoint_count"], 3)
        self.assertEqual(
            [item["run_id"] for item in manifest["checkpoints"]],
            [
                "phase4b-cifar10-custom-cnn-baseline-001",
                "phase6b2-cifar10-resnet18-frozen-feature-001",
                "phase6c-cifar10-resnet18-layer4-finetune-001",
            ],
        )
        self.assertTrue(all(item["checkpoint_sha256"] for item in manifest["checkpoints"]))

    def test_preprocessing_order_preserves_raw_unit_tensor(self):
        report = verify_phase8c2_preprocessing_order(TinyCifar101Dataset(count=2))

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["model_specific_preprocessing_after_raw_unit_tensor"])
        self.assertTrue(all(row["raw_inputs_identical_before_preprocessing"] for row in report["checks"]))
        self.assertEqual(report["checks"][0]["custom_preprocessing_id"], "phase4-cifar10-normalization")
        self.assertEqual(report["checks"][0]["transfer_preprocessing_id"], "phase6a-resnet18-imagenet1k-v1-preprocessing")

    def test_prediction_views_preserve_metadata_and_different_preprocessing_shapes(self):
        dataset = TinyCifar101Dataset(count=1)
        custom = Phase8C2CustomPredictionView(dataset)[0]
        transfer = Phase8C2TransferPredictionView(dataset)[0]

        self.assertEqual(custom["sample_id"], transfer["sample_id"])
        self.assertEqual(custom["label"], transfer["label"])
        self.assertTrue(torch.equal(custom["raw_cross_source_input"], transfer["raw_cross_source_input"]))
        self.assertEqual(tuple(custom["input"].shape), (3, 32, 32))
        self.assertEqual(tuple(transfer["input"].shape), (3, 224, 224))

    def test_material_contract_records_future_shape_and_forbidden_flags(self):
        dataset_report = {
            "dataset_id": "cifar10-1",
            "version": "v6",
            "split": "cross_source_test",
            "sample_count": 2000,
            "sample_label_digest": PHASE8C1_SAMPLE_LABEL_DIGEST,
        }
        checkpoint_manifest = {"checkpoint_count": 3, "checkpoints": [], "status": "passed"}
        historical = {"rows": [{"run_id": item.run_id} for item in fake_references()]}
        contract = build_phase8c2_material_contract(
            dataset_report=dataset_report,
            checkpoint_manifest=checkpoint_manifest,
            historical_reference=historical,
        )

        self.assertEqual(verify_phase8c2_material_contract(contract), {"status": "passed"})
        self.assertEqual(contract["future_material_run_id"], PHASE8C2B_RUN_ID)
        self.assertEqual(contract["expected_future_model_metric_rows"], PHASE8C2_EXPECTED_MODEL_ROWS)
        self.assertEqual(contract["expected_future_cross_source_delta_rows"], PHASE8C2_EXPECTED_DELTA_ROWS)
        self.assertFalse(contract["material_cross_source_evaluation_performed"])
        self.assertFalse(contract["official_cifar10_test_rerun_performed"])
        self.assertFalse(contract["phase8c2b_started"])

    def test_material_contract_rejects_wrong_digest_or_started_flags(self):
        dataset_report = {
            "dataset_id": "cifar10-1",
            "version": "v6",
            "split": "cross_source_test",
            "sample_count": 2000,
            "sample_label_digest": "bad",
        }
        contract = build_phase8c2_material_contract(
            dataset_report=dataset_report,
            checkpoint_manifest={"checkpoint_count": 3, "checkpoints": [], "status": "passed"},
            historical_reference={"rows": []},
        )
        with self.assertRaisesRegex(ValueError, "digest"):
            verify_phase8c2_material_contract(contract)

        dataset_report["sample_label_digest"] = PHASE8C1_SAMPLE_LABEL_DIGEST
        contract = build_phase8c2_material_contract(
            dataset_report=dataset_report,
            checkpoint_manifest={"checkpoint_count": 3, "checkpoints": [], "status": "passed"},
            historical_reference={"rows": []},
        )
        contract["phase8c2b_started"] = True
        with self.assertRaisesRegex(ValueError, "phase8c2b_started"):
            verify_phase8c2_material_contract(contract)

    def test_output_isolation_rejects_prior_phase_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_phase8c2_output_isolation(Path(tmp) / "phase8c2a")
        self.assertEqual(report["status"], "passed")

        with self.assertRaisesRegex(ValueError, "must not overlap"):
            verify_phase8c2_output_isolation(PHASE8C1_OUTPUT_DIR)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            verify_phase8c2_output_isolation(PHASE8C2A_OUTPUT_DIR.parent / "phase8b2b-fixed-checkpoint-validation-robustness-sweep")

    def test_artifact_schema_and_tiny_smoke_are_non_material(self):
        schema = phase8c2_expected_artifact_schema()
        smoke = build_phase8c2_tiny_smoke_report(TinyCifar101Dataset(count=3))

        self.assertEqual(schema["expected_future_model_metric_rows"], 3)
        self.assertEqual(schema["expected_future_cross_source_delta_rows"], 3)
        self.assertIn("accuracy_delta_from_phase7_test", schema["delta_columns"])
        self.assertFalse(smoke["metrics_are_material_results"])
        self.assertFalse(smoke["metrics_are_conclusive"])
        self.assertFalse(smoke["model_checkpoint_evaluation_performed"])

    def test_artifact_validation_rejects_missing_or_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.json"
            empty = root / "empty.json"
            good.write_text("{}", encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            self.assertEqual(verify_phase8c2a_artifacts([good])["status"], "passed")
            with self.assertRaises(ValueError):
                verify_phase8c2a_artifacts([empty])
            with self.assertRaises(FileNotFoundError):
                verify_phase8c2a_artifacts([root / "missing.json"])

    def test_preflight_runner_writes_artifacts_without_material_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase8c2a_preflight(Path(tmp) / "phase8c2a", data_root="data")

            payload = result.to_dict()
            self.assertEqual(payload["status"], "completed_preflight_only")
            for path in payload["artifact_paths"].values():
                self.assertTrue(Path(path).exists())
            contract = Path(payload["artifact_paths"]["material_contract"]).read_text(encoding="utf-8")
            self.assertIn('"material_cross_source_evaluation_performed": false', contract)
            self.assertIn('"official_cifar10_test_rerun_performed": false', contract)


if __name__ == "__main__":
    unittest.main()
