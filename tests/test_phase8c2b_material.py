import tempfile
import unittest
from pathlib import Path

from visionlab.data.cifar10_1 import (
    CIFAR10_1_DATASET_ID,
    CIFAR10_1_EXPECTED_SAMPLE_COUNT,
    CIFAR10_1_SPLIT_NAME,
    CIFAR10_1_VERSION,
)
from visionlab.experiments.phase7 import phase7_references
from visionlab.experiments.phase8c import (
    PHASE8C1_SAMPLE_LABEL_DIGEST,
    PHASE8C2B_RUN_ID,
    build_phase8c2_checkpoint_manifest,
    build_phase8c2b_material_contract,
    build_phase8c2b_runtime_projection,
    load_phase7_historical_test_reference,
    phase8c2b_historical_reference_delta_rows,
    verify_phase8c2b_artifacts,
    verify_phase8c2b_material_contract,
    verify_phase8c2b_runtime_guard,
    verify_phase8c2b_sample_alignment,
)


class Phase8C2BMaterialContractTests(unittest.TestCase):
    def _dataset_report(self):
        return {
            "dataset_id": CIFAR10_1_DATASET_ID,
            "version": CIFAR10_1_VERSION,
            "split": CIFAR10_1_SPLIT_NAME,
            "sample_count": CIFAR10_1_EXPECTED_SAMPLE_COUNT,
            "sample_label_digest": PHASE8C1_SAMPLE_LABEL_DIGEST,
        }

    def test_material_contract_accepts_only_registered_cifar10_1_v6(self):
        references = phase7_references()
        contract = build_phase8c2b_material_contract(
            dataset_report=self._dataset_report(),
            checkpoint_manifest=build_phase8c2_checkpoint_manifest(references),
            historical_reference=load_phase7_historical_test_reference(),
            runtime_projection=build_phase8c2b_runtime_projection(),
        )
        self.assertEqual(contract["run_id"], PHASE8C2B_RUN_ID)
        self.assertEqual(contract["dataset"]["sample_count"], 2000)
        self.assertFalse(contract["official_cifar10_test_rerun_performed"])
        self.assertEqual(verify_phase8c2b_material_contract(contract)["status"], "passed")

    def test_material_contract_rejects_official_test_or_wrong_count(self):
        references = phase7_references()
        dataset_report = self._dataset_report()
        dataset_report["split"] = "test"
        contract = build_phase8c2b_material_contract(
            dataset_report=dataset_report,
            checkpoint_manifest=build_phase8c2_checkpoint_manifest(references),
            historical_reference=load_phase7_historical_test_reference(),
            runtime_projection=build_phase8c2b_runtime_projection(),
        )
        with self.assertRaises(ValueError):
            verify_phase8c2b_material_contract(contract)

    def test_runtime_guard_accepts_approved_projection(self):
        projection = build_phase8c2b_runtime_projection()
        self.assertLess(projection["estimated_minutes"], projection["runtime_guard_minutes"])
        self.assertEqual(verify_phase8c2b_runtime_guard(projection)["status"], "passed")

    def test_runtime_guard_rejects_excessive_projection(self):
        projection = build_phase8c2b_runtime_projection()
        projection["guard_status"] = "failed"
        with self.assertRaises(TimeoutError):
            verify_phase8c2b_runtime_guard(projection)

    def test_historical_reference_deltas_use_matching_run(self):
        historical = load_phase7_historical_test_reference()
        rows = []
        for reference_row in historical["rows"]:
            rows.append(
                {
                    "run_id": reference_row["run_id"],
                    "split": CIFAR10_1_SPLIT_NAME,
                    "total_examples": 2000,
                    "accuracy": reference_row["accuracy"] - 0.1,
                    "balanced_accuracy": reference_row["balanced_accuracy"] - 0.1,
                    "macro_f1": reference_row["macro_f1"] - 0.1,
                    "ece": reference_row["ece"] + 0.01,
                    "average_confidence": reference_row["average_confidence"] - 0.02,
                    "incorrect_average_confidence": reference_row["incorrect_average_confidence"] - 0.03,
                }
            )
        deltas = phase8c2b_historical_reference_delta_rows(rows, historical)
        self.assertEqual(len(deltas), 3)
        self.assertAlmostEqual(deltas[0]["accuracy_delta_from_phase7_test"], -0.1)
        self.assertFalse(deltas[0]["reference_was_rerun_in_phase8c2b"])

    def test_sample_alignment_requires_same_ids_labels_and_sources(self):
        source = {
            "sample_ids": ["a", "b"],
            "true_labels": ["airplane", "truck"],
            "source_ids": ["cifar10-1/v6/000000", "cifar10-1/v6/000001"],
        }
        split_results = {
            "phase4b-cifar10-custom-cnn-baseline-001": source,
            "phase6b2-cifar10-resnet18-frozen-feature-001": dict(source),
            "phase6c-cifar10-resnet18-layer4-finetune-001": dict(source),
        }
        with self.assertRaises(ValueError):
            verify_phase8c2b_sample_alignment(split_results, dataset_report=self._dataset_report())

    def test_artifact_validation_checks_rows_and_non_ood_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ["a.json", "b.csv"]:
                path = Path(tmp) / name
                path.write_text("ok", encoding="utf-8")
                paths.append(path)
            historical = load_phase7_historical_test_reference()
            metrics_rows = []
            for row in historical["rows"]:
                metrics_rows.append(
                    {
                        "run_id": row["run_id"],
                        "total_examples": 2000,
                        "metrics_are_ood_detection_results": False,
                    }
                )
            delta_rows = [{"run_id": row["run_id"]} for row in historical["rows"]]
            alignment_report = {"status": "passed"}
            report = verify_phase8c2b_artifacts(
                paths,
                metrics_rows=metrics_rows,
                delta_rows=delta_rows,
                alignment_report=alignment_report,
            )
            self.assertEqual(report["cross_source_metric_rows"], 3)
            self.assertFalse(report["official_cifar10_test_rerun_performed"])


if __name__ == "__main__":
    unittest.main()
