import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from visionlab.experiments.phase8b import (
    PHASE8B1_SEED,
    Phase8BCustomPredictionView,
    Phase8BTransferPredictionView,
    ValidationSubset,
    clean_delta_rows,
    condition_raw_tensor,
    phase8b_conditions,
    verify_phase8b1_artifacts,
    verify_phase8b_sample_alignment,
    verify_preprocessing_contracts,
    verify_raw_input_equivalence,
)


class TinyPredictionDataset(Dataset):
    def __init__(self, count=3):
        self.samples = []
        for index in range(count):
            raw = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(3, 32, 32)
            raw = torch.roll(raw, shifts=index, dims=2)
            self.samples.append(
                {
                    "input": raw * 2.0 - 1.0,
                    "raw_input": raw,
                    "label": index % 10,
                    "sample_id": f"tiny-val-{index:05d}",
                    "split": "val",
                    "source_id": f"tiny-source-{index:05d}",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.samples[index].items()
        }


class Phase8BPlumbingTest(unittest.TestCase):
    def test_condition_contract_contains_clean_and_frozen_phase8a_conditions(self):
        conditions = phase8b_conditions()

        self.assertEqual(len(conditions), 21)
        self.assertEqual(conditions[0].condition_id, "clean")
        self.assertTrue(conditions[0].is_clean)
        condition_ids = [condition.condition_id for condition in conditions]
        self.assertIn("phase8a-gaussian-noise__S1", condition_ids)
        self.assertIn("phase8a-gaussian-blur__S5", condition_ids)
        self.assertIn("phase8a-brightness-shift__S3", condition_ids)
        self.assertIn("phase8a-contrast-reduction__S4", condition_ids)
        noise_s1 = next(item for item in conditions if item.condition_id == "phase8a-gaussian-noise__S1")
        self.assertEqual(noise_s1.profile_version, "1.0")
        self.assertEqual(noise_s1.severity_parameters, {"std": 0.03})

    def test_validation_subset_is_fixed_prefix_and_bounded(self):
        dataset = TinyPredictionDataset(count=3)
        subset = ValidationSubset(dataset, sample_count=2)

        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0]["sample_id"], "tiny-val-00000")
        self.assertEqual(subset[1]["sample_id"], "tiny-val-00001")
        with self.assertRaisesRegex(ValueError, "positive"):
            ValidationSubset(dataset, sample_count=0)
        with self.assertRaisesRegex(ValueError, "exceed"):
            ValidationSubset(dataset, sample_count=4)

    def test_raw_condition_input_is_identical_before_model_specific_preprocessing(self):
        dataset = TinyPredictionDataset(count=2)
        conditions = phase8b_conditions()

        report = verify_raw_input_equivalence(dataset, conditions, seed=PHASE8B1_SEED)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["condition_count"], 21)
        self.assertEqual(report["sample_count"], 2)

    def test_custom_and_transfer_preprocessing_share_raw_condition_input_then_diverge(self):
        dataset = TinyPredictionDataset(count=1)
        condition = next(item for item in phase8b_conditions() if item.condition_id == "phase8a-gaussian-blur__S3")
        raw = condition_raw_tensor(dataset[0], condition, seed=PHASE8B1_SEED)

        custom_sample = Phase8BCustomPredictionView(dataset, condition, seed=PHASE8B1_SEED)[0]
        transfer_sample = Phase8BTransferPredictionView(dataset, condition, seed=PHASE8B1_SEED)[0]

        self.assertTrue(torch.equal(custom_sample["raw_condition_input"], raw))
        self.assertTrue(torch.equal(transfer_sample["raw_condition_input"], raw))
        self.assertEqual(tuple(custom_sample["input"].shape), (3, 32, 32))
        self.assertEqual(tuple(transfer_sample["input"].shape), (3, 224, 224))
        self.assertEqual(custom_sample["preprocessing_id"], "phase4-cifar10-normalization")
        self.assertEqual(
            transfer_sample["preprocessing_id"],
            "phase6a-resnet18-imagenet1k-v1-preprocessing",
        )

    def test_preprocessing_verification_preserves_identity_and_shapes(self):
        dataset = TinyPredictionDataset(count=1)
        conditions = (phase8b_conditions()[0], phase8b_conditions()[1])

        report = verify_preprocessing_contracts(dataset, conditions, seed=PHASE8B1_SEED)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["condition_count"], 2)
        self.assertTrue(all(row["raw_condition_inputs_identical"] for row in report["checks"]))

    def test_clean_delta_rows_require_clean_baseline_and_compute_deltas(self):
        rows = [
            {
                "run_id": "run-a",
                "condition_id": "clean",
                "profile_id": "",
                "profile_version": "",
                "severity_id": "",
                "accuracy": 0.8,
                "balanced_accuracy": 0.75,
                "macro_f1": 0.7,
                "ece": 0.1,
                "average_confidence": 0.9,
                "incorrect_average_confidence": 0.6,
            },
            {
                "run_id": "run-a",
                "condition_id": "phase8a-gaussian-noise__S1",
                "profile_id": "phase8a-gaussian-noise",
                "profile_version": "1.0",
                "severity_id": "S1",
                "accuracy": 0.7,
                "balanced_accuracy": 0.65,
                "macro_f1": 0.55,
                "ece": 0.2,
                "average_confidence": 0.85,
                "incorrect_average_confidence": 0.7,
            },
        ]

        deltas = clean_delta_rows(rows)

        self.assertAlmostEqual(deltas[1]["accuracy_delta_from_clean"], -0.1)
        self.assertAlmostEqual(deltas[1]["ece_delta_from_clean"], 0.1)
        self.assertFalse(deltas[1]["metrics_are_robustness_results"])
        rows[1]["incorrect_average_confidence"] = None
        self.assertIsNone(clean_delta_rows(rows)[1]["incorrect_average_confidence_delta_from_clean"])
        with self.assertRaisesRegex(ValueError, "missing clean"):
            clean_delta_rows([rows[1]])

    def test_sample_alignment_checks_conditions_and_runs(self):
        split_results = {
            "run-a": [
                {"condition_id": "clean", "sample_ids": ["s1", "s2"], "true_labels": ["cat", "dog"]},
                {"condition_id": "noise", "sample_ids": ["s1", "s2"], "true_labels": ["cat", "dog"]},
            ],
            "run-b": [
                {"condition_id": "clean", "sample_ids": ["s1", "s2"], "true_labels": ["cat", "dog"]},
            ],
        }

        report = verify_phase8b_sample_alignment(split_results)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["sample_ids_identical_across_runs"])
        bad = {"run-a": [{"condition_id": "clean", "sample_ids": ["s1"], "true_labels": ["cat"]}, {"condition_id": "noise", "sample_ids": ["s2"], "true_labels": ["cat"]}]}
        with self.assertRaisesRegex(ValueError, "sample identity mismatch"):
            verify_phase8b_sample_alignment(bad)

    def test_artifact_validation_rejects_missing_or_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.json"
            empty = root / "empty.json"
            good.write_text("{}", encoding="utf-8")
            empty.write_text("", encoding="utf-8")

            report = verify_phase8b1_artifacts([good])

            self.assertEqual(report["status"], "passed")
            with self.assertRaises(ValueError):
                verify_phase8b1_artifacts([empty])
            with self.assertRaises(FileNotFoundError):
                verify_phase8b1_artifacts([root / "missing.json"])


if __name__ == "__main__":
    unittest.main()
