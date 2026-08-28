import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

from visionlab.data.cifar10 import CIFAR10_CLASSES
from visionlab.data.degradations import degradation_registry_dict
from visionlab.experiments.phase8b import (
    PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS,
    phase8b_conditions,
)
from visionlab.data.cifar10_1 import (
    CIFAR10_1_DATASET_ID,
    CIFAR10_1_EXPECTED_SAMPLE_COUNT,
    CIFAR10_1_IMAGE_SHAPE_HWC,
    CIFAR10_1_SPLIT_NAME,
    CIFAR10_1_USAGE,
    CIFAR10_1_VERSION,
    Cifar101Dataset,
    build_cifar10_1_manifest,
    cifar10_1_contract_dict,
    find_local_cifar10_1_v6,
    phase8c1_tiny_fixture_dataset,
    validate_cifar10_1_split,
    validate_cifar10_1_usage,
    verify_cifar10_1_dataset_contract,
    write_phase8c1_preflight_artifacts,
)


class Phase8C1Cifar101Test(unittest.TestCase):
    def test_contract_records_exact_dataset_identity_and_version(self):
        contract = cifar10_1_contract_dict()

        self.assertEqual(contract["identity"]["dataset_id"], CIFAR10_1_DATASET_ID)
        self.assertEqual(contract["identity"]["version"], CIFAR10_1_VERSION)
        self.assertEqual(contract["expected_sample_count"], CIFAR10_1_EXPECTED_SAMPLE_COUNT)
        self.assertEqual(contract["image_shape"], list(CIFAR10_1_IMAGE_SHAPE_HWC))
        self.assertEqual(contract["split_name"], CIFAR10_1_SPLIT_NAME)
        self.assertEqual(contract["usage"], CIFAR10_1_USAGE)

    def test_class_map_exactly_matches_cifar10(self):
        contract = cifar10_1_contract_dict()

        self.assertEqual(tuple(contract["classes"]), CIFAR10_CLASSES)

    def test_dataset_requires_expected_v6_sample_count_when_material(self):
        images = torch.zeros(4, 32, 32, 3, dtype=torch.uint8)
        labels = torch.arange(4)

        with self.assertRaisesRegex(ValueError, "2,000"):
            Cifar101Dataset(images, labels, require_expected_count=True)

    def test_tiny_fixture_preserves_stable_ids_and_metadata(self):
        dataset = phase8c1_tiny_fixture_dataset(sample_count=3)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]["sample_id"], "cifar10-1-v6-00000")
        self.assertEqual(dataset[1]["sample_id"], "cifar10-1-v6-00001")
        self.assertEqual(dataset[0]["dataset_id"], "cifar10-1")
        self.assertEqual(dataset[0]["dataset_version"], "v6")
        self.assertEqual(dataset[0]["split"], CIFAR10_1_SPLIT_NAME)
        self.assertEqual(dataset[0]["usage"], CIFAR10_1_USAGE)

    def test_raw_tensor_contract_shape_range_and_finite_values(self):
        dataset = phase8c1_tiny_fixture_dataset(sample_count=2)

        for index in range(len(dataset)):
            sample = dataset[index]
            raw = sample["raw_input"]
            self.assertEqual(tuple(raw.shape), (3, 32, 32))
            self.assertGreaterEqual(float(raw.min().item()), 0.0)
            self.assertLessEqual(float(raw.max().item()), 1.0)
            self.assertTrue(torch.isfinite(raw).all().item())

    def test_invalid_split_and_usage_are_rejected(self):
        validate_cifar10_1_split(CIFAR10_1_SPLIT_NAME)
        validate_cifar10_1_usage("evaluation")

        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            validate_cifar10_1_split("train")
        with self.assertRaisesRegex(ValueError, "only be used for evaluation"):
            validate_cifar10_1_usage("training")

    def test_manifest_serialization_preserves_identity_and_samples(self):
        dataset = phase8c1_tiny_fixture_dataset(sample_count=2)
        manifest = build_cifar10_1_manifest(dataset).to_dict()

        self.assertEqual(manifest["identity"]["dataset_id"], "cifar10-1")
        self.assertEqual(manifest["identity"]["version"], "v6")
        self.assertEqual(manifest["split_names"], [CIFAR10_1_SPLIT_NAME])
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual(manifest["samples"][0]["sample_id"], "cifar10-1-v6-00000")

    def test_verify_dataset_contract_accepts_fixture_without_material_count(self):
        dataset = phase8c1_tiny_fixture_dataset(sample_count=2)
        report = verify_cifar10_1_dataset_contract(dataset, require_expected_count=False)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["dataset_id"], "cifar10-1")
        self.assertEqual(report["version"], "v6")
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(len(report["sample_label_digest"]), 64)

    def test_local_availability_reports_missing_without_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            availability = find_local_cifar10_1_v6(tmp)

        self.assertFalse(availability.available)
        self.assertIn("approval required", availability.reason)
        self.assertIsNone(availability.data_path)
        self.assertIsNone(availability.labels_path)

    def test_preflight_artifacts_stop_before_real_visual_qa_when_data_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase8c1"
            data_root = Path(tmp) / "data"
            result = write_phase8c1_preflight_artifacts(output, data_root=data_root)

            self.assertEqual(result["status"], "blocked_external_data_required")
            self.assertEqual(result["real_cifar10_1_registration_complete"], False)
            self.assertEqual(result["real_visual_qa_complete"], False)
            self.assertEqual(result["model_evaluation_performed"], False)
            for path in result.values():
                if isinstance(path, str) and path.endswith((".json", ".md")):
                    self.assertTrue(Path(path).exists())

    def test_existing_phase8a_and_phase8b_contracts_are_unchanged(self):
        registry = degradation_registry_dict()
        conditions = phase8b_conditions()

        self.assertEqual(registry["version"], "1.0")
        self.assertEqual(len(registry["profiles"]), 4)
        self.assertEqual(len(conditions), 21)
        self.assertEqual(PHASE8B2_EXPECTED_MODEL_CONDITION_ROWS, 63)
        noise_s5 = next(
            condition for condition in conditions
            if condition.condition_id == "phase8a-gaussian-noise__S5"
        )
        self.assertEqual(noise_s5.severity_parameters, {"std": 0.15})

    def test_preflight_registers_available_v6_files_and_writes_visual_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            local = data_root / "cifar10.1"
            local.mkdir(parents=True)
            images = np.zeros((2000, 32, 32, 3), dtype=np.uint8)
            labels = np.repeat(np.arange(10, dtype=np.int32), 200)
            for index in range(2000):
                images[index, :, :, index % 3] = index % 255
            np.save(local / "cifar10.1_v6_data.npy", images)
            np.save(local / "cifar10.1_v6_labels.npy", labels)

            result = write_phase8c1_preflight_artifacts(root / "outputs", data_root=data_root)

            self.assertEqual(result["status"], "real_registration_and_visual_qa_complete")
            self.assertTrue(result["real_cifar10_1_registration_complete"])
            self.assertTrue(result["real_visual_qa_complete"])
            self.assertFalse(result["model_evaluation_performed"])
            self.assertTrue(Path(result["real_registration"]).exists())
            self.assertTrue(Path(result["real_manifest"]).exists())
            self.assertTrue(Path(result["visual_sample_manifest"]).exists())
            self.assertTrue(Path(result["visual_grid"]).exists())

if __name__ == "__main__":
    unittest.main()
