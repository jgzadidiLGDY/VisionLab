import unittest

import torch
from torch.utils.data import Dataset

from visionlab.data.degradations import (
    PHASE8A_DEFAULT_SEED,
    PHASE8A_DEGRADATION_PROFILES,
    PHASE8A_DEGRADATION_REGISTRY_ID,
    PHASE8A_DEGRADATION_REGISTRY_VERSION,
    DegradedSampleDataset,
    apply_degradation,
    degradation_registry_dict,
    get_degradation_profile,
    get_degradation_severity,
)


class TinyPredictionDataset(Dataset):
    def __init__(self):
        base = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(3, 32, 32)
        self.samples = [
            {
                "input": base.clone(),
                "raw_input": base.clone(),
                "label": 3,
                "sample_id": "tiny-sample-00001",
                "split": "test",
                "source_id": "tiny-source-00001",
            }
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in self.samples[index].items()}


class Phase8ADegradationTest(unittest.TestCase):
    def test_registry_identity_and_exact_profiles_are_pinned(self):
        registry = degradation_registry_dict()

        self.assertEqual(registry["registry_id"], PHASE8A_DEGRADATION_REGISTRY_ID)
        self.assertEqual(registry["version"], PHASE8A_DEGRADATION_REGISTRY_VERSION)
        self.assertFalse(registry["model_evaluation_allowed"])
        profiles = {profile["profile_id"]: profile for profile in registry["profiles"]}
        self.assertEqual(
            sorted(profiles),
            [
                "phase8a-brightness-shift",
                "phase8a-contrast-reduction",
                "phase8a-gaussian-blur",
                "phase8a-gaussian-noise",
            ],
        )
        self.assertEqual(
            [item["parameters"] for item in profiles["phase8a-gaussian-noise"]["severity_levels"]],
            [{"std": 0.03}, {"std": 0.06}, {"std": 0.09}, {"std": 0.12}, {"std": 0.15}],
        )
        self.assertEqual(
            [item["parameters"] for item in profiles["phase8a-gaussian-blur"]["severity_levels"]],
            [
                {"kernel_size": 3, "sigma": 0.4},
                {"kernel_size": 3, "sigma": 0.7},
                {"kernel_size": 5, "sigma": 1.0},
                {"kernel_size": 5, "sigma": 1.3},
                {"kernel_size": 7, "sigma": 1.6},
            ],
        )
        self.assertEqual(
            [item["parameters"] for item in profiles["phase8a-brightness-shift"]["severity_levels"]],
            [{"delta": -0.08}, {"delta": -0.16}, {"delta": -0.24}, {"delta": -0.32}, {"delta": -0.40}],
        )
        self.assertEqual(
            [item["parameters"] for item in profiles["phase8a-contrast-reduction"]["severity_levels"]],
            [{"factor": 0.9}, {"factor": 0.8}, {"factor": 0.7}, {"factor": 0.6}, {"factor": 0.5}],
        )

    def test_severity_lookup_and_invalid_profile_or_severity_rejection(self):
        profile = get_degradation_profile("phase8a-gaussian-blur")
        severity = get_degradation_severity(profile, "S3")

        self.assertEqual(severity.parameters, {"kernel_size": 5, "sigma": 1.0})
        with self.assertRaisesRegex(ValueError, "unknown degradation profile"):
            get_degradation_profile("missing")
        with self.assertRaisesRegex(ValueError, "unknown severity"):
            get_degradation_severity(profile, "S6")

    def test_all_profiles_and_severities_preserve_shape_range_and_finite_values(self):
        tensor = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(3, 32, 32)

        for profile in PHASE8A_DEGRADATION_PROFILES:
            for severity in profile.severities:
                degraded = apply_degradation(
                    tensor,
                    profile_id=profile.profile_id,
                    severity_id=severity.severity_id,
                    seed=PHASE8A_DEFAULT_SEED,
                    sample_id="sample-a",
                    source_id="source-a",
                )
                self.assertEqual(tuple(degraded.shape), (3, 32, 32))
                self.assertTrue(torch.isfinite(degraded).all())
                self.assertGreaterEqual(float(degraded.min().item()), 0.0)
                self.assertLessEqual(float(degraded.max().item()), 1.0)

    def test_degradation_does_not_mutate_input_tensor(self):
        tensor = torch.rand(3, 32, 32)
        original = tensor.clone()

        _ = apply_degradation(
            tensor,
            profile_id="phase8a-brightness-shift",
            severity_id="S2",
            seed=PHASE8A_DEFAULT_SEED,
        )

        self.assertTrue(torch.equal(tensor, original))

    def test_seeded_stochastic_behavior_is_deterministic(self):
        tensor = torch.rand(3, 32, 32)

        first = apply_degradation(
            tensor,
            profile_id="phase8a-gaussian-noise",
            severity_id="S4",
            seed=123,
            sample_id="sample-a",
            source_id="source-a",
        )
        second = apply_degradation(
            tensor,
            profile_id="phase8a-gaussian-noise",
            severity_id="S4",
            seed=123,
            sample_id="sample-a",
            source_id="source-a",
        )
        different_sample = apply_degradation(
            tensor,
            profile_id="phase8a-gaussian-noise",
            severity_id="S4",
            seed=123,
            sample_id="sample-b",
            source_id="source-a",
        )

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different_sample))
        with self.assertRaisesRegex(ValueError, "requires an explicit seed"):
            apply_degradation(
                tensor,
                profile_id="phase8a-gaussian-noise",
                severity_id="S1",
            )

    def test_invalid_tensor_contract_and_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 RGB channels"):
            apply_degradation(
                torch.zeros(1, 32, 32),
                profile_id="phase8a-gaussian-blur",
                severity_id="S1",
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            apply_degradation(
                torch.full((3, 32, 32), 1.5),
                profile_id="phase8a-gaussian-blur",
                severity_id="S1",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_degradation(
                torch.full((3, 32, 32), float("nan")),
                profile_id="phase8a-gaussian-blur",
                severity_id="S1",
            )

    def test_degraded_dataset_preserves_metadata_and_propagates_registry_identity(self):
        dataset = DegradedSampleDataset(
            TinyPredictionDataset(),
            profile_id="phase8a-contrast-reduction",
            severity_id="S5",
            seed=PHASE8A_DEFAULT_SEED,
        )

        sample = dataset[0]

        self.assertEqual(sample["sample_id"], "tiny-sample-00001")
        self.assertEqual(sample["label"], 3)
        self.assertEqual(sample["split"], "test")
        self.assertEqual(sample["source_id"], "tiny-source-00001")
        self.assertEqual(sample["degradation_profile_id"], "phase8a-contrast-reduction")
        self.assertEqual(sample["degradation_profile_version"], "1.0")
        self.assertEqual(sample["degradation_severity_id"], "S5")
        self.assertEqual(sample["degradation_seed"], PHASE8A_DEFAULT_SEED)
        self.assertEqual(tuple(sample["raw_input"].shape), (3, 32, 32))
        self.assertTrue(torch.equal(sample["input"], sample["raw_input"]))


if __name__ == "__main__":
    unittest.main()
