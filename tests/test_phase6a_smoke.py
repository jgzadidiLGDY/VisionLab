import json
import tempfile
import unittest
from pathlib import Path

from visionlab.experiments.phase6a import (
    PHASE6A_BASELINE_REFERENCE_RUN_ID,
    PHASE6A_MECHANICS_SMOKE_RUN_ID,
    run_phase6a_model_mechanics_smoke,
    run_phase6a_pretrained_frozen_smoke,
)
from visionlab.models import (
    EXPECTED_PHASE6A_PARAMETER_COUNTS,
    RESNET18_WEIGHT_ENUM,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
)
from visionlab.training import load_checkpoint


class Phase6ASmokeTest(unittest.TestCase):
    def test_mechanics_smoke_writes_contract_without_pretrained_weights_or_material_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase6a_model_mechanics_smoke(Path(tmp))

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.run_id, PHASE6A_MECHANICS_SMOKE_RUN_ID)
            self.assertFalse(result.pretrained_weights_loaded)
            self.assertIn("run_contract", result.artifact_paths)
            self.assertIn("checkpoint", result.artifact_paths)

            contract = json.loads(
                Path(result.artifact_paths["run_contract"]).read_text(encoding="utf-8")
            )
            self.assertEqual(contract["phase"], "6A")
            self.assertEqual(
                contract["baseline_reference"]["run_id"],
                PHASE6A_BASELINE_REFERENCE_RUN_ID,
            )
            self.assertTrue(contract["baseline_reference"]["preserved_unchanged"])
            self.assertFalse(contract["pretrained_weights_loaded"])
            self.assertFalse(contract["official_test_evaluation"])
            self.assertFalse(contract["material_cifar10_training"])
            self.assertFalse(contract["fine_tuning"])
            self.assertIn("mechanics evidence only", contract["evidence_boundary"])
            self.assertEqual(
                contract["model_identity"]["config"]["weight_identity"],
                RESNET18_WEIGHT_ENUM,
            )
            self.assertEqual(
                contract["model_identity"]["parameter_counts"],
                EXPECTED_PHASE6A_PARAMETER_COUNTS,
            )
            self.assertEqual(contract["smoke_checks"]["logits_shape"], [2, 10])
            self.assertTrue(contract["smoke_checks"]["loss_finite"])
            self.assertTrue(contract["smoke_checks"]["classifier_head_updated"])
            self.assertTrue(contract["smoke_checks"]["frozen_backbone_unchanged"])
            self.assertTrue(contract["smoke_checks"]["frozen_gradients_blocked"])
            self.assertFalse(contract["weight_cache_probe"]["download_attempted"])

            metadata = json.loads(
                Path(result.artifact_paths["metadata"]).read_text(encoding="utf-8")
            )
            self.assertFalse(metadata["pretrained_weights_loaded"])

    def test_mechanics_smoke_checkpoint_restores_with_matching_transfer_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase6a_model_mechanics_smoke(Path(tmp))
            model = build_phase6a_transfer_model(load_pretrained=False)

            checkpoint = load_checkpoint(
                Path(result.artifact_paths["checkpoint"]),
                model=model,
                expected_run_id=PHASE6A_MECHANICS_SMOKE_RUN_ID,
            )

            self.assertEqual(checkpoint["run_id"], PHASE6A_MECHANICS_SMOKE_RUN_ID)
            self.assertEqual(checkpoint["tag"], "phase6a-mechanics-smoke")
            self.assertEqual(
                checkpoint["model_identity"]["config"]["weight_identity"],
                RESNET18_WEIGHT_ENUM,
            )

    def test_checkpoint_rejects_incompatible_transfer_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase6a_model_mechanics_smoke(Path(tmp))
            incompatible_config = TransferModelConfig()
            object.__setattr__(
                incompatible_config,
                "config_version",
                "phase6a-transfer-model-incompatible",
            )
            incompatible = build_phase6a_transfer_model(
                load_pretrained=False,
                config=incompatible_config,
            )

            with self.assertRaisesRegex(ValueError, "model identity"):
                load_checkpoint(Path(result.artifact_paths["checkpoint"]), model=incompatible)

    def test_pretrained_smoke_skips_when_exact_weights_are_not_cached(self):
        probe = probe_resnet18_weight_cache()
        if probe.exists:
            self.skipTest("cached ResNet-18 weights are present; skip absent-cache check")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "download approval"):
                run_phase6a_pretrained_frozen_smoke(Path(tmp))


if __name__ == "__main__":
    unittest.main()
