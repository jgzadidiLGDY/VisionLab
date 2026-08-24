import json
import tempfile
import unittest
from pathlib import Path

from visionlab.experiments.phase6b import (
    PHASE6B1_BASELINE_REFERENCE_RUN_ID,
    PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
    run_phase6b1_pretrained_smoke,
)
from visionlab.models import (
    EXPECTED_PHASE6A_PARAMETER_COUNTS,
    RESNET18_WEIGHT_ENUM,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
)
from visionlab.training import load_checkpoint


class Phase6BSmokeTest(unittest.TestCase):
    def setUp(self):
        probe = probe_resnet18_weight_cache()
        if not probe.exists:
            self.skipTest("ResNet-18 weights are not cached; Phase 6B-1 smoke cannot run")

    def test_pretrained_smoke_records_cached_weight_identity_and_no_material_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase6b1_pretrained_smoke(Path(tmp))

            self.assertEqual(result.status, "completed")
            self.assertTrue(result.pretrained_weights_loaded)
            self.assertEqual(result.run_id, PHASE6B1_PRETRAINED_SMOKE_RUN_ID)
            contract = json.loads(
                Path(result.artifact_paths["run_contract"]).read_text(encoding="utf-8")
            )

            self.assertEqual(contract["phase"], "6B-1")
            self.assertEqual(
                contract["baseline_reference"]["run_id"],
                PHASE6B1_BASELINE_REFERENCE_RUN_ID,
            )
            self.assertTrue(contract["baseline_reference"]["preserved_unchanged"])
            self.assertTrue(contract["pretrained_weights_loaded"])
            self.assertTrue(contract["weight_cache_probe"]["exists"])
            self.assertFalse(contract["weight_cache_probe"]["download_attempted"])
            self.assertEqual(
                contract["model_identity"]["config"]["weight_identity"],
                RESNET18_WEIGHT_ENUM,
            )
            self.assertEqual(
                contract["model_identity"]["parameter_counts"],
                EXPECTED_PHASE6A_PARAMETER_COUNTS,
            )
            self.assertFalse(contract["official_test_evaluation"])
            self.assertFalse(contract["material_cifar10_training"])
            self.assertFalse(contract["fine_tuning"])
            self.assertEqual(contract["smoke_checks"]["raw_input_shape"], [2, 3, 32, 32])
            self.assertEqual(
                contract["smoke_checks"]["preprocessed_batch_shape"],
                [2, 3, 224, 224],
            )
            self.assertEqual(contract["smoke_checks"]["logits_shape"], [2, 10])
            self.assertTrue(contract["smoke_checks"]["loss_finite"])
            self.assertTrue(contract["smoke_checks"]["classifier_head_updated"])
            self.assertTrue(contract["smoke_checks"]["frozen_backbone_unchanged"])
            self.assertTrue(contract["smoke_checks"]["frozen_gradients_blocked"])

    def test_pretrained_smoke_checkpoint_restores_with_matching_transfer_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase6b1_pretrained_smoke(Path(tmp))
            model = build_phase6a_transfer_model(load_pretrained=True)

            checkpoint = load_checkpoint(
                Path(result.artifact_paths["checkpoint"]),
                model=model,
                expected_run_id=PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
            )

            self.assertEqual(checkpoint["run_id"], PHASE6B1_PRETRAINED_SMOKE_RUN_ID)
            self.assertEqual(checkpoint["tag"], "phase6b1-pretrained-frozen-smoke")
            self.assertTrue(checkpoint["model_identity"]["pretrained_weights_loaded"])


if __name__ == "__main__":
    unittest.main()
