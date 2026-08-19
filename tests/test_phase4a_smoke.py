import json
import tempfile
import unittest
from pathlib import Path

from visionlab.experiments.phase4a import run_phase4a_smoke


class Phase4ASmokeTest(unittest.TestCase):
    def test_smoke_run_writes_plumbing_artifacts_without_official_test_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_phase4a_smoke(Path(tmp))

            self.assertEqual(result.status, "completed")
            self.assertIn("history", result.artifact_paths)
            self.assertIn("curve_data", result.artifact_paths)
            self.assertIn("val_predictions", result.artifact_paths)
            self.assertTrue(Path(result.artifact_paths["run_contract"]).exists())
            contract = json.loads(
                Path(result.artifact_paths["run_contract"]).read_text(encoding="utf-8")
            )

            self.assertEqual(contract["phase"], "4A")
            self.assertFalse(contract["official_test_evaluation"])
            self.assertEqual(contract["dataloader_policy"]["num_workers"], 0)
            self.assertEqual(contract["training_config"]["selection_metric"], "val_loss")
            self.assertEqual(contract["test_loader_constructed_count"], 10)
            self.assertEqual(contract["checkpoint_selection_metric"], "val_loss")
            self.assertTrue(contract["selected_checkpoint_for_evaluation"].endswith("best.pt"))
            self.assertEqual(result.artifact_paths["selected_checkpoint_tag"], "best")
            self.assertNotIn("test_predictions", result.artifact_paths)


if __name__ == "__main__":
    unittest.main()
