import json
import tempfile
import unittest
from pathlib import Path

from visionlab.experiments.phase4a import TinyCIFARLikeDataset
from visionlab.experiments.phase5b import (
    PHASE5B_BASELINE_REFERENCE_RUN_ID,
    PHASE5B_RUN_ID,
    PHASE5B_SEED,
    PHASE5B_BASELINE_REFERENCE,
    prepare_phase5b_material_run,
    write_phase5b_comparison_report,
)


class Phase5BPlumbingTest(unittest.TestCase):
    def test_prepare_phase5b_material_run_records_exact_approved_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "phase5b-run"
            prepared = prepare_phase5b_material_run(
                run_dir,
                upstream_train=TinyCIFARLikeDataset(repeats_per_class=5000),
                upstream_test=TinyCIFARLikeDataset(repeats_per_class=1000, offset=0.05),
            )

            contract = prepared.run_contract
            self.assertEqual(contract["phase"], "5B")
            self.assertEqual(contract["run_id"], PHASE5B_RUN_ID)
            self.assertEqual(contract["baseline_reference"]["run_id"], PHASE5B_BASELINE_REFERENCE_RUN_ID)
            self.assertEqual(
                contract["dataset_contract"]["split_counts"],
                {"train": 45000, "val": 5000, "test": 10000},
            )
            self.assertEqual(contract["model_config"]["input_channels"], 3)
            self.assertEqual(contract["model_config"]["image_size"], [32, 32])
            self.assertEqual(contract["model_config"]["num_classes"], 10)
            self.assertEqual(contract["model_config"]["feature_channels"], [32, 64, 128])
            self.assertEqual(contract["model_config"]["dropout"], 0.0)
            self.assertEqual(contract["training_config"]["seed"], PHASE5B_SEED)
            self.assertEqual(contract["training_config"]["max_epochs"], 10)
            self.assertEqual(contract["training_config"]["optimizer"]["name"], "adam")
            self.assertEqual(contract["training_config"]["optimizer"]["learning_rate"], 0.001)
            self.assertEqual(contract["training_config"]["optimizer"]["weight_decay"], 0.0)
            self.assertIsNone(contract["training_config"]["scheduler"])
            self.assertEqual(contract["dataloader_policy"]["batch_size"], 128)
            self.assertEqual(contract["dataloader_policy"]["seed"], PHASE5B_SEED)
            self.assertTrue(contract["dataloader_policy"]["train_shuffle"])
            self.assertFalse(contract["dataloader_policy"]["eval_shuffle"])
            self.assertEqual(contract["dataloader_policy"]["num_workers"], 0)
            self.assertEqual(contract["checkpoint_selection_metric"], "val_loss")
            self.assertEqual(
                contract["augmentation_profile"]["profile_id"],
                "phase5a-candidate-horizontal-flip-random-crop",
            )
            self.assertEqual(contract["augmentation_profile"]["version"], "1.0")
            self.assertEqual(
                contract["experimental_variable"]["phase4b_control_profile_id"],
                "phase4-control-no-augmentation",
            )
            self.assertTrue(contract["official_test_evaluation"]["enabled"])
            self.assertTrue(contract["official_test_evaluation"]["occurs_after_best_checkpoint_restore"])
            self.assertEqual(contract["official_test_evaluation"]["count"], 1)
            self.assertEqual(
                contract["evaluation_sequence"],
                [
                    "train with candidate train-only augmentation",
                    "select best by val_loss",
                    "restore best checkpoint",
                    "generate final validation artifacts",
                    "evaluate test once",
                    "generate test artifacts",
                ],
            )
            self.assertTrue(prepared.profile_registry_snapshot_path.exists())
            self.assertTrue(prepared.run_contract_path.exists())
            self.assertEqual(prepared.preflight_report["status"], "passed")

    def test_comparison_report_records_phase4b_reference_and_candidate_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            val_summary_path = output_dir / "val_summary.json"
            test_summary_path = output_dir / "test_summary.json"
            history_path = output_dir / "history.json"
            artifact_paths = {
                "history": str(history_path),
                "val_summary": str(val_summary_path),
                "test_summary": str(test_summary_path),
            }
            val_summary_path.write_text(
                json.dumps({"loss": 0.9, "accuracy": 0.7}),
                encoding="utf-8",
            )
            test_summary_path.write_text(
                json.dumps({"loss": 1.0, "accuracy": 0.64}),
                encoding="utf-8",
            )
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "train_loss": 1.2,
                            "train_accuracy": 0.5,
                            "val_loss": 0.9,
                            "val_accuracy": 0.7,
                            "learning_rate": 0.001,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            report_path = write_phase5b_comparison_report(
                output_dir,
                run_id=PHASE5B_RUN_ID,
                best_epoch=1,
                artifact_paths=artifact_paths,
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn(PHASE5B_BASELINE_REFERENCE_RUN_ID, report)
            self.assertIn("phase5a-candidate-horizontal-flip-random-crop", report)
            self.assertIn("The candidate profile remains a candidate", report)
            self.assertIn(f"{PHASE5B_BASELINE_REFERENCE['official_test_accuracy']:.6f}", report)


if __name__ == "__main__":
    unittest.main()
