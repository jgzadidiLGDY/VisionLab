import tempfile
import unittest
from pathlib import Path

import torch

from visionlab.data import DataLoaderPolicy, build_cifar10_split_datasets, build_transfer_dataloaders
from visionlab.experiments.phase4a import TinyCIFARLikeDataset
from visionlab.experiments.phase6b import (
    PHASE6B2_BASELINE_REFERENCE,
    PHASE6B2_RUN_ID,
    prepare_phase6b2_material_run,
    write_selected_transfer_checkpoint_evaluation_artifacts,
)
from visionlab.models import (
    EXPECTED_PHASE6A_PARAMETER_COUNTS,
    RESNET18_WEIGHT_ENUM,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
)
from visionlab.training import save_checkpoint


class Phase6BMaterialPlumbingTest(unittest.TestCase):
    def setUp(self):
        probe = probe_resnet18_weight_cache()
        if not probe.exists:
            self.skipTest("ResNet-18 weights are not cached; Phase 6B-2 plumbing requires them")

    def test_prepare_phase6b2_material_run_records_exact_approved_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_phase6b2_material_run(
                Path(tmp) / "phase6b2",
                batch_size=4,
                upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
                upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
                validation_per_class=1,
                expected_counts={"train": 10, "val": 10, "test": 10},
            )

            contract = prepared.run_contract
            self.assertEqual(contract["phase"], "6B-2")
            self.assertEqual(contract["run_id"], PHASE6B2_RUN_ID)
            self.assertEqual(
                contract["baseline_reference"]["run_id"],
                PHASE6B2_BASELINE_REFERENCE["run_id"],
            )
            self.assertEqual(
                contract["dataset_contract"]["split_counts"],
                {"train": 10, "val": 10, "test": 10},
            )
            self.assertEqual(contract["model_identity"]["config"]["architecture"], "torchvision.models.resnet18")
            self.assertEqual(contract["model_identity"]["config"]["weight_identity"], RESNET18_WEIGHT_ENUM)
            self.assertTrue(contract["model_identity"]["pretrained_weights_loaded"])
            self.assertEqual(contract["model_identity"]["parameter_counts"], EXPECTED_PHASE6A_PARAMETER_COUNTS)
            self.assertEqual(contract["training_config"]["optimizer"]["name"], "adam")
            self.assertEqual(contract["training_config"]["optimizer"]["learning_rate"], 0.001)
            self.assertEqual(contract["training_config"]["optimizer"]["weight_decay"], 0.0)
            self.assertEqual(contract["training_config"]["max_epochs"], 5)
            self.assertEqual(contract["training_config"]["selection_metric"], "val_loss")
            self.assertEqual(contract["dataloader_policy"]["batch_size"], 4)
            self.assertEqual(contract["preprocessing_contract"]["actual_transform_source"], "ResNet18_Weights.IMAGENET1K_V1.transforms()")
            self.assertEqual(contract["preflight_report"], str(prepared.preflight_path))
            self.assertTrue(contract["weight_cache_probe"]["exists"])
            self.assertFalse(contract["fine_tuning"])
            self.assertFalse(contract["partial_unfreezing"])
            self.assertFalse(contract["differential_learning_rate_groups"])
            self.assertFalse(contract["seed_sweep"])
            self.assertFalse(contract["hyperparameter_search"])
            self.assertEqual(contract["augmentation"], "none")
            self.assertTrue(contract["official_test_evaluation"]["enabled_for_material_run_only"])
            self.assertTrue(contract["official_test_evaluation"]["occurs_after_best_checkpoint_restore"])
            self.assertEqual(contract["official_test_evaluation"]["count"], 1)
            self.assertTrue(contract["official_test_evaluation"]["not_performed_during_preflight"])
            self.assertIn("pretrained_source_data", contract["comparison_asymmetry"])
            self.assertEqual(prepared.preflight_report["status"], "passed")
            self.assertEqual(
                prepared.preflight_report["preprocessing_probe"]["preprocessed_shape"],
                [3, 224, 224],
            )

    def test_selected_transfer_checkpoint_restore_writes_val_and_test_artifacts(self):
        datasets = build_cifar10_split_datasets(
            upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
            upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
            validation_per_class=1,
        )
        loaders = build_transfer_dataloaders(
            datasets,
            DataLoaderPolicy(batch_size=5, seed=20260820),
        )
        model = build_phase6a_transfer_model(load_pretrained=True)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.001,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_path = save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=1,
                run_id=PHASE6B2_RUN_ID,
                seed=20260820,
                metrics=[{"val_loss": 1.0}],
                tag="best",
            )

            artifacts = write_selected_transfer_checkpoint_evaluation_artifacts(
                checkpoint_path=checkpoint_path,
                run_id=PHASE6B2_RUN_ID,
                val_loader=loaders.prediction_val,
                test_loader=loaders.prediction_test,
                output_dir=output_dir / "artifacts",
                include_test=True,
            )

            self.assertEqual(artifacts["selected_checkpoint_tag"], "best")
            self.assertTrue(Path(artifacts["val_summary"]).exists())
            self.assertTrue(Path(artifacts["val_predictions"]).exists())
            self.assertTrue(Path(artifacts["test_summary"]).exists())
            self.assertTrue(Path(artifacts["test_predictions"]).exists())


if __name__ == "__main__":
    unittest.main()
