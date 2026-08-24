import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from visionlab.experiments.phase4a import TinyCIFARLikeDataset
from visionlab.experiments.phase6b import PHASE6B2_RUN_ID
from visionlab.experiments.phase6c import (
    PHASE6C_FINETUNE_MODE,
    PHASE6C_LEARNING_RATE,
    PHASE6C_RUN_ID,
    build_phase6c_finetune_model_from_phase6b2,
    build_phase6c_optimizer,
    finalize_phase6c2_material_contract,
    mark_phase6c2_material_contract_started,
    phase6c_initialization_identity,
    prepare_phase6c1_finetuning_preflight,
    run_phase6c1_mechanics_smoke,
    write_phase6c_comparison_report,
    write_selected_phase6c_checkpoint_evaluation_artifacts,
    verify_phase6c_optimizer_scope,
)
from visionlab.models import (
    PHASE6A_FREEZE_MODE,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
    transfer_parameter_summary,
)
from visionlab.training import save_checkpoint


class Phase6CFineTuningTest(unittest.TestCase):
    def setUp(self):
        if not probe_resnet18_weight_cache().exists:
            self.skipTest("ResNet-18 weights are not cached; Phase 6C requires them")

    def test_phase6c_model_restores_phase6b2_best_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write_source_checkpoint(Path(tmp) / "best.pt")

            model, initialization = build_phase6c_finetune_model_from_phase6b2(checkpoint)

            self.assertEqual(initialization.source_run_id, PHASE6B2_RUN_ID)
            self.assertEqual(initialization.checkpoint_tag, "best")
            self.assertEqual(initialization.checkpoint_epoch, 4)
            self.assertEqual(len(initialization.checkpoint_sha256), 64)
            self.assertEqual(model.config.freeze_mode, PHASE6C_FINETUNE_MODE)
            self.assertEqual(model.identity_dict()["config"]["freeze_mode"], PHASE6C_FINETUNE_MODE)

    def test_layer4_and_fc_are_the_only_trainable_parameters(self):
        model = build_phase6a_transfer_model(
            load_pretrained=True,
            config=TransferModelConfig(freeze_mode=PHASE6C_FINETUNE_MODE),
        )

        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("model.layer4.", "model.fc.")) for name in trainable))
        self.assertIn("model.fc.weight", trainable)
        self.assertIn("model.fc.bias", trainable)
        self.assertTrue(any(name.startswith("model.layer4.") for name in trainable))
        self.assertFalse(
            any(
                name.startswith(("model.conv1.", "model.bn1.", "model.layer1.", "model.layer2.", "model.layer3."))
                for name in trainable
            )
        )
        counts = transfer_parameter_summary(model)["counts"]
        self.assertEqual(counts["total"], 11_181_642)
        self.assertGreater(counts["trainable"], 5_130)

    def test_optimizer_scope_exactly_matches_trainable_parameters(self):
        model = build_phase6a_transfer_model(
            load_pretrained=True,
            config=TransferModelConfig(freeze_mode=PHASE6C_FINETUNE_MODE),
        )

        optimizer = build_phase6c_optimizer(model)
        scope = verify_phase6c_optimizer_scope(model, optimizer)

        self.assertTrue(scope["optimizer_matches_trainable_scope"])
        self.assertEqual(scope["frozen_parameters_in_optimizer"], [])
        self.assertEqual(optimizer.param_groups[0]["lr"], PHASE6C_LEARNING_RATE)
        self.assertEqual(
            set(scope["optimizer_parameter_names"]),
            set(scope["trainable_parameter_names"]),
        )

    def test_mechanics_smoke_proves_frozen_and_trainable_update_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_source_checkpoint(root / "best.pt")

            smoke = run_phase6c1_mechanics_smoke(root / "smoke", checkpoint_path=checkpoint)

            self.assertEqual(smoke["status"], "passed")
            self.assertFalse(smoke["checks"]["official_test_evaluation"])
            self.assertFalse(smoke["checks"]["material_fine_tuning"])
            self.assertTrue(smoke["checks"]["frozen_gradients_blocked"])
            self.assertTrue(smoke["checks"]["frozen_parameters_unchanged"])
            self.assertTrue(smoke["checks"]["trainable_parameters_updated"])
            self.assertEqual(smoke["checks"]["logits_shape"], [2, 10])
            self.assertEqual(
                smoke["optimizer"]["parameter_scope"],
                "exactly parameters marked trainable by finetune_layer4_head",
            )

    def test_preflight_records_initialization_and_evidence_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_source_checkpoint(root / "best.pt")

            prepared = prepare_phase6c1_finetuning_preflight(
                root / "phase6c1",
                batch_size=4,
                checkpoint_path=checkpoint,
                upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
                upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
                validation_per_class=1,
                expected_counts={"train": 10, "val": 10, "test": 10},
            )

            contract = prepared.run_contract
            self.assertEqual(contract["phase"], "6C-1")
            self.assertEqual(contract["run_id"], PHASE6C_RUN_ID)
            self.assertEqual(
                contract["phase6b2_reference"]["frozen_feature_run_id"],
                PHASE6B2_RUN_ID,
            )
            self.assertEqual(contract["phase6b2_reference"]["official_test_loss"], 0.413686)
            self.assertEqual(contract["phase6b2_reference"]["official_test_accuracy"], 0.8561)
            self.assertEqual(
                contract["phase4b_reference"]["run_id"],
                "phase4b-cifar10-custom-cnn-baseline-001",
            )
            self.assertEqual(contract["initialization"]["initialization_source_run_id"], PHASE6B2_RUN_ID)
            self.assertEqual(contract["initialization"]["initialization_checkpoint_tag"], "best")
            self.assertEqual(contract["initialization"]["initialization_checkpoint_epoch"], 4)
            self.assertIn("initialization_checkpoint_sha256", contract["initialization"])
            self.assertEqual(contract["model_identity"]["config"]["freeze_mode"], PHASE6C_FINETUNE_MODE)
            self.assertTrue(contract["fine_tuning"])
            self.assertEqual(contract["partial_unfreezing"], "layer4_plus_fc_only")
            self.assertFalse(contract["differential_learning_rate_groups"])
            self.assertFalse(contract["official_test_evaluation"]["performed_in_phase6c1"])
            self.assertIn("fixed_reference_not_training_target", contract["phase6b2_reference"])
            self.assertEqual(
                contract["preflight_report"],
                str(prepared.preflight_path),
            ) if "preflight_report" in contract else None
            self.assertEqual(prepared.preflight_report["status"], "passed")
            self.assertEqual(
                prepared.preflight_report["preprocessing_probe"]["preprocessed_shape"],
                [3, 224, 224],
            )
            self.assertTrue(prepared.mechanics_path.exists())
            self.assertTrue(prepared.run_contract_path.exists())


    def test_phase6c2_material_contract_metadata_is_promoted_and_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_source_checkpoint(root / "best.pt")
            prepared = prepare_phase6c1_finetuning_preflight(
                root / "phase6c2",
                batch_size=4,
                checkpoint_path=checkpoint,
                upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
                upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
                validation_per_class=1,
                expected_counts={"train": 10, "val": 10, "test": 10},
            )

            started = mark_phase6c2_material_contract_started(prepared.run_contract_path)

            self.assertEqual(started["phase"], "6C-2")
            self.assertTrue(started["material_fine_tuning"])
            self.assertFalse(started["official_test_evaluation"]["performed"])
            self.assertEqual(started["batch_size"], 4)
            self.assertEqual(started["device"], "cpu")
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            history_path = artifact_dir / "history.json"
            val_path = artifact_dir / "val_summary.json"
            test_path = artifact_dir / "test_summary.json"
            history_path.write_text(json.dumps([{"epoch": 1}, {"epoch": 2}]), encoding="utf-8")
            val_path.write_text(json.dumps({"loss": 0.246512, "accuracy": 0.9258}), encoding="utf-8")
            test_path.write_text(json.dumps({"loss": 0.272485, "accuracy": 0.9147}), encoding="utf-8")
            result = SimpleNamespace(status="completed", best_epoch=2)

            finalized = finalize_phase6c2_material_contract(
                prepared.run_contract_path,
                result=result,
                artifact_paths={
                    "history": str(history_path),
                    "val_summary": str(val_path),
                    "test_summary": str(test_path),
                    "selected_checkpoint": "checkpoints/best.pt",
                },
            )

            self.assertEqual(finalized["phase"], "6C-2")
            self.assertTrue(finalized["official_test_evaluation"]["performed"])
            self.assertEqual(finalized["official_test_evaluation"]["count"], 1)
            self.assertTrue(
                finalized["official_test_evaluation"]["performed_after_best_checkpoint_restore"]
            )
            self.assertEqual(finalized["material_run_result"]["best_checkpoint_epoch"], 2)
            self.assertAlmostEqual(
                finalized["material_run_result"]["test_accuracy_delta_vs_phase6b2"],
                0.0586,
                places=4,
            )
            self.assertFalse(finalized["metadata_correction"]["training_rerun"])
            self.assertFalse(finalized["metadata_correction"]["additional_test_evaluation"])

    def test_selected_phase6c_checkpoint_restore_writes_val_and_test_artifacts(self):
        from visionlab.data import DataLoaderPolicy, build_cifar10_split_datasets, build_transfer_dataloaders

        datasets = build_cifar10_split_datasets(
            upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
            upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
            validation_per_class=1,
        )
        loaders = build_transfer_dataloaders(
            datasets,
            DataLoaderPolicy(batch_size=5, seed=20260820),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_checkpoint = self._write_source_checkpoint(root / "phase6b2_best.pt")
            model, _ = build_phase6c_finetune_model_from_phase6b2(source_checkpoint)
            optimizer = build_phase6c_optimizer(model)
            phase6c_checkpoint = save_checkpoint(
                root / "phase6c_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=1,
                run_id=PHASE6C_RUN_ID,
                seed=20260820,
                metrics=[{"val_loss": 0.25}],
                tag="best",
            )

            artifacts = write_selected_phase6c_checkpoint_evaluation_artifacts(
                checkpoint_path=phase6c_checkpoint,
                run_id=PHASE6C_RUN_ID,
                val_loader=loaders.prediction_val,
                test_loader=loaders.prediction_test,
                output_dir=root / "artifacts",
                include_test=True,
            )

            self.assertEqual(artifacts["selected_checkpoint_tag"], "best")
            self.assertTrue(Path(artifacts["val_summary"]).exists())
            self.assertTrue(Path(artifacts["val_predictions"]).exists())
            self.assertTrue(Path(artifacts["test_summary"]).exists())
            self.assertTrue(Path(artifacts["test_predictions"]).exists())

    def test_phase6c_comparison_report_keeps_references_separate(self):
        from visionlab.experiments.phase6c import phase6c_initialization_identity
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_checkpoint = self._write_source_checkpoint(root / "phase6b2_best.pt")
            initialization = phase6c_initialization_identity(source_checkpoint)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            history_path = artifact_dir / "history.json"
            val_path = artifact_dir / "val_summary.json"
            test_path = artifact_dir / "test_summary.json"
            history_path.write_text(json.dumps([{"train_loss": 0.3, "train_accuracy": 0.9, "val_loss": 0.4, "val_accuracy": 0.8}]), encoding="utf-8")
            val_path.write_text(json.dumps({"loss": 0.4, "accuracy": 0.8}), encoding="utf-8")
            test_path.write_text(json.dumps({"loss": 0.5, "accuracy": 0.75}), encoding="utf-8")

            report = write_phase6c_comparison_report(
                root,
                run_id=PHASE6C_RUN_ID,
                best_epoch=1,
                artifact_paths={
                    "history": str(history_path),
                    "val_summary": str(val_path),
                    "test_summary": str(test_path),
                },
                initialization=initialization,
            )

            text = report.read_text(encoding="utf-8")
            self.assertIn("Fixed Phase 6B-2 Reference", text)
            self.assertIn("Phase 4B Historical Reference", text)
            self.assertIn("0.856100", text)
            self.assertIn("0.635900", text)

    def test_incompatible_initialization_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write_source_checkpoint(Path(tmp) / "terminal.pt", epoch=5, tag="terminal")

            with self.assertRaises(ValueError):
                phase6c_initialization_identity(checkpoint)

    def _write_source_checkpoint(self, path: Path, *, epoch: int = 4, tag: str = "best") -> Path:
        model = build_phase6a_transfer_model(
            load_pretrained=True,
            config=TransferModelConfig(freeze_mode=PHASE6A_FREEZE_MODE),
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.001,
        )
        return save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            run_id=PHASE6B2_RUN_ID,
            seed=20260820,
            metrics=[{"val_loss": 0.3983015718460083}],
            tag=tag,
        )


if __name__ == "__main__":
    unittest.main()
