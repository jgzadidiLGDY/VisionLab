import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visionlab.models import CustomCNN, CustomCNNConfig
from visionlab.training import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    fit,
    load_checkpoint,
    train_one_epoch,
    validate,
)


def easy_color_dataset(repeats=4):
    red = torch.zeros(repeats, 3, 32, 32)
    red[:, 0, :, :] = 1.0
    green = torch.zeros(repeats, 3, 32, 32)
    green[:, 1, :, :] = 1.0
    inputs = torch.cat([red, green], dim=0)
    labels = torch.tensor([0] * repeats + [1] * repeats)
    return TensorDataset(inputs, labels)


class NanLoss(nn.Module):
    def forward(self, logits, labels):
        return logits.sum() * torch.tensor(float("nan"))


class TrainingEngineTest(unittest.TestCase):
    def test_train_one_epoch_updates_trainable_parameters(self):
        torch.manual_seed(3)
        model = CustomCNN(CustomCNNConfig(num_classes=2, feature_channels=(4, 8)))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        loader = DataLoader(easy_color_dataset(), batch_size=4, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        metrics = train_one_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
        )

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in model.named_parameters()
            )
        )

    def test_validation_uses_eval_no_grad_and_does_not_mutate_parameters(self):
        torch.manual_seed(4)
        model = CustomCNN(CustomCNNConfig(num_classes=2, feature_channels=(4, 8)))
        model.train()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        loader = DataLoader(easy_color_dataset(), batch_size=4, shuffle=False)

        metrics = validate(model, loader, nn.CrossEntropyLoss())

        self.assertTrue(model.training)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter.detach()))
            self.assertIsNone(parameter.grad)

    def test_tiny_synthetic_dataset_is_explicitly_overfit(self):
        config = TrainingConfig(
            run_id="phase3-tiny-overfit",
            seed=19,
            max_epochs=35,
            optimizer=OptimizerConfig(name="adam", learning_rate=0.02),
            selection_metric="train_loss",
        )
        model = CustomCNN(CustomCNNConfig(num_classes=2, feature_channels=(4, 8)))
        loader = DataLoader(easy_color_dataset(repeats=5), batch_size=2, shuffle=False)

        result = fit(model, loader, config=config)

        self.assertEqual(result.status, "completed")
        final = result.metadata.epoch_history[-1]
        self.assertEqual(final.train_accuracy, 1.0)
        self.assertLess(final.train_loss, 0.02)
        self.assertIsNotNone(result.best_epoch)

    def test_fit_records_metadata_learning_rates_and_checkpoints(self):
        config = TrainingConfig(
            run_id="phase3-checkpoint",
            seed=23,
            max_epochs=3,
            optimizer=OptimizerConfig(name="adam", learning_rate=0.01),
            scheduler=SchedulerConfig(step_size=1, gamma=0.5),
            selection_metric="train_loss",
        )
        model_config = CustomCNNConfig(num_classes=2, feature_channels=(4, 8))
        model = CustomCNN(model_config)
        loader = DataLoader(easy_color_dataset(), batch_size=4, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, loader, config=config, run_dir=Path(tmp))

            self.assertEqual(result.status, "completed")
            self.assertEqual(
                [metric.learning_rate for metric in result.metadata.epoch_history],
                [0.01, 0.005, 0.0025],
            )
            self.assertEqual(result.metadata.stop_reason, "max_epochs_reached")
            self.assertIn("torch", result.metadata.environment)
            self.assertTrue((Path(tmp) / "metadata.json").exists())
            self.assertTrue(Path(result.metadata.checkpoint_references["best"]).exists())
            self.assertTrue(Path(result.metadata.checkpoint_references["terminal"]).exists())

            restored = CustomCNN(model_config)
            optimizer = build_optimizer(restored, config.optimizer)
            scheduler = build_scheduler(optimizer, config.scheduler)
            checkpoint = load_checkpoint(
                Path(result.metadata.checkpoint_references["terminal"]),
                model=restored,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_run_id=config.run_id,
            )

            self.assertEqual(checkpoint["run_id"], config.run_id)
            with torch.no_grad():
                inputs, _ = next(iter(loader))
                self.assertTrue(torch.allclose(model(inputs), restored(inputs)))

    def test_checkpoint_rejects_incompatible_model_identity(self):
        config = TrainingConfig(
            run_id="phase3-incompatible-checkpoint",
            seed=29,
            max_epochs=1,
            optimizer=OptimizerConfig(name="adam", learning_rate=0.01),
            selection_metric="train_loss",
        )
        model = CustomCNN(CustomCNNConfig(num_classes=2, feature_channels=(4, 8)))
        loader = DataLoader(easy_color_dataset(), batch_size=4, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, loader, config=config, run_dir=Path(tmp))
            incompatible = CustomCNN(
                CustomCNNConfig(num_classes=3, feature_channels=(4, 8))
            )

            with self.assertRaisesRegex(ValueError, "model identity"):
                load_checkpoint(
                    Path(result.metadata.checkpoint_references["terminal"]),
                    model=incompatible,
                )

    def test_non_finite_loss_leaves_inspectable_failed_status(self):
        config = TrainingConfig(
            run_id="phase3-nonfinite",
            seed=31,
            max_epochs=2,
            optimizer=OptimizerConfig(name="adam", learning_rate=0.01),
            selection_metric="train_loss",
        )
        model = CustomCNN(CustomCNNConfig(num_classes=2, feature_channels=(4, 8)))
        loader = DataLoader(easy_color_dataset(), batch_size=4, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp:
            result = fit(
                model,
                loader,
                config=config,
                loss_fn=NanLoss(),
                run_dir=Path(tmp),
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.metadata.stop_reason, "failure")
            self.assertIn("non-finite training loss", result.metadata.failure_reason)
            self.assertEqual(result.metadata.epoch_history, ())
            self.assertTrue((Path(tmp) / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
