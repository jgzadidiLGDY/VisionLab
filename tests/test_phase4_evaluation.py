import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionlab.evaluation import (
    evaluate_classification,
    write_evaluation_artifacts,
    write_history_artifacts,
)
from visionlab.training import EpochMetrics, TrainingRunMetadata


class DictDataset(Dataset):
    def __init__(self):
        self.rows = [
            (torch.tensor([1.0, 0.0]), 0, "sample-0"),
            (torch.tensor([0.0, 1.0]), 1, "sample-1"),
            (torch.tensor([0.5, 0.5]), 1, "sample-2"),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        inputs, label, sample_id = self.rows[index]
        return {
            "input": inputs,
            "label": label,
            "sample_id": sample_id,
            "split": "val",
            "source_id": f"source-{index}",
        }


class FixedModel(nn.Module):
    def forward(self, inputs):
        return torch.stack((inputs[:, 0], inputs[:, 1]), dim=1)


class Phase4EvaluationTest(unittest.TestCase):
    def test_evaluate_records_predictions_per_class_and_confusion(self):
        evaluation = evaluate_classification(
            FixedModel(),
            DataLoader(DictDataset(), batch_size=2),
            class_names=("red", "green"),
            split="val",
        )

        self.assertEqual(evaluation.summary.total_examples, 3)
        self.assertEqual(evaluation.summary.confusion_matrix, [[1, 0], [1, 1]])
        self.assertEqual(evaluation.summary.per_class["green"]["total"], 2)
        self.assertEqual(evaluation.predictions[0].sample_id, "sample-0")
        self.assertEqual(evaluation.predictions[0].split, "val")
        self.assertIn(evaluation.predictions[0].predicted_label, {"red", "green"})
        self.assertGreaterEqual(evaluation.predictions[0].confidence, 0.0)
        self.assertLessEqual(evaluation.predictions[0].confidence, 1.0)

    def test_writes_machine_readable_evaluation_and_history_artifacts(self):
        evaluation = evaluate_classification(
            FixedModel(),
            DataLoader(DictDataset(), batch_size=3),
            class_names=("red", "green"),
            split="val",
        )
        metadata = TrainingRunMetadata(
            run_id="phase4a-test",
            config={"selection_metric": "val_loss"},
            seed=5,
            environment={"device": "cpu"},
            status="completed",
            epoch_history=(
                EpochMetrics(
                    epoch=1,
                    train_loss=1.0,
                    train_accuracy=0.5,
                    val_loss=0.9,
                    val_accuracy=0.66,
                    learning_rate=0.01,
                ),
            ),
            checkpoint_references={"best": "best.pt"},
            stop_reason="max_epochs_reached",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            eval_paths = write_evaluation_artifacts(evaluation, output_dir, prefix="val")
            history_paths = write_history_artifacts(metadata, output_dir)

            summary = json.loads(Path(eval_paths["summary"]).read_text(encoding="utf-8"))
            with Path(eval_paths["predictions"]).open(encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))
            history = json.loads(Path(history_paths["history"]).read_text(encoding="utf-8"))

            self.assertEqual(summary["split"], "val")
            self.assertEqual(prediction_rows[0]["sample_id"], "sample-0")
            self.assertEqual(history[0]["epoch"], 1)
            self.assertTrue(Path(history_paths["curve_data"]).exists())


if __name__ == "__main__":
    unittest.main()
