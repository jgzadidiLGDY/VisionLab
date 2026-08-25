import csv
import json
import tempfile
import unittest
from pathlib import Path

from visionlab.evaluation import PredictionRecord
from visionlab.experiments.phase7 import Phase7SplitResult, verify_phase7_sample_alignment
from visionlab.evaluation.classification import ClassificationEvaluation, ClassificationSummary, write_evaluation_artifacts


class Phase7PlumbingTest(unittest.TestCase):
    def test_prediction_export_preserves_logits_and_probabilities(self):
        evaluation = ClassificationEvaluation(
            summary=ClassificationSummary(
                split="val",
                total_examples=1,
                loss=0.2,
                accuracy=1.0,
                per_class={"a": {"total": 1, "correct": 1, "accuracy": 1.0}},
                confusion_matrix=[[1]],
            ),
            predictions=(
                PredictionRecord(
                    sample_id="sample-1",
                    split="val",
                    true_label="a",
                    predicted_label="a",
                    confidence=0.75,
                    correct=True,
                    source_id="source-1",
                    true_index=0,
                    predicted_index=0,
                    logits=(1.0,),
                    probabilities=(0.75,),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_evaluation_artifacts(evaluation, Path(tmp), prefix="val")
            with Path(paths["predictions"]).open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["true_index"], "0")
        self.assertEqual(json.loads(rows[0]["logits"]), [1.0])
        self.assertEqual(json.loads(rows[0]["probabilities"]), [0.75])

    def test_sample_alignment_passes_for_identical_split_records(self):
        result_a = Phase7SplitResult(
            run_id="a",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s2"),
            true_labels=("cat", "dog"),
        )
        result_b = Phase7SplitResult(
            run_id="b",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s2"),
            true_labels=("cat", "dog"),
        )

        report = verify_phase7_sample_alignment({"test": [result_a, result_b]})

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["splits"]["test"]["sample_ids_identical"])

    def test_sample_alignment_fails_for_mismatched_ids(self):
        result_a = Phase7SplitResult(
            run_id="a",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s2"),
            true_labels=("cat", "dog"),
        )
        result_b = Phase7SplitResult(
            run_id="b",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s3"),
            true_labels=("cat", "dog"),
        )

        with self.assertRaises(ValueError):
            verify_phase7_sample_alignment({"test": [result_a, result_b]})

    def test_sample_alignment_fails_for_mismatched_labels(self):
        result_a = Phase7SplitResult(
            run_id="a",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s2"),
            true_labels=("cat", "dog"),
        )
        result_b = Phase7SplitResult(
            run_id="b",
            split="test",
            artifacts={},
            summary={},
            sample_ids=("s1", "s2"),
            true_labels=("cat", "horse"),
        )

        with self.assertRaises(ValueError):
            verify_phase7_sample_alignment({"test": [result_a, result_b]})


if __name__ == "__main__":
    unittest.main()
