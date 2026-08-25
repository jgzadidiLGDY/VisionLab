import unittest

from visionlab.evaluation.metrics import (
    binary_average_precision,
    binary_roc_auc,
    classification_metrics_from_predictions,
)


class Phase7MetricsTest(unittest.TestCase):
    def test_confusion_metrics_macro_weighted_and_micro(self):
        metrics = classification_metrics_from_predictions(
            true_indices=[0, 0, 1, 1, 1, 2],
            predicted_indices=[0, 1, 1, 1, 2, 1],
            probabilities=[
                [0.8, 0.1, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
                [0.2, 0.5, 0.3],
            ],
            class_names=("a", "b", "c"),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 1], [0, 1, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 3 / 6)
        self.assertAlmostEqual(metrics["per_class"]["a"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["b"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["b"]["recall"], 2 / 3)
        self.assertEqual(metrics["per_class"]["c"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["c"]["recall"], 0.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (0.5 + 2 / 3 + 0.0) / 3)
        self.assertAlmostEqual(metrics["averages"]["micro"]["f1"], 0.5)
        self.assertAlmostEqual(
            metrics["averages"]["f1"]["weighted"],
            ((2 / 3) * 2 + (4 / 7) * 3 + 0.0 * 1) / 6,
        )

    def test_zero_predicted_or_actual_support_is_explicitly_undefined(self):
        metrics = classification_metrics_from_predictions(
            true_indices=[0, 0, 1],
            predicted_indices=[0, 0, 0],
            probabilities=[
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
            ],
            class_names=("a", "b", "c"),
        )

        self.assertIsNone(metrics["per_class"]["b"]["precision"])
        self.assertIsNone(metrics["per_class"]["c"]["precision"])
        self.assertIsNone(metrics["per_class"]["c"]["recall"])
        self.assertLess(len(metrics["warnings"]), 10)

    def test_binary_auc_definitions_are_pinned(self):
        labels = [0, 1, 1, 0]
        scores = [0.1, 0.4, 0.35, 0.8]

        self.assertAlmostEqual(binary_roc_auc(labels, scores).value, 0.5)
        self.assertAlmostEqual(binary_average_precision(labels, scores).value, (1 / 2 + 2 / 3) / 2)

    def test_auc_tied_scores_are_grouped_not_order_dependent(self):
        labels = [1, 0]
        scores = [0.5, 0.5]

        self.assertAlmostEqual(binary_roc_auc(labels, scores).value, 0.5)
        self.assertAlmostEqual(binary_average_precision(labels, scores).value, 0.5)
    def test_auc_undefined_cases_return_warning_not_zero(self):
        roc = binary_roc_auc([1, 1], [0.2, 0.8])
        pr = binary_average_precision([0, 0], [0.2, 0.8])

        self.assertIsNone(roc.value)
        self.assertIn("undefined", roc.warnings[0])
        self.assertIsNone(pr.value)
        self.assertIn("undefined", pr.warnings[0])


if __name__ == "__main__":
    unittest.main()
