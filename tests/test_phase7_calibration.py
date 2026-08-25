import unittest

from visionlab.evaluation.calibration import calibration_summary, confidence_histogram


class Phase7CalibrationTest(unittest.TestCase):
    def test_ece_bin_boundaries_and_empty_bins_are_explicit(self):
        summary = calibration_summary(
            [0.0, 0.2, 0.399, 0.4, 0.8, 1.0],
            [False, True, True, False, True, True],
            num_bins=5,
        )

        self.assertEqual([item.count for item in summary.bins], [1, 2, 1, 0, 2])
        self.assertEqual(summary.bins[3].accuracy, None)
        self.assertEqual(summary.bins[3].average_confidence, None)
        self.assertAlmostEqual(summary.bins[1].average_confidence, (0.2 + 0.399) / 2)
        self.assertAlmostEqual(summary.bins[2].accuracy, 0.0)
        self.assertAlmostEqual(summary.bins[4].accuracy, 1.0)
        expected_ece = (
            1 / 6 * abs(0.0 - 0.0)
            + 2 / 6 * abs(1.0 - ((0.2 + 0.399) / 2))
            + 1 / 6 * abs(0.0 - 0.4)
            + 2 / 6 * abs(1.0 - 0.9)
        )
        self.assertAlmostEqual(summary.expected_calibration_error, expected_ece)

    def test_confidence_histogram_splits_correct_and_incorrect(self):
        rows = confidence_histogram(
            [0.1, 0.1, 0.5, 1.0],
            [True, False, False, True],
            num_bins=2,
        )

        self.assertEqual(rows[0]["correct_count"], 1)
        self.assertEqual(rows[0]["incorrect_count"], 1)
        self.assertEqual(rows[1]["correct_count"], 1)
        self.assertEqual(rows[1]["incorrect_count"], 1)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            calibration_summary([1.01], [True])


if __name__ == "__main__":
    unittest.main()
