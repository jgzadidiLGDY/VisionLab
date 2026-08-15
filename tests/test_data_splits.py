import unittest

from visionlab.data.splits import stratified_validation_indices


class StratifiedSplitTest(unittest.TestCase):
    def test_selects_fixed_count_per_class(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        selected = stratified_validation_indices(labels, validation_per_class=1, seed=7)

        self.assertEqual(len(selected), 3)
        self.assertEqual({labels[index] for index in selected}, {0, 1, 2})

    def test_is_deterministic_for_same_seed(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]

        first = stratified_validation_indices(labels, validation_per_class=2, seed=42)
        second = stratified_validation_indices(labels, validation_per_class=2, seed=42)

        self.assertEqual(first, second)

    def test_changes_when_seed_changes(self):
        labels = [0] * 20 + [1] * 20

        first = stratified_validation_indices(labels, validation_per_class=5, seed=1)
        second = stratified_validation_indices(labels, validation_per_class=5, seed=2)

        self.assertNotEqual(first, second)

    def test_rejects_too_large_validation_count(self):
        with self.assertRaisesRegex(ValueError, "cannot select"):
            stratified_validation_indices([0, 0, 1], validation_per_class=2, seed=1)


if __name__ == "__main__":
    unittest.main()
