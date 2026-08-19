import unittest

import torch
from torch.utils.data import Dataset

from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    DataLoaderPolicy,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
    verify_material_cifar10_contract,
)


class TinyCIFARLikeDataset(Dataset):
    classes = CIFAR10_CLASSES

    def __init__(self, repeats_per_class):
        self.targets = []
        self.images = []
        for label in range(len(self.classes)):
            for _ in range(repeats_per_class):
                image = torch.zeros(3, 32, 32)
                image[label % 3, :, :] = 1.0
                self.targets.append(label)
                self.images.append(image)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class TargetOnlyCIFARLikeDataset(Dataset):
    classes = CIFAR10_CLASSES

    def __init__(self, repeats_per_class):
        self.targets = [
            label
            for label in range(len(self.classes))
            for _ in range(repeats_per_class)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        raise AssertionError("preflight count verification should not load images")


class Phase4DataLoaderTest(unittest.TestCase):
    def test_registered_split_policy_keeps_test_partition_separate(self):
        bundle = build_cifar10_split_datasets(
            upstream_train=TinyCIFARLikeDataset(repeats_per_class=3),
            upstream_test=TinyCIFARLikeDataset(repeats_per_class=1),
            validation_per_class=1,
        )

        self.assertEqual(bundle.split_counts(), {"train": 20, "val": 10, "test": 10})
        self.assertTrue(
            all(sample_id.startswith("cifar10-test-") for sample_id in _sample_ids(bundle.test))
        )
        self.assertTrue(
            all(sample_id.startswith("cifar10-train-") for sample_id in _sample_ids(bundle.val))
        )
        self.assertEqual(set(bundle.train.indices).isdisjoint(bundle.val.indices), True)

    def test_loader_policy_shuffles_train_and_preserves_eval_order(self):
        bundle = build_cifar10_split_datasets(
            upstream_train=TinyCIFARLikeDataset(repeats_per_class=3),
            upstream_test=TinyCIFARLikeDataset(repeats_per_class=1),
            validation_per_class=1,
        )
        policy = DataLoaderPolicy(batch_size=4, seed=11, num_workers=0)

        first = build_phase4_dataloaders(bundle, policy)
        second = build_phase4_dataloaders(bundle, policy)
        first_train_labels = next(iter(first.train))[1].tolist()
        second_train_labels = next(iter(second.train))[1].tolist()
        first_val_batch = next(iter(first.prediction_val))
        second_val_batch = next(iter(second.prediction_val))

        self.assertEqual(first_train_labels, second_train_labels)
        self.assertEqual(first_val_batch["sample_id"], second_val_batch["sample_id"])
        self.assertEqual(first_val_batch["split"], ["val"] * len(first_val_batch["split"]))

    def test_rejects_wrong_class_order(self):
        dataset = TinyCIFARLikeDataset(repeats_per_class=1)
        dataset.classes = tuple(reversed(CIFAR10_CLASSES))

        with self.assertRaisesRegex(ValueError, "class order"):
            build_cifar10_split_datasets(
                upstream_train=dataset,
                upstream_test=TinyCIFARLikeDataset(repeats_per_class=1),
                validation_per_class=1,
            )

    def test_material_preflight_verifies_expected_cifar10_counts(self):
        bundle = build_cifar10_split_datasets(
            upstream_train=TargetOnlyCIFARLikeDataset(repeats_per_class=5000),
            upstream_test=TargetOnlyCIFARLikeDataset(repeats_per_class=1000),
        )

        report = verify_material_cifar10_contract(bundle)

        self.assertEqual(
            report["split_counts"],
            {"train": 45000, "val": 5000, "test": 10000},
        )
        self.assertEqual(report["classes"], list(CIFAR10_CLASSES))
        self.assertEqual(report["test_partition"], "test")

    def test_material_preflight_rejects_unexpected_counts(self):
        bundle = build_cifar10_split_datasets(
            upstream_train=TargetOnlyCIFARLikeDataset(repeats_per_class=3),
            upstream_test=TargetOnlyCIFARLikeDataset(repeats_per_class=1),
            validation_per_class=1,
        )

        with self.assertRaisesRegex(ValueError, "split counts"):
            verify_material_cifar10_contract(bundle)


def _sample_ids(dataset):
    return [dataset[index]["sample_id"] for index in range(len(dataset))]


if __name__ == "__main__":
    unittest.main()
