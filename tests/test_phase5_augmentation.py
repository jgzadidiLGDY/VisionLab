import unittest

import torch
from torch.utils.data import Dataset

from visionlab.data.augmentation import (
    PHASE4_NO_AUGMENTATION_PROFILE,
    PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
    apply_augmentation_profile,
    get_augmentation_profile,
    profile_registry_dict,
)
from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    VisionLabSplitDataset,
    build_cifar10_split_datasets,
)


class TinyCIFARLikeDataset(Dataset):
    classes = CIFAR10_CLASSES

    def __init__(self):
        self.targets = list(range(len(self.classes))) * 3
        self.images = []
        for index, label in enumerate(self.targets):
            image = torch.zeros(3, 32, 32)
            image[label % 3, :, :] = 0.25
            image[:, 8:24, 8:24] = float(index % 5) / 4.0
            self.images.append(image)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class Phase5AugmentationTest(unittest.TestCase):
    def test_profile_registry_is_machine_readable_and_versioned(self):
        registry = profile_registry_dict()

        self.assertEqual(registry["registry_id"], "visionlab-phase5a-augmentation-profiles")
        profiles = {profile["profile_id"]: profile for profile in registry["profiles"]}
        self.assertEqual(profiles["phase4-control-no-augmentation"]["version"], "1.0")
        candidate = profiles["phase5a-candidate-horizontal-flip-random-crop"]
        self.assertEqual(candidate["version"], "1.0")
        self.assertEqual(
            candidate["transforms"],
            [
                {
                    "name": "random_horizontal_flip",
                    "parameters": {"probability": 0.5},
                },
                {
                    "name": "random_crop_with_padding",
                    "parameters": {
                        "output_size": [32, 32],
                        "padding": 4,
                        "padding_mode": "constant",
                        "fill": 0.0,
                    },
                },
            ],
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown augmentation profile"):
            get_augmentation_profile("missing-profile")

    def test_no_augmentation_control_preserves_tensor_values(self):
        tensor = torch.rand(3, 32, 32)

        augmented = apply_augmentation_profile(tensor, PHASE4_NO_AUGMENTATION_PROFILE)

        self.assertTrue(torch.equal(augmented, tensor))
        self.assertIsNot(augmented, tensor)

    def test_candidate_profile_preserves_cifar_shape_and_unit_range(self):
        tensor = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(3, 32, 32)
        generator = torch.Generator().manual_seed(20260819)

        augmented = apply_augmentation_profile(
            tensor,
            PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
            generator=generator,
        )

        self.assertEqual(tuple(augmented.shape), (3, 32, 32))
        self.assertGreaterEqual(float(augmented.min().item()), 0.0)
        self.assertLessEqual(float(augmented.max().item()), 1.0)

    def test_augmentation_profile_is_train_only_for_registered_splits(self):
        upstream = TinyCIFARLikeDataset()

        with self.assertRaisesRegex(ValueError, "train split"):
            VisionLabSplitDataset(
                upstream,
                split="val",
                upstream_partition="train",
                indices=[0, 1],
                augmentation_profile=PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
            )

    def test_train_can_use_augmentation_while_eval_splits_remain_deterministic(self):
        bundle = build_cifar10_split_datasets(
            upstream_train=TinyCIFARLikeDataset(),
            upstream_test=TinyCIFARLikeDataset(),
            validation_per_class=1,
            train_augmentation_profile=PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
        )

        train_sample = bundle.train[0]
        first_val = bundle.val[0]
        second_val = bundle.val[0]
        first_test = bundle.test[0]
        second_test = bundle.test[0]

        self.assertEqual(tuple(train_sample["input"].shape), (3, 32, 32))
        self.assertGreaterEqual(float(train_sample["input"].min().item()), -1.0)
        self.assertLessEqual(float(train_sample["input"].max().item()), 1.0)
        self.assertTrue(torch.equal(first_val["input"], second_val["input"]))
        self.assertTrue(torch.equal(first_test["input"], second_test["input"]))
        self.assertIs(bundle.val.augmentation_profile, None)
        self.assertIs(bundle.test.augmentation_profile, None)


if __name__ == "__main__":
    unittest.main()
