import unittest

import torch

from visionlab.data import (
    phase6b_preprocessing_contract_dict,
    preprocess_resnet18_imagenet_tensor,
)
from visionlab.models import PHASE6A_PREPROCESSING_ID, RESNET18_WEIGHT_ENUM


class Phase6BPreprocessingTest(unittest.TestCase):
    def test_preprocessing_uses_selected_weight_contract_and_outputs_resnet_shape(self):
        image = torch.zeros(3, 32, 32)
        image[0, :, :] = 1.0

        tensor = preprocess_resnet18_imagenet_tensor(image)

        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_preprocessing_is_deterministic_for_same_input(self):
        image = torch.rand(3, 32, 32)

        first = preprocess_resnet18_imagenet_tensor(image)
        second = preprocess_resnet18_imagenet_tensor(image)

        self.assertTrue(torch.equal(first, second))

    def test_preprocessing_contract_records_phase6b_actual_transform_source(self):
        contract = phase6b_preprocessing_contract_dict()

        self.assertEqual(contract["profile_id"], PHASE6A_PREPROCESSING_ID)
        self.assertEqual(contract["source"], RESNET18_WEIGHT_ENUM)
        self.assertEqual(contract["resize_size"], 256)
        self.assertEqual(contract["crop_size"], 224)
        self.assertEqual(contract["input_size"], [224, 224])
        self.assertEqual(contract["normalization_mean"], [0.485, 0.456, 0.406])
        self.assertEqual(contract["normalization_std"], [0.229, 0.224, 0.225])
        self.assertEqual(
            contract["actual_transform_source"],
            "ResNet18_Weights.IMAGENET1K_V1.transforms()",
        )
        self.assertTrue(contract["phase6b_verified_application"])


if __name__ == "__main__":
    unittest.main()
