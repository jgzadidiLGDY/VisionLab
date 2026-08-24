import unittest

import torch
from torch import nn

from visionlab.models import (
    EXPECTED_PHASE6A_PARAMETER_COUNTS,
    PHASE6A_PREPROCESSING_ID,
    RESNET18_ARCHITECTURE,
    RESNET18_WEIGHT_ENUM,
    ResNet18PreprocessingContract,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
    transfer_parameter_summary,
)


class TransferModelTest(unittest.TestCase):
    def test_config_binds_exact_resnet18_imagenet1k_v1_identity(self):
        config = TransferModelConfig()

        self.assertEqual(config.architecture, RESNET18_ARCHITECTURE)
        self.assertEqual(config.weight_identity, RESNET18_WEIGHT_ENUM)
        self.assertNotIn("DEFAULT", config.weight_identity)
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.input_size, (224, 224))
        self.assertEqual(config.preprocessing_id, PHASE6A_PREPROCESSING_ID)

    def test_config_rejects_unapproved_backbone_or_weight_identity(self):
        with self.assertRaisesRegex(ValueError, "resnet18"):
            TransferModelConfig(architecture="torchvision.models.mobilenet_v3_small")
        with self.assertRaisesRegex(ValueError, "IMAGENET1K_V1"):
            TransferModelConfig(weight_identity="ResNet18_Weights.DEFAULT")

    def test_preprocessing_contract_records_resnet18_imagenet_requirements(self):
        contract = ResNet18PreprocessingContract().to_dict()

        self.assertEqual(contract["profile_id"], PHASE6A_PREPROCESSING_ID)
        self.assertEqual(contract["resize_size"], 256)
        self.assertEqual(contract["crop_size"], 224)
        self.assertEqual(contract["input_size"], [224, 224])
        self.assertEqual(contract["interpolation"], "bilinear")
        self.assertEqual(contract["normalization_mean"], [0.485, 0.456, 0.406])
        self.assertEqual(contract["normalization_std"], [0.229, 0.224, 0.225])
        self.assertTrue(contract["separate_from_phase4b_custom_cnn_preprocessing"])

    def test_model_replaces_classifier_with_512_to_10_head_and_returns_logits(self):
        model = build_phase6a_transfer_model(load_pretrained=False)

        self.assertIsInstance(model.model.fc, nn.Linear)
        self.assertEqual(model.model.fc.in_features, 512)
        self.assertEqual(model.model.fc.out_features, 10)
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))

        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_model_rejects_invalid_input_contract(self):
        model = build_phase6a_transfer_model(load_pretrained=False)

        with self.assertRaisesRegex(ValueError, "rank 3"):
            model(torch.randn(3, 224, 224))
        with self.assertRaisesRegex(ValueError, "expected 3 input channels"):
            model(torch.randn(2, 1, 224, 224))
        with self.assertRaisesRegex(ValueError, "expected spatial size"):
            model(torch.randn(2, 3, 32, 32))

    def test_parameter_counts_and_frozen_trainable_summary_match_contract(self):
        model = build_phase6a_transfer_model(load_pretrained=False)

        summary = transfer_parameter_summary(model)

        self.assertEqual(summary["counts"], EXPECTED_PHASE6A_PARAMETER_COUNTS)
        self.assertEqual(summary["groups"]["trainable"], ["model.fc.weight", "model.fc.bias"])
        self.assertIn("conv1", summary["groups"]["frozen_prefixes"])
        self.assertIn("layer4", summary["groups"]["frozen_prefixes"])

    def test_frozen_backbone_does_not_update_and_head_does_update(self):
        torch.manual_seed(7)
        model = build_phase6a_transfer_model(load_pretrained=False)
        model.train()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.05,
        )
        inputs = torch.zeros(2, 3, 224, 224)
        inputs[0, 0, :, :] = 1.0
        inputs[1, 1, :, :] = 1.0
        labels = torch.tensor([0, 1])

        optimizer.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(model(inputs), labels)
        loss.backward()

        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.parameters()
                if not parameter.requires_grad
            )
        )
        optimizer.step()

        frozen_unchanged = all(
            torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        )
        head_changed = any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.assertTrue(frozen_unchanged)
        self.assertTrue(head_changed)

    def test_cache_probe_does_not_require_weighted_model_construction(self):
        probe = probe_resnet18_weight_cache()

        self.assertEqual(probe.weight_identity, RESNET18_WEIGHT_ENUM)
        self.assertEqual(probe.expected_filename, "resnet18-f37072fd.pth")
        self.assertFalse(probe.to_dict()["download_attempted"])


if __name__ == "__main__":
    unittest.main()
