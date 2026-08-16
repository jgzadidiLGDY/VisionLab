import unittest

import torch
from torch import nn

from visionlab.models import CustomCNN, CustomCNNConfig, count_parameters


class CustomCNNTest(unittest.TestCase):
    def test_forward_returns_raw_logits_for_cifar10_shape(self):
        model = CustomCNN()
        inputs = torch.randn(4, 3, 32, 32)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_cross_entropy_accepts_forward_logits_and_integer_labels(self):
        model = CustomCNN()
        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 9])

        loss = nn.CrossEntropyLoss()(model(inputs), labels)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))

    def test_intermediate_shapes_are_concise_and_stable(self):
        model = CustomCNN()

        shapes = model.intermediate_shapes(batch_size=2)

        self.assertEqual(
            shapes,
            {
                "input": (2, 3, 32, 32),
                "block1": (2, 32, 16, 16),
                "block2": (2, 64, 8, 8),
                "block3": (2, 128, 4, 4),
                "pooled": (2, 128, 1, 1),
                "flattened": (2, 128),
                "logits": (2, 10),
            },
        )

    def test_parameter_count_is_positive_and_fully_trainable_by_default(self):
        model = CustomCNN()

        counts = count_parameters(model)

        self.assertEqual(counts, {"total": 94538, "trainable": 94538})

    def test_eval_no_grad_forward_runs_on_cpu(self):
        model = CustomCNN().eval()

        with torch.no_grad():
            logits = model(torch.randn(1, 3, 32, 32))

        self.assertEqual(tuple(logits.shape), (1, 10))

    def test_forward_rejects_invalid_rank(self):
        model = CustomCNN()

        with self.assertRaisesRegex(ValueError, "rank 3"):
            model(torch.randn(3, 32, 32))

    def test_forward_rejects_invalid_channel_count(self):
        model = CustomCNN()

        with self.assertRaisesRegex(ValueError, "expected 3 input channels"):
            model(torch.randn(2, 1, 32, 32))

    def test_forward_rejects_invalid_spatial_size(self):
        model = CustomCNN()

        with self.assertRaisesRegex(ValueError, "expected spatial size"):
            model(torch.randn(2, 3, 28, 28))

    def test_config_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            CustomCNNConfig(num_classes=1)
        with self.assertRaisesRegex(ValueError, "feature_channels"):
            CustomCNNConfig(feature_channels=())
        with self.assertRaisesRegex(ValueError, "dropout"):
            CustomCNNConfig(dropout=1.0)

    def test_intermediate_shapes_rejects_invalid_batch_size(self):
        model = CustomCNN()

        with self.assertRaisesRegex(ValueError, "batch_size"):
            model.intermediate_shapes(batch_size=0)


if __name__ == "__main__":
    unittest.main()
