import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import PanCAN


class TinyBackbone(nn.Module):
    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, self.feature_dim, kernel_size=1)

    def forward(self, images):
        return self.projection(images)


class EndToEndSmokeTest(unittest.TestCase):
    def test_package_import_and_tiny_forward(self):
        config = PanCAN.NetworkConfig(
            backbone_name="resnet34",
            backbone_feature_dim=4,
            grid_rows=2,
            grid_cols=2,
            num_classes=3,
            max_order=1,
            attention_heads=1,
            scales=[(2, 2), (1, 1)],
            anchor_sizes=[(1, 1)],
            kernel_feature_dims=[8],
            final_feature_dim=8,
        )
        model = PanCAN.MultiScaleContextAwareNetwork(
            backbone=TinyBackbone(),
            num_classes=3,
            config=config,
            device=torch.device("cpu"),
        )
        model.eval()
        deep_kernel_inputs = []
        model.deep_kernel_network.register_forward_pre_hook(
            lambda _module, inputs: deep_kernel_inputs.append(inputs[0].shape)
        )
        coarse_context_inputs = []
        model.coarse_scale_context[0].register_forward_pre_hook(
            lambda _module, inputs: coarse_context_inputs.append(inputs[0].shape)
        )

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 8, 8))

        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(deep_kernel_inputs, [torch.Size((2, 4, 8))])
        self.assertEqual(coarse_context_inputs, [torch.Size((2, 1, 8))])


if __name__ == "__main__":
    unittest.main()
