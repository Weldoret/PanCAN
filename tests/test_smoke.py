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
    def test_paper_orders_are_wired_per_scale(self):
        config = PanCAN.NetworkConfig(
            backbone_name="resnet34",
            backbone_feature_dim=4,
            grid_rows=4,
            grid_cols=5,
            num_classes=3,
            max_order=2,
            coarse_max_order=1,
            attention_heads=1,
            scales=[(4, 5), (2, 3), (1, 2), (1, 1)],
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

        self.assertEqual(model.random_walk.max_order, 2)
        self.assertEqual(
            [block.random_walk.max_order for block in model.coarse_scale_context],
            [1, 1, 2],
        )

    def test_eight_by_ten_keeps_second_order_at_four_by_five(self):
        config = PanCAN.NetworkConfig(
            backbone_name="resnet34",
            backbone_feature_dim=4,
            grid_rows=8,
            grid_cols=10,
            num_classes=3,
            max_order=2,
            coarse_max_order=1,
            attention_heads=1,
            kernel_feature_dims=[8],
            final_feature_dim=8,
        )
        model = PanCAN.MultiScaleContextAwareNetwork(
            backbone=TinyBackbone(), num_classes=3, config=config,
            device=torch.device("cpu"),
        )
        self.assertEqual(
            [block.random_walk.max_order for block in model.coarse_scale_context],
            [2, 1, 1, 2],
        )

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
        scale_inputs = []
        model.multi_scale_aggregator.register_forward_pre_hook(
            lambda _module, inputs: scale_inputs.append(inputs[0].shape)
        )
        coarse_context_inputs = []
        model.coarse_scale_context[0].register_forward_pre_hook(
            lambda _module, inputs: coarse_context_inputs.append(inputs[0].shape)
        )

        with torch.no_grad():
            model.context_kernel.neighborhood_residuals[-1, 0, 0, 3] = 0.5

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 8, 8))

        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(hasattr(model, "deep_kernel_network"))
        self.assertEqual(scale_inputs, [torch.Size((2, 4, 8))])
        self.assertEqual(coarse_context_inputs, [torch.Size((2, 1, 8))])


if __name__ == "__main__":
    unittest.main()
