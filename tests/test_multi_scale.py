import unittest

import torch

from models.multi_scale import MultiScaleFeatureAggregator


class CrossScaleGroupingTest(unittest.TestCase):
    def test_paper_hierarchy_uses_overlapping_groups_with_full_coverage(self):
        groups = MultiScaleFeatureAggregator._build_groups(
            8, 10, 4, 5, stride=(2, 2)
        )

        self.assertEqual(len(groups), 20)
        self.assertTrue(all(len(group) == 9 for group in groups))
        self.assertTrue(set(groups[0]).intersection(groups[1]))
        self.assertEqual(
            set().union(*map(set, groups)), set(range(8 * 10))
        )

    def test_reported_scale_hierarchy_is_supported(self):
        hierarchy = [
            ((8, 10), (4, 5)),
            ((4, 5), (2, 3)),
            ((2, 3), (1, 2)),
            ((1, 2), (1, 1)),
        ]

        for (source_rows, source_cols), (target_rows, target_cols) in hierarchy:
            groups = MultiScaleFeatureAggregator._build_groups(
                source_rows,
                source_cols,
                target_rows,
                target_cols,
                stride=2,
            )
            self.assertEqual(len(groups), target_rows * target_cols)
            self.assertEqual(
                set().union(*map(set, groups)),
                set(range(source_rows * source_cols)),
            )

    def test_stride_changes_the_micro_cell_windows(self):
        stride_two = MultiScaleFeatureAggregator._build_groups(
            8, 10, 4, 5, stride=2
        )
        stride_three = MultiScaleFeatureAggregator._build_groups(
            8, 10, 4, 5, stride=3
        )

        self.assertNotEqual(stride_two[0], stride_three[0])

    def test_anchor_stays_inside_its_macro_cell(self):
        groups = MultiScaleFeatureAggregator._build_groups(
            8, 10, 4, 5, stride=2
        )
        features = torch.zeros(1, 80, 4)
        features[0, groups[0][0], 0] = 10.0

        anchors = MultiScaleFeatureAggregator._select_anchor_indices(
            features, groups, 8, 10, suppression_radius=0
        )

        for group_index, group in enumerate(groups):
            self.assertIn(int(anchors[0, group_index]), group)
        self.assertEqual(int(anchors[0, 0]), groups[0][0])

    def test_overlapping_forward_path_backpropagates(self):
        aggregator = MultiScaleFeatureAggregator(
            feature_dim=4,
            scales=[(8, 10), (4, 5)],
            anchor_sizes=[(2, 2)],
            attention_heads=1,
            dropout=0.0,
            stride=2,
        )
        features = torch.randn(2, 80, 4, requires_grad=True)
        global_features = torch.randn(2, 4, requires_grad=True)
        attention_tokens = []
        aggregator.multi_head_attention.register_forward_pre_hook(
            lambda _module, inputs: attention_tokens.append(inputs[1].detach())
        )

        output = aggregator(
            features, (8, 10), global_features=global_features
        )
        output.sum().backward()

        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(global_features.grad).all())
        self.assertTrue(torch.equal(attention_tokens[0][:, -1], global_features.detach()))


if __name__ == "__main__":
    unittest.main()
