import unittest

import torch

from models.random_walk import RandomWalkAttention


class RandomWalkAttentionTest(unittest.TestCase):
    def test_accepts_dense_directional_adjacency_and_backpropagates(self):
        features = torch.randn(2, 4, 16, requires_grad=True)
        adjacency = [
            torch.tensor(
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                dtype=torch.float32,
            ),
        ]
        module = RandomWalkAttention(
            feature_dim=16,
            num_heads=4,
            dropout=0.0,
            threshold=0.71,
            num_directions=2,
        )

        output = module(features, features, adjacency)
        self.assertEqual(output.shape, features.shape)
        self.assertTrue(torch.isfinite(output).all())

        output.mean().backward()
        self.assertIsNotNone(features.grad)

    def test_accepts_index_adjacency(self):
        features = torch.randn(1, 3, 8)
        adjacency_index = torch.tensor([[1, -1], [0, 2], [1, -1]])
        module = RandomWalkAttention(
            feature_dim=8,
            num_heads=2,
            dropout=0.0,
            use_threshold=False,
            num_directions=2,
        )

        output = module(features, features, adjacency_index)
        self.assertEqual(output.shape, features.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_matches_scaled_dot_product_random_walk_for_one_order(self):
        features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
        )
        adjacency = [
            torch.tensor(
                [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                dtype=torch.float32,
            )
        ]
        module = RandomWalkAttention(
            feature_dim=2,
            num_heads=1,
            dropout=0.0,
            use_threshold=False,
            max_order=1,
            num_directions=1,
        )

        transition = module.transition_calculators[0]
        value_projection = module.value_projections[0]
        with torch.no_grad():
            for projection in (transition.query_proj, transition.key_proj,
                               value_projection):
                projection.weight.copy_(torch.eye(2))
                projection.bias.zero_()
            module.dimension_reduction.weight.zero_()
            module.dimension_reduction.bias.zero_()
            module.dimension_reduction.weight[:, 2:, 0].copy_(torch.eye(2))

        output = module(features, features, adjacency)

        scores = torch.tensor([0.0, 1.0]) / (2.0 ** 0.5)
        expected_node_zero = torch.softmax(scores, dim=0) @ features[0, [1, 2]]
        expected = torch.stack([
            expected_node_zero,
            features[0, 2],
            torch.zeros(2),
        ]).unsqueeze(0)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
