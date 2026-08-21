import unittest
import math

import torch

from models.random_walk import (
    MultiOrderContextMappingNetwork,
    RandomWalkAttention,
)


class RandomWalkAttentionTest(unittest.TestCase):
    def test_threshold_keeps_original_transition_probability(self):
        attention_features = torch.tensor(
            [[[math.sqrt(2.0), 0.0], [math.log(4.0), 0.0], [0.0, 0.0]]]
        )
        module = RandomWalkAttention(
            feature_dim=2,
            dropout=0.0,
            threshold=0.6,
            max_order=1,
            num_directions=1,
        )

        with torch.no_grad():
            transition = module.transition_calculators[0]
            transition.query_proj.weight.copy_(torch.eye(2))
            transition.query_proj.bias.zero_()
            transition.key_proj.weight.copy_(torch.eye(2))
            transition.key_proj.bias.zero_()
            module.value_projections[0].weight.copy_(torch.eye(2))
            module.value_projections[0].bias.zero_()

        aggregated = module._aggregate_order(
            attention_features, [[1, 2], [], []], order=1
        )

        self.assertTrue(torch.allclose(
            aggregated[0, 0],
            torch.tensor([0.8 * math.log(4.0), 0.0]),
            atol=1e-6,
        ))

    def test_threshold_can_select_no_neighbor(self):
        attention_features = torch.zeros(1, 3, 2)
        module = RandomWalkAttention(
            feature_dim=2,
            dropout=0.0,
            threshold=0.6,
            max_order=1,
            num_directions=1,
        )

        with torch.no_grad():
            transition = module.transition_calculators[0]
            transition.query_proj.weight.copy_(torch.eye(2))
            transition.query_proj.bias.zero_()
            transition.key_proj.weight.copy_(torch.eye(2))
            transition.key_proj.bias.zero_()
            module.value_projections[0].weight.copy_(torch.eye(2))
            module.value_projections[0].bias.zero_()

        aggregated = module._aggregate_order(
            attention_features, [[1, 2], [], []], order=1
        )

        self.assertTrue(torch.equal(aggregated[0, 0], torch.zeros(2)))

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

        output = module(features, adjacency)
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

        output = module(features, adjacency_index)
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

        output = module(features, adjacency)

        expected = torch.stack([
            features[0, 2],
            torch.zeros(2),
            torch.zeros(2),
        ]).unsqueeze(0)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_third_order_uses_papers_recursive_neighborhood(self):
        neighbors = [[1], [2], [3], [4], []]
        self.assertEqual(
            RandomWalkAttention._get_order_neighbors(neighbors, 0, 3),
            [4],
        )


class MultiOrderContextMappingNetworkTest(unittest.TestCase):
    def test_every_context_layer_runs_rwca_with_independent_parameters(self):
        module = MultiOrderContextMappingNetwork(
            input_dim=2,
            feature_dim=2,
            num_nodes=3,
            num_layers=3,
            max_order=1,
            num_directions=1,
            threshold=0.0,
            dropout=0.0,
        )
        calls = []
        layer_adjacencies = []
        for index, layer in enumerate(module.layers):
            layer.register_forward_hook(
                lambda _module, _inputs, _output, index=index: calls.append(index)
            )
            layer.register_forward_pre_hook(
                lambda _module, inputs: layer_adjacencies.append(
                    inputs[1][0].detach().clone()
                )
            )

        adjacency = [torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )]
        features = torch.randn(1, 3, 2, requires_grad=True)
        with torch.no_grad():
            for index in range(3):
                module.neighborhood_residuals[index, 0, 0, 1] = index + 1
        gram, output = module(features, adjacency)
        output.sum().backward()

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(gram.shape, (1, 3, 3))
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(
            [float(matrix[0, 1]) for matrix in layer_adjacencies],
            [2.0, 3.0, 4.0],
        )
        self.assertIsNot(
            module.layers[0].transition_calculators[0].query_proj.weight,
            module.layers[1].transition_calculators[0].query_proj.weight,
        )
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_each_layer_reuses_phi_zero_intrinsic_branch(self):
        module = MultiOrderContextMappingNetwork(
            input_dim=2,
            feature_dim=2,
            num_nodes=1,
            num_layers=2,
            max_order=1,
            num_directions=1,
            dropout=0.0,
        )
        with torch.no_grad():
            first, second = module.layers
            first.dimension_reduction.weight.zero_()
            first.dimension_reduction.bias.fill_(7.0)
            second.dimension_reduction.weight.zero_()
            second.dimension_reduction.bias.zero_()
            second.dimension_reduction.weight[:, :2, 0].copy_(torch.eye(2))

        features = torch.tensor([[[1.0, 2.0]]])
        _, output = module(features, [torch.zeros(1, 1)])
        self.assertTrue(torch.equal(output, features))


if __name__ == "__main__":
    unittest.main()
