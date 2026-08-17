import unittest

import torch

from models.context_kernel import ContextAwareKernelMap


class ContextAwareKernelMapTest(unittest.TestCase):
    def test_explicit_map_uses_adjacency_and_backpropagates(self):
        module = ContextAwareKernelMap(
            feature_dim=2,
            kernel_dim=2,
            alpha=0.25,
            beta=1.0,
            num_directions=1,
            kernel_type="gaussian",
        )
        module.eval()

        with torch.no_grad():
            layer = module.mapping_layers[0]
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[:, 2:, 0].copy_(torch.eye(2))

        features = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            requires_grad=True,
        )
        zero_adjacency = [torch.zeros(2, 2)]
        identity_adjacency = [torch.eye(2)]

        _, without_context = module(features, zero_adjacency)
        gram, with_context = module(features, identity_adjacency)

        self.assertTrue(torch.allclose(without_context, torch.zeros_like(features)))
        self.assertTrue(torch.allclose(with_context, features * 0.5))
        self.assertEqual(gram.shape, (1, 2, 2))

        with_context.sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(layer.weight.grad)

    def test_each_layer_reuses_the_initial_map(self):
        module = ContextAwareKernelMap(
            feature_dim=2,
            kernel_dim=2,
            alpha=0.25,
            beta=1.0,
            num_directions=1,
            kernel_type="gaussian",
            num_layers=2,
        )
        module.eval()

        with torch.no_grad():
            first, second = module.mapping_layers
            first.weight.zero_()
            first.bias.zero_()
            first.weight[:, 2:, 0].copy_(torch.eye(2))
            second.weight.zero_()
            second.bias.zero_()
            second.weight[:, :2, 0].copy_(torch.eye(2))

        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        _, mapped = module(features, [torch.eye(2)])

        self.assertTrue(torch.allclose(mapped, features))

    def test_learnable_neighborhoods_start_as_the_fixed_grid(self):
        module = ContextAwareKernelMap(
            feature_dim=2,
            kernel_dim=2,
            alpha=0.25,
            beta=1.0,
            num_directions=1,
            kernel_type="gaussian",
            num_nodes=2,
        )

        adjacency = [torch.eye(2)]
        learned = module.get_adjacency_matrices(adjacency)
        self.assertTrue(torch.allclose(learned[0], adjacency[0], atol=1e-6))

        features = torch.randn(1, 2, 2, requires_grad=True)
        _, mapped = module(features, adjacency)
        mapped.sum().backward()
        self.assertIsNotNone(module.neighborhood_logits.grad)


if __name__ == "__main__":
    unittest.main()
