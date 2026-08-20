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
        expected = 0.5 * features
        self.assertTrue(torch.allclose(with_context, expected))
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
        self.assertEqual(learned[0].shape, (2, 2))

        with torch.no_grad():
            module.neighborhood_residuals[0, 0, 0, 1] = 0.5
            module.neighborhood_residuals[0, 0, 0, 0] = 0.5
        learned = module.get_adjacency_matrices(adjacency)
        self.assertEqual(learned[0][0, 1], 0)
        self.assertEqual(learned[0][0, 0], 1.5)

        features = torch.randn(1, 2, 2, requires_grad=True)
        _, mapped = module(features, adjacency)
        mapped.sum().backward()
        self.assertIsNotNone(module.neighborhood_residuals.grad)

    def test_learned_neighborhoods_preserve_spatial_support(self):
        module = ContextAwareKernelMap(
            feature_dim=2,
            kernel_dim=2,
            num_directions=1,
            num_layers=2,
            num_nodes=3,
        )
        adjacency = [torch.eye(3)]
        features = torch.randn(1, 3, 2, requires_grad=True)

        _, mapped = module(features, adjacency)
        mapped.sum().backward()

        gradients = module.neighborhood_residuals.grad
        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, (2, 1, 3, 3))
        self.assertTrue(torch.isfinite(gradients).all())
        support = adjacency[0].bool()
        self.assertGreater(gradients[:, 0, support].abs().sum(), 0)
        self.assertEqual(gradients[:, 0, ~support].abs().sum(), 0)

    def test_fixed_point_solver_accepts_batched_similarity_matrices(self):
        from models.context_kernel import ContextAwareKernel

        solver = ContextAwareKernel(
            alpha=0.25,
            beta=1.0,
            num_directions=1,
            max_iterations=1,
        )
        similarity = torch.eye(2).repeat(3, 1, 1)
        adjacency = [torch.eye(2)]

        optimized, _ = solver(similarity, adjacency)

        self.assertEqual(optimized.shape, (3, 2, 2))
        self.assertTrue(torch.allclose(optimized, similarity * 1.25))


if __name__ == "__main__":
    unittest.main()
