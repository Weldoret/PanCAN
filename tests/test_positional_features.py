import unittest

import torch

from models.network import RotatedGridPositionFeatures


class RotatedGridPositionFeaturesTest(unittest.TestCase):
    def test_preserves_shape_and_changes_by_grid_position(self):
        module = RotatedGridPositionFeatures(rows=2, cols=2, feature_dim=4)
        features = torch.ones(1, 4, 4)

        encoded = module(features)

        self.assertEqual(encoded.shape, features.shape)
        self.assertTrue(torch.allclose(encoded[:, 0], features[:, 0]))
        self.assertFalse(torch.allclose(encoded[:, 0], encoded[:, 1]))


if __name__ == "__main__":
    unittest.main()
