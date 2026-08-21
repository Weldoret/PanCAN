import unittest

import torch

from models.neighborhood import NeighborhoodSystem


class DirectionalNeighborhoodTest(unittest.TestCase):
    def test_paper_directions_have_expected_orientation_and_boundaries(self):
        system = NeighborhoodSystem(3, 3)
        up, down, left, right = system.adjacency_matrices

        self.assertEqual(system.directions, [(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.assertEqual(up[4, 1], 1)
        self.assertEqual(down[4, 7], 1)
        self.assertEqual(left[4, 3], 1)
        self.assertEqual(right[4, 5], 1)
        self.assertEqual(up[1].sum(), 0)
        self.assertEqual(left[3].sum(), 0)
        self.assertTrue(torch.equal(up, down.T))
        self.assertTrue(torch.equal(left, right.T))

    def test_direction_types_are_not_degree_normalized(self):
        system = NeighborhoodSystem(3, 3)
        self.assertTrue(torch.equal(
            system.adjacency_weight[4], torch.ones(4)
        ))

    def test_higher_orders_follow_equation_five_per_direction(self):
        system = NeighborhoodSystem(5, 5)
        center = 2 * 5 + 2

        self.assertEqual(system.get_directional_neighbors(center, 0, 1), [7])
        self.assertEqual(system.get_directional_neighbors(center, 0, 2), [2])
        self.assertEqual(system.get_directional_neighbors(center, 0, 3), [])
        self.assertEqual(
            system.get_higher_order_neighbors(center, 2),
            [2, 10, 14, 22],
        )


if __name__ == "__main__":
    unittest.main()
