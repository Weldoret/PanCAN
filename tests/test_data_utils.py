import tempfile
import unittest
from pathlib import Path

import torch

from config import DatasetConfig, ExperimentConfig, TrainingConfig
from models.neighborhood import NeighborhoodSystem
from utils import create_data_loaders, setup_experiment


class DataUtilsTest(unittest.TestCase):
    def test_adjacency_matrices_are_registered_buffers(self):
        system = NeighborhoodSystem(2, 2)

        self.assertEqual(
            {matrix.device.type for matrix in system.adjacency_matrices},
            {"cpu"},
        )
        self.assertEqual(
            {
                name for name in system.state_dict()
                if name.startswith("adjacency_matrix_")
            },
            {f"adjacency_matrix_{index}" for index in range(4)},
        )
        system.to("meta")
        self.assertEqual(
            {matrix.device.type for matrix in system.adjacency_matrices},
            {"meta"},
        )

    def test_tensor_split_loaders_and_seed_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, size in (("train", 5), ("val", 3), ("test", 2)):
                torch.save(
                    {
                        "images": torch.randn(size, 3, 8, 8),
                        "labels": torch.randint(0, 2, (size, 4)),
                    },
                    root / f"{split}.pt",
                )

            config = ExperimentConfig(
                dataset=DatasetConfig(data_root=str(root), batch_size=2, num_workers=0),
                training=TrainingConfig(save_dir=str(root / "runs"), log_dir=str(root / "logs")),
            )
            setup = setup_experiment(config, "smoke")
            first_random_value = torch.rand(1)
            setup_experiment(config, "smoke")
            self.assertTrue(torch.equal(first_random_value, torch.rand(1)))

            loaders = create_data_loaders(config, use_features=False)
            images, labels = next(iter(loaders["train"]))

            self.assertEqual(set(loaders), {"train", "val", "test"})
            self.assertEqual(images.shape[1:], (3, 8, 8))
            self.assertEqual(labels.dtype, torch.float32)
            self.assertEqual(setup["name"], "smoke")


if __name__ == "__main__":
    unittest.main()
