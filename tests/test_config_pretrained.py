import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from config import DatasetConfig, ExperimentConfig, NetworkConfig, TrainingConfig
from models import BACKBONES, load_pretrained_backbone, load_resnet_backbone


class ConfigAndPretrainedTest(unittest.TestCase):
    def test_paper_training_defaults(self):
        dataset = DatasetConfig()
        training = TrainingConfig()

        self.assertEqual(dataset.batch_size, 6)
        self.assertEqual(dataset.image_size, (400, 500))
        self.assertTrue(dataset.use_augmentation)
        self.assertEqual(training.num_epochs, 200)
        self.assertEqual(training.learning_rate, 1e-4)
        self.assertEqual(training.optimizer, "adamw")
        self.assertEqual(training.ema_decay, 0.9997)

    def test_paper_neighborhood_order_defaults(self):
        config = NetworkConfig()

        self.assertEqual(config.max_order, 2)
        self.assertEqual(config.coarse_max_order, 1)

        with self.assertRaises(ValueError):
            NetworkConfig(max_order=1, coarse_max_order=2)

    def test_config_round_trip_and_backbone_registry(self):
        config = ExperimentConfig(network=NetworkConfig(backbone_name="resnet34"))
        self.assertIs(config.network.network, config.network)
        self.assertEqual(config.network.backbone_feature_dim, 512)
        self.assertEqual(config.network.num_blocks, 80)
        self.assertEqual(set(BACKBONES), {"resnet34", "resnet50", "resnet101"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config.save(path)
            loaded = ExperimentConfig.from_file(path)
            loaded_network = NetworkConfig.from_file(path)
        self.assertEqual(loaded.to_dict(), config.to_dict())
        self.assertEqual(loaded_network.to_dict(), config.network.to_dict())

        with self.assertRaises(ValueError):
            load_pretrained_backbone("cvt_w24", pretrained=False)

    def test_resnet_loader_removes_the_classification_head(self):
        class FakeSequential:
            def __init__(self, *children):
                self.children = children

        class FakeModel:
            def children(self):
                return iter(("stem", "body", "pool", "classifier"))

        calls = []

        def build_resnet34(**kwargs):
            calls.append(kwargs)
            return FakeModel()

        torch = ModuleType("torch")
        torch.__path__ = []
        torch_nn = ModuleType("torch.nn")
        torch_nn.Sequential = FakeSequential
        torchvision = ModuleType("torchvision")
        torchvision.models = SimpleNamespace(
            resnet34=build_resnet34,
            ResNet34_Weights=SimpleNamespace(DEFAULT="default-weights"),
        )

        with patch.dict(
            "sys.modules",
            {"torch": torch, "torch.nn": torch_nn, "torchvision": torchvision},
        ):
            backbone = load_resnet_backbone("resnet34", pretrained=True)

        self.assertEqual(calls, [{"weights": "default-weights"}])
        self.assertEqual(backbone.children, ("stem", "body"))
        self.assertEqual(backbone.feature_dim, 512)


if __name__ == "__main__":
    unittest.main()
