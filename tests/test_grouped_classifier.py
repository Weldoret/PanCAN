import unittest
import tempfile

import torch
from torch import nn

from config import ExperimentConfig, TrainingConfig
from models.network import MultiLabelClassifier
from training import TrainingManager
from training.grouping import build_label_groups
from training.losses import GroupedMultiLabelLoss


class GroupedClassifierTest(unittest.TestCase):
    def test_groups_follow_cooccurrence_and_cover_each_class(self):
        labels = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ])

        groups, weights = build_label_groups(labels, num_groups=2)

        self.assertEqual(sorted(index for group in groups for index in group), [0, 1, 2, 3])
        self.assertTrue(any(set((0, 1)).issubset(group) for group in groups))
        self.assertTrue(any(set((2, 3)).issubset(group) for group in groups))
        self.assertEqual(len(weights), 2)

    def test_group_heads_restore_original_class_order(self):
        classifier = MultiLabelClassifier(
            input_dim=4,
            num_classes=4,
            dropout_rate=0.0,
            class_groups=[[2, 0], [1, 3]],
        )
        with torch.no_grad():
            classifier.group_classifiers[0].weight.zero_()
            classifier.group_classifiers[0].bias.copy_(torch.tensor([1.0, 2.0]))
            classifier.group_classifiers[1].weight.zero_()
            classifier.group_classifiers[1].bias.copy_(torch.tensor([3.0, 4.0]))

        logits = classifier(torch.zeros(1, 4))
        self.assertTrue(torch.equal(logits, torch.tensor([[2.0, 3.0, 1.0, 4.0]])))

    def test_grouped_loss_backpropagates(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        labels = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        loss = GroupedMultiLabelLoss([[0, 1], [2, 3]])(logits, labels)

        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_training_manager_selects_grouped_objective(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Linear(3, 4)
                self.classifier = MultiLabelClassifier(
                    input_dim=4,
                    num_classes=4,
                    dropout_rate=0.0,
                    class_groups=[[0, 1], [2, 3]],
                )

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                training=TrainingConfig(
                    save_dir=directory,
                    log_dir=directory,
                    device="cpu",
                )
            )
            model = TinyModel()
            trainer = TrainingManager(model, config, torch.device("cpu"))
            loss = trainer._compute_training_loss(
                model(torch.randn(2, 3)),
                torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]),
            )

        self.assertIsInstance(trainer.criterion, GroupedMultiLabelLoss)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
