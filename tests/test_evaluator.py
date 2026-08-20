import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DatasetConfig, ExperimentConfig, TrainingConfig
from training.evaluator import Evaluator
from training.metrics import MultiLabelMetrics


class FixedLogitModel(nn.Module):
    num_classes = 4

    def __init__(self):
        super().__init__()
        self.register_buffer("logits", torch.tensor([[2.0, 1.0, -1.0, -2.0]]))

    def forward(self, inputs):
        return self.logits.expand(inputs.size(0), -1)


class EvaluatorTest(unittest.TestCase):
    def test_cf1_and_of1_use_paper_aggregation(self):
        labels = np.array([[1, 1, 1, 0], [0, 0, 0, 1]])
        predictions = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        metrics = MultiLabelMetrics()

        self.assertAlmostEqual(metrics.compute_cf1(labels, predictions), 0.5)
        self.assertAlmostEqual(metrics.compute_of1(labels, predictions), 2 / 3)
        self.assertNotEqual(
            metrics.compute_of1(labels, predictions),
            metrics.compute_f1_score(labels, predictions, average="samples"),
        )

    def _loader(self):
        inputs = torch.zeros(1, 1)
        labels = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        return DataLoader(TensorDataset(inputs, labels), batch_size=1)

    def _evaluator(self, dataset_name):
        config = ExperimentConfig(
            dataset=DatasetConfig(dataset_name=dataset_name),
            training=TrainingConfig(device="cpu"),
        )
        return Evaluator(config)

    def test_non_coco_uses_threshold_predictions(self):
        results = self._evaluator("voc2007").evaluate(
            FixedLogitModel(), self._loader()
        )

        np.testing.assert_array_equal(results["predictions"], [[1.0, 1.0, 0.0, 0.0]])
        self.assertNotIn("top3_cf1", results["metrics"])

    def test_coco_reports_all_and_top3_metrics(self):
        results = self._evaluator("coco").evaluate(
            FixedLogitModel(), self._loader()
        )
        metrics = results["metrics"]

        np.testing.assert_array_equal(results["predictions"], [[1.0, 1.0, 0.0, 0.0]])
        self.assertEqual(metrics["all_cf1"], metrics["cf1"])
        self.assertEqual(metrics["all_map"], metrics["map"])
        self.assertIn("top3_cf1", metrics)
        self.assertNotEqual(metrics["top3_cf1"], metrics["all_cf1"])


if __name__ == "__main__":
    unittest.main()
