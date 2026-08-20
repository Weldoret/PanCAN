import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.trainer import ExponentialMovingAverage
from utils.checkpoint import load_checkpoint, save_checkpoint


class ExponentialMovingAverageTest(unittest.TestCase):
    def test_update_and_restore(self):
        model = nn.Linear(2, 1)
        ema = ExponentialMovingAverage(model, decay=0.5)
        initial_weight = model.weight.detach().clone()

        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)

        expected_weight = (initial_weight + 2.0) / 2
        self.assertTrue(torch.allclose(ema.shadow["weight"], expected_weight))

        ema.apply_to(model)
        self.assertTrue(torch.allclose(model.weight, expected_weight))
        ema.restore(model)
        self.assertTrue(torch.allclose(model.weight, torch.full_like(model.weight, 2.0)))

    def test_checkpoint_round_trip(self):
        model = nn.Linear(2, 1)
        ema = ExponentialMovingAverage(model, decay=0.75)
        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update(model)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pth"
            save_checkpoint(model, path, ema_state_dict=ema.state_dict())
            checkpoint = load_checkpoint(model, path)

        restored = ExponentialMovingAverage(model, decay=0.1)
        restored.load_state_dict(checkpoint["ema_state_dict"])
        self.assertEqual(restored.decay, 0.75)
        for name, value in ema.shadow.items():
            self.assertTrue(torch.equal(restored.shadow[name], value))


if __name__ == "__main__":
    unittest.main()
