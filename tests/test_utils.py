import tempfile
import unittest
from pathlib import Path

import torch

from utils.checkpoint import load_checkpoint, save_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        model = torch.nn.Linear(2, 1)
        expected = {name: value.clone() for name, value in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(model, path, epoch=3)
            for parameter in model.parameters():
                parameter.data.zero_()
            checkpoint = load_checkpoint(model, path)

        self.assertEqual(checkpoint["epoch"], 3)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))


if __name__ == "__main__":
    unittest.main()
