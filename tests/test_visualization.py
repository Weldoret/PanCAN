import tempfile
import unittest
from pathlib import Path

from utils.visualization import plot_metrics


class VisualizationTest(unittest.TestCase):
    def test_plot_metrics_writes_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plots" / "metrics.png"
            result = plot_metrics({"loss": [2.0, 1.0], "mAP": [0.4, 0.6]}, path)
            self.assertEqual(result, path)
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
