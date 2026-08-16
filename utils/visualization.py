"""Headless plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_visualization(figure, save_path, dpi=150):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_metrics(metrics_history, save_path, title="Training Metrics"):
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, values in metrics_history.items():
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            values = [values]
        axis.plot(range(1, len(values) + 1), values, label=name)
    axis.set(title=title, xlabel="Epoch", ylabel="Value")
    axis.grid(alpha=0.3)
    if axis.lines:
        axis.legend()
    return save_visualization(figure, save_path)


def _plot_heatmap(values, save_path=None, title="Heatmap"):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    while getattr(values, "ndim", 0) > 2:
        values = values[0]
    figure, axis = plt.subplots()
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    return save_visualization(figure, save_path) if save_path else figure


def visualize_attention_maps(attention_maps, save_path=None, title="Attention Maps"):
    return _plot_heatmap(attention_maps, save_path, title)


def visualize_neighborhood(neighborhood, save_path=None, title="Neighborhood"):
    return _plot_heatmap(neighborhood, save_path, title)


def visualize_multi_scale_features(features, save_path=None, title="Multi-scale Features"):
    return _plot_heatmap(features, save_path, title)
