"""DataLoader creation for tensor splits."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_split(path, use_features):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        input_key = "features" if use_features else "images"
        if input_key not in payload or "labels" not in payload:
            raise ValueError(f"{path} must contain '{input_key}' and 'labels' tensors")
        inputs, labels = payload[input_key], payload["labels"]
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        inputs, labels = payload
    else:
        raise ValueError(f"{path} must contain a tensor pair or mapping")

    if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError(f"{path} inputs and labels must be tensors")
    if len(inputs) != len(labels):
        raise ValueError(f"{path} has {len(inputs)} inputs but {len(labels)} labels")
    return TensorDataset(inputs, labels.float())


def create_data_loaders(config, use_features=False):
    """Load train/val/test datasets from ``<data_root>/<split>.pt`` files."""
    root = Path(config.dataset.data_root)
    loaders = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")
        loaders[split] = DataLoader(
            _load_split(path, use_features),
            batch_size=config.dataset.batch_size,
            shuffle=split == "train",
            num_workers=config.dataset.num_workers,
            pin_memory=str(config.training.device).startswith("cuda"),
        )
    return loaders
