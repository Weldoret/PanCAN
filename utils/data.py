"""DataLoader creation for tensor splits."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandAugment


class Cutout:
    """Mask one square region of an image tensor."""

    def __init__(self, size: int = 16):
        if size < 0:
            raise ValueError("cutout size cannot be negative")
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return image

        height, width = image.shape[-2:]
        size = min(self.size, height, width)
        top = torch.randint(height - size + 1, (), device=image.device).item()
        left = torch.randint(width - size + 1, (), device=image.device).item()
        output = image.clone()
        output[..., top:top + size, left:left + size] = 0
        return output


class ImageTransform:
    """Resize images and apply the paper's training-time augmentations."""

    def __init__(
            self,
            image_size: tuple[int, int],
            train: bool,
            use_augmentation: bool,
            randaugment_num_ops: int,
            randaugment_magnitude: int,
            cutout_size: int):
        self.image_size = image_size
        self.train = train
        self.use_augmentation = use_augmentation
        self.randaugment = RandAugment(
            num_ops=randaugment_num_ops,
            magnitude=randaugment_magnitude,
        ) if randaugment_num_ops else None
        self.cutout = Cutout(cutout_size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            raise ValueError("images must have shape [channels, height, width]")

        image = image.float()
        if tuple(image.shape[-2:]) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.train and self.use_augmentation:
            if self.randaugment is not None:
                image = self.randaugment(image)
            image = self.cutout(image)
        return image


class TensorSplitDataset(Dataset):
    """Tensor split with an optional per-image transform."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor, transform=None):
        self.inputs = inputs
        self.labels = labels.float()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    @property
    def tensors(self):
        """Expose the TensorDataset-compatible backing tensors."""
        return self.inputs, self.labels

    def __getitem__(self, index):
        image = self.inputs[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def _load_split(path, use_features, transform=None):
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
    return TensorSplitDataset(inputs, labels, transform=transform)


def create_data_loaders(config, use_features=False):
    """Load train/val/test datasets from ``<data_root>/<split>.pt`` files."""
    root = Path(config.dataset.data_root)
    loaders = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")

        transform = None
        if not use_features:
            transform = ImageTransform(
                image_size=tuple(config.dataset.image_size),
                train=split == "train",
                use_augmentation=config.dataset.use_augmentation,
                randaugment_num_ops=config.dataset.randaugment_num_ops,
                randaugment_magnitude=config.dataset.randaugment_magnitude,
                cutout_size=config.dataset.cutout_size,
            )
        loaders[split] = DataLoader(
            _load_split(path, use_features, transform=transform),
            batch_size=config.dataset.batch_size,
            shuffle=split == "train",
            num_workers=config.dataset.num_workers,
            pin_memory=str(config.training.device).startswith("cuda"),
        )
    return loaders
