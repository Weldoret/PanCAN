"""Dataset-specific configuration defaults."""

from dataclasses import dataclass

from .base_config import DatasetConfig


@dataclass
class NUSWIDEConfig(DatasetConfig):
    dataset_name: str = "nuswide"
    num_classes: int = 81


@dataclass
class VOC2007Config(DatasetConfig):
    dataset_name: str = "voc2007"
    num_classes: int = 20


@dataclass
class COCOConfig(DatasetConfig):
    dataset_name: str = "coco"
    num_classes: int = 80
