"""Small, serializable configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping


def _read_mapping(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {path} must be a mapping")
    return data


def _write_mapping(path: str | Path, data: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        # JSON is valid YAML 1.2 and keeps configuration usable without PyYAML.
        destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        destination.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")


class ConfigMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__} configuration must be a mapping")
        return cls(**dict(data))

    @classmethod
    def from_file(cls, path: str | Path):
        return cls.from_dict(_read_mapping(path))

    def save(self, path: str | Path) -> None:
        _write_mapping(path, self.to_dict())


@dataclass
class NetworkConfig(ConfigMixin):
    model_name: str = "MultiScaleContextAwareNetwork"
    backbone_name: str = "resnet101"
    backbone_pretrained: bool = True
    backbone_feature_dim: int | None = None
    num_classes: int = 81
    grid_rows: int = 8
    grid_cols: int = 10
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.5
    num_directions: int = 4
    context_layers: int = 3
    max_order: int = 2
    coarse_max_order: int = 1
    attention_heads: int = 8
    attention_dropout: float = 0.1
    random_walk_threshold: float = 0.71
    scales: list[tuple[int, int]] | None = None
    anchor_sizes: list[tuple[int, int]] = field(default_factory=lambda: [(2, 2), (3, 3)])
    sliding_window_stride: int | tuple[int, int] = 2
    kernel_feature_dims: list[int] | None = None
    final_feature_dim: int | None = None
    classifier_dropout: float = 0.5
    use_grouped_fc: bool = True
    num_groups: int = 5

    FEATURE_DIMS: ClassVar[dict[str, int]] = {
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }

    def __post_init__(self) -> None:
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid dimensions must be positive")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.num_directions not in (4, 8):
            raise ValueError("num_directions must be 4 or 8")
        if self.context_layers < 1:
            raise ValueError("context_layers must be positive")
        if self.max_order < 1 or self.coarse_max_order < 1:
            raise ValueError("neighborhood orders must be positive")
        if self.coarse_max_order > self.max_order:
            raise ValueError("coarse_max_order cannot exceed max_order")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if self.num_groups < 1:
            raise ValueError("num_groups must be positive")

        stride = self.sliding_window_stride
        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            try:
                stride = tuple(stride)
            except TypeError as exc:
                raise ValueError(
                    "sliding_window_stride must be an int or a pair of ints"
                ) from exc
            if len(stride) != 2:
                raise ValueError("sliding_window_stride must contain two values")
        if any(not isinstance(value, int) or value < 1 for value in stride):
            raise ValueError("sliding_window_stride values must be positive integers")
        self.sliding_window_stride = stride

        if self.backbone_feature_dim is None:
            self.backbone_feature_dim = self.FEATURE_DIMS.get(self.backbone_name, 2048)
        if self.final_feature_dim is None:
            self.final_feature_dim = self.backbone_feature_dim * 2
        if self.kernel_feature_dims is None:
            self.kernel_feature_dims = [self.final_feature_dim]
        if self.scales is None:
            rows, cols = self.grid_rows, self.grid_cols
            self.scales = []
            while True:
                self.scales.append((rows, cols))
                if (rows, cols) == (1, 1):
                    break
                rows, cols = max(1, (rows + 1) // 2), max(1, (cols + 1) // 2)

        self.scales = [tuple(scale) for scale in self.scales]
        self.anchor_sizes = [tuple(size) for size in self.anchor_sizes]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkConfig":
        if not isinstance(data, Mapping):
            raise TypeError("Network configuration must be a mapping")
        if "network" in data and isinstance(data["network"], Mapping):
            data = data["network"]
        return super().from_dict(data)

    @property
    def network(self) -> "NetworkConfig":
        """Support model code that receives either NetworkConfig or ExperimentConfig."""
        return self

    @property
    def num_blocks(self) -> int:
        return self.grid_rows * self.grid_cols


@dataclass
class DatasetConfig(ConfigMixin):
    dataset_name: str = "nuswide"
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    num_classes: int = 81

    def __post_init__(self) -> None:
        if self.dataset_name not in {"nuswide", "voc2007", "coco"}:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers cannot be negative")


@dataclass
class TrainingConfig(ConfigMixin):
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    device: str = "cpu"
    seed: int = 42
    save_dir: str = "./experiments"
    log_dir: str = "./logs"
    loss_function: str = "bce"
    scheduler: str = "plateau"
    monitor: str = "mAP"
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.0
    save_best_only: bool = True
    save_frequency: int = 1
    accumulation_steps: int = 1
    use_amp: bool = False
    momentum: float = 0.9
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    asymmetric_gamma_neg: float = 4.0
    asymmetric_gamma_pos: float = 1.0
    group_l2: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_epochs < 1 or self.learning_rate <= 0:
            raise ValueError("num_epochs and learning_rate must be positive")
        if self.optimizer.lower() not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.accumulation_steps < 1 or self.save_frequency < 1:
            raise ValueError("accumulation_steps and save_frequency must be positive")
        if self.group_l2 < 0:
            raise ValueError("group_l2 must be non-negative")


@dataclass
class ExperimentConfig(ConfigMixin):
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.network, Mapping):
            self.network = NetworkConfig.from_dict(self.network)
        if isinstance(self.dataset, Mapping):
            self.dataset = DatasetConfig.from_dict(self.dataset)
        if isinstance(self.training, Mapping):
            self.training = TrainingConfig.from_dict(self.training)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(data, Mapping):
            raise TypeError("Experiment configuration must be a mapping")
        return cls(
            network=NetworkConfig.from_dict(data.get("network", {})),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
        )
