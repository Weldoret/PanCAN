"""Experiment initialization."""

import random
from pathlib import Path

import numpy as np
import torch


def setup_experiment(config, experiment_name):
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path(config.training.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)
    return {
        "name": experiment_name,
        "seed": seed,
        "device": config.training.device,
    }
