"""PyTorch checkpoint helpers."""

from pathlib import Path

import torch


def save_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, **metadata):
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict(), **metadata}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, checkpoint_path, device="cpu", optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_model(model, model_path):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(model, model_path, device="cpu"):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, filename="checkpoint.pth", **kwargs):
        return save_checkpoint(model, self.save_dir / filename, **kwargs)

    def load_checkpoint(self, model, checkpoint_path, device="cpu", optimizer=None, scheduler=None):
        return load_checkpoint(model, checkpoint_path, device, optimizer, scheduler)
