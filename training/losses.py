"""Losses used by the training loop."""

import torch
from torch import nn
from torch.nn import functional as F


class GroupedMultiLabelLoss(nn.Module):
    """Group-wise multi-label cross-entropy with paper-style sample groups."""

    def __init__(self, class_groups, group_weights=None):
        super().__init__()
        self.class_groups = tuple(tuple(group) for group in class_groups)
        if not self.class_groups:
            raise ValueError("class_groups must not be empty")
        if group_weights is None:
            group_weights = [1.0] * len(self.class_groups)
        if len(group_weights) != len(self.class_groups):
            raise ValueError("group_weights must match class_groups")
        self.register_buffer(
            "group_weights",
            torch.as_tensor(group_weights, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = (labels > 0).to(dtype=logits.dtype)
        total = logits.new_zeros(())
        weight_total = logits.new_zeros(())

        for group, weight in zip(self.class_groups, self.group_weights):
            indices = torch.as_tensor(group, device=logits.device)
            group_labels = targets.index_select(1, indices)
            sample_mask = group_labels.any(dim=1)
            if not sample_mask.any():
                continue
            group_loss = F.binary_cross_entropy_with_logits(
                logits.index_select(1, indices)[sample_mask],
                group_labels[sample_mask],
            )
            total = total + weight.to(logits.device) * group_loss
            weight_total = weight_total + weight.to(logits.device)

        if weight_total.item() == 0:
            return F.binary_cross_entropy_with_logits(logits, targets)
        return total / weight_total
