"""Deterministic label-cooccurrence grouping for grouped classification."""

import torch


def build_label_groups(labels: torch.Tensor, num_groups: int = 5):
    """Partition labels into balanced groups using normalized co-occurrence."""
    if labels.dim() != 2:
        raise ValueError("labels must have shape [samples, classes]")
    if num_groups < 1:
        raise ValueError("num_groups must be positive")

    present = labels > 0
    num_classes = present.size(1)
    if num_classes < 1:
        raise ValueError("labels must contain at least one class")

    group_count = min(num_groups, num_classes)
    frequencies = present.sum(dim=0).float()
    cooccurrence = present.float().t() @ present.float()
    denominator = torch.sqrt(frequencies[:, None] * frequencies[None, :]).clamp_min(1.0)
    similarity = cooccurrence / denominator
    similarity.fill_diagonal_(0)

    groups = [[label] for label in range(num_classes)]
    while len(groups) > group_count:
        left, right = max(
            (
                (left, right)
                for left in range(len(groups))
                for right in range(left + 1, len(groups))
            ),
            key=lambda pair: (
                sum(
                    float(similarity[first, second])
                    for first in groups[pair[0]]
                    for second in groups[pair[1]]
                ) / (len(groups[pair[0]]) * len(groups[pair[1]])),
                -pair[0],
                -pair[1],
            ),
        )
        groups[left].extend(groups.pop(right))

    groups = [sorted(group) for group in groups]
    sample_counts = torch.stack([
        present[:, group].any(dim=1).sum().float() for group in groups
    ])
    weights = labels.size(0) / sample_counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    return groups, weights.tolist()
