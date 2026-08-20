

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import math


class CenteredSelfAttention(nn.Module):
    """Centered self-attention module"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, 
                center_feature: torch.Tensor, 
                connected_features: torch.Tensor) -> torch.Tensor:
        if center_feature.dim() == 1:
            center_feature = center_feature.unsqueeze(0)
        
        Q = self.query(center_feature)
        K = self.key(connected_features)
        V = self.value(connected_features)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_probs = self.softmax(attention_scores)
        
        max_score_index = torch.argmax(attention_probs, dim=-1, keepdim=True)
        
        if V.dim() == 2:
            V = V.unsqueeze(0)
        
        most_similar_feature = torch.gather(
            V, 
            1, 
            max_score_index.unsqueeze(-1).expand(-1, -1, V.size(-1))
        )
        
        max_attention_prob = attention_probs.gather(1, max_score_index)
        
        weighted_most_similar_feature = most_similar_feature * max_attention_prob.unsqueeze(-1)
        
        return weighted_most_similar_feature.squeeze(1)


class AnchorBoxGenerator(nn.Module):
    """Anchor box generator"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 anchor_sizes: List[Tuple[int, int]],
                 strides: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.grid_rows, self.grid_cols = grid_size
        self.anchor_sizes = anchor_sizes
        self.strides = strides if strides else (1, 1)
    
    def generate_anchor_boxes(self) -> List[Tuple[int, int, int, int]]:
        """
        Generate anchor boxes
        
        Returns:
            List of anchor boxes, each as (top, left, bottom, right)
        """
        anchor_boxes = []
        stride_h, stride_w = self.strides
        
        for i in range(0, self.grid_rows, stride_h):
            for j in range(0, self.grid_cols, stride_w):
                for (h, w) in self.anchor_sizes:
                    top = i
                    left = j
                    bottom = min(i + h, self.grid_rows)
                    right = min(j + w, self.grid_cols)
                    
                    if bottom > top and right > left:
                        anchor_boxes.append((top, left, bottom, right))
        
        return anchor_boxes
    
    def get_cells_in_anchor(self, 
                           anchor_box: Tuple[int, int, int, int]) -> List[int]:
        """
        Get cell indices within anchor box
        
        Args:
            anchor_box: Anchor box (top, left, bottom, right)
            
        Returns:
            List of cell indices
        """
        top, left, bottom, right = anchor_box
        cell_indices = []
        
        for i in range(top, bottom):
            for j in range(left, right):
                cell_idx = i * self.grid_cols + j
                cell_indices.append(cell_idx)
        
        return cell_indices


class SubregionSelector(nn.Module):
    """Subregion selector"""
    
    def __init__(self, 
                 feature_dim: int,
                 top_k: int = 3,
                 use_nms: bool = True):
        """
        Initialize subregion selector
        
        Args:
            feature_dim: Feature dimension
            top_k: Number of top-k anchors to select
            use_nms: Whether to use non-maximum suppression
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.use_nms = use_nms
        
    def forward(self, 
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select saliency high anchors
        
        Args:
            features: Features [batch_size, num_cells, feature_dim]
            
        Returns:
            anchors: Anchor indices [batch_size, top_k]
            saliency: Saliency scores [batch_size, num_cells]
        """
        batch_size, num_cells, feature_dim = features.shape
        
        saliency = torch.norm(features, p=2, dim=-1)
        
        if self.use_nms:
            anchors = self.non_maximum_suppression(saliency)
        else:
            _, anchors = torch.topk(saliency, k=self.top_k, dim=-1)
        
        return anchors, saliency
    
    def non_maximum_suppression(self, 
                               saliency: torch.Tensor,
                               suppression_radius: int = 2) -> torch.Tensor:
        """
        Non-maximum suppression
        
        Args:
            saliency: Saliency scores [batch_size, num_cells]
            suppression_radius: Suppression radius
            
        Returns:
            Anchor indices [batch_size, top_k]
        """
        batch_size, num_cells = saliency.shape
        device = saliency.device
        
        anchors_list = []
        
        for b in range(batch_size):
            batch_saliency = saliency[b]
            selected_indices = []
            
            remaining_saliency = batch_saliency.clone()
            
            for _ in range(self.top_k):
                if remaining_saliency.max() <= 0:
                    break
                
                max_idx = torch.argmax(remaining_saliency).item()
                selected_indices.append(max_idx)
                
                max_idx = min(max_idx, num_cells - 1)
                
                for i in range(max(0, max_idx - suppression_radius), min(num_cells, max_idx + suppression_radius + 1)):
                    if i != max_idx:
                        remaining_saliency[i] = 0
            
            while len(selected_indices) < self.top_k:
                selected_indices.append(0)
            
            anchors_list.append(torch.tensor(selected_indices, device=device))
        
        return torch.stack(anchors_list, dim=0)


class MultiHeadScaleAttention(nn.Module):
    """Multi-head scale attention"""
    
    def __init__(self, 
                 feature_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        _, num_keys, _ = key.shape
        
        Q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, num_queries, self.feature_dim)
        
        output = self.out_proj(context)
        
        return output


class MultiScaleFeatureAggregator(nn.Module):
    """Cascaded micro-to-macro feature aggregation from PanCAN Eqs. (11)-(18)."""
    
    def __init__(self,
                 feature_dim: int,
                 scales: List[Tuple[int, int]],
                 anchor_sizes: List[Tuple[int, int]],
                 top_k: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_heads: Optional[int] = None,
                 stride: Optional[int | Tuple[int, int]] = None):
        super().__init__()
        if attention_heads is not None:
            num_heads = attention_heads

        self.feature_dim = feature_dim
        self.scales = scales
        # Retained for configuration compatibility; CSCAMN membership is
        # determined by the scale transition and its overlap stride.
        self.anchor_sizes = anchor_sizes
        self.top_k = top_k
        self.stride = self._normalize_stride(stride)

        if not scales:
            raise ValueError("scales must contain at least one grid")
        if any(rows < 1 or cols < 1 for rows, cols in scales):
            raise ValueError("scale dimensions must be positive")

        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        self.fusion = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.multi_head_attention = MultiHeadScaleAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self,
                features: torch.Tensor,
                grid_size: Tuple[int, int],
                coarse_context_processors: Optional[List[nn.Module]] = None,
                global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_cells, feature_dim = features.shape
        grid_rows, grid_cols = grid_size
        if num_cells != grid_rows * grid_cols:
            raise ValueError("Number of features doesn't match grid size")
        if tuple(self.scales[0]) != (grid_rows, grid_cols):
            raise ValueError("the first configured scale must match grid_size")
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, got {feature_dim}"
            )
        if global_features is None or global_features.shape != (batch_size, feature_dim):
            raise ValueError(
                f"global_features must have shape {(batch_size, feature_dim)}"
            )
        if coarse_context_processors is not None and len(coarse_context_processors) != len(self.scales) - 1:
            raise ValueError("one context processor is required for each coarser scale")

        current = features
        current_rows, current_cols = grid_rows, grid_cols
        scale_tokens = [current.sum(dim=1)]

        for scale_index, (target_rows, target_cols) in enumerate(self.scales[1:]):
            groups = self._build_groups(
                current_rows,
                current_cols,
                target_rows,
                target_cols,
                stride=self.stride,
            )
            anchor_indices = self._select_anchor_indices(
                current, groups, current_rows, current_cols
            )

            macro_features = []
            for group_idx, group in enumerate(groups):
                cell_indices = torch.tensor(group, device=features.device)
                cells = current[:, cell_indices, :]
                anchor = current[
                    torch.arange(batch_size, device=features.device),
                    anchor_indices[:, group_idx],
                ]

                query = self.query_projection(anchor).unsqueeze(1)
                keys = self.key_projection(cells)
                values = self.value_projection(cells)
                scores = torch.matmul(query, keys.transpose(1, 2))
                scores = scores / math.sqrt(self.feature_dim)
                weights = F.softmax(scores, dim=-1)
                fused = torch.matmul(weights, values).squeeze(1)

                combined = torch.cat((fused, anchor), dim=-1).unsqueeze(-1)
                macro = self.fusion(combined).squeeze(-1)
                macro_features.append(macro)

            current = torch.stack(macro_features, dim=1)
            if coarse_context_processors is not None:
                current = coarse_context_processors[scale_index](current)
            current_rows, current_cols = target_rows, target_cols
            scale_tokens.append(current.sum(dim=1))

        scale_tokens.append(global_features)
        tokens = torch.stack(scale_tokens, dim=1)
        query = tokens.mean(dim=1, keepdim=True)
        return self.multi_head_attention(query, tokens, tokens).squeeze(1)

    @staticmethod
    def _build_groups(
        current_rows: int,
        current_cols: int,
        target_rows: int,
        target_cols: int,
        stride: int | Tuple[int, int] = 2,
    ) -> List[List[int]]:
        """Build stride-aware overlapping groups for one scale transition.

        PanCAN defines the coarser grid by overlap strides and reports a 2x2
        scale interval. The paper does not spell out boundary padding or an
        exact window size, so this uses the smallest window that both starts
        from the configured stride and guarantees overlap plus full coverage.
        The requested target grid remains authoritative for non-divisible
        transitions such as 5 -> 3.
        """
        if (
            current_rows < 1
            or current_cols < 1
            or target_rows < 1
            or target_cols < 1
        ):
            raise ValueError("grid dimensions must be positive")
        if target_rows > current_rows or target_cols > current_cols:
            raise ValueError("scales must progress from fine to coarse")

        row_stride, col_stride = MultiScaleFeatureAggregator._normalize_stride(stride)
        row_starts, row_window = MultiScaleFeatureAggregator._window_layout(
            current_rows, target_rows, row_stride
        )
        col_starts, col_window = MultiScaleFeatureAggregator._window_layout(
            current_cols, target_cols, col_stride
        )

        groups = []
        for row_start in row_starts:
            for col_start in col_starts:
                groups.append([
                    source_row * current_cols + source_col
                    for source_row in range(row_start, row_start + row_window)
                    for source_col in range(col_start, col_start + col_window)
                ])
        return groups

    @staticmethod
    def _normalize_stride(
        stride: Optional[int | Tuple[int, int]],
    ) -> Tuple[int, int]:
        if stride is None:
            return 2, 2
        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = tuple(stride)
        if len(stride) != 2 or any(
            not isinstance(value, int) or value < 1 for value in stride
        ):
            raise ValueError("stride must be a positive int or pair of ints")
        return stride

    @staticmethod
    def _window_layout(length: int, count: int, stride: int) -> Tuple[List[int], int]:
        if count < 1 or count > length:
            raise ValueError("target grid must fit inside source grid")

        # ponytail: infer only the missing window size; keep the paper's
        # reported target hierarchy and use the smallest overlapping window.
        minimum_overlap_window = (length + count - 1) // count
        window = min(
            length - count + 1,
            max(stride + 1, minimum_overlap_window),
        )
        if count == 1:
            return [0], window

        max_start = length - window
        starts = [
            int(round(index * max_start / (count - 1)))
            for index in range(count)
        ]
        return starts, window

    @staticmethod
    def _select_anchor_indices(
        features: torch.Tensor,
        groups: List[List[int]],
        rows: int,
        cols: int,
        suppression_radius: int = 1,
    ) -> torch.Tensor:
        """Select salient anchors while suppressing nearby candidates."""
        batch_size = features.size(0)
        saliency = torch.linalg.vector_norm(features, dim=-1)
        selected = torch.empty(
            batch_size, len(groups), dtype=torch.long, device=features.device
        )

        for batch_idx in range(batch_size):
            candidates = []
            for group_idx, group in enumerate(groups):
                indices = torch.tensor(group, device=features.device)
                order = torch.argsort(saliency[batch_idx, indices], descending=True)
                ranked = [group[index] for index in order.tolist()]
                candidates.append((saliency[batch_idx, ranked[0]].item(), group_idx, ranked))

            chosen_coordinates = []
            for _, group_idx, ranked in sorted(candidates, reverse=True):
                chosen = ranked[0]
                for candidate in ranked:
                    row, col = divmod(candidate, cols)
                    if all(
                        max(abs(row - old_row), abs(col - old_col))
                        > suppression_radius
                        for old_row, old_col in chosen_coordinates
                    ):
                        chosen = candidate
                        break
                selected[batch_idx, group_idx] = chosen
                chosen_coordinates.append(divmod(chosen, cols))

        return selected
    
    def aggregate_anchor_features(self,
                                 anchor_features: torch.Tensor,
                                 anchors: torch.Tensor) -> torch.Tensor:
        batch_size, num_cells, feature_dim = anchor_features.shape
        top_k = anchors.size(1)
        
        anchor_feat_list = []
        
        for b in range(batch_size):
            batch_anchors = anchors[b]
            batch_features = anchor_features[b]
            
            batch_anchors = torch.clamp(batch_anchors, 0, num_cells - 1)
            
            anchor_feat = batch_features[batch_anchors]
            anchor_feat_list.append(anchor_feat)
        
        all_anchor_feat = torch.stack(anchor_feat_list, dim=0)
        
        query = all_anchor_feat[:, 0:1, :]
        key = all_anchor_feat
        value = all_anchor_feat
        
        aggregated = self.multi_head_attention(query, key, value)
        
        return aggregated.squeeze(1)
